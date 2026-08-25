#!/usr/bin/env python
import argparse
import dataclasses
import logging
import os
import pathlib
import queue
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)

import lmdb
import torch
import tqdm
import transformers
from d3text import corpus, logs, utils
from d3text.embeddings_store import tensor_to_bytes

logger = logging.getLogger(__name__)

CPU_COUNT = os.cpu_count() or 1
COMP_THREADS = max(1, CPU_COUNT // 2)
MAX_BACKLOG = max(8, COMP_THREADS * 2)

# The whole corpus measures 100.8 GiB through this store's codec, so the 100 GiB
# this used to reserve ran out near the end of a full pass. On Linux `map_size`
# reserves address space rather than allocating it, and LMDB writes the file
# sparsely, so the headroom costs nothing until the pages are written.
DEFAULT_MAP_SIZE_GIB = 256.0


class StoreFullError(RuntimeError):
    """The LMDB ran out of `map_size` before every document was written."""


@dataclasses.dataclass
class WriterState:
    """How the writer thread reports a failure back to `main`.

    A thread has no return value and an exception raised inside one is invisible
    to the caller, so the writer records the failure here, keeps draining its
    queue so no producer can block on it, and lets `main` raise it after the
    join.
    """

    failure: StoreFullError | None = None


def read_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("base_model")
    p.add_argument("output_path")
    p.add_argument("datasets", nargs="+")
    p.add_argument(
        "-f",
        "--force-regenerate",
        action="store_true",
        help="re-embed documents already stored in the output LMDB",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="token windows per forward pass; tune for your VRAM",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="tokens per window (default: the base model's context window)",
    )
    p.add_argument("--commit_every", type=int, default=100)
    p.add_argument(
        "--map_size",
        type=float,
        default=DEFAULT_MAP_SIZE_GIB,
        help=(
            "GiB of address space to reserve for the LMDB; a pass that needs "
            "more than this stops and says so"
        ),
    )
    p.add_argument(
        "--stream_batch", type=int, default=1000
    )  # rows per Polars slice
    return p.parse_args()


def window_size(
    max_length: int | None, model_config: transformers.PretrainedConfig
) -> int:
    """Resolve the number of tokens per window `embed_document` splits into.

    The tokenizer cannot be asked for this. `model_max_length` is a ~1e30
    sentinel whenever the tokenizer config declares no limit — which is the
    case for the default base model — and `split_and_tokenize` pads *to*
    `max_length`, so that sentinel asks for an impossible tensor. The position
    embeddings are the real cap: a longer window indexes past the table.
    """
    limit: int = model_config.max_position_embeddings
    if max_length is None:
        return limit
    if not 1 <= max_length <= limit:
        msg = (
            f"--max_length must be between 1 and {limit}, the context window "
            f"of {model_config.name_or_path}; got {max_length}."
        )
        raise ValueError(msg)
    return max_length


def stored_keys(env: lmdb.Environment) -> set[bytes]:
    """The pubmed ids already embedded in `env`.

    Keys only: the values are the compressed embeddings, and pulling those in
    just to test for presence would defeat the point of skipping them.
    """
    with env.begin() as txn:
        return set(txn.cursor().iternext(keys=True, values=False))


def store_full(env: lmdb.Environment, key: bytes) -> StoreFullError:
    budget = env.info()["map_size"]
    return StoreFullError(
        f"the embeddings store at {env.path()} ran out of its map_size of "
        f"{budget:,} bytes ({budget / 1024**3:.1f} GiB) while writing document "
        f"{key.decode()}. The documents already committed are kept and are "
        f"skipped on a rerun, so rerunning with a larger --map_size resumes "
        f"from them."
    )


def writer_thread(
    env: lmdb.Environment,
    in_q: queue.Queue[tuple[bytes, bytes]],
    stop_evt: threading.Event,
    commit_every: int,
    pbar_written: tqdm.tqdm,
    state: WriterState,
) -> None:
    tdb = env.begin(write=True)
    n_since = 0
    try:
        while True:
            if stop_evt.is_set() and in_q.empty():
                break
            try:
                k, v = in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if state.failure is not None:
                # The map is full and the transaction is closed, so this value
                # cannot be stored. Draining it anyway is what lets a producer
                # blocked on a full queue reach its own stop check.
                continue

            try:
                tdb.put(k, v)
            except lmdb.MapFullError:
                state.failure = store_full(env, k)
                stop_evt.set()
                # Try to keep this transaction rather than abandoning it: the
                # commit needs pages of its own and may run out as well, but
                # when it fits it saves up to `commit_every` documents of GPU
                # time from being embedded again.
                try:
                    tdb.commit()
                except lmdb.Error:
                    tdb.abort()
                env.sync()
                continue

            n_since += 1
            pbar_written.update(1)
            if n_since >= commit_every:
                tdb.commit()
                tdb = env.begin(write=True)
                n_since = 0

        try:
            tdb.commit()
        except lmdb.Error:
            pass
        env.sync()
    except Exception:
        try:
            tdb.abort()
        except Exception:
            pass
        raise


def main() -> None:
    logs.configure()

    # help CUDA memory fragmentation a bit
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = read_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    model = (
        transformers.AutoModel.from_pretrained(args.base_model)
        .to(device)
        .eval()
    )
    max_len = window_size(args.max_length, model.config)

    # LMDB env
    env = lmdb.open(args.output_path, map_size=int(args.map_size * 1024**3))

    # Snapshot taken before any writing, so a document is judged against what
    # a *previous* run stored, not against this run's own output.
    already_embedded = set() if args.force_regenerate else stored_keys(env)

    # Shared across datasets only because the first failure ends the run: the
    # writer that recorded it is the last one started.
    writer_state = WriterState()

    for dataset in args.datasets:
        path = pathlib.Path(dataset)
        logger.info("\nProcessing %s", path)

        total_rows, row_iter = corpus.stream_rows(path, args.stream_batch)
        skipped = 0

        # In-flight compression jobs -> the pmid key each will be stored under.
        # Local to the dataset: a shared dict would let one dataset's undrained
        # leftovers be written while the next is processed.
        futures: dict[Future[bytes], bytes] = {}

        # queues + bars
        out_q: queue.Queue[tuple[bytes, bytes]] = queue.Queue(maxsize=124)
        stop_evt = threading.Event()

        pbar_emb = tqdm.tqdm(
            total=total_rows,
            desc="Embedded",
            position=0,
            leave=False,
            dynamic_ncols=True,
        )
        pbar_written = tqdm.tqdm(
            total=total_rows,
            desc="Written ",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        # start writer
        wt = threading.Thread(
            target=writer_thread,
            args=(
                env,
                out_q,
                stop_evt,
                args.commit_every,
                pbar_written,
                writer_state,
            ),
            daemon=True,
        )
        wt.start()

        # compression pool
        with (
            ThreadPoolExecutor(max_workers=COMP_THREADS) as pool,
            torch.inference_mode(),
        ):
            for pmid, text in row_iter:
                if stop_evt.is_set():
                    break

                key = str(pmid).encode()
                if key in already_embedded:
                    skipped += 1
                    pbar_emb.update(1)
                    pbar_written.total = total_rows - skipped
                    continue

                emb = utils.embed_document(
                    text,
                    tokenizer=tokenizer,
                    model=model,
                    stride=20,
                    batch_size=args.batch_size,
                    max_len=max_len,
                )
                pbar_emb.update(1)

                # submit for compression
                f = pool.submit(tensor_to_bytes, emb)
                futures[f] = key

                if len(futures) >= MAX_BACKLOG:
                    done, _ = wait(
                        list(futures.keys()), return_when=FIRST_COMPLETED
                    )
                    for d in done:
                        out_q.put((futures.pop(d), d.result()))

            # Drain unconditionally: the in-loop flush above is what keeps the
            # backlog *below* MAX_BACKLOG, so repeating that guard here would
            # be false exactly when there is still work in flight. A dataset
            # shorter than MAX_BACKLOG would then write nothing at all.
            for done_future in as_completed(list(futures)):
                if stop_evt.is_set():
                    # The writer stopped consuming (map full), so further puts
                    # would block forever once out_q fills.
                    break
                out_q.put((futures.pop(done_future), done_future.result()))

        # signal writer to finish; join
        stop_evt.set()
        wt.join()

        # close bars
        pbar_emb.close()
        pbar_written.close()

        if writer_state.failure is not None:
            break

        if skipped:
            logger.info(
                "Skipped %d documents already embedded in %s; "
                "pass -f to re-embed them.",
                skipped,
                args.output_path,
            )

    env.close()

    # A truncated store must not be reachable from a command that reported
    # success: the resume path reads every document that did not fit as one
    # already embedded, so the next run walks into the same wall.
    if writer_state.failure is not None:
        raise writer_state.failure

    logger.info("Done.")


if __name__ == "__main__":
    main()
