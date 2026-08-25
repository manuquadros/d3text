#!/usr/bin/env python
import argparse
import dataclasses
import logging
import math
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
    to the caller, so the writer records the failure here and lets `main` raise
    it after the join. Whether the writer can carry on draining its queue (a
    full map) or cannot (anything else), it is `stop_evt` that tells a producer
    to stop waiting for it.
    """

    failure: Exception | None = None


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


def map_size_bytes(map_size: float) -> int:
    """Resolve `--map_size` in GiB to the reservation `lmdb.open` takes.

    A value that does not come out as at least one byte has to be refused
    here, because LMDB does not refuse it: it reads a `map_size` of zero as
    "keep the size this store already has", which for a new store is LMDB's
    own 1 MiB default. The run would then spend its GPU time embedding and
    die at the first write against a budget nobody asked for.

    There is no floor above one byte. Any positive reservation is honoured
    literally, and a store that outgrows it stops and names the budget, so a
    map too small to be useful already fails loudly; a floor would have to
    guess at a document's embedded size from a hidden width and a token count
    that are not known when the flag is read.
    """
    reserved = int(map_size * 1024**3) if math.isfinite(map_size) else 0
    if reserved < 1:
        msg = (
            f"--map_size must be a finite number of GiB reserving at least "
            f"one byte; got {map_size}. LMDB reads a non-positive map_size as "
            f"the size the store already has, which for a new store is its "
            f"1 MiB default."
        )
        raise ValueError(msg)
    return reserved


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
    """Drain `in_q` into `env`, and set `stop_evt` however this ends.

    This thread is the queue's only consumer, so a producer waiting for room in
    a full queue is really waiting for this thread; if it dies without saying
    so, that wait never ends. Every exit therefore goes through `stop_evt`, and
    every failure is recorded for `main` to raise after the join.
    """
    tdb: lmdb.Transaction | None = None
    try:
        tdb = env.begin(write=True)
        n_since = 0
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
                # Committing what this transaction holds cannot rescue it:
                # LMDB marks a transaction invalid on the `put` that overflows
                # the map, so its `commit` answers `BadTxnError`. Up to
                # `commit_every - 1` documents are embedded again on the rerun.
                state.failure = store_full(env, k)
                stop_evt.set()
                tdb.abort()
                env.sync()
                continue

            n_since += 1
            pbar_written.update(1)
            if n_since >= commit_every:
                tdb.commit()
                tdb = env.begin(write=True)
                n_since = 0

        if state.failure is None:
            tdb.commit()
        env.sync()
    except Exception as exc:
        if tdb is not None:
            try:
                tdb.abort()
            except lmdb.Error:
                pass
        if state.failure is None:
            state.failure = exc
    finally:
        stop_evt.set()


def put_or_stop(
    out_q: queue.Queue[tuple[bytes, bytes]],
    item: tuple[bytes, bytes],
    stop_evt: threading.Event,
) -> bool:
    """Hand `item` to the writer; return False once the writer has stopped.

    A plain `put` on a full queue waits for a consumer that may already be
    gone, and setting an event does not wake it. Breaking the wait into
    timeouts is what lets `stop_evt` be seen at all.
    """
    while not stop_evt.is_set():
        try:
            out_q.put(item, timeout=0.1)
        except queue.Full:
            continue
        return True
    return False


def main() -> None:
    logs.configure()

    # help CUDA memory fragmentation a bit
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = read_args()
    map_size = map_size_bytes(args.map_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = utils.load_fast_tokenizer(args.base_model)
    model = (
        transformers.AutoModel.from_pretrained(args.base_model)
        .to(device)
        .eval()
    )
    max_len = window_size(args.max_length, model.config)

    # LMDB env
    env = lmdb.open(args.output_path, map_size=map_size)

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
                        item = (futures.pop(d), d.result())
                        if not put_or_stop(out_q, item, stop_evt):
                            break

            # Drain unconditionally: the in-loop flush above is what keeps the
            # backlog *below* MAX_BACKLOG, so repeating that guard here would
            # be false exactly when there is still work in flight. A dataset
            # shorter than MAX_BACKLOG would then write nothing at all.
            for done_future in as_completed(list(futures)):
                item = (futures.pop(done_future), done_future.result())
                if not put_or_stop(out_q, item, stop_evt):
                    break

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
    # success: the resume path reads every document that was written as one
    # already embedded, so the next run walks past the gap the writer left.
    if writer_state.failure is not None:
        raise writer_state.failure

    logger.info("Done.")


if __name__ == "__main__":
    main()
