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
from d3text.embeddings_store import (
    StoreProvenance,
    read_provenance,
    tensor_to_bytes,
    write_provenance,
)

logger = logging.getLogger(__name__)

CPU_COUNT = os.cpu_count() or 1
COMP_THREADS = max(1, CPU_COUNT // 2)
MAX_BACKLOG = max(8, COMP_THREADS * 2)

# The whole corpus measures 100.8 GiB through this store's codec, so the 100 GiB
# this used to reserve ran out near the end of a full pass. On Linux `map_size`
# reserves address space rather than allocating it, and LMDB writes the file
# sparsely, so the headroom costs nothing until the pages are written.
DEFAULT_MAP_SIZE_GIB = 256.0

# The overlap between consecutive windows, and not a flag: the encodings the
# training run reads are tokenized by `split_and_tokenize`'s own default, and
# a store striding differently from them is a store of different rows.
STRIDE = utils.WINDOW_STRIDE


class StoreFullError(RuntimeError):
    """The LMDB ran out of `map_size` before every document was written."""


@dataclasses.dataclass
class WriterState:
    """How the writer thread reports a failure back to `main`.

    A thread has no return value and an exception raised inside one is
    invisible to the caller, so the writer records the failure here and `main`
    raises it after the join.
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

    The tokenizer cannot be asked: `model_max_length` is a ~1e30 sentinel
    whenever its config declares no limit, and the tokenizer pads *to*
    `max_length`. The position embeddings are the real cap.

    :param max_length: the requested window, or None for the model's own cap.
    :param model_config: the base model's config.
    :return: the window size to use.
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

    Anything under a byte truncates to zero, which LMDB reads as "keep the size
    this store already has" — its own 1 MiB default for a new store, so the run
    dies at the first write against a budget nobody asked for. A floor above
    one byte belongs to `check_map_size_for_one_document`, which knows the
    hidden width and the window this function never sees.

    :param map_size: the reservation, in GiB.
    :return: the reservation in bytes.
    :raises ValueError: if it does not come out as at least one byte.
    """
    reserved = int(map_size * 1024**3) if math.isfinite(map_size) else 0
    if reserved < 1:
        msg = (
            f"--map_size must be a finite number of GiB reserving at least "
            f"one byte; got {map_size}. A reservation smaller than a byte "
            f"truncates to zero, and zero is no error to LMDB: it reads it as "
            f"the size the store already has, for a new store its own 1 MiB "
            f"default. A negative one lmdb.open does refuse, but with an "
            f"OverflowError naming neither this flag nor its value."
        )
        raise ValueError(msg)
    return reserved


def positive_int(name: str, value: int) -> int:
    """Reject a non-positive count before the tokenizer and base model load.

    The three flags fail differently and only one loudly: a negative
    `--stream_batch` step yields no rows at all, so the command writes zero
    documents and reports `Done.`; `--commit_every <= 0` commits once per
    document instead of once per batch, a silent throughput cliff.

    :param name: the flag being validated, for the message.
    :param value: the value given.
    :return: the value.
    :raises ValueError: if it is not positive.
    """
    if value < 1:
        msg = f"--{name} must be a positive integer; got {value}."
        raise ValueError(msg)
    return value


def record_provenance(
    env: lmdb.Environment, provenance: StoreProvenance
) -> None:
    """Stamp `env` with what this run is about to write into it.

    A pass that appends under another model or window produces one LMDB holding
    two kinds of matrix nothing downstream can separate, since the widths agree
    between encoders of the same hidden size. An unstamped store that already
    holds documents is refused for the same reason: what wrote them is unknown.
    `-f` is no way past it — it re-embeds only the documents these datasets
    name, leaving the rest behind under the new stamp.

    :param env: the open LMDB environment.
    :param provenance: what this run will write.
    :raises ValueError: if the store records another geometry, or holds
        documents and records none.
    """
    recorded = read_provenance(env)
    if recorded == provenance:
        return

    if recorded is not None:
        msg = (
            f"{env.path()} was written by {recorded.base_model} at window "
            f"{recorded.max_length}, stride {recorded.stride}, and this run "
            f"writes {provenance.base_model} at window "
            f"{provenance.max_length}, stride {provenance.stride}. One store "
            f"holding both is one no reader can tell apart, and -f does not "
            f"help: it rewrites only the documents these datasets name. "
            f"Build this into a store of its own."
        )
        raise ValueError(msg)

    if env.stat()["entries"]:
        msg = (
            f"{env.path()} holds documents but does not record which model "
            f"wrote them, so nothing can show them to be "
            f"{provenance.base_model} activations. Build this into a store of "
            f"its own; the documents here are readable only by whatever "
            f"wrote them."
        )
        raise ValueError(msg)

    write_provenance(env, provenance)


_PROBE_KEY = b"\x00probe"
_BF16_ITEMSIZE = 2


def check_map_size_for_one_document(
    env: lmdb.Environment, max_len: int, hidden_size: int
) -> None:
    """Refuse a `map_size` that opens but cannot hold one document.

    `lmdb.open` accepts any reservation LMDB can mmap, so a map merely too
    small for the data was caught nowhere until the first real `put`, hours of
    GPU time later. The probe is sized at `max_len * hidden_size` bf16 values —
    one full window uncompressed, a lower bound on one document — and lands in
    a transaction that is aborted either way.

    :param env: the open LMDB environment.
    :param max_len: the resolved window size.
    :param hidden_size: the base model's hidden width.
    :raises StoreFullError: if the probe does not fit.
    """
    txn = env.begin(write=True)
    try:
        txn.put(_PROBE_KEY, bytes(max_len * hidden_size * _BF16_ITEMSIZE))
    except lmdb.MapFullError:
        txn.abort()
        budget = env.info()["map_size"]
        msg = (
            f"{env.path()} was opened with a map_size of {budget:,} bytes "
            f"({budget / 1024**3:.2f} GiB), which cannot hold even one "
            f"document: at {max_len} tokens and a hidden size of "
            f"{hidden_size}, a single window of bf16 activations alone is "
            f"{max_len * hidden_size * _BF16_ITEMSIZE:,} bytes uncompressed, "
            f"before whatever a real document beyond one window adds. Pass a "
            f"larger --map_size."
        )
        raise ValueError(msg)
    else:
        txn.abort()


def stored_keys(env: lmdb.Environment) -> set[bytes]:
    """The pubmed ids already embedded in `env`.

    Keys only: pulling the compressed embeddings in just to test for presence
    would defeat the point of skipping them. The provenance record rides along
    harmlessly, keyed on bytes no pubmed id can spell.

    :param env: the open LMDB environment.
    :return: the keys already stored.
    """
    with env.begin() as txn:
        return set(txn.cursor().iternext(keys=True, values=False))


def store_full(
    env: lmdb.Environment, key: bytes, *, deleting: bool
) -> StoreFullError:
    """Name the operation that ran out, not the only one the queue carries.

    A delete has to grow the map the same way a put does — LMDB rewrites the
    pages it touches rather than editing them in place.

    :param env: the open LMDB environment, for the budget it reports.
    :param key: the key whose write failed.
    :param deleting: whether the failed operation was a delete.
    :return: the error to raise.
    """
    budget = env.info()["map_size"]
    operation = "deleting the stale entry for" if deleting else "writing"
    return StoreFullError(
        f"the embeddings store at {env.path()} ran out of its map_size of "
        f"{budget:,} bytes ({budget / 1024**3:.1f} GiB) while {operation} "
        f"document {key.decode()}. The documents already committed are kept "
        f"and are skipped on a rerun, so rerunning with a larger --map_size "
        f"resumes from them."
    )


def writer_thread(
    env: lmdb.Environment,
    in_q: queue.Queue[tuple[bytes, bytes | None]],
    stop_evt: threading.Event,
    commit_every: int,
    pbar_written: tqdm.tqdm,
    state: WriterState,
) -> None:
    """Drain `in_q` into `env`, and set `stop_evt` however this ends.

    This thread is the queue's only consumer, so a producer waiting for room is
    really waiting for it; every exit therefore goes through `stop_evt`. A
    value of `None` asks for the key to be deleted, which has to travel this
    queue because LMDB allows one writer at a time and a second transaction
    would wait on a commit only the blocked producer could supply.

    :param env: the open LMDB environment.
    :param in_q: the key/value pairs to write, `None` meaning delete.
    :param stop_evt: set on every exit, so producers stop waiting.
    :param commit_every: documents per write transaction.
    :param pbar_written: advanced once per stored document.
    :param state: records a failure for `main` to raise after the join.
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
                if v is None:
                    # LMDB answers False for a key it did not hold, which is
                    # the one item that leaves the transaction untouched.
                    changed = tdb.delete(k)
                else:
                    tdb.put(k, v)
                    changed = True
            except lmdb.MapFullError:
                # Committing what this transaction holds cannot rescue it:
                # LMDB marks a transaction invalid on the `put` that overflows
                # the map, so its `commit` answers `BadTxnError`. Up to
                # `commit_every - 1` documents are embedded again on the rerun.
                state.failure = store_full(env, k, deleting=v is None)
                stop_evt.set()
                tdb.abort()
                env.sync()
                continue

            if changed:
                # `commit_every` bounds the work a transaction accumulates, so
                # an item that accumulated none must not advance it.
                n_since += 1
            if v is not None:
                # A deleted document is one this pass writes nothing for, and
                # the bar's total has already been reduced by it.
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
    out_q: queue.Queue[tuple[bytes, bytes | None]],
    item: tuple[bytes, bytes | None],
    stop_evt: threading.Event,
) -> bool:
    """Hand `item` to the writer; return False once the writer has stopped.

    A plain `put` on a full queue waits for a consumer that may already be
    gone, and setting an event does not wake it.

    :param out_q: the writer's queue.
    :param item: the key/value pair to hand over.
    :param stop_evt: set once the writer has stopped.
    :return: whether the item was handed over.
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
    positive_int("batch_size", args.batch_size)
    positive_int("commit_every", args.commit_every)
    positive_int("stream_batch", args.stream_batch)

    # Everything that can refuse this run is settled before the base model is
    # read: a reservation lmdb.open itself rejects, a store some other model
    # wrote, and a map too small to hold even one document all used to be
    # found only once the weights were on the device. The context window and
    # hidden size are the only things needed from the model here, and the
    # config alone carries both, so nothing waits on the weights.
    model_config = transformers.AutoConfig.from_pretrained(args.base_model)
    max_len = window_size(args.max_length, model_config)
    env = lmdb.open(args.output_path, map_size=map_size)
    try:
        record_provenance(
            env,
            StoreProvenance(
                base_model=args.base_model, max_length=max_len, stride=STRIDE
            ),
        )
        check_map_size_for_one_document(env, max_len, model_config.hidden_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = utils.load_fast_tokenizer(args.base_model)
        model = (
            transformers.AutoModel.from_pretrained(args.base_model)
            .to(device)
            .eval()
        )

        # Snapshot taken before any writing, so a document is judged against
        # what a *previous* run stored, not against this run's own output.
        already_embedded = set() if args.force_regenerate else stored_keys(env)

        # Shared across datasets only because the first failure ends the run:
        # the writer that recorded it is the last one started.
        writer_state = WriterState()

        for dataset in args.datasets:
            path = pathlib.Path(dataset)
            logger.info("\nProcessing %s", path)

            total_rows, row_iter = corpus.stream_rows(path, args.stream_batch)
            skipped = 0
            empty = 0

            # In-flight compression jobs -> the pmid key each will be stored
            # under. Local to the dataset: a shared dict would let one
            # dataset's undrained leftovers be written while the next is
            # processed.
            futures: dict[Future[bytes], bytes] = {}

            # queues + bars
            out_q: queue.Queue[tuple[bytes, bytes | None]] = queue.Queue(
                maxsize=124
            )
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

            try:
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
                            pbar_written.total = total_rows - skipped - empty
                            continue

                        if not text:
                            logger.warning(
                                "%s has neither an abstract nor a fulltext; "
                                "storing no embedding for it.",
                                key.decode(),
                            )
                            empty += 1
                            pbar_emb.update(1)
                            pbar_written.total = total_rows - skipped - empty
                            # Only reachable with -f, since a stored key is
                            # skipped above otherwise. The corpus now says this
                            # document has no text, and -f exists to make the
                            # store agree with the corpus, so the stale entry
                            # goes too. Deleting a key the store does not hold
                            # is a no-op, which is why this asks unconditionally
                            # rather than reading the store to find out.
                            if not put_or_stop(out_q, (key, None), stop_evt):
                                break
                            continue

                        emb = utils.embed_document(
                            text,
                            tokenizer=tokenizer,
                            model=model,
                            stride=STRIDE,
                            batch_size=args.batch_size,
                            max_len=max_len,
                        )
                        pbar_emb.update(1)

                        # submit for compression
                        f = pool.submit(tensor_to_bytes, emb)
                        futures[f] = key

                        if len(futures) >= MAX_BACKLOG:
                            done, _ = wait(
                                list(futures.keys()),
                                return_when=FIRST_COMPLETED,
                            )
                            for d in done:
                                item = (futures.pop(d), d.result())
                                if not put_or_stop(out_q, item, stop_evt):
                                    break

                    # Drain unconditionally: the in-loop flush above is what
                    # keeps the backlog *below* MAX_BACKLOG, so repeating that
                    # guard here would be false exactly when there is still
                    # work in flight. A dataset shorter than MAX_BACKLOG would
                    # then write nothing at all.
                    for done_future in as_completed(list(futures)):
                        item = (
                            futures.pop(done_future),
                            done_future.result(),
                        )
                        if not put_or_stop(out_q, item, stop_evt):
                            break
            finally:
                # Reached on every exit from the block above, including an
                # exception out of `embed_document` or the row iterator: the
                # writer is this dataset's only consumer, and leaving it
                # running un-joined would leak a daemon thread still holding
                # `env`'s write lock for as long as the process (or, in a
                # caller that runs `main` more than once, the next dataset)
                # lasts.
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
    finally:
        # Reached whether the loop above finished, broke on a writer failure,
        # or an exception left it early — `record_provenance` refusing a store
        # built by a different run is the most-travelled way that happens.
        # Without this, every path but the successful one left the lock file
        # held and, for a caller that opens the store again in the same
        # process, unreachable.
        env.close()

    # A truncated store must not be reachable from a command that reported
    # success: the resume path reads every document that was written as one
    # already embedded, so the next run walks past the gap the writer left.
    if writer_state.failure is not None:
        raise writer_state.failure

    logger.info("Done.")


if __name__ == "__main__":
    main()
