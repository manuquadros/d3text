#!/usr/bin/env python
import argparse
import dataclasses
import logging
import math
import os
import pathlib
import queue
import threading
import typing
from collections.abc import Collection, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)

import lmdb
import torch
import torch.nn as nn
import tqdm
import transformers
from d3text import corpus, logs, utils
from d3text.cli import args as cli_args
from d3text.constraints import NonNegative, Positive
from d3text.embeddings_store import (
    AGGREGATED,
    DEFAULT_MAP_SIZE_GIB,
    MAX_SUB_DATABASES,
    StoreProvenance,
    boundary_name,
    read_provenance,
    tensor_to_bytes,
    windowed_tensor_to_bytes,
    write_provenance,
)
from d3text.runtime import select_amp_dtype
from transformers.masking_utils import create_bidirectional_mask

logger = logging.getLogger(__name__)

CPU_COUNT = os.cpu_count() or 1
COMP_THREADS = max(1, CPU_COUNT // 2)
MAX_BACKLOG = max(8, COMP_THREADS * 2)

# `embed_document` mostly waits (tokenizing, then `.cpu()`), both GIL-free.
# Two workers overlap one document's tokenizing with the previous one's GPU
# wait; one accelerator runs the forward passes, so more would not help.
EMBED_WORKERS = 2
EMBED_BACKLOG = EMBED_WORKERS

# The overlap between consecutive windows, and not a flag: the encodings the
# training run reads are tokenized by `split_and_tokenize`'s own default, and
# a store striding differently from them is a store of different rows.
STRIDE = utils.WINDOW_STRIDE

# Not a flag either, for the same reason: `precompute_encodings.MAX_LENGTH`
# and training's live forward fallback use this constant, so a store built
# at another window mixes silently with them on every miss.
MAX_LENGTH = utils.WINDOW_LENGTH


class StoreFullError(RuntimeError):
    """The LMDB ran out of `map_size` before every document was written."""


# What the writer queue carries: the sub-database, the document's key, and
# the value to store there, `None` meaning delete.
WriteItem = tuple[lmdb._Database, bytes, bytes | None]

# What an embedding future resolves to: `utils.embed_document`'s aggregated
# row when no boundary is asked for, or `embed_document_and_prefix`'s
# `(aggregated, prefixes)` pair when one is.
EmbedResult = torch.Tensor | tuple[torch.Tensor | None, dict[int, torch.Tensor]]


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
    p.add_argument(
        "datasets",
        nargs="*",
        type=cli_args.readable_path,
        help=(
            "corpus files to embed; defaults to the splits and noise pools "
            "`brenda_references` is configured with"
        ),
    )
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
    p.add_argument("--commit_every", type=int, default=100)
    p.add_argument(
        "--no_compress",
        dest="compress",
        action="store_false",
        help=(
            "store the matrices uncompressed, trading disk for write and "
            "read speed; both stores read either way"
        ),
    )
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
        "--stream_batch",
        type=int,
        default=corpus.STREAM_BATCH,
        help="rows per Polars slice",
    )
    p.add_argument(
        "--unfrozen_top_layers",
        type=int,
        nargs="+",
        default=[],
        help=(
            "also store, for each count given, one row of hidden states per "
            "window at the boundary a training run leaving that many top "
            "encoder layers trainable resumes from"
        ),
    )
    p.add_argument(
        "--no_aggregated",
        dest="aggregated",
        action="store_false",
        help=(
            "skip the aggregated last-layer rows; a frozen run derives them "
            "from the stored boundary with the fewest unfrozen layers"
        ),
    )
    args = p.parse_args()
    args.datasets = cli_args.resolve_datasets(p, args.datasets)
    return args


def window_size(model_config: transformers.PretrainedConfig) -> int:
    """Refuse a base model whose context window is narrower than the pin.

    Unlike `precompute-encodings`, this command forwards through the model,
    so a narrower position table would be indexed past instead of failing
    loudly.

    :param model_config: the base model's config.
    :return: `MAX_LENGTH`.
    :raises ValueError: if the base model's context window is narrower than
        `MAX_LENGTH`.
    """
    limit: int = model_config.max_position_embeddings
    if MAX_LENGTH > limit:
        msg = (
            f"{model_config.name_or_path}'s context window is {limit} "
            f"tokens, narrower than the {MAX_LENGTH}-token window "
            f"`precompute-encodings` and training both use; embedding it at "
            f"that window would index past its position table."
        )
        raise ValueError(msg)
    return MAX_LENGTH


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

    Applies to `--batch_size`, `--commit_every` and `--stream_batch`, so a
    bad value costs nothing.

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

    A store of another model or window, or an unstamped one holding
    documents, is refused: nothing downstream can separate the two kinds of
    matrix, and `-f` re-embeds only the documents these datasets name.

    :param env: the open LMDB environment.
    :param provenance: what this run will write.
    :raises ValueError: if the store records another geometry, or holds
        documents and records none.
    :raises ProvenanceError: from `read_provenance`, if the record cannot be
        read or the env is in the older one-cut-per-env layout.
    """
    recorded = read_provenance(env)
    # Identity, not equality: `forward_dtype` says how, not what, so a store
    # resumes under a build of another precision and keeps its older stamp
    # rather than claiming a uniformity it does not have.
    if recorded is not None and recorded.identity == provenance.identity:
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
    env: lmdb.Environment, max_len: Positive, hidden_size: Positive
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
    :raises ValueError: if the probe does not fit.
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


def stored_keys(env: lmdb.Environment, db: lmdb._Database) -> set[bytes]:
    """The pubmed ids already embedded in one sub-database of `env`.

    Keys only: pulling the compressed embeddings in just to test for presence
    would defeat the point of skipping them.

    :param env: the open LMDB environment.
    :param db: the sub-database to list.
    :return: the keys already stored.
    """
    with env.begin(db=db) as txn:
        return set(txn.cursor().iternext(keys=True, values=False))


def embed_document_and_prefix(
    doc: str,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    frozen_layers: Collection[NonNegative],
    need_full: bool,
    stride: NonNegative = STRIDE,
    batch_size: Positive = 50,
    max_len: Positive = utils.WINDOW_LENGTH,
) -> tuple[torch.Tensor | None, dict[int, torch.Tensor]]:
    """Run `doc` through the base model once, for every sub-database.

    Fuses `utils.embed_document`'s pass with the layer-boundary ones: all
    tokenize, window and run the same frozen layers, so running them apart
    forwards every document through those layers once per boundary. Keeps
    a per-window prefix at each of `frozen_layers`, and continues through
    the remaining layers only if `need_full`.

    :param doc: the document text.
    :param tokenizer: the tokenizer the windows are cut with.
    :param model: the base model, in eval mode.
    :param frozen_layers: the leading encoder layer counts to keep a
        per-window prefix after.
    :param need_full: whether to continue past the deepest boundary and
        aggregate a full-trunk row.
    :param stride: tokens of overlap between adjacent windows.
    :param batch_size: windows per forward pass.
    :param max_len: tokens per window.
    :return: `(aggregated, prefixes)`; `aggregated` is `None` unless
        `need_full`, and `prefixes` maps each of `frozen_layers` to its
        prefix.
    :raises ValueError: if a boundary exceeds the base model's encoder.
    """
    encoder_layers = typing.cast(
        nn.ModuleList, model.get_submodule("encoder.layer")
    )
    deepest = max(frozen_layers, default=0)
    if deepest > len(encoder_layers):
        msg = (
            f"frozen_layers={deepest} exceeds "
            f"{type(model).__name__}'s {len(encoder_layers)} encoder layers"
        )
        raise ValueError(msg)

    encoding = utils.split_and_tokenize(
        tokenizer=tokenizer,
        inputs=doc,
        stride=stride,
        max_length=max_len,
        return_offsets_mapping=False,
    )
    input_ids_all = typing.cast(torch.Tensor, encoding["input_ids"])
    attention_mask_all = typing.cast(torch.Tensor, encoding["attention_mask"])

    prefix_windows: dict[int, list[torch.Tensor]] = {
        boundary: [] for boundary in frozen_layers
    }
    full_windows: list[torch.Tensor] = []
    n_windows = input_ids_all.size(0)

    with torch.inference_mode():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            ids = input_ids_all[start:end].to(model.device, non_blocking=True)
            mask = attention_mask_all[start:end].to(
                model.device, non_blocking=True
            )
            with torch.amp.autocast(
                device_type=model.device.type,
                dtype=select_amp_dtype(model.device.type),
            ):
                hidden_states = model.get_submodule("embeddings")(input_ids=ids)
                extended_mask = create_bidirectional_mask(
                    config=model.config,
                    inputs_embeds=hidden_states,
                    attention_mask=mask,
                )
                if 0 in prefix_windows:
                    prefix_windows[0].append(hidden_states.detach().cpu())
                for i, layer in enumerate(encoder_layers):
                    if i >= deepest and not need_full:
                        break
                    hidden_states = layer(hidden_states, extended_mask)
                    if i + 1 in prefix_windows:
                        prefix_windows[i + 1].append(
                            hidden_states.detach().cpu()
                        )

            if need_full:
                full_windows.append(hidden_states.detach().cpu())
            del hidden_states, ids, mask

    aggregated = (
        utils.aggregate_embeddings(
            embeddings=torch.cat(full_windows, dim=0),
            attention_mask=attention_mask_all,
            stride=stride,
        )
        if need_full
        else None
    )
    return aggregated, {
        boundary: torch.cat(windows, dim=0)
        for boundary, windows in prefix_windows.items()
    }


def as_rows(
    result: EmbedResult, boundaries: Mapping[int, str]
) -> list[tuple[str, torch.Tensor]]:
    """Pair each row an embedding future computed with its sub-database.

    :param result: `utils.embed_document`'s aggregated row when no boundary
        was asked for, or `embed_document_and_prefix`'s pair when one was.
    :param boundaries: the sub-database of each frozen count asked for.
    :return: `(sub-database, row)` pairs.
    """
    if isinstance(result, torch.Tensor):
        return [(AGGREGATED, result)]
    aggregated, prefixes = result
    rows = [(boundaries[frozen], prefix) for frozen, prefix in prefixes.items()]
    if aggregated is not None:
        rows.append((AGGREGATED, aggregated))
    return rows


def encode_row(name: str, row: torch.Tensor, *, compress: bool) -> bytes:
    """Compress one sub-database's row with the codec that sub-database reads.

    :param name: the sub-database the row goes to.
    :param row: the aggregated row, or a boundary's per-window prefix.
    :param compress: whether the frame is zstd-compressed or stored raw.
    :return: the value to store.
    """
    if name == AGGREGATED:
        return tensor_to_bytes(row, compress=compress)
    return windowed_tensor_to_bytes(row, compress=compress)


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
    in_q: queue.Queue[WriteItem],
    stop_evt: threading.Event,
    commit_every: Positive,
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
    :param in_q: the sub-database/key/value triples to write, `None`
        meaning delete.
    :param stop_evt: set on every exit, so producers stop waiting.
    :param commit_every: rows per write transaction.
    :param pbar_written: advanced once per stored row.
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
                db, k, v = in_q.get(timeout=0.1)
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
                    changed = tdb.delete(k, db=db)
                else:
                    tdb.put(k, v, db=db)
                    changed = True
            except lmdb.MapFullError:
                # Not committed: LMDB invalidates a transaction on the `put`
                # that overflows (`commit` answers `BadTxnError`), so up to
                # `commit_every - 1` documents are embedded again on rerun.
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
    out_q: queue.Queue[WriteItem],
    item: WriteItem,
    stop_evt: threading.Event,
) -> bool:
    """Hand `item` to the writer; return False once the writer has stopped.

    A plain `put` on a full queue waits for a consumer that may already be
    gone, and setting an event does not wake it.

    :param out_q: the writer's queue.
    :param item: the sub-database/key/value triple to hand over.
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

    # Everything that can refuse this run is settled before the weights load;
    # the config alone carries the context window and hidden size it needs.
    model_config = transformers.AutoConfig.from_pretrained(args.base_model)
    max_len = window_size(model_config)
    # Chosen before the stamp is written, since the stamp records it. Naming
    # the device costs nothing and loads nothing; the weights still wait
    # until every refusal below has had its chance.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = model_config.num_hidden_layers
    for count in args.unfrozen_top_layers:
        if not 0 <= count <= num_layers:
            msg = (
                f"--unfrozen_top_layers={count} is not between 0 and "
                f"{args.base_model}'s {num_layers} encoder layers."
            )
            raise ValueError(msg)
    # Frozen count -> sub-database, one per distinct boundary asked for.
    boundaries = {
        num_layers - count: boundary_name(count)
        for count in args.unfrozen_top_layers
    }
    names = [AGGREGATED] if args.aggregated else []
    names += sorted(boundaries.values())
    if not names:
        msg = "--no_aggregated leaves nothing to write without a boundary."
        raise ValueError(msg)

    env = lmdb.open(
        args.output_path, map_size=map_size, max_dbs=MAX_SUB_DATABASES
    )
    try:
        record_provenance(
            env,
            StoreProvenance(
                base_model=args.base_model,
                max_length=max_len,
                stride=STRIDE,
                forward_dtype=str(select_amp_dtype(device.type)),
            ),
        )
        check_map_size_for_one_document(env, max_len, model_config.hidden_size)
        dbs = {name: env.open_db(name.encode()) for name in names}

        tokenizer = utils.load_fast_tokenizer(args.base_model)
        model = (
            transformers.AutoModel.from_pretrained(args.base_model)
            .to(device)
            .eval()
        )

        # Snapshot taken before any writing, so a document is judged against
        # what a *previous* run stored, not against this run's own output.
        # One per sub-database: a document already in one still needs the
        # others, so the skip sets cannot be merged into one.
        already_embedded: dict[str, set[bytes]] = {
            name: set() if args.force_regenerate else stored_keys(env, db)
            for name, db in dbs.items()
        }

        # Shared across datasets only because the first failure ends the run:
        # the writer that recorded it is the last one started.
        writer_state = WriterState()

        for dataset in args.datasets:
            path = pathlib.Path(dataset)
            logger.info("\nProcessing %s", path)

            total_rows, row_iter = corpus.stream_rows(path, args.stream_batch)
            skipped = 0
            planned = 0

            # Compression future -> where its value goes. Per dataset, else
            # one dataset's undrained leftovers are written while the next is
            # processed.
            futures: dict[Future[bytes], tuple[lmdb._Database, bytes]] = {}

            # Embedding future -> the document's key and the boundaries it
            # was asked for, scoped like `futures`.
            embed_futures: dict[
                Future[EmbedResult], tuple[bytes, dict[int, str]]
            ] = {}

            # queues + bars
            out_q: queue.Queue[WriteItem] = queue.Queue(maxsize=124)
            stop_evt = threading.Event()

            pbar_emb = tqdm.tqdm(
                total=total_rows,
                desc="Embedded",
                position=0,
                leave=False,
                dynamic_ncols=True,
            )
            # Counts rows, one per sub-database a document still lacks, so
            # its total grows as the documents are read.
            pbar_written = tqdm.tqdm(
                total=0,
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

            def compress_rows(
                done: Future[EmbedResult],
                pool: ThreadPoolExecutor,
            ) -> None:
                """Queue each row of a finished embedding for compression."""
                pbar_emb.update(1)
                doc_key, asked = embed_futures.pop(done)
                for name, row in as_rows(done.result(), asked):
                    f = pool.submit(
                        encode_row, name, row, compress=args.compress
                    )
                    futures[f] = (dbs[name], doc_key)

            try:
                # embedding and compression pools
                with (
                    ThreadPoolExecutor(max_workers=EMBED_WORKERS) as embed_pool,
                    ThreadPoolExecutor(max_workers=COMP_THREADS) as pool,
                    torch.inference_mode(),
                ):
                    for pmid, text in row_iter:
                        if stop_evt.is_set():
                            break

                        key = str(pmid).encode()
                        targets = [
                            name
                            for name in names
                            if key not in already_embedded[name]
                        ]
                        if not targets:
                            skipped += 1
                            pbar_emb.update(1)
                            continue

                        if not text:
                            logger.warning(
                                "%s has neither an abstract nor a fulltext; "
                                "storing no embedding for it.",
                                key.decode(),
                            )
                            pbar_emb.update(1)
                            # Drops a stale entry `-f` would otherwise leave.
                            deleted = all(
                                put_or_stop(
                                    out_q, (dbs[name], key, None), stop_evt
                                )
                                for name in targets
                            )
                            if not deleted:
                                break
                            continue

                        planned += len(targets)
                        pbar_written.total = planned
                        asked = {
                            frozen: name
                            for frozen, name in boundaries.items()
                            if name in targets
                        }
                        ef: Future[EmbedResult]
                        if not asked:
                            ef = embed_pool.submit(
                                utils.embed_document,
                                text,
                                tokenizer=tokenizer,
                                model=model,
                                stride=STRIDE,
                                batch_size=args.batch_size,
                                max_len=max_len,
                            )
                        else:
                            ef = embed_pool.submit(
                                embed_document_and_prefix,
                                text,
                                tokenizer=tokenizer,
                                model=model,
                                frozen_layers=sorted(asked),
                                need_full=AGGREGATED in targets,
                                stride=STRIDE,
                                batch_size=args.batch_size,
                                max_len=max_len,
                            )
                        embed_futures[ef] = (key, asked)

                        if len(embed_futures) >= EMBED_BACKLOG:
                            done_embed, _ = wait(
                                list(embed_futures.keys()),
                                return_when=FIRST_COMPLETED,
                            )
                            for de in done_embed:
                                compress_rows(de, pool)

                        # submit whatever compression jobs are ready
                        if len(futures) >= MAX_BACKLOG:
                            done, _ = wait(
                                list(futures.keys()),
                                return_when=FIRST_COMPLETED,
                            )
                            for d in done:
                                db, doc_key = futures.pop(d)
                                item = (db, doc_key, d.result())
                                if not put_or_stop(out_q, item, stop_evt):
                                    break

                    # Drain the embedding backlog first: a document still
                    # embedding when the row loop ends has not reached the
                    # compression stage yet, so it is invisible to `futures`.
                    for done_embed_future in as_completed(list(embed_futures)):
                        compress_rows(done_embed_future, pool)

                    # Drain unconditionally: the in-loop flush keeps the
                    # backlog below MAX_BACKLOG, so that guard here would
                    # drop every dataset shorter than MAX_BACKLOG.
                    for done_future in as_completed(list(futures)):
                        db, doc_key = futures.pop(done_future)
                        item = (db, doc_key, done_future.result())
                        if not put_or_stop(out_q, item, stop_evt):
                            break
            finally:
                # On every exit, exceptions included: an un-joined writer
                # would leak a daemon thread holding `env`'s write lock for
                # the rest of the process.
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
        # On every exit (a `record_provenance` refusal is the usual one): else
        # the lock file stays held, and the store unreachable to a caller
        # that reopens it in the same process.
        env.close()

    # A truncated store must not be reachable from a command that reported
    # success: the resume path reads every document that was written as one
    # already embedded, so the next run walks past the gap the writer left.
    if writer_state.failure is not None:
        raise writer_state.failure

    logger.info("Done.")


if __name__ == "__main__":
    main()
