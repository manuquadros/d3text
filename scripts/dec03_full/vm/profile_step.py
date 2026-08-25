"""Where a training step's time and VRAM go once the store serves everything.

Two beliefs decide how the arms are configured, and neither has been
measured in the regime the arms actually run in:

**That `batch_max_chunks` is near the card's ceiling.** An earlier P100
baseline put the card at 99.2% of its 16 GiB at a budget of 512 — but that was
at `--limit 200`, whose entity head is a few hundred columns wide rather than
the full corpus's 6862, and it was before the store existed. A separate
residency measurement established that the high-water mark was set by the
*base-model forward*, which the store removes entirely. Both corrections move the ceiling,
and they move it in opposite directions, so the number has to be measured
rather than reasoned about.

**That the run is GPU-bound.** With the forward gone, what is left on the
critical path is `EmbeddingsStore.get` — an LMDB read plus a zstd decompress —
called from inside `Model.forward` on the main thread, with `num_workers = 0`
above it and the full HDF5 encodings still being read and discarded every
epoch. If that is the majority of a step, then raising the chunk budget buys
almost nothing and the loader's worker count is what matters.

The sweep answers both at once: peak VRAM and a phase breakdown per budget,
with an OOM recorded rather than raised so that finding the ceiling is what
the sweep is *for*. `read_bytes` from `/proc/self/io` is included because at a
101 GiB store on a 121 GiB machine, whether the store is served from page
cache or from the disk is genuinely unclear and changes the answer.

Structure follows `scripts/benchmarks/bench_embedding_residency.py`: batches
are drawn once and reused, warmup is untimed, rounds are medianed, and the
instrumentation is monkeypatched here rather than added to `src/` — the arms
must run the code they would have run.
"""

import argparse
import contextlib
import json
import pathlib
import statistics
import time
import typing

import torch
from d3text import data, embeddings_store, factory, runtime
from d3text.models import models as M
from d3text.models.config import encodings, load_model_config
from d3text.training.trainer import Trainer


def read_io_bytes() -> int:
    """Bytes this process has fetched from the storage layer, or 0.

    `read_bytes` counts what actually reached the block device, so it is the
    one number that tells a store served from page cache from one served off
    disk. Absent outside Linux, and its absence is not worth failing over.
    """
    try:
        with open("/proc/self/io") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


class Timers:
    """Seconds accumulated per phase by the patched call sites."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds

    def reset(self) -> None:
        self.totals.clear()

    def snapshot(self) -> dict[str, float]:
        return dict(self.totals)


@contextlib.contextmanager
def instrumented(timers: Timers) -> typing.Iterator[None]:
    """Time the four calls that make up the store path, then put them back.

    Patched rather than edited in place: `get_token_embeddings` and
    `EmbeddingsStore.get` are on the arms' critical path, and a timer left in
    `src/` would be measuring a library the run no longer ships.
    """
    original_embeddings = M.Model.get_token_embeddings
    original_token_count = M.document_token_count
    original_decode = embeddings_store.bytes_to_tensor
    original_get = embeddings_store.EmbeddingsStore.get

    def timed_embeddings(self, batch):  # type: ignore[no-untyped-def]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_embeddings(self, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timers.add("get_token_embeddings", time.perf_counter() - start)
        return result

    def timed_token_count(item):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = original_token_count(item)
        timers.add("document_token_count", time.perf_counter() - start)
        return result

    def timed_decode(packed):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = original_decode(packed)
        timers.add("decompress", time.perf_counter() - start)
        return result

    def timed_get(self, pubmed_id, expected_tokens):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = original_get(self, pubmed_id, expected_tokens)
        timers.add("store_get", time.perf_counter() - start)
        return result

    M.Model.get_token_embeddings = timed_embeddings  # type: ignore[method-assign]
    M.document_token_count = timed_token_count  # type: ignore[assignment]
    # `EmbeddingsStore.get` resolves `bytes_to_tensor` as a module global, so
    # patching the module attribute is what puts the decompress timer inside
    # the `store_get` one rather than beside it.
    embeddings_store.bytes_to_tensor = timed_decode  # type: ignore[assignment]
    embeddings_store.EmbeddingsStore.get = timed_get  # type: ignore[method-assign]
    try:
        yield
    finally:
        M.Model.get_token_embeddings = original_embeddings  # type: ignore[method-assign]
        M.document_token_count = original_token_count  # type: ignore[assignment]
        embeddings_store.bytes_to_tensor = original_decode  # type: ignore[assignment]
        embeddings_store.EmbeddingsStore.get = original_get  # type: ignore[method-assign]


def draw_batches(
    dataset: object, batch_size: int, budget: int, count: int
) -> tuple[list[object], float]:
    """`count` batches from a fresh loader, and the seconds drawing them took.

    That time is the loader's own — HDF5 read, Zstd filter, collate, pin — and
    at `num_workers = 0` it is paid on the same thread as the forward, so it
    belongs in the step's budget even though the batches are reused below.
    """
    loader = data.get_batch_loader(
        dataset=typing.cast(typing.Any, dataset),
        batch_size=batch_size,
        max_chunks=budget,
    )
    start = time.perf_counter()
    batches: list[object] = []
    iterator = iter(loader)
    for _ in range(count):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break
    return batches, time.perf_counter() - start


def measure_budget(
    model: typing.Any,
    update: typing.Any,
    dataset: object,
    batch_size: int,
    budget: int,
    *,
    batches: int,
    warmup: int,
    rounds: int,
) -> dict[str, object]:
    """Peak VRAM and a phase breakdown at one chunk budget."""
    drawn, draw_seconds = draw_batches(
        dataset, batch_size, budget, warmup + batches
    )
    if len(drawn) <= warmup:
        return {"budget": budget, "error": "not enough batches"}

    warm, measured = drawn[:warmup], drawn[warmup:]
    documents = sum(len(typing.cast(list, b)) for b in measured)
    timers = Timers()

    def step(batch: object) -> None:
        # The trainer's own update, not a bare `backward()`: it is what
        # carries the GradScaler's unscale, the `clip_grad_norm_` over every
        # parameter — the frozen base model's included — and the optimizer
        # step. Those are per-step costs on the arms' critical path, and a
        # profile that skipped them would attribute their share to the heads.
        update.zero_grad()
        losses = model.compute_batch_losses(batch)
        update(*(losses if isinstance(losses, tuple) else (losses,)))

    result: dict[str, object] = {
        "budget": budget,
        "batches": len(measured),
        "documents": documents,
        "documents_per_batch": round(documents / len(measured), 2),
        "draw_seconds_per_batch": round(draw_seconds / max(len(drawn), 1), 4),
    }

    try:
        with instrumented(timers):
            for batch in warm:
                step(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            totals, io_deltas = [], []
            for _ in range(rounds):
                timers.reset()
                io_before = read_io_bytes()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for batch in measured:
                    step(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                totals.append(
                    {
                        "total": time.perf_counter() - start,
                        **timers.snapshot(),
                    }
                )
                io_deltas.append(read_io_bytes() - io_before)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        result["error"] = f"OOM: {str(error)[:160]}"
        return result

    keys = sorted({key for row in totals for key in row})
    phases = {
        key: round(statistics.median([row.get(key, 0.0) for row in totals]), 4)
        for key in keys
    }
    total = phases.get("total", 0.0)
    # Everything the step does that is not the embedding lookup: the heads'
    # forward, the losses, the backward and the optimizer step. Derived rather
    # than timed, so it cannot disagree with the total.
    phases["heads_and_backward"] = round(
        total - phases.get("get_token_embeddings", 0.0), 4
    )

    result.update(
        {
            "peak_MiB": (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else None
            ),
            "seconds": phases,
            "share": {
                key: round(value / total, 4)
                for key, value in phases.items()
                if key != "total"
            }
            if total
            else {},
            "documents_per_second": (documents / total if total else None),
            "disk_read_MiB": round(statistics.median(io_deltas) / 2**20, 2),
        }
    )
    return result


def integers(text: str) -> list[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Peak VRAM and a per-phase time breakdown of a training step at "
            "several chunk budgets, with the embeddings store serving."
        )
    )
    parser.add_argument("config", help="the arm's model config TOML")
    parser.add_argument(
        "--budgets",
        type=integers,
        default=[64, 128, 256, 512],
        help="batch_max_chunks values to sweep; 64 is what the arms run",
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "training documents to load. Omit it: the limit sizes the entity "
            "head, and the head is the largest tensor in the step, so a "
            "limited run measures a different card."
        ),
    )
    parser.add_argument("--out", help="write the report here as JSON")
    return parser.parse_args()


def main() -> int:
    args = read_args()
    runtime.configure()

    config = load_model_config(args.config)
    dataset = data.brenda_dataset(
        encodings=encodings[config.base_model], limit=args.limit
    )
    train = dataset.data["train"]
    model = factory.build_model(
        config,
        dataset,
        entity_freqs=data.compute_frequencies(train, column="entities"),
        class_freqs=data.compute_frequencies(train, column="classes"),
    )
    model.to(model.device)
    model.train()
    # What `Trainer` builds before its first epoch. Without it there is no
    # optimizer for the update to step, and the step being measured would not
    # be the step the arms run.
    update = Trainer(model).update

    report: dict[str, object] = {
        "config": args.config,
        "limit": args.limit,
        "entity_columns": len(dataset.entity_index),
        "model_class": config.model_class,
        "amp_dtype": str(model.amp_dtype),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "vram_MiB": (
            torch.cuda.get_device_properties(0).total_memory / 2**20
            if torch.cuda.is_available()
            else None
        ),
        "store_configured": M.mconfig.embeddings_store,
        "cpu_embeddings_cache_size": M.mconfig.cpu_embeddings_cache_size,
    }

    results = []
    for budget in args.budgets:
        row = measure_budget(
            model,
            update,
            train,
            config.batch_size,
            budget,
            batches=args.batches,
            warmup=args.warmup,
            rounds=args.rounds,
        )
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)
    report["budgets"] = results

    store = M.embeddings_store()
    report["store"] = store.summary() if store is not None else "not configured"
    print(report["store"])

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
