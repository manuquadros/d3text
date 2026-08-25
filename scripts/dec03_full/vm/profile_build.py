"""Where the store build's wall clock actually goes, per document.

The build was called "a GPU-bound job again" once the codec changed, and the
whole stage was sized from a 0.14 s/doc forward. That claim is worth
checking before spending two hours on it, because the loop in
`precompute-embeddings` has four places the GPU can sit idle and none of them
appear in that arithmetic:

- the D2H copy after every forward is blocking and unpinned (`utils.py:294`);
- `aggregate_embeddings` is a Python loop over windows, on the main thread,
  after the forward and before anything is handed to the compression pool;
- the corpus is streamed synchronously — every document pays a Polars slice,
  `remove_tags` and a tokenize on the same thread that drives the GPU;
- the forward batches the windows of *one* document, and no document in this
  corpus has more than 29 of them, so the `--batch_size` knob never fires.

This mirrors that loop over a sample of documents and times each phase, then
re-runs the GPU half two other ways: batching windows *across* documents to a
fixed budget, and staging the transfers through pinned buffers. The shipped
command is not imported and not modified — this is a measurement, and a
measurement that edits the thing it measures is not one.

Corpus reading and tokenization are timed once and shared by every arm: they
are identical in all three, and paying for them three times would only make
the script slower without making the comparison fairer.
"""

import argparse
import json
import pathlib
import statistics
import time
import typing

import torch
import transformers
from d3text import corpus, logs
from d3text.embeddings_store import tensor_to_bytes
from d3text.models.config import ModelConfig
from d3text.utils.utils import aggregate_embeddings, split_and_tokenize

# What `precompute-embeddings` passes; the store's aggregation depends on it,
# so an arm that used a different one would not be measuring the same job.
STRIDE = 20

DTYPES = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Phases:
    """Accumulated seconds per phase, and a context manager to add to them."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}

    def __call__(self, name: str) -> typing.ContextManager[None]:
        return self._Timer(self, name)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds

    class _Timer:
        def __init__(self, phases: "Phases", name: str) -> None:
            self.phases = phases
            self.name = name

        def __enter__(self) -> None:
            self.start = time.perf_counter()

        def __exit__(self, *exc: object) -> None:
            self.phases.add(self.name, time.perf_counter() - self.start)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def autocast(
    device: torch.device, dtype: torch.dtype | None
) -> typing.ContextManager[None]:
    if dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


def read_corpus(
    path: pathlib.Path, count: int, stream_batch: int
) -> tuple[list[str], float]:
    """`count` documents' text, and the seconds the corpus layer took.

    Empty documents are skipped rather than counted: `document_text` returns
    `""` for a body that is markup around whitespace, and the real command
    skips those too, so charging the sample for them would measure a document
    the store never holds.
    """
    start = time.perf_counter()
    _, rows = corpus.stream_rows(path, stream_batch)
    texts = []
    for _, text in rows:
        if not text:
            continue
        texts.append(text)
        if len(texts) >= count:
            break
    return texts, time.perf_counter() - start


def tokenize_all(
    texts: list[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_length: int,
) -> tuple[list[dict[str, torch.Tensor]], float]:
    """Every document's windows, and the seconds tokenization took."""
    start = time.perf_counter()
    encodings = []
    for text in texts:
        encoding = split_and_tokenize(
            tokenizer=tokenizer,
            inputs=text,
            stride=STRIDE,
            max_length=max_length,
        )
        encodings.append(
            {
                "input_ids": typing.cast(torch.Tensor, encoding["input_ids"]),
                "attention_mask": typing.cast(
                    torch.Tensor, encoding["attention_mask"]
                ),
            }
        )
    return encodings, time.perf_counter() - start


def per_document_arm(
    encodings: list[dict[str, torch.Tensor]],
    model: transformers.PreTrainedModel,
    device: torch.device,
    dtype: torch.dtype | None,
    *,
    pinned: bool,
) -> Phases:
    """The shipped loop: one forward per document, blocking D2H, then pack.

    `pinned=True` is the same loop with the ids staged through page-locked
    memory, which is what makes `non_blocking=True` mean anything — in the
    shipped path the source is whatever the tokenizer allocated, so CUDA
    silently falls back to a synchronous copy.
    """
    phases = Phases()

    for encoding in encodings:
        ids_cpu = encoding["input_ids"]
        mask_cpu = encoding["attention_mask"]
        if pinned:
            ids_cpu = ids_cpu.pin_memory()
            mask_cpu = mask_cpu.pin_memory()

        synchronize()
        with phases("h2d"):
            ids = ids_cpu.to(device, non_blocking=True)
            mask = mask_cpu.to(device, non_blocking=True)
            synchronize()

        with phases("forward"):
            with torch.inference_mode(), autocast(device, dtype):
                out = model(ids, mask).last_hidden_state
            synchronize()

        with phases("d2h"):
            host = out.detach().to("cpu", non_blocking=pinned)
            synchronize()

        with phases("aggregate"):
            embedding = aggregate_embeddings(host, encoding["attention_mask"])

        with phases("pack"):
            tensor_to_bytes(embedding)

        del ids, mask, out, host

    return phases


def cross_document_arm(
    encodings: list[dict[str, torch.Tensor]],
    model: transformers.PreTrainedModel,
    device: torch.device,
    dtype: torch.dtype | None,
    *,
    budget: int,
) -> Phases:
    """Windows batched across documents to a fixed budget, then split back.

    The aggregation and the packing still happen per document — they have to,
    the store is keyed on documents — so only the forward changes shape. That
    is the point: it isolates what a cross-document batcher would buy from
    what it would not.
    """
    phases = Phases()
    group: list[dict[str, torch.Tensor]] = []
    windows = 0

    def flush(group: list[dict[str, torch.Tensor]]) -> None:
        if not group:
            return
        ids_cpu = torch.cat([item["input_ids"] for item in group], dim=0)
        mask_cpu = torch.cat([item["attention_mask"] for item in group], dim=0)

        synchronize()
        with phases("h2d"):
            ids = ids_cpu.to(device, non_blocking=True)
            mask = mask_cpu.to(device, non_blocking=True)
            synchronize()

        with phases("forward"):
            with torch.inference_mode(), autocast(device, dtype):
                out = model(ids, mask).last_hidden_state
            synchronize()

        with phases("d2h"):
            host = out.detach().to("cpu")
            synchronize()

        offset = 0
        for item in group:
            rows = item["input_ids"].shape[0]
            with phases("aggregate"):
                embedding = aggregate_embeddings(
                    host[offset : offset + rows], item["attention_mask"]
                )
            with phases("pack"):
                tensor_to_bytes(embedding)
            offset += rows

        del ids, mask, out, host

    for encoding in encodings:
        rows = int(encoding["input_ids"].shape[0])
        if group and windows + rows > budget:
            flush(group)
            group, windows = [], 0
        group.append(encoding)
        windows += rows
    flush(group)

    return phases


def summarize(phases: Phases, documents: int, tokens: int) -> dict[str, object]:
    total = sum(phases.totals.values())
    return {
        "seconds": total,
        "documents_per_second": documents / total if total else None,
        "tokens_per_second": tokens / total if total else None,
        "peak_MiB": (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else None
        ),
        "phases": {
            name: round(value, 4)
            for name, value in sorted(phases.totals.items())
        },
        "phase_share": {
            name: round(value / total, 4)
            for name, value in sorted(phases.totals.items())
        }
        if total
        else {},
    }


def read_args() -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parents[3]
    default_corpus = (
        repo / "brenda_references/src/brenda_references/data/training_data.csv"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Time each phase of the precompute-embeddings loop, and compare "
            "its per-document batching against batching across documents."
        )
    )
    parser.add_argument("--corpus", type=pathlib.Path, default=default_corpus)
    parser.add_argument(
        "--docs",
        type=int,
        default=150,
        help="documents sampled from the head of the corpus",
    )
    parser.add_argument(
        "--base-model",
        default=ModelConfig().base_model,
        help="must match the arms' config, or the windows differ",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="fp16",
        help="autocast dtype for the forward; fp16 is what the shipped "
        "command hardcodes",
    )
    parser.add_argument(
        "--window-budget",
        type=int,
        default=128,
        help="windows per forward in the cross-document arm",
    )
    parser.add_argument("--stream-batch", type=int, default=1000)
    parser.add_argument("--out", help="write the report here as JSON")
    return parser.parse_args()


def main() -> int:
    args = read_args()
    # `logs.configure()` and not `runtime.configure()`, because that is what
    # `precompute-embeddings` itself calls: the build runs on torch's own
    # defaults, so a profile that applied `config.toml`'s matmul precision
    # would be measuring a stage nobody runs.
    logs.configure()

    if not args.corpus.exists():
        print(f"corpus file missing: {args.corpus}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        transformers.AutoModel.from_pretrained(args.base_model)
        .to(device)
        .eval()
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    max_length = int(model.config.max_position_embeddings)
    dtype = DTYPES[args.dtype]

    texts, corpus_seconds = read_corpus(
        args.corpus, args.docs, args.stream_batch
    )
    encodings, tokenize_seconds = tokenize_all(texts, tokenizer, max_length)
    windows = [int(item["input_ids"].shape[0]) for item in encodings]
    tokens = sum(windows) * max_length

    report: dict[str, object] = {
        "corpus": str(args.corpus),
        "documents": len(encodings),
        "base_model": args.base_model,
        "dtype": args.dtype,
        "max_length": max_length,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        # Shared by every arm below, and reported once for that reason. Both
        # are on the shipped loop's critical path with nothing overlapping
        # them, so they belong in any end-to-end ratio computed from this.
        "shared": {
            "corpus_seconds": round(corpus_seconds, 4),
            "tokenize_seconds": round(tokenize_seconds, 4),
            "per_document": round(
                (corpus_seconds + tokenize_seconds) / max(len(encodings), 1), 4
            ),
        },
        "windows_per_document": {
            "mean": round(statistics.mean(windows), 2) if windows else None,
            "median": statistics.median(windows) if windows else None,
            "min": min(windows) if windows else None,
            "max": max(windows) if windows else None,
            "total": sum(windows),
        },
    }

    arms: dict[str, dict[str, object]] = {}
    plans: list[tuple[str, typing.Callable[[], Phases]]] = [
        (
            "per_document",
            lambda: per_document_arm(
                encodings, model, device, dtype, pinned=False
            ),
        ),
        (
            "per_document_pinned",
            lambda: per_document_arm(
                encodings, model, device, dtype, pinned=True
            ),
        ),
        (
            f"cross_document_{args.window_budget}",
            lambda: cross_document_arm(
                encodings, model, device, dtype, budget=args.window_budget
            ),
        ),
    ]

    for name, run in plans:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            # One untimed document first: the first forward of a process pays
            # for cuBLAS handles and autocast's weight cache, and charging
            # that to whichever arm ran first is how an ordering artifact gets
            # written down as a result.
            run_warmup(encodings, model, device, dtype)
            arms[name] = summarize(run(), len(encodings), tokens)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            arms[name] = {
                "error": f"{type(error).__name__}: {str(error)[:160]}"
            }
        print(f"{name}: {json.dumps(arms[name], indent=2)}")

    report["arms"] = arms

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")

    return 0


def run_warmup(
    encodings: list[dict[str, torch.Tensor]],
    model: transformers.PreTrainedModel,
    device: torch.device,
    dtype: torch.dtype | None,
) -> None:
    if not encodings:
        return
    encoding = encodings[0]
    with torch.inference_mode(), autocast(device, dtype):
        model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        ).last_hidden_state
    synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
