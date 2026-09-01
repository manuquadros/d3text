"""What this GPU is actually fast at, in the two shapes this run cares about.

**Which 16-bit format?** `torch.cuda.is_bf16_supported()` answers yes by
*emulation* on a Pascal card, while `precompute-embeddings` hardcodes fp16 — so
the two halves of the run disagree about the dtype on a card that may be slow
at one and has tensor cores for neither. **How many windows per forward?** The
store build's `--batch_size` never fires, so whether a cross-document batcher
is worth writing depends on where throughput stops climbing.

Needs no dataset, store or checkpoint, and runs in a couple of minutes *before*
the store build. Every arm is guarded: a dtype the card cannot do and a batch
that does not fit are both results.
"""

import argparse
import json
import pathlib
import statistics
import time
import typing

import torch
import transformers
from d3text import runtime
from d3text.models.config import ModelConfig

# The entity head's width at the full corpus vocabulary. It is the dimension
# that makes `[T, E]` the largest tensor in a training step, so a head-shaped
# GEMM has to use the real number.
ENTITY_COLUMNS = 6862

# Tokens a `batch_max_chunks` budget implies: a 512-token window contributes
# ~477 after `aggregate_embeddings` trims [CLS]/[SEP] and half the stride
# overlap. 64 is what the arms run; 512 is what the earlier P100 baseline
# ran, before the store removed the base-model forward.
TOKENS_PER_CHUNK = 477


def failure(error: BaseException) -> str:
    """A one-line reason, distinguishing "too big" from "unsupported"."""
    text = str(error)
    if (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in text
    ):
        return "OOM"
    return f"{type(error).__name__}: {text[:160]}"


def timed(
    call: typing.Callable[[], None], *, warmup: int, repeats: int
) -> dict[str, float]:
    """Median seconds over `repeats` runs, and the peak VRAM they reached.

    The median rather than the mean: this is a shared, thermally throttled
    card and a single slow iteration is not the number anyone wants.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)

    return {
        "seconds": statistics.median(samples),
        "peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }


def autocast_arms(
    device: torch.device,
) -> dict[str, typing.Callable[[], typing.ContextManager[None]]]:
    """The dtype arms, as context managers over the region being measured.

    `.half()` weights are a fourth arm handled separately, since it casts the
    module rather than wrapping a region.
    """

    def plain() -> typing.ContextManager[None]:
        return torch.autocast(device_type=device.type, enabled=False)

    def half() -> typing.ContextManager[None]:
        return torch.autocast(device_type=device.type, dtype=torch.float16)

    def bfloat() -> typing.ContextManager[None]:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    return {"fp32": plain, "fp16_autocast": half, "bf16_autocast": bfloat}


def forward_sweep(
    model: transformers.PreTrainedModel,
    device: torch.device,
    windows: list[int],
    *,
    warmup: int,
    repeats: int,
    max_length: int,
) -> list[dict[str, object]]:
    """Base-model throughput against windows per forward, in each dtype.

    The ids are random: the forward's cost is a function of shape alone, which
    is what keeps this script independent of the corpus.
    """
    rows: list[dict[str, object]] = []
    arms = autocast_arms(device)
    generator = torch.Generator(device="cpu").manual_seed(0)

    for count in windows:
        ids = torch.randint(
            0,
            int(model.config.vocab_size),
            (count, max_length),
            generator=generator,
            dtype=torch.long,
        ).to(device)
        mask = torch.ones_like(ids)
        tokens = count * max_length

        for name, arm in arms.items():
            row: dict[str, object] = {
                "windows": count,
                "tokens": tokens,
                "dtype": name,
            }
            try:

                def once(arm=arm, ids=ids, mask=mask) -> None:
                    with torch.inference_mode(), arm():
                        model(ids, mask).last_hidden_state

                result = timed(once, warmup=warmup, repeats=repeats)
                row.update(result)
                row["tokens_per_second"] = tokens / result["seconds"]
            except (
                torch.cuda.OutOfMemoryError,
                RuntimeError,
                NotImplementedError,
            ) as error:
                torch.cuda.empty_cache()
                row["error"] = failure(error)
            rows.append(row)

        del ids, mask
        torch.cuda.empty_cache()

    return rows


def half_weights_sweep(
    model: transformers.PreTrainedModel,
    device: torch.device,
    windows: list[int],
    *,
    warmup: int,
    repeats: int,
    max_length: int,
) -> list[dict[str, object]]:
    """The same sweep with fp16 *weights* rather than fp16 autocast.

    Autocast keeps an fp32 master copy and caches an fp16 cast of every weight,
    so it costs about 1.5x the footprint; the base model is frozen here and
    there is no master copy to protect.
    """
    model.half()
    try:
        rows = []
        generator = torch.Generator(device="cpu").manual_seed(0)
        for count in windows:
            ids = torch.randint(
                0,
                int(model.config.vocab_size),
                (count, max_length),
                generator=generator,
                dtype=torch.long,
            ).to(device)
            mask = torch.ones_like(ids)
            tokens = count * max_length
            row: dict[str, object] = {
                "windows": count,
                "tokens": tokens,
                "dtype": "fp16_weights",
            }
            try:

                def once(ids=ids, mask=mask) -> None:
                    with torch.inference_mode():
                        model(ids, mask).last_hidden_state

                result = timed(once, warmup=warmup, repeats=repeats)
                row.update(result)
                row["tokens_per_second"] = tokens / result["seconds"]
            except (
                torch.cuda.OutOfMemoryError,
                RuntimeError,
                NotImplementedError,
            ) as error:
                torch.cuda.empty_cache()
                row["error"] = failure(error)
            rows.append(row)
            del ids, mask
            torch.cuda.empty_cache()
        return rows
    finally:
        # Left half, every later arm would silently measure a different model.
        model.float()


def head_gemm_sweep(
    device: torch.device,
    token_counts: list[int],
    *,
    hidden: int,
    columns: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    """Forward+backward of the entity head's matmul, in each dtype.

    The shape that dominates a training step once the store removes the
    base-model forward. Measured separately, which is what separates "the card
    is slow at bf16" from "the base model is slow".
    """
    rows: list[dict[str, object]] = []
    arms = autocast_arms(device)

    for tokens in token_counts:
        for name, arm in arms.items():
            row: dict[str, object] = {
                "tokens": tokens,
                "hidden": hidden,
                "columns": columns,
                "dtype": name,
            }
            layer: torch.nn.Linear | None = None
            inputs: torch.Tensor | None = None
            try:
                layer = torch.nn.Linear(hidden, columns).to(device)
                inputs = torch.randn(
                    tokens, hidden, device=device, requires_grad=True
                )

                def once(arm=arm, layer=layer, inputs=inputs) -> None:
                    with arm():
                        out = layer(inputs)
                    # `.float()` so the reduction is not the thing being
                    # measured; the GEMM and its backward are.
                    out.float().sum().backward()
                    layer.zero_grad(set_to_none=True)
                    inputs.grad = None

                result = timed(once, warmup=warmup, repeats=repeats)
                row.update(result)
                # Forward plus backward is three GEMMs of the same size.
                row["TFLOPs"] = (
                    3 * 2 * tokens * hidden * columns / result["seconds"] / 1e12
                )
            except (
                torch.cuda.OutOfMemoryError,
                RuntimeError,
                NotImplementedError,
            ) as error:
                row["error"] = failure(error)
            finally:
                del layer, inputs
                torch.cuda.empty_cache()
            rows.append(row)

    return rows


def device_report() -> dict[str, object]:
    """What the card says about itself, including the claims that mislead.

    `is_bf16_supported()` answers for emulation as well as hardware, which is
    how a Pascal card comes to run its whole training loop in a format it
    converts on every op.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    major, minor = torch.cuda.get_device_capability()
    return {
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "capability": [major, minor],
        "vram_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        ),
        "torch": torch.__version__,
        "bf16_supported_reports": torch.cuda.is_bf16_supported(),
        "bf16_hardware": (major, minor) >= (8, 0),
        "triton_compatible": runtime.is_triton_compatible(),
    }


def integers(text: str) -> list[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Base-model and entity-head throughput per dtype and per batch "
            "size on this GPU. Needs no dataset and no store."
        )
    )
    parser.add_argument(
        "--base-model",
        default=ModelConfig().base_model,
        help="the model the store is keyed on; must match the arms' config",
    )
    parser.add_argument(
        "--windows",
        type=integers,
        default=[8, 18, 32, 64, 128, 256],
        help=(
            "512-token windows per forward. 18 is this corpus's mean per "
            "document and 29 its maximum, so everything above 32 is only "
            "reachable by a batcher that crosses documents."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=integers,
        default=[64 * TOKENS_PER_CHUNK, 512 * TOKENS_PER_CHUNK],
        help="token counts for the head GEMM; the defaults are the token "
        "counts batch_max_chunks 64 and 512 imply",
    )
    parser.add_argument("--entity-columns", type=int, default=ENTITY_COLUMNS)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", help="write the report here as JSON")
    return parser.parse_args()


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    print("  " + "  ".join(f"{name:>16}" for name in columns))
    for row in rows:
        cells = []
        for name in columns:
            value = row.get(name, "")
            if isinstance(value, float):
                cells.append(f"{value:>16.4g}")
            else:
                cells.append(f"{str(value):>16}")
        print("  " + "  ".join(cells))


def main() -> int:
    args = read_args()
    runtime.configure()

    report: dict[str, object] = {"device": device_report()}
    print(json.dumps(report["device"], indent=2))

    if not torch.cuda.is_available():
        print("\nno CUDA device: nothing to measure", flush=True)
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
        return 1

    device = torch.device("cuda")
    model = (
        transformers.AutoModel.from_pretrained(args.base_model)
        .to(device)
        .eval()
    )
    max_length = int(model.config.max_position_embeddings)
    report["base_model"] = args.base_model
    report["max_length"] = max_length

    print(f"\n=== base-model forward, {args.base_model} ===")
    forward = forward_sweep(
        model,
        device,
        args.windows,
        warmup=args.warmup,
        repeats=args.repeats,
        max_length=max_length,
    )
    forward += half_weights_sweep(
        model,
        device,
        args.windows,
        warmup=args.warmup,
        repeats=args.repeats,
        max_length=max_length,
    )
    report["forward"] = forward
    print_table(
        forward,
        [
            "windows",
            "dtype",
            "seconds",
            "tokens_per_second",
            "peak_MiB",
            "error",
        ],
    )

    print(
        f"\n=== entity-head GEMM, [T, {args.hidden}] @ "
        f"[{args.hidden}, {args.entity_columns}], fwd+bwd ==="
    )
    head = head_gemm_sweep(
        device,
        args.tokens,
        hidden=args.hidden,
        columns=args.entity_columns,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    report["head_gemm"] = head
    print_table(
        head, ["tokens", "dtype", "seconds", "TFLOPs", "peak_MiB", "error"]
    )

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
