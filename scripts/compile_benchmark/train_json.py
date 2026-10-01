#!/usr/bin/env python
"""Run `train` and keep the per-epoch timings it logs, as JSON.

`tracking.log_metrics` is a no-op without a tracking server, so the timings
are captured by wrapping it; wrapping keeps the timed path the shipped one.

    pdm run python scripts/compile_benchmark/train_json.py \\
        out/metrics.json cfg.toml out/model.pt --limit 500
"""

import json
import pathlib
import sys
import traceback
from collections.abc import Mapping

import torch
from d3text import runtime, tracking

#: Where the metrics logged with no epoch — the dataset and model sizes — go.
NO_STEP = "none"

#: How much of a failing run's exception to keep. The full traceback is in the
#: run's log; this is the line a table has room to quote.
ERROR_CHARACTERS = 400


def capture_epoch_metrics() -> dict[str, dict[str, float]]:
    """Keep every metric `train` logs, filed under the epoch it belongs to.

    Per epoch and not merged into one dict: the first epoch is the one that
    pays for tracing, so a capture that kept only the last value would hide
    exactly the cost this benchmark is asking about.

    :return: the store the wrapper fills, keyed by epoch as a string.
    """
    collected: dict[str, dict[str, float]] = {}
    logged = tracking.log_metrics

    def capture(metrics: Mapping[str, float], step: int | None = None) -> None:
        key = NO_STEP if step is None else str(step)
        collected.setdefault(key, {}).update(metrics)
        logged(metrics, step)

    tracking.log_metrics = capture  # type: ignore[assignment]
    return collected


def capture_compilation() -> dict[str, bool]:
    """Keep what `compile_trunk` installed and what the epochs ran.

    A later backend failure silently removes the installed graph, so the
    retag `train` makes after `fit` is what says the epochs ran compiled. A
    frozen trunk never reaches `runtime.compile_model`, so
    `"graph_installed"` stays absent rather than False.

    :return: the store the wrappers fill, under `"graph_installed"` and, once
        `fit` has returned, `"compiled"` and `"recompile_limit_hit"`.
    """
    result: dict[str, bool] = {}
    compile_model = runtime.compile_model
    set_tags = tracking.set_tags

    def compile_wrapper(model: torch.nn.Module) -> bool:
        result["graph_installed"] = compile_model(model)
        return result["graph_installed"]

    def tag_wrapper(tags: Mapping[str, str]) -> None:
        if "compiled" in tags:
            result["compiled"] = tags["compiled"] == "true"
        if "recompile_limit_hit" in tags:
            result["recompile_limit_hit"] = (
                tags["recompile_limit_hit"] == "true"
            )
        set_tags(tags)

    runtime.compile_model = compile_wrapper  # type: ignore[assignment]
    tracking.set_tags = tag_wrapper  # type: ignore[assignment]
    return result


def error_summary(error: BaseException) -> str:
    """The exception, in the one line a comparison table can quote.

    :param error: the exception the run died of.
    :return: its type and message, whitespace collapsed and truncated.
    """
    rendered = " ".join(
        "".join(traceback.format_exception_only(type(error), error)).split()
    )
    return rendered[:ERROR_CHARACTERS]


def summarize(
    collected: Mapping[str, dict[str, float]],
    compilation: Mapping[str, bool],
    error: str | None,
) -> dict[str, object]:
    """What this run did, as the record the comparison reads.

    `compiled` is the post-`fit` retag wherever the run got that far and
    `compile_trunk`'s return only where it did not. The two disagree exactly
    when a graph was installed and the eager fallback later cleared it, and
    both are kept because that arm is unusable for a different reason than one
    which never compiled at all: its epochs are a mixture of the two arms.

    :param collected: the metrics the run logged, by epoch.
    :param compilation: what `compile_trunk` and the post-`fit` retag said.
    :param error: the exception that ended the run, or None if it finished.
    :return: the record to write.
    """
    installed = compilation.get("graph_installed")
    return {
        "compiled": compilation.get("compiled", installed),
        "graph_installed": installed,
        "recompile_limit_hit": compilation.get("recompile_limit_hit"),
        "completed": error is None,
        "error": error,
        "epochs": {
            epoch: metrics
            for epoch, metrics in collected.items()
            if epoch != NO_STEP
        },
        "context": dict(collected.get(NO_STEP, {})),
    }


def main(argv: list[str] | None = None) -> None:
    """Time a `train` run, writing what it logged to a JSON file.

    :param argv: the command line, `sys.argv` by default; its first argument
        is the destination and everything after it is `train`'s own.
    :raises SystemExit: with the destination missing, or after a run that
        raised — the file is written either way.
    """
    arguments = list(sys.argv if argv is None else argv)
    if len(arguments) < 2:
        raise SystemExit(
            "usage: train_json.py <metrics.json> <config> <output> [options]"
        )

    destination = pathlib.Path(arguments[1])
    collected = capture_epoch_metrics()
    compilation = capture_compilation()

    # Deferred: `d3text.cli.train` reaches `d3text.data`, which attaches a log
    # handler to a relative path at import time, so importing it at module
    # scope would write a file in whatever directory a reader imported from.
    from d3text.cli import train

    sys.argv = [arguments[0], *arguments[2:]]

    # A run that dies is a result: compile fails at the first forward, not
    # at install. `completed` keeps it from reading as a whole run.
    error = None
    try:
        train.main()
    except Exception as exc:
        error = error_summary(exc)
    finally:
        destination.write_text(
            json.dumps(
                summarize(collected, compilation, error),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    if error is not None:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
