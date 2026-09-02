"""Run `evaluate` and keep every metric it logs, as JSON beside its log.

`evaluate_model` returns the dict it prints, but the console carries only the
three overall detection numbers; the per-type precision and recall — the
columns this comparison is about — reach MLflow and nowhere else, so a machine
with no tracking server scores the arms and keeps none of the answer. Wrapping
the one logging call rather than re-implementing `main` keeps the path being
scored the shipped one.

    FEAT10_METRICS_JSON=out/eval_balanced.json pdm run python \\
        scripts/feat10_recall/evaluate_json.py cfg.toml model.pt
"""

import json
import os
import pathlib
from collections.abc import Mapping

from d3text import tracking

DESTINATION = os.environ.get("FEAT10_METRICS_JSON")
if not DESTINATION:
    raise SystemExit("FEAT10_METRICS_JSON must name the file to write")

collected: dict[str, float] = {}
log_metrics = tracking.log_metrics


def capture(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Record `metrics` here as well as wherever tracking sends them."""
    collected.update(metrics)
    log_metrics(metrics, step)


tracking.log_metrics = capture  # type: ignore[assignment]

from d3text.cli import evaluate  # noqa: E402

evaluate.main()

# After `main`, not at exit: a run that died mid-evaluation has scored part of
# a split, and a file of half a split's metrics is indistinguishable from a
# file of a whole one.
pathlib.Path(DESTINATION).write_text(
    json.dumps(collected, indent=2, sort_keys=True) + "\n"
)
