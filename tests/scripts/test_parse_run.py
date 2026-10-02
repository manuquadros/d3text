"""``parse_run.py``: per-epoch times recovered from a run's progress bars."""

import importlib.util
import pathlib
from types import ModuleType

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/benchmarks/parse_run.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("parse_run", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bar(label: str, done: int, total: int, elapsed: str) -> str:
    return f"{label}: 100%|██████████| {done}/{total} [{elapsed}<00:00]"


def test_validation_bars_are_read_as_validation_passes() -> None:
    """Validation draws an `Evaluating` bar, not a `Batches` one. Reading
    only `Batches` bars and pairing them by order labelled epoch 2's
    training pass as epoch 1's validation and dropped the last epoch."""
    log = "\r".join(
        _bar(label, docs, docs, elapsed)
        for _ in range(3)
        for label, docs, elapsed in (
            ("Batches", 21, "00:02"),
            ("Evaluating", 23, "00:01"),
        )
    )
    rows = [
        line.split()
        for line in _load().report(log).splitlines()
        if line.strip()[:1].isdigit()
    ]
    assert rows == [
        [str(epoch), "21", "0:02", "23", "0:01", "0:03", "33%"]
        for epoch in (1, 2, 3)
    ]
