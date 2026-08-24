"""Per-epoch wall times, recovered from a training run's progress bars.

PERF-07 asks for epoch wall time. The training loop computes it — `seconds`
in `print_epoch_stats` — but only ever sends it to MLflow, so a run without a
tracking server keeps the number to itself and the log shows losses and nothing
else. Until that changes, the tqdm bars are the only record on disk, and this
turns them back into numbers rather than leaving them to be read off by hand.

Both passes draw a bar labelled `Batches`, and the training pass always runs
first within an epoch (`self.train()` before `validate_model`'s `self.eval()`),
so the completed bars alternate training, validation, training, validation.
That ordering is what assigns them, not the document counts — a full training
split is larger than validation, a `--limit`ed one is smaller, and reading the
roles off the sizes would silently swap them at some limit.
"""

import argparse
import pathlib
import re
import sys

# tqdm redraws in place with a carriage return, so one "line" holds many bar
# states; the count and the elapsed clock are what identify a finished pass.
_BAR = re.compile(
    r"(?P<label>Batches|Epochs): *\d+%\|[^|]*\| *"
    r"(?P<done>\d+)/(?P<total>\d+) \[(?P<elapsed>[\d:]+)<"
)


def _seconds(clock: str) -> int:
    """`MM:SS` or `H:MM:SS` as seconds."""
    parts = [int(part) for part in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def completed_passes(text: str) -> list[tuple[int, int]]:
    """The `(documents, seconds)` of each finished `Batches` bar, in order.

    A bar at 100% is redrawn several times with the same clock, so identical
    consecutive states are one pass. Two passes cannot collide here even when
    they take the same number of seconds, because a training bar and a
    validation bar always separate them.
    """
    passes: list[tuple[int, int]] = []
    for line in text.replace("\r", "\n").splitlines():
        found = _BAR.search(line)
        if found is None or found["label"] != "Batches":
            continue
        done, total = int(found["done"]), int(found["total"])
        if done != total:
            # tqdm rounds to 100% before the last batch lands; that state
            # carries a stale clock.
            continue
        state = (total, _seconds(found["elapsed"]))
        if not passes or passes[-1] != state:
            passes.append(state)
    return passes


def epoch_total(text: str) -> int | None:
    """The whole run's wall clock, from the outer `Epochs` bar."""
    last = None
    for line in text.replace("\r", "\n").splitlines():
        found = _BAR.search(line)
        if found is not None and found["label"] == "Epochs":
            last = _seconds(found["elapsed"])
    return last


def _clock(seconds: int) -> str:
    return f"{seconds // 60:d}:{seconds % 60:02d}"


def report(text: str) -> str:
    passes = completed_passes(text)
    if not passes:
        return "no completed progress bars found — was the run interrupted?"

    lines = [
        f"{'epoch':>5s} {'train docs':>10s} {'train':>8s} "
        f"{'val docs':>9s} {'val':>8s} {'epoch':>8s} {'val share':>10s}"
    ]
    for index in range(0, len(passes), 2):
        epoch = index // 2 + 1
        train_docs, train_seconds = passes[index]
        if index + 1 < len(passes):
            val_docs, val_seconds = passes[index + 1]
        else:
            # An interrupted run can end mid-validation.
            val_docs, val_seconds = 0, 0
        total = train_seconds + val_seconds
        share = f"{val_seconds / total:.0%}" if total else "-"
        lines.append(
            f"{epoch:5d} {train_docs:10d} {_clock(train_seconds):>8s} "
            f"{val_docs:9d} {_clock(val_seconds):>8s} "
            f"{_clock(total):>8s} {share:>10s}"
        )

    # Slicing by parity rather than indexing from the end: a run killed during
    # a training pass leaves an odd number of bars, and `passes[-2]` would then
    # be a validation bar wearing a training label.
    trainings = passes[0::2]
    validations = passes[1::2]

    lines.append("")
    for label, group in (("training", trainings), ("validation", validations)):
        rates = [docs / seconds for docs, seconds in group if seconds]
        if not rates:
            continue
        lines.append(
            f"{label + ':':12s} first {rates[0]:5.1f} doc/s, "
            f"last {rates[-1]:5.1f} doc/s"
        )

    whole = epoch_total(text)
    if whole is not None:
        lines.append(f"run wall clock:      {_clock(whole)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        type=pathlib.Path,
        nargs="+",
        help="the run's log; the first one holding progress bars is used",
    )
    args = parser.parse_args()

    for path in args.log:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        if completed_passes(text):
            print(f"# from {path}")
            print(report(text))
            return

    named = ", ".join(str(path) for path in args.log)
    print(f"no completed progress bars in any of: {named}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
