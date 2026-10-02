"""Per-epoch wall times, recovered from a training run's progress bars.

Without a tracking server the tqdm bars are the only record. Training draws
a `Batches` bar and validation an `Evaluating` one; the label assigns each
pass, since sizes swap under `--limit`.
"""

import argparse
import pathlib
import re
import sys

# tqdm redraws in place with a carriage return, so one "line" holds many bar
# states; the count and the elapsed clock are what identify a finished pass.
_BAR = re.compile(
    r"(?P<label>Batches|Evaluating|Epochs): *\d+%\|[^|]*\| *"
    r"(?P<done>\d+)/(?P<total>\d+) \[(?P<elapsed>[\d:]+)<"
)


def _seconds(clock: str) -> int:
    """`MM:SS` or `H:MM:SS` as seconds."""
    parts = [int(part) for part in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def completed_passes(text: str) -> list[tuple[str, int, int]]:
    """The `(label, documents, seconds)` of each pass's bar, in order.

    A pass's duration is its **last** frame, not the one where
    `done == total`: tqdm redraws on a timer, so a pass may never draw full
    or draw 100% early. Dropping one would re-label every later bar.
    """
    frames = []
    for line in text.replace("\r", "\n").splitlines():
        found = _BAR.search(line)
        if found is not None and found["label"] != "Epochs":
            frames.append(
                (
                    found["label"],
                    int(found["done"]),
                    int(found["total"]),
                    _seconds(found["elapsed"]),
                )
            )

    passes: list[tuple[str, int, int]] = []
    for index, (label, done, total, seconds) in enumerate(frames):
        last = index + 1 == len(frames)
        if not last:
            next_label, next_done, next_total, _ = frames[index + 1]
            # Counter went back or a new bar took over: the pass ended. Strict,
            # since a bar redraws without advancing within one pass.
            last = (
                next_label != label or next_total != total or next_done < done
            )
        if last:
            passes.append((label, total, seconds))
    return passes


def epoch_passes(
    passes: list[tuple[str, int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int] | None]]:
    """Each epoch's training pass, with the validation pass after it if any."""
    epochs: list[tuple[tuple[int, int], tuple[int, int] | None]] = []
    for label, docs, seconds in passes:
        if label == "Batches":
            epochs.append(((docs, seconds), None))
        elif epochs and epochs[-1][1] is None:
            epochs[-1] = (epochs[-1][0], (docs, seconds))
    return epochs


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
    epochs = epoch_passes(completed_passes(text))
    if not epochs:
        return "no completed progress bars found — was the run interrupted?"

    lines = [
        f"{'epoch':>5s} {'train docs':>10s} {'train':>8s} "
        f"{'val docs':>9s} {'val':>8s} {'epoch':>8s} {'val share':>10s}"
    ]
    for epoch, ((train_docs, train_seconds), validation) in enumerate(
        epochs, start=1
    ):
        # An interrupted run can end mid-validation.
        val_docs, val_seconds = validation or (0, 0)
        total = train_seconds + val_seconds
        share = f"{val_seconds / total:.0%}" if total else "-"
        lines.append(
            f"{epoch:5d} {train_docs:10d} {_clock(train_seconds):>8s} "
            f"{val_docs:9d} {_clock(val_seconds):>8s} "
            f"{_clock(total):>8s} {share:>10s}"
        )

    trainings = [training for training, _ in epochs]
    validations = [validation for _, validation in epochs if validation]

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
