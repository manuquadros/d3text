#!/usr/bin/env python
"""Put the compiled and eager arms' epoch timings side by side.

Medians over repeats rather than one pair of numbers: the arms were
interleaved because a card throttles under load, and a mean would hand a single
thermal outlier the answer. The first epoch is tabulated apart from the rest
because it is the one paying for tracing — whether that cost is worth it
depends on how many epochs a real run has to amortize it over.

The verdict is as much about which arm survived as about which was faster. An
arm with no completed run is reported as the finding it is, never quietly
dropped so that the remaining arm's column reads like a comparison.

    pdm run python scripts/compile_benchmark/compare_arms.py out/run.json
"""

import argparse
import json
import pathlib
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

COMPILED = "compiled"
EAGER = "eager"
ARMS = (COMPILED, EAGER)

SECONDS = "training/epoch_seconds"
RATE = "training/batches_per_second"

MISSING = "—"


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_arms",
        description="Did compiling pay, and did the compiled arm survive?",
    )
    parser.add_argument("run", help="`run.json`, as `run_arms.py` writes it")
    parser.add_argument("--out", help="where to write the merged table")
    return parser.parse_args()


def arm_runs(runs: Iterable[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    """The runs of one arm that finished, in the order they were made.

    :param runs: every run the benchmark made.
    :param arm: the arm to select.
    :return: that arm's completed runs.
    """
    return [run for run in runs if run["arm"] == arm and run.get("completed")]


def series(runs: Sequence[dict[str, Any]], key: str) -> dict[int, list[float]]:
    """One metric's values by epoch, gathered across the repeats.

    :param runs: the runs to read.
    :param key: the metric key.
    :return: every value of that key, keyed by epoch index.
    """
    values: dict[int, list[float]] = {}
    for run in runs:
        for epoch, metrics in run["epochs"].items():
            if key in metrics:
                values.setdefault(int(epoch), []).append(float(metrics[key]))

    return values


def totals(runs: Sequence[dict[str, Any]]) -> list[float]:
    """Each repeat's whole-run training seconds.

    :param runs: the runs to read.
    :return: one total per run.
    """
    return [
        sum(
            float(metrics[SECONDS])
            for metrics in run["epochs"].values()
            if SECONDS in metrics
        )
        for run in runs
    ]


def after_first(values: dict[int, list[float]]) -> list[float]:
    """Every value from an epoch other than the first.

    The first epoch carries the tracing cost, so it is not a sample from the
    same population as the ones after it and must not be pooled with them.

    :param values: the metric by epoch.
    :return: the steady-state values.
    """
    return [
        value
        for epoch, epoch_values in values.items()
        if epoch > 0
        for value in epoch_values
    ]


def median(values: Sequence[float]) -> float | None:
    """`values`' median, or None when there is nothing to take one of.

    :param values: the samples.
    :return: the median, or None.
    """
    return statistics.median(values) if values else None


def number(value: float | None, digits: int = 2) -> str:
    return MISSING if value is None else f"{value:.{digits}f}"


def ratio(slower: float | None, faster: float | None) -> str:
    """`slower / faster`, the speedup, where both are known and non-zero.

    :param slower: the eager arm's seconds.
    :param faster: the compiled arm's seconds.
    :return: the ratio, formatted, or a dash.
    """
    if slower is None or not faster:
        return MISSING

    return f"{slower / faster:.2f}x"


def table(rows: Sequence[Sequence[str]]) -> str:
    """`rows` as a markdown table, the first row being the header.

    :param rows: the header followed by the body.
    :return: the rendered table.
    """
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        for row in rows
    ]
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([lines[0], rule, *lines[1:]])


def summarize(runs: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """One arm's headline timings.

    :param runs: that arm's completed runs.
    :return: the medians the summary table quotes.
    """
    seconds = series(runs, SECONDS)
    rate = series(runs, RATE)

    return {
        "first_epoch_seconds": median(seconds.get(0, [])),
        "later_epoch_seconds": median(after_first(seconds)),
        "whole_run_seconds": median(totals(runs)),
        "later_batches_per_second": median(after_first(rate)),
    }


def report_epochs(
    by_arm: dict[str, list[dict[str, Any]]], *, comparable: bool
) -> None:
    """Print the per-epoch medians, one row an epoch.

    :param by_arm: each arm's completed runs.
    :param comparable: whether a ratio between the columns prices the switch;
        where it does not, the column is left empty rather than disclaimed.
    """
    seconds = {arm: series(runs, SECONDS) for arm, runs in by_arm.items()}
    rate = {arm: series(runs, RATE) for arm, runs in by_arm.items()}
    epochs = sorted({epoch for values in seconds.values() for epoch in values})
    if not epochs:
        print("no epoch finished in either arm.")
        return

    rows = [
        [
            "epoch",
            "compiled s",
            "eager s",
            "eager/compiled",
            "compiled batch/s",
            "eager batch/s",
        ]
    ]
    for epoch in epochs:
        compiled = median(seconds[COMPILED].get(epoch, []))
        eager = median(seconds[EAGER].get(epoch, []))
        rows.append(
            [
                str(epoch),
                number(compiled),
                number(eager),
                ratio(eager, compiled) if comparable else MISSING,
                number(median(rate[COMPILED].get(epoch, []))),
                number(median(rate[EAGER].get(epoch, []))),
            ]
        )
    print(table(rows))


def report_summary(
    by_arm: dict[str, list[dict[str, Any]]], *, comparable: bool
) -> None:
    """Print the headline medians and what they mean.

    :param by_arm: each arm's completed runs.
    :param comparable: whether a ratio between the columns prices the switch.
    """
    summaries = {arm: summarize(runs) for arm, runs in by_arm.items()}
    rows = [["", "compiled", "eager", "eager/compiled"]]
    for key, label in (
        ("first_epoch_seconds", "first epoch (s)"),
        ("later_epoch_seconds", "later epochs (s)"),
        ("whole_run_seconds", "whole run (s)"),
        ("later_batches_per_second", "later epochs (batch/s)"),
    ):
        compiled, eager = summaries[COMPILED][key], summaries[EAGER][key]
        quotable = comparable and key.endswith("seconds")
        rows.append(
            [
                label,
                number(compiled),
                number(eager),
                ratio(eager, compiled) if quotable else MISSING,
            ]
        )
    print(table(rows))
    if comparable:
        print(
            "\n`eager/compiled` above 1.00 means compiling paid; below 1.00 "
            "means it cost."
        )
    else:
        print(
            "\nNo `eager/compiled` is quoted above: an arm did not do what "
            "its name says, so a ratio between these columns would price "
            "something other than the compile switch."
        )


def not_comparable(runs: Sequence[dict[str, Any]]) -> str | None:
    """The first arm that did not do what its name says, if there is one.

    :param runs: every run the benchmark made.
    :return: that arm's diagnostic, or None where the comparison stands.
    """
    for run in runs:
        failure = run.get("switch_failure")
        if failure:
            return str(failure)

    return None


def verdict(runs: Sequence[dict[str, Any]]) -> tuple[str, int]:
    """What this benchmark actually established, and what to exit with.

    :param runs: every run the benchmark made.
    :return: the verdict, and the exit status — non-zero only where the
        comparison is invalid rather than merely negative.
    """
    broken = not_comparable(runs)
    if broken:
        return (
            "THE ARMS ARE NOT COMPARABLE: "
            + broken
            + " No timing below can be read as a compilation effect.",
            1,
        )

    lines = []
    status = 0
    for arm in ARMS:
        attempted = [run for run in runs if run["arm"] == arm]
        finished = [run for run in attempted if run.get("completed")]
        lines.append(f"{arm}: {len(finished)}/{len(attempted)} runs completed")
        if finished or not attempted:
            continue
        # `nn.Module.compile` returns before inductor has built anything, so a
        # compiled arm that never finished an epoch is a card on which
        # compiling does not work at all — the benchmark's answer, not its
        # failure.
        lines.append(
            f"  the {arm} arm never survived an epoch: "
            f"{attempted[0].get('error')}"
        )
        lines.append(f"  its log: {attempted[0].get('log')}")

    if any(not arm_runs(runs, arm) for arm in ARMS):
        lines.append(
            "One arm produced no timings, so there is no speedup to quote; "
            "the columns below are one arm's numbers, not a comparison."
        )
    return "\n".join(lines), status


def main() -> int:
    args = read_args()
    loaded = json.loads(pathlib.Path(args.run).read_text())
    runs = loaded["runs"]
    if not runs:
        raise SystemExit(f"{args.run} records no runs")

    print(
        f"\n=== {loaded['repeats']} repeats, {loaded['epochs']} epochs, "
        f"limit {loaded['limit']} ===\n"
    )
    message, status = verdict(runs)
    print(message)

    by_arm = {arm: arm_runs(runs, arm) for arm in ARMS}
    comparable = not_comparable(runs) is None

    print("\n=== Per epoch, median over the repeats ===\n")
    report_epochs(by_arm, comparable=comparable)

    print("\n=== Summary ===\n")
    report_summary(by_arm, comparable=comparable)

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(
                {arm: summarize(runs) for arm, runs in by_arm.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
