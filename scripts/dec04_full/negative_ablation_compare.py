#!/usr/bin/env python
"""The document-level measurement DEC-04's run left untaken.

The 2026-08-27 full-split run falsified this ticket's *mechanism* — the
organism channels localize well despite the false-negative label noise, so
the noise is not visibly damaging them — but never measured whether the noise
costs anything at all once it is actually removed. `class_negative_abstention`
removes it (masks a document-level class negative wherever the token-label
store's dictionary matched that type anywhere in the text); this script reads
two `evaluate` runs' logs — one with it off, one with it on, everything else
identical — and diffs the class head's per-class document precision/recall/F1,
which is `evaluate_model`'s own "Entity CLASS metrics" table and needs no
separate probe.

Read `bacteria` and `strains` first: those are the two channels DEC-04 measures
the false-negative rate on (roughly half and a third of their document
negatives respectively). A recall increase with precision roughly held is the
result this ablation is for; a real drop in `other_organisms` or `enzymes`
recall is the trade-off this run makes it possible to see.
"""

import argparse
import json
import pathlib
import re

CLASSES = ("strains", "bacteria", "other_organisms", "enzymes")

_ROW = re.compile(
    r"^\s*(?P<name>" + "|".join(CLASSES) + r")\s+"
    r"(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+(?P<f1>[\d.]+)\s+"
    r"(?P<support>\d+)\s*$",
    re.MULTILINE,
)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dec04-negative-ablation-compare")
    parser.add_argument("before", help="evaluate log without the abstention")
    parser.add_argument("after", help="evaluate log with the abstention")
    parser.add_argument("--out", help="write the verdict as JSON here too")
    return parser.parse_args()


def class_table(log_text: str) -> dict[str, dict[str, float]]:
    """This log's `Entity CLASS metrics` row per class.

    :raises ValueError: if a class the report always prints is missing, which
        means the log is not what `evaluate` produces (a crash, a truncated
        file, or the wrong log entirely) rather than a class this run happens
        to score zero.
    """
    table = {
        match["name"]: {
            "precision": float(match["precision"]),
            "recall": float(match["recall"]),
            "f1": float(match["f1"]),
            "support": int(match["support"]),
        }
        for match in _ROW.finditer(log_text)
    }
    missing = [name for name in CLASSES if name not in table]
    if missing:
        msg = f"no class-metrics row for {missing} — not an evaluate log?"
        raise ValueError(msg)
    return table


def verdict(
    before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]
) -> dict[str, object]:
    rows = [
        {
            "class": name,
            "recall_before": before[name]["recall"],
            "recall_after": after[name]["recall"],
            "recall_delta": after[name]["recall"] - before[name]["recall"],
            "precision_before": before[name]["precision"],
            "precision_after": after[name]["precision"],
            "precision_delta": (
                after[name]["precision"] - before[name]["precision"]
            ),
            "support": after[name]["support"],
        }
        for name in CLASSES
    ]
    return {"rows": rows}


def render(result: dict[str, object]) -> str:
    header = (
        f"{'class':<16}{'recall (before->after)':<26}"
        f"{'precision (before->after)':<28}support"
    )
    lines = [header, "-" * len(header)]
    for row in result["rows"]:  # type: ignore[index]
        lines.append(
            f"{row['class']:<16}"
            f"{row['recall_before']:.3f} -> {row['recall_after']:.3f}"
            f" ({row['recall_delta']:+.3f}){'':<3}"
            f"{row['precision_before']:.3f} -> {row['precision_after']:.3f}"
            f" ({row['precision_delta']:+.3f}){'':<3}"
            f"{row['support']}"
        )
    return "\n".join(lines)


def main() -> int:
    args = read_args()
    before = class_table(pathlib.Path(args.before).read_text())
    after = class_table(pathlib.Path(args.after).read_text())

    result = verdict(before, after)
    print(render(result))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
