"""Put the arms' detection scores side by side, one table per entity type.

The report is a table and not a verdict. FEAT-10's question is what the recall
lever costs in precision, and which way `other_organisms` moves now that its
surface forms carry abbreviated genera; both are tradeoffs to be read, not
thresholds to be passed.

    pdm run python scripts/feat10_recall/compare_arms.py \\
        unweighted=out/eval_unweighted.json balanced=out/eval_balanced.json \\
        --out out/arms.json
"""

import argparse
import json
import pathlib

# The detection block is scored against the same distant labels the arms train
# on, so these are agreement with the matcher rather than correctness, and they
# are blind to entities BRENDA does not carry.
HEADLINE = (
    "test/entity_micro_f1",
    "test/entity_lrap",
    "test/class_micro_f1",
    "test/relation_macro_f1_typed",
    "test/relation_micro_f1_typed",
)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_arms",
        description="Compare the arms' detection and document metrics.",
    )
    parser.add_argument(
        "arms",
        nargs="+",
        metavar="NAME=PATH",
        help="each arm's metrics JSON, as `evaluate_json.py` writes it",
    )
    parser.add_argument("--out", help="where to write the merged table")
    return parser.parse_args()


def load(arms: list[str]) -> dict[str, dict[str, float]]:
    """The arms' metrics, keyed by arm name, in the order given."""
    loaded = {}
    for argument in arms:
        name, _, path = argument.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {argument!r}")
        loaded[name] = json.loads(pathlib.Path(path).read_text())
    return loaded


def types_scored(metrics: dict[str, float]) -> list[str]:
    """The entity types the detection block carries, from the keys themselves.

    Read off the metrics rather than the label space so an arm scored under a
    schema this script does not import still tabulates.
    """
    prefix, suffix = "test/detection_", "_recall"
    # `test/detection_recall` matches both affixes and names no type: the
    # overall score is already tabulated on its own, and an empty type name
    # would tabulate it a second time under a blank heading.
    names = (
        key[len(prefix) : -len(suffix)]
        for key in metrics
        if key.startswith(prefix) and key.endswith(suffix)
    )
    return sorted(name for name in names if name)


def table(rows: list[list[str]]) -> str:
    """`rows` as a markdown table, the first row being the header."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        for row in rows
    ]
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([lines[0], rule, *lines[1:]])


def cell(metrics: dict[str, float], key: str) -> str:
    value = metrics.get(key)
    return "—" if value is None else f"{value:.3f}"


def main() -> None:
    args = read_args()
    arms = load(args.arms)
    names = list(arms)
    first = arms[names[0]]

    print("\n=== Detection, overall (span level, against distant labels) ===\n")
    rows = [["arm", "precision", "recall", "f1", "TP", "FP", "FN"]]
    for name, metrics in arms.items():
        rows.append(
            [
                name,
                cell(metrics, "test/detection_precision"),
                cell(metrics, "test/detection_recall"),
                cell(metrics, "test/detection_f1"),
                f"{int(metrics.get('test/detection_true_positives', 0))}",
                f"{int(metrics.get('test/detection_false_positives', 0))}",
                f"{int(metrics.get('test/detection_false_negatives', 0))}",
            ]
        )
    print(table(rows))

    for entity_type in types_scored(first):
        print(f"\n=== Detection, {entity_type} ===\n")
        rows = [["arm", "precision", "recall", "f1"]]
        for name, metrics in arms.items():
            rows.append(
                [
                    name,
                    cell(metrics, f"test/detection_{entity_type}_precision"),
                    cell(metrics, f"test/detection_{entity_type}_recall"),
                    cell(metrics, f"test/detection_{entity_type}_f1"),
                ]
            )
        print(table(rows))

    print("\n=== The document-level heads, to see what the lever cost ===\n")
    rows = [["metric", *names]]
    for key in HEADLINE:
        rows.append([key, *(cell(arms[name], key) for name in names)])
    print(table(rows))

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(arms, indent=2, sort_keys=True) + "\n"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
