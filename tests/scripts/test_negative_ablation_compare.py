"""Parsing and diffing two `evaluate` logs' class-metrics tables."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts/dec04_full")
)
from negative_ablation_compare import class_table, verdict  # noqa: E402

REPORT = """
=== Entity CLASS metrics (multilabel, document-level) ===
micro-F1: 0.908
                 precision    recall  f1-score   support

        strains       0.61      0.88      0.72       419
       bacteria       0.68      0.93      0.79       267
other_organisms       0.96      0.92      0.94       819
        enzymes       1.00      1.00      1.00      1174

      micro avg       0.87      0.95      0.91      2679
"""


def test_class_table_reads_the_four_class_rows() -> None:
    table = class_table(REPORT)

    assert table["bacteria"] == {
        "precision": 0.68,
        "recall": 0.93,
        "f1": 0.79,
        "support": 267,
    }
    # The micro/macro/weighted/samples averages are not one of the four
    # entity classes and must not be read as one.
    assert set(table) == {"strains", "bacteria", "other_organisms", "enzymes"}


def test_class_table_rejects_a_log_with_no_class_report() -> None:
    with pytest.raises(ValueError, match="strains"):
        class_table("evaluate crashed before it got anywhere\n")


def test_verdict_reports_the_recall_delta_per_class() -> None:
    before = class_table(REPORT)
    after_text = REPORT.replace(
        "bacteria       0.68      0.93", "bacteria       0.70      0.97"
    )
    after = class_table(after_text)

    result = verdict(before, after)

    bacteria = next(row for row in result["rows"] if row["class"] == "bacteria")
    assert bacteria["recall_before"] == 0.93
    assert bacteria["recall_after"] == 0.97
    assert bacteria["recall_delta"] == pytest.approx(0.04)
