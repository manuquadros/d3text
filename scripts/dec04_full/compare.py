#!/usr/bin/env python
"""Does token supervision fix the class channel's mention ranking?

The `other_organisms` channel was measured scoring gold mention tokens *below*
ordinary prose, which is what the label noise predicts: a positive document
pushes up one token, a false-negative document pushes down all of them. If
token supervision supplies the localization the pooled loss cannot, that
inversion must go.

`lift` is the statistic — mean probability on gold mention tokens over mean
probability on background, below 1.0 meaning the channel ranks mentions worse
than prose. The two arms differ in exactly one config line. Read the enzyme row
with the caveat that at 98.3% textual anchoring it is the least independent of
the four.
"""

import argparse
import json
import pathlib

CLASSES = ("enzymes", "bacteria", "strains", "other_organisms")

# The channel the prediction is about. DEC-02 measured it anti-localized under
# both poolings — 0.0015 against 0.0110 under logsumexp, and lift 0.822 under
# the logmeanexp that shipped — so it is the one whose sign carries the answer.
DECIDING_CLASS = "other_organisms"


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dec04-compare",
        description="Did token supervision undo the anti-localization?",
    )
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("tagger", type=pathlib.Path)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def summary(path: pathlib.Path) -> dict[str, dict[str, float]]:
    loaded = json.loads(path.read_text())
    scores: dict[str, dict[str, float]] = loaded["summary"]
    return scores


def main() -> int:
    args = read_args()
    baseline, tagger = summary(args.baseline), summary(args.tagger)

    rows = []
    for name in CLASSES:
        before, after = baseline.get(name), tagger.get(name)
        if before is None or after is None:
            continue
        rows.append(
            {
                "class": name,
                "lift_baseline": round(before["lift"], 3),
                "lift_tagger": round(after["lift"], 3),
                "auc_baseline": round(before["token_auc"], 3),
                "auc_tagger": round(after["token_auc"], 3),
                "gold_baseline": round(before["mean_prob_gold"], 4),
                "background_baseline": round(before["mean_prob_background"], 4),
                "gold_tagger": round(after["mean_prob_gold"], 4),
                "background_tagger": round(after["mean_prob_background"], 4),
            }
        )

    header = (
        f"{'class':18s} {'lift before':>12s} {'lift after':>11s} "
        f"{'AUC before':>11s} {'AUC after':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['class']:18s} {row['lift_baseline']:12.3f} "
            f"{row['lift_tagger']:11.3f} {row['auc_baseline']:11.3f} "
            f"{row['auc_tagger']:10.3f}"
        )

    deciding = next(
        (row for row in rows if row["class"] == DECIDING_CLASS), None
    )
    verdict: dict[str, object]
    if deciding is None:
        verdict = {"verdict": "inconclusive", "why": f"no {DECIDING_CLASS} row"}
    elif deciding["lift_baseline"] >= 1.0:
        # The premise did not reproduce, so this run cannot test the
        # prediction: there was no inversion for the supervision to undo.
        verdict = {
            "verdict": "premise absent",
            "why": (
                f"{DECIDING_CLASS} was not anti-localized in the baseline "
                f"(lift {deciding['lift_baseline']}), so option 3's "
                "prediction has nothing to bite on here. Compare against "
                "DEC-02's arm before concluding anything."
            ),
        }
    elif deciding["lift_tagger"] > 1.0:
        verdict = {
            "verdict": "option 3 survives",
            "why": (
                f"{DECIDING_CLASS} lift moved "
                f"{deciding['lift_baseline']} -> {deciding['lift_tagger']}, "
                "crossing 1.0: token supervision undid the inversion the "
                "document-level label noise produced, which is what option 3 "
                "predicted. The document term can stay as it is."
            ),
        }
    else:
        verdict = {
            "verdict": "option 3 falsified",
            "why": (
                f"{DECIDING_CLASS} lift is still {deciding['lift_tagger']} "
                "with token supervision in place, so a supervised token loss "
                "does not override the noisy pooled one. The choice is now "
                "between option 1 (abstain at document level) and option 2 "
                "(down-weight), and the designation guard that option 1 "
                "waited on has landed."
            ),
        }

    print(f"\n{verdict['verdict'].upper()}\n{verdict['why']}")

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps({"rows": rows, **verdict}, indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
