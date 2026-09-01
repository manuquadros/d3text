#!/usr/bin/env python
"""Can this machine finish the run?

Reuses the earlier preflight's `encodings_agree_with_the_corpus` outright,
which matters more here: the token labels are placed against offsets from a
*re-tokenization*, so an encodings file that no longer reproduces what the
corpus reader produces puts every code on the wrong token.

**It deliberately does not inherit the disk gate.** This run never builds an
embeddings store — it reuses the earlier one if the volume still has it and
falls back to the live base-model forward if not — so the gate would refuse a
perfectly runnable machine over space it was never going to use.
"""

import json
import os
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
CORPUS = REPO / "brenda_references/src/brenda_references/data"
ENCODINGS = pathlib.Path(
    os.environ.get("DEC04_ENCODINGS")
    or REPO / "data/biolinkbert-base-zstd-22-encodings.hdf5"
)

sys.path.insert(0, str(REPO / "scripts/dec03_full/vm"))

AGREEMENT_DOCS = int(os.environ.get("DEC04_AGREEMENT_DOCS", "30"))

# The label store is small — int8 codes and mention spans, tens of MB over the
# whole corpus — but a volume with nothing left on it fails in the middle of
# the build rather than here.
LABEL_STORE_HEADROOM_GIB = 2.0


def main() -> int:
    report: dict[str, object] = {}
    problems: list[str] = []
    notes: list[str] = []

    import torch

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1
        )
    else:
        notes.append("no CUDA device; the arms will train on CPU and crawl")

    # The one precondition that is new, and the one a `git pull` alone does not
    # satisfy. Without the frequency table the index is built unguarded, and
    # the run then measures the mislabelling it exists to remove — silently,
    # since an unguarded index is a perfectly valid one.
    try:
        from d3text.surface_forms import COMMON_WORD_ZIPF, is_common_word

        report["common_word_zipf"] = COMMON_WORD_ZIPF
        if not is_common_word("sensitive") or is_common_word("catalase"):
            problems.append(
                "the designation guard is installed but does not separate "
                "'sensitive' from 'catalase'; COMMON_WORD_ZIPF is miscalibrated"
            )
    except ImportError as error:
        report["common_word_zipf"] = None
        problems.append(
            f"this checkout has no designation guard ({error}); run "
            "`pdm install -L locks/<flavour>.lock --frozen-lockfile`"
        )

    # The tokenizer version is not a version check but a behaviour one: 5.16.1
    # installs and imports perfectly and returns two windows where 5.15.1
    # returns thirteen. `tests/test_utils.py` pins this; repeating it here is
    # what keeps a VM whose environment drifted from finding out four hours in.
    try:
        from d3text import utils

        tokenizer = utils.load_fast_tokenizer(
            "hf-internal-testing/tiny-random-BertModel"
        )
        text = " ".join(f"token{n} of the sequence," for n in range(600))
        windows = len(
            utils.split_and_tokenize(
                tokenizer=tokenizer, inputs=text, max_length=64, stride=20
            )["input_ids"]
        )
        report["windows_for_probe_text"] = windows
        if windows < 100:
            problems.append(
                f"split_and_tokenize returned {windows} windows for a "
                "13,000-token text; this transformers release truncates "
                "documents (see the pin in pyproject.toml)"
            )
    except Exception as error:  # noqa: BLE001 - reported, not handled
        problems.append(f"could not check the tokenizer's windowing: {error}")

    # Every script a later stage shells out to, checked now. `probe_baseline`
    # runs after ninety minutes of training, so a path that does not resolve
    # has to be caught here or it is paid for twice — which is exactly how the
    # probe's old home under the untracked `design/` went unnoticed until a VM
    # checkout reached that stage.
    for helper in (
        REPO / "scripts/dec02_probe/localization_probe.py",
        REPO / "scripts/dec03_full/seeded_train.py",
        REPO / "scripts/dec04_full/label_audit.py",
        REPO / "scripts/dec04_full/compare.py",
    ):
        if not helper.is_file():
            problems.append(f"no {helper.relative_to(REPO)} in this checkout")

    for name in (
        "documents.json",
        "training_data.csv",
        "validation_data.csv",
        "test_data.csv",
        "pmc_linguistics_articles.json",
    ):
        if not (CORPUS / name).is_file():
            problems.append(f"no {name} in {CORPUS}")

    report["encodings"] = str(ENCODINGS)
    if ENCODINGS.is_file():
        from preflight import encodings_agree_with_the_corpus

        checked, disagreeing, examples = encodings_agree_with_the_corpus(
            AGREEMENT_DOCS
        )
        report["encodings_checked"] = checked
        report["encodings_disagreeing"] = disagreeing
        report["encodings_disagreement_examples"] = examples
        if disagreeing:
            problems.append(
                f"{disagreeing} of {checked} documents tokenize to something "
                "other than their stored encodings; the labels would land on "
                f"the wrong tokens. Examples: {examples[:3]}"
            )
    else:
        notes.append(
            f"no encodings at {ENCODINGS}; the probe will not cross-check "
            "tokenization, and nothing verifies the labels line up"
        )

    labels = pathlib.Path(os.environ.get("DEC04_LABELS") or REPO / "labels.h5")
    target = labels.parent if labels.parent.exists() else REPO
    free_gib = shutil.disk_usage(target).free / 2**30
    report["label_store_target"] = str(target)
    report["label_store_free_gib"] = round(free_gib, 1)
    if free_gib < LABEL_STORE_HEADROOM_GIB:
        problems.append(
            f"{free_gib:.1f} GiB free at {target}, which is not enough for "
            "the token-label store"
        )

    store = pathlib.Path(os.environ.get("DEC04_STORE") or "")
    report["embeddings_store"] = str(store)
    report["embeddings_store_present"] = store.is_dir()
    if not store.is_dir():
        notes.append(
            f"no embeddings store at {store}; the arms will run the base "
            "model live, which is correct and hours slower"
        )

    report["notes"] = notes
    print(json.dumps(report, indent=2))
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2))

    if problems:
        print("\nPREFLIGHT FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\npreflight OK")
    for note in notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
