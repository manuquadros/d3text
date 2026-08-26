"""Refuse to start the VM run unless the machine can finish it.

Every check here is one that fails *late* otherwise: a missing corpus file
after the store is half built, a checkout without the store reader after two
hours of embedding, a disk that runs out at 90 GiB. The report is written to
JSON as well as printed, because it is the record of what the run was measured
on.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
CORPUS = REPO / "brenda_references/src/brenda_references/data"
ENCODINGS = REPO / "data/biolinkbert-base-zstd-22-encodings.hdf5"
BASE_MODEL = "michiyasunaga/BioLinkBERT-base"
SOURCES = (
    "training_data.csv",
    "validation_data.csv",
    "test_data.csv",
    "pmc_linguistics_articles.json",
)

# The store needs 100.8 GiB by measurement; the margin is for the LMDB's own
# pages and for not filling the filesystem.
STORE_GIB = 100.8
HEADROOM_GIB = 15.0


def gib(n: float) -> str:
    return f"{n:,.1f} GiB"


def card_notes(capability: tuple[int, int], bf16_reported: bool) -> list[str]:
    """What this card silently ignores, said once rather than never.

    The run writes `float32_matmul_precision` and `cudnn_allow_tf32` into
    `config.toml` and calls `torch.compile` behind a capability gate, and on a
    pre-Volta or pre-Ampere card some of that is inert. None of it fails, none
    of it warns, and none of it appears in a log — so a five-hour run can be
    tuned entirely by knobs the hardware has no units for.

    `is_bf16_supported()` is the one that misleads hardest: it answers for
    *emulation* as well as hardware, so a True on a Pascal card is what makes
    `Model.amp_dtype` choose a format that card has to convert on every op.
    """
    notes = [f"compute capability {capability[0]}.{capability[1]}"]

    if capability < (7, 0):
        notes.append(
            "Triton cannot target this card, so torch.compile is skipped "
            "(runtime.is_triton_compatible)"
        )
    if capability < (8, 0):
        notes.append(
            "no TF32 units: float32_matmul_precision and cudnn_allow_tf32 "
            "have no effect here"
        )
        if bf16_reported:
            notes.append(
                "torch.cuda.is_bf16_supported() reports True by emulation, "
                "not hardware — Model.amp_dtype asks by compute capability "
                "instead and trains this card in fp16, which is what "
                "precompute-embeddings already used. Emulated bf16 measured "
                "27% slower at three times the peak memory; see the "
                "profile_card stage for this card's own numbers"
            )

    return notes


def default_store() -> pathlib.Path:
    """Where `run.sh` would put the store, for a hand-run preflight.

    It exports `DEC03_STORE`, so this only matters when someone checks a
    machine before starting — and then it has to name the same filesystem the
    run will fill, or the free-disk check answers about the wrong one.
    """
    volume = pathlib.Path("/vol/storage")
    if volume.is_dir():
        return volume / "d3text-embeddings"
    return pathlib.Path.home() / "d3text-embeddings"


def encodings_agree_with_the_corpus(sample: int) -> tuple[int, int, list[str]]:
    """Do the encodings still tokenize to what today's corpus reader produces?

    They are two recordings of the same text made at different times, and the
    reader has been fixed since some of them were written -- an empty abstract
    used to reach the tokenizer as the literal string "nan", one token of it,
    in about 3% of documents. Nothing downstream compares them: training reads
    the encodings and the store is built from the text, so a stale file is a
    silent disagreement that surfaces, at best, as every document falling back
    to the base model seventy minutes into a store build.

    Returns (checked, disagreeing, examples). Documents absent from the file
    are skipped rather than counted: they are the loader's business, not this
    check's.
    """
    import h5py
    import hdf5plugin  # noqa: F401  registers the Zstd filter h5py needs
    import polars
    import torch

    from d3text import corpus, utils
    from d3text.utils.utils import aggregate_embeddings, split_and_tokenize

    tokenizer = utils.load_fast_tokenizer(BASE_MODEL)
    checked = disagreeing = 0
    examples: list[str] = []

    def aggregated_rows(mask) -> int:
        """The document length the aggregation yields, counted with no values:
        a zero-width feature dimension makes it free and keeps it honest."""
        return aggregate_embeddings(
            torch.empty((mask.shape[0], mask.shape[1], 0)), mask
        ).shape[0]

    with h5py.File(ENCODINGS, "r") as encodings:
        for name in SOURCES:
            path = CORPUS / name
            scan = (
                polars.scan_csv(path)
                if path.suffix == ".csv"
                else polars.scan_ndjson(path).rename({"body": "fulltext"})
            )
            frame = scan.select("pubmed_id", "abstract", "fulltext")
            rows = frame.collect()
            step = max(1, len(rows) // max(sample, 1))

            for pubmed_id, abstract, fulltext in rows[::step].iter_rows():
                group = encodings.get(str(pubmed_id))
                if group is None or "attention_mask" not in group:
                    continue
                stored = torch.from_numpy(
                    group["attention_mask"][:].astype("int64")
                )
                text = corpus.document_text(abstract, fulltext)
                if not text:
                    continue
                fresh = split_and_tokenize(
                    tokenizer=tokenizer, inputs=text, stride=20, max_length=512
                )["attention_mask"]

                checked += 1
                if aggregated_rows(stored) != aggregated_rows(fresh):
                    disagreeing += 1
                    if len(examples) < 5:
                        examples.append(
                            f"{pubmed_id}: encodings {aggregated_rows(stored)}"
                            f" vs corpus {aggregated_rows(fresh)}"
                        )

    return checked, disagreeing, examples


def main() -> int:
    store = pathlib.Path(os.environ.get("DEC03_STORE") or default_store())
    report: dict[str, object] = {}
    problems: list[str] = []

    import torch

    report["host"] = os.uname().nodename
    report["cores"] = os.cpu_count()
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()

    notes: list[str] = []

    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
        capability = torch.cuda.get_device_capability()
        report["capability"] = list(capability)
        report["bf16"] = torch.cuda.is_bf16_supported()
        report["bf16_hardware"] = capability >= (8, 0)
        report["triton_compatible"] = capability >= (7, 0)
        notes.extend(card_notes(capability, report["bf16"]))
    else:
        problems.append("no CUDA device: this run needs a GPU")

    with open("/proc/meminfo") as meminfo:
        total_kb = int(meminfo.readline().split()[1])
    report["ram_gib"] = round(total_kb / 1024**2, 1)

    # The store's own filesystem, which is not necessarily the repo's.
    target = store if store.exists() else store.parent
    usage = shutil.disk_usage(target)
    free_gib = usage.free / 1024**3
    report["store_path"] = str(store)
    report["store_filesystem_free_gib"] = round(free_gib, 1)
    already = (
        sum(f.stat().st_size for f in store.glob("*.mdb") if f.is_file())
        / 1024**3
        if store.exists()
        else 0.0
    )
    report["store_existing_gib"] = round(already, 1)

    needed = STORE_GIB + HEADROOM_GIB - already
    if free_gib < needed:
        problems.append(
            f"{gib(free_gib)} free at {target}, but the store needs about "
            f"{gib(needed)} more ({gib(STORE_GIB)} measured, plus headroom)"
        )

    for name in (
        "training_data.csv",
        "validation_data.csv",
        "test_data.csv",
        "pmc_linguistics_articles.json",
    ):
        path = CORPUS / name
        if not path.exists():
            problems.append(f"corpus file missing: {path}")
    report["corpus"] = str(CORPUS)

    if not ENCODINGS.exists():
        problems.append(
            f"encodings missing: {ENCODINGS}. The store is verified against "
            "the token counts these imply, and training reads them directly."
        )
    report["encodings"] = str(ENCODINGS)

    if ENCODINGS.exists():
        checked, disagreeing, examples = encodings_agree_with_the_corpus(
            int(os.environ.get("DEC03_AGREEMENT_DOCS", "30"))
        )
        report["encodings_checked"] = checked
        report["encodings_disagreeing"] = disagreeing
        report["encodings_disagreement_examples"] = examples
        if disagreeing:
            problems.append(
                f"{disagreeing} of {checked} sampled documents tokenize to a "
                f"different length than the encodings hold ({examples}). The "
                "encodings and the corpus reader are out of step -- most "
                "likely the file predates a fix to the reader, and one token "
                "of difference where a document has no abstract is the "
                'signature of the one that wrote "nan" into it. Training '
                "would read that text, and the store, built from the corpus, "
                "would disagree with it document by document. Rebuild with "
                "`precompute-encodings`, or copy a file built since the fix."
            )

    # The reader is what the whole plan rests on; a checkout without it would
    # run the experiment at the speed the store exists to avoid.
    try:
        from d3text.embeddings_store import EmbeddingsStore  # noqa: F401
        from d3text.models.models import document_token_count  # noqa: F401

        report["store_reader"] = True
    except ImportError as error:
        report["store_reader"] = False
        problems.append(
            f"this checkout has no embeddings-store reader ({error}); "
            "pull the branch that carries it before running"
        )

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked=no"],
        capture_output=True,
        text=True,
    )
    report["commit"] = head.stdout.strip() or "unknown"
    report["dirty"] = bool(dirty.stdout.strip())

    config = REPO / "config.toml"
    report["config_toml"] = config.read_text() if config.exists() else None

    report["notes"] = notes
    print(json.dumps(report, indent=2))
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
