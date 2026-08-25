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
                "not hardware, so Model.amp_dtype picks bf16 while "
                "precompute-embeddings uses fp16 — see the profile_card stage"
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
