"""Fetch the BRENDA data blobs from the Hugging Face Hub and verify them.

Paths are resolved from ``__file__`` rather than through
``resources.files("brenda_references")`` on purpose: this is a bootstrap
script, so it has to run before the package is necessarily importable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys

DATA_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "brenda_references"
    / "data"
)
MANIFEST = DATA_DIR / "SHA256SUMS"
DEFAULT_REPO = "manuquadros/brenda-references-data"
CHUNK_SIZE = 1 << 20


def read_manifest(path: pathlib.Path) -> dict[str, str]:
    """Parse a ``sha256sum``-format manifest into {filename: digest}."""
    if not path.is_file():
        msg = f"No manifest at {path}"
        raise SystemExit(msg)

    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if not name:
            msg = f"Malformed manifest line: {line!r}"
            raise SystemExit(msg)
        entries[name.strip()] = digest.strip()

    return entries


def file_digest(path: pathlib.Path) -> str:
    """Return the hex sha256 of `path`, read incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def verify(expected: dict[str, str]) -> list[str]:
    """Return one message per file that is missing or has the wrong digest."""
    problems: list[str] = []
    for name, want in sorted(expected.items()):
        path = DATA_DIR / name
        if not path.is_file():
            problems.append(f"{name}: MISSING")
            continue

        got = file_digest(path)
        status = "OK" if got == want else "FAILED"
        print(f"{name}: {status}")
        if got != want:
            problems.append(f"{name}: expected {want}, got {got}")

    return problems


def download(repo: str, names: list[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = (
            "huggingface_hub is required to download the data."
            " Install it with `pdm install` or `pip install huggingface_hub`."
        )
        raise SystemExit(msg) from exc

    # Restricted to the manifest's names so the Hub repo's own README.md
    # cannot overwrite the one tracked in this directory.
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=names,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the files already on disk without downloading",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("BRENDA_DATA_REPO", DEFAULT_REPO),
        help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})",
    )
    args = parser.parse_args()

    expected = read_manifest(MANIFEST)

    if not args.check:
        print(f"Downloading {len(expected)} files from {args.repo}...")
        download(args.repo, sorted(expected))

    problems = verify(expected)
    if problems:
        print("\nVerification failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"\nAll {len(expected)} files verified against {MANIFEST.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
