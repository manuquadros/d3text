"""Publish changed BRENDA data blobs as one Hub commit and re-pin them.

Hashes every file the manifest lists and uploads the ones whose digest
changed in a single commit whose parent is the revision in ``HUB_REVISION``,
refusing to start unless that revision is still the repo's head. It then
writes the new digests to ``SHA256SUMS`` and the id the upload returned to
``HUB_REVISION``, rather than asking the Hub for its head afterwards, which
could name a concurrent upload. When every changed file is already
identical on the Hub, no commit is made and only ``SHA256SUMS`` changes.
Nothing is written locally unless the upload succeeds. Exits 1 when no file
changed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

try:
    from scripts.pull_data import (
        DATA_DIR,
        DEFAULT_REPO,
        HUB_REVISION,
        MANIFEST,
        file_digest,
        read_manifest,
        read_revision,
    )
except ModuleNotFoundError:
    # Run as a script, only this directory is on the path, not its parent.
    from pull_data import (  # type: ignore[no-redef,import-not-found]
        DATA_DIR,
        DEFAULT_REPO,
        HUB_REVISION,
        MANIFEST,
        file_digest,
        read_manifest,
        read_revision,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--message",
        help="Hub commit message (required unless --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which files changed without uploading or writing",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("BRENDA_DATA_REPO", DEFAULT_REPO),
        help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})",
    )
    args = parser.parse_args()
    if not args.dry_run and not args.message:
        parser.error("-m/--message is required unless --dry-run")

    old = read_manifest(MANIFEST)
    parent = read_revision(HUB_REVISION)

    missing = sorted(n for n in old if not (DATA_DIR / n).is_file())
    if missing:
        msg = f"Manifest files not found in {DATA_DIR}: {', '.join(missing)}"
        raise SystemExit(msg)

    new = {name: file_digest(DATA_DIR / name) for name in old}
    changed = [name for name in old if new[name] != old[name]]
    for name in old:
        print(f"{name}: {'changed' if name in changed else 'unchanged'}")

    if not changed:
        print("Nothing changed; nothing to publish.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    stale = (
        f"If {args.repo} has moved past {parent}, pull it, rebuild and run"
        " this again. Nothing was written locally."
    )
    head = api.repo_info(args.repo, repo_type="dataset").sha
    if head != parent:
        msg = f"The Hub's head is {head}, not the pinned {parent}.\n{stale}"
        raise SystemExit(msg)

    try:
        info = api.create_commit(
            args.repo,
            operations=[
                CommitOperationAdd(
                    path_in_repo=name, path_or_fileobj=DATA_DIR / name
                )
                for name in changed
            ],
            commit_message=args.message,
            repo_type="dataset",
            parent_commit=parent,
        )
    except HfHubHTTPError as exc:
        msg = f"The Hub refused the commit: {exc}\n{stale}"
        raise SystemExit(msg) from exc

    if not re.fullmatch(r"[0-9a-f]{40}", info.oid):
        msg = f"The Hub returned {info.oid!r}, not a commit sha; not pinned."
        raise SystemExit(msg)

    MANIFEST.write_text("".join(f"{new[n]}  {n}\n" for n in old))
    if info.oid == parent:
        print(f"The Hub already holds these files at {parent}; no commit made.")
        print(f"Commit {MANIFEST}.")
        return 0
    HUB_REVISION.write_text(f"{info.oid}\n")
    print(f"Published {len(changed)} file(s) as {info.oid}.")
    print(f"Commit {MANIFEST} and {HUB_REVISION} together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
