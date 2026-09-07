#!/usr/bin/env bash
# Run wily without letting it check out revisions in the working tree.
#
# `wily build` walks the history with a real `git checkout` per revision, in
# whatever directory it was pointed at. Pointed at this repo it would swing the
# live tree through two hundred detached-HEAD checkouts: a concurrent test run
# or editing session would be reading whatever revision the walk had reached,
# and a build killed outright leaves the tree detached. A dedicated worktree
# takes that churn instead.
#
# Two things therefore have to be pinned by hand. The cache, because wily
# derives its path from a hash of the directory it ran in, which would give the
# worktree and the repo separate caches and leave `diff` — the one command that
# must see the edited files, so the one command that runs here — reading an
# empty one. And the config, because the worktree's own `wily.cfg` is whatever
# the checked-out revision carried, not what this checkout says.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
WORKTREE=${WILY_WORKTREE:-$HOME/.cache/wily/$(basename "$ROOT")}
CACHE=${WILY_CACHE:-$HOME/.wily/$(basename "$ROOT")}

# Resolve the venv's wily rather than going through `pdm run`: the build runs
# in the worktree, and pdm would take the pyproject.toml sitting there as its
# project root and look for a different venv. `.pdm-python` holds the
# interpreter's absolute path, so its directory is the venv's bin. A wily on
# PATH — a standalone `uv tool install` — is the fallback.
WILY=""
if [ -r "$ROOT/.pdm-python" ]; then
    candidate="$(dirname "$(cat "$ROOT/.pdm-python")")/wily"
    if [ -x "$candidate" ]; then WILY=$candidate; fi
fi
if [ -z "$WILY" ]; then WILY=$(command -v wily || true); fi
if [ -z "$WILY" ]; then
    cat >&2 <<'MSG'
wily was not found in the project venv or on PATH. It is a dev dependency, so
a synced venv has it:

    TMPDIR=~/.cache/pdm-tmp pdm install -L locks/<flavour>.lock --frozen-lockfile
MSG
    exit 1
fi

if [ $# -eq 0 ]; then
    cat >&2 <<'MSG'
usage: scripts/wily.sh <subcommand> [args]

  build            index src/d3text over the last `max_revisions` commits
  build [args]     passed to `wily build` verbatim, e.g. build -n 1200 src
  rank | report | diff | graph | index | list-metrics
                   read the cache built above
MSG
    exit 2
fi

wily=("$WILY" --config "$ROOT/wily.cfg" --cache "$CACHE")

if [ "$1" = build ]; then
    shift
    head=$(git -C "$ROOT" rev-parse HEAD)
    git -C "$ROOT" worktree prune
    if [ -e "$WORKTREE/.git" ]; then
        # A build killed before its cleanup leaves the worktree on a historical
        # revision, and wily refuses to start on a dirty one; --force is what
        # makes the next build recover rather than report someone else's mess.
        git -C "$WORKTREE" checkout --force --detach "$head" --
    else
        rm -rf "$WORKTREE"
        mkdir -p "$(dirname "$WORKTREE")"
        git -C "$ROOT" worktree add --detach "$WORKTREE" "$head"
    fi
    cd "$WORKTREE"
    if [ $# -eq 0 ]; then set -- src/d3text; fi
    # Always build from scratch. wily treats the first revision of each
    # invocation as a seed and skips its carry-forward of unchanged files, so
    # an incremental run writes a latest-revision snapshot holding only that
    # commit's own files — and `rank`, which reads the latest revision, then
    # lists two files instead of the tree. A full build is ~25 seconds.
    rm -rf "$CACHE"
    exec "${wily[@]}" build "$@"
fi

cd "$ROOT"
exec "${wily[@]}" "$@"
