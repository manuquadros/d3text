#!/bin/bash
# The measurement DEC-04's full-split run left untaken: does removing the
# document-level false-negative label noise (class_negative_abstention) buy
# anything, once it is actually removed rather than merely carried?
#
#   bash scripts/dec04_full/vm/run_negative_ablation.sh
#
# Requires the earlier run.sh to have completed: this reuses its label store,
# its embeddings store, and its "tagger" arm's config and checkpoint as the
# "before" side of the comparison, rather than re-deriving any of them.
# Stages resume the same way run.sh's do: a stamp in $OUT/stamps skips a
# finished stage on rerun; `rm` it, or set DEC04NA_FORCE=1, to redo one.
set -uo pipefail

REPO="${DEC04NA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

PDM="${DEC04NA_PDM:-$HOME/.local/bin/pdm}"
BASE_OUT="${DEC04_OUT:-$REPO/scripts/dec04_full/vm/out}"
OUT="${DEC04NA_OUT:-$REPO/scripts/dec04_full/vm/out_negative_ablation}"
D3="$REPO/scripts/dec03_full"
COMPARE="$REPO/scripts/dec04_full/negative_ablation_compare.py"

TAGGER_CONFIG="$BASE_OUT/cfg_tagger.toml"
TAGGER_CHECKPOINT="$BASE_OUT/model_tagger.pt"

mkdir -p "$OUT" "$OUT/stamps"
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"

log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

stage () {  # stage <name> <command...>
  local name=$1; shift
  if [[ -f "$OUT/stamps/$name" && -z "${DEC04NA_FORCE:-}" ]]; then
    log "SKIP  $name (already done; rm $OUT/stamps/$name to redo)"
    return 0
  fi
  log "START $name"
  local started=$SECONDS
  "$@"
  local status=$?
  local elapsed=$((SECONDS - started))
  if [[ $status -ne 0 ]]; then
    log "FAIL  $name after ${elapsed}s (exit $status) — stopping here"
    exit $status
  fi
  echo "$name $(date -Is) ${elapsed}s" > "$OUT/stamps/$name"
  log "DONE  $name in ${elapsed}s"
}

# --- 0. the earlier run's artifacts have to exist ----------------------------
require_prior_run () {
  local missing=0
  for f in "$TAGGER_CONFIG" "$TAGGER_CHECKPOINT"; do
    if [[ ! -f "$f" ]]; then
      log "missing $f — run scripts/dec04_full/vm/run.sh first"
      missing=1
    fi
  done
  [[ $missing -eq 0 ]]
}
log "START preflight (every run, never skipped)"
if ! require_prior_run; then
  log "FAIL  preflight — stopping here"
  exit 1
fi
log "DONE  preflight"

# --- 0b. stamp the commit both arms train under ------------------------------
# Without this, a resumed run after a `git checkout` of a different commit
# would silently train the two arms on different code and nothing would catch
# it — the config diff a few lines down only checks the one line the ablation
# means to isolate, not the training/eval code each checkpoint actually ran.
check_commit () {
  local current
  current=$(git -C "$REPO" log -1 --format="%H %ad %s" --date=iso)
  if [[ -f "$OUT/commit.txt" ]]; then
    local stamped
    stamped=$(cat "$OUT/commit.txt")
    if [[ "$stamped" != "$current" ]]; then
      log "FAIL  HEAD changed since this run started:"
      log "      stamped: $stamped"
      log "      current: $current"
      log "      a resumed run must not mix commits across its two arms — checkout the stamped commit, or start a fresh \$OUT"
      return 1
    fi
    log "commit unchanged since this run started: $current"
  else
    echo "$current" > "$OUT/commit.txt"
    log "commit: $current"
  fi
}
log "START commit stamp (every run, never skipped)"
if ! check_commit; then
  log "FAIL  commit stamp — stopping here"
  exit 1
fi
log "DONE  commit stamp"

# --- 1. the ablation arm's config --------------------------------------------
# One line added to the tagger config that already exists: token_labels_store
# is unchanged, so this reuses the same label store and the same base
# checkpoint's config, and the two arms below differ in exactly one line —
# class_negative_abstention — same attributability argument run.sh makes for
# token_labels_store against the baseline.
write_abstain_config () {
  cp "$TAGGER_CONFIG" "$OUT/cfg_abstain.toml"
  echo "class_negative_abstention = true" >> "$OUT/cfg_abstain.toml"
  local difference
  difference=$(diff <(sed '/^#/d;/^$/d' "$TAGGER_CONFIG") \
                    <(sed '/^#/d;/^$/d' "$OUT/cfg_abstain.toml") \
                 | grep -c '^>')
  if [[ "$difference" -ne 1 ]]; then
    log "the two arms differ in $difference lines, not 1 — see $OUT/cfg_abstain.toml"
    return 1
  fi
  log "arms differ in exactly one line: class_negative_abstention"
}
stage abstain_config write_abstain_config

# --- 1b. the baseline arm's config -------------------------------------------
# Symmetric with write_abstain_config: an explicit `class_negative_abstention
# = false` line, so the two configs differ in exactly the one line the
# ablation means to isolate, and neither arm's behaviour depends on the
# option's default.
write_baseline_config () {
  cp "$TAGGER_CONFIG" "$OUT/cfg_baseline_fresh.toml"
  echo "class_negative_abstention = false" >> "$OUT/cfg_baseline_fresh.toml"
  local difference
  difference=$(diff <(sed '/^#/d;/^$/d' "$TAGGER_CONFIG") \
                    <(sed '/^#/d;/^$/d' "$OUT/cfg_baseline_fresh.toml") \
                 | grep -c '^>')
  if [[ "$difference" -ne 1 ]]; then
    log "the two arms differ in $difference lines, not 1 — see $OUT/cfg_baseline_fresh.toml"
    return 1
  fi
  log "arms differ in exactly one line: class_negative_abstention"
}
stage baseline_config write_baseline_config

# --- 2. train both arms fresh, at the same commit ----------------------------
# $BASE_OUT/model_tagger.pt was trained by the earlier run.sh, potentially at
# an older commit — reusing it as the "before" side would compare a stale
# checkpoint's training code against train_abstain's current one, not just
# the one config line the comparison is meant to isolate. Training both arms
# here, under the commit stamp above, keeps that the only difference.
train_baseline_fresh () {
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_baseline_fresh.toml" "$OUT/model_baseline_fresh.pt" \
    > "$OUT/train_baseline_fresh.log" 2>&1
}
stage train_baseline_fresh train_baseline_fresh

# Seeded through the same wrapper the original two arms used, so
# initialization and batch order match theirs — full training split, no
# --limit, same as run.sh.
train_abstain () {
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_abstain.toml" "$OUT/model_abstain.pt" \
    > "$OUT/train_abstain.log" 2>&1
}
stage train_abstain train_abstain

# --- 3. evaluate both arms fresh ---------------------------------------------
# The "before" side is evaluated off the freshly trained baseline checkpoint,
# not $TAGGER_CHECKPOINT: $TAGGER_CHECKPOINT was trained by the earlier run.sh
# and its weights carry whatever training code was checked out then, which
# current evaluation code cannot retroactively correct — training it fresh at
# the same commit train_abstain runs at is what keeps the two arms comparable
# on exactly the one config line they're meant to differ by.
evaluate_arm () {  # evaluate_arm <name> <config> <checkpoint>
  "$PDM" run evaluate "$2" "$3" > "$OUT/evaluate_$1.log" 2>&1
}
stage evaluate_tagger  evaluate_arm tagger  "$OUT/cfg_baseline_fresh.toml" "$OUT/model_baseline_fresh.pt"
stage evaluate_abstain evaluate_arm abstain "$OUT/cfg_abstain.toml" "$OUT/model_abstain.pt"

# --- 4. the verdict -----------------------------------------------------------
compare () {
  "$PDM" run python "$COMPARE" \
      "$OUT/evaluate_tagger.log" "$OUT/evaluate_abstain.log" \
      --out "$OUT/verdict.json" 2>&1 | tee "$OUT/verdict.log"
  return "${PIPESTATUS[0]}"
}
stage compare compare

log "ALL STAGES DONE"
[[ -f "$OUT/verdict.log" ]] && cat "$OUT/verdict.log"
