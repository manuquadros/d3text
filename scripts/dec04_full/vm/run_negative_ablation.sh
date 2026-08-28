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

# --- 2. train the ablation arm ------------------------------------------------
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
# The tagger arm is re-evaluated here too, not read off the original run's
# log: that log predates BUG-79's fix and carries no detection block, and
# re-running costs one inference pass (~15-20 min) against a checkpoint
# that already exists — cheaper than trusting two runs on different code to
# agree.
evaluate_arm () {  # evaluate_arm <name> <config> <checkpoint>
  "$PDM" run evaluate "$2" "$3" > "$OUT/evaluate_$1.log" 2>&1
}
stage evaluate_tagger  evaluate_arm tagger  "$TAGGER_CONFIG" "$TAGGER_CHECKPOINT"
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
