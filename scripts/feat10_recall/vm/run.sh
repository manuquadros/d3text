#!/bin/bash
# FEAT-10's re-measurement on the VM, end to end.
#
#   bash scripts/feat10_recall/vm/run.sh
#
# Three arms differing in one config line — `token_loss_weighting` — each
# trained on the full split and then scored with `evaluate`, whose per-type
# detection block is the number FEAT-01 is waiting on.
#
# Stages run in order and each records a stamp in $OUT/stamps; a rerun skips
# the stages already stamped, so an interrupted run resumes where it stopped.
# Force one with `rm $OUT/stamps/<stage>`, or all with FEAT10_FORCE=1.
set -uo pipefail

REPO="${FEAT10_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

VOL="${FEAT10_VOL:-/vol/storage}"
[[ -d "$VOL" ]] || VOL="$HOME"

PDM="${FEAT10_PDM:-$HOME/.local/bin/pdm}"
OUT="${FEAT10_OUT:-$REPO/scripts/feat10_recall/vm/out}"
CORPUS="$REPO/brenda_references/src/brenda_references/data"
D="$REPO/scripts/feat10_recall"
D3="$REPO/scripts/dec03_full"
D4="$REPO/scripts/dec04_full"

# A new filename, not DEC-04's. `8cb932b` stamps the surface-form index a store
# was built from and made that the store's format 3, so every earlier store is
# refused on open rather than silently reused — and `eb3addc` changed what the
# index holds for `other_organisms`, which is the class this run is about.
LABELS="${FEAT10_LABELS:-$VOL/d3text-token-labels-fmt3.hdf5}"

# Reused if the earlier runs left it. Without it every document falls back to
# the live base-model forward: correct, hours slower, and `configure` says
# which way it went.
STORE="${FEAT10_STORE:-$VOL/d3text-embeddings}"

ENCODINGS="${FEAT10_ENCODINGS:-$REPO/data/biolinkbert-base-zstd-22-encodings.hdf5}"

ARMS="${FEAT10_ARMS:-unweighted balanced focal}"
AUDIT_DOCS="${FEAT10_AUDIT_DOCS:-400}"
SMOKE_DOCS="${FEAT10_SMOKE_DOCS:-20}"

STAMP="${FEAT10_STAMP:-$(date +%Y%m%d)}"
BUNDLE="${FEAT10_BUNDLE:-$VOL/feat10-vm-$STAMP.tar.gz}"

# Read from the config rather than taken as a knob: the labels are placed by
# re-tokenizing with this model's tokenizer, so a store built under one model
# addresses another's encodings nowhere at all, and misses silently.
BASE_MODEL="$(sed -n 's/^base_model *= *"\(.*\)" *$/\1/p' "$D/cfg_base.toml")"
if [[ -z "$BASE_MODEL" ]]; then
  echo "no base_model in $D/cfg_base.toml" >&2; exit 1
fi

mkdir -p "$OUT" "$OUT/stamps"
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"

log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

log "repo     $REPO"
log "labels   $LABELS"
log "store    $STORE"
log "arms     $ARMS"
log "out      $OUT"
log "bundle   $BUNDLE"

stage () {  # stage <name> <command...>
  local name=$1; shift
  if [[ -n "${FEAT10_STOPPED:-}" ]]; then
    log "HOLD  $name (FEAT10_UNTIL=${FEAT10_UNTIL} was reached)"
    return 0
  fi
  if [[ -f "$OUT/stamps/$name" && -z "${FEAT10_FORCE:-}" ]]; then
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
  if [[ "${FEAT10_UNTIL:-}" == "$name" ]]; then
    FEAT10_STOPPED=1
    log "HOLD  stopping after $name, as asked"
  fi
}

# --- 0. can this machine finish the run? ------------------------------------
# DEC-04's, unchanged and called with this run's paths: the checks are the same
# ones — a GPU, the designation guard, and that the encodings still tokenize to
# what the corpus reader produces, which is what the labels are placed against.
# Never stamped: the machine is not what it was when a stamp was written, and
# that is usually why a run is being resumed.
preflight () {
  DEC04_LABELS="$LABELS" DEC04_STORE="$STORE" DEC04_ENCODINGS="$ENCODINGS" \
    "$PDM" run python "$D4/vm/preflight.py" "$OUT/preflight.json" 2>&1 \
    | tee "$OUT/preflight.log"
  return "${PIPESTATUS[0]}"
}

log "START preflight (every run, never skipped)"
if ! preflight; then
  log "FAIL  preflight — stopping here"
  exit 1
fi
log "DONE  preflight"

# --- 1. the token labels, rebuilt under format 3 -----------------------------
# All four sources, the noise pool included: its documents carry no entity
# columns, so every one of their tokens becomes a negative, which is what
# teaches the tagger what text with no mention in it looks like.
#
# This stage depends on nothing about the arms, which is why it is worth
# starting before the comparison is settled — FEAT10_UNTIL=audit runs exactly
# this much.
token_labels () {
  "$PDM" run precompute-token-labels \
      "$BASE_MODEL" "$CORPUS/documents.json" "$LABELS" \
      "$CORPUS/training_data.csv" \
      "$CORPUS/validation_data.csv" \
      "$CORPUS/test_data.csv" \
      "$CORPUS/pmc_linguistics_articles.json" \
    > "$OUT/precompute_token_labels.log" 2>&1
}
stage token_labels token_labels

# --- 2. was it built with the guarded dictionary? ---------------------------
audit () {
  "$PDM" run python "$D4/label_audit.py" \
      "$CORPUS/documents.json" "$LABELS" \
      "$CORPUS/training_data.csv" \
      "$CORPUS/validation_data.csv" \
      "$CORPUS/test_data.csv" \
      --documents "$AUDIT_DOCS" \
      --out "$OUT/label_audit.json" \
    > "$OUT/label_audit.log" 2>&1 || {
      log "the label store was not built with the guarded dictionary"
      log "  see $OUT/label_audit.log; rm $OUT/stamps/token_labels and rerun"
      log "  with FEAT10_FORCE=1 to rebuild it"
      return 1
    }
  grep -E "^(index|  )" "$OUT/label_audit.log" | head -20 \
    | while read -r line; do log "audit: $line"; done
  # Explicit, because `pipefail` is on and `head` closing the pipe early gives
  # `grep` a SIGPIPE: without it a clean audit reports itself as a failure.
  return 0
}
stage audit audit

# --- 3. the arms' configs ----------------------------------------------------
# Generated rather than tracked: `token_labels_store` is an absolute path, so a
# committed config could only ever name one machine's. The check below is what
# makes the comparison attributable — any two arms must differ in exactly the
# weighting line.
write_configs () {
  local arm
  for arm in $ARMS; do
    {
      cat "$D/cfg_base.toml"
      echo ""
      echo "# written by scripts/feat10_recall/vm/run.sh on $(date -Is)"
      echo "token_labels_store = \"$LABELS\""
      echo "token_loss_weighting = \"$arm\""
    } > "$OUT/cfg_$arm.toml"
  done

  local first="" difference
  for arm in $ARMS; do
    if [[ -z "$first" ]]; then first=$arm; continue; fi
    difference=$(diff <(sed '/^#/d;/^$/d' "$OUT/cfg_$first.toml") \
                      <(sed '/^#/d;/^$/d' "$OUT/cfg_$arm.toml") \
                   | grep -c '^>')
    if [[ "$difference" -ne 1 ]]; then
      log "$first and $arm differ in $difference lines, not 1 — they would"
      log "not be comparable; see $OUT/cfg_$arm.toml"
      return 1
    fi
  done
  log "arms differ in exactly one line: token_loss_weighting"
}
stage configs write_configs

# --- 4. point this machine's config at the embeddings store ------------------
configure () {
  local config="$REPO/config.toml"
  if [[ -f "$config" && ! -f "$OUT/config.toml.before" ]]; then
    cp "$config" "$OUT/config.toml.before"
  fi
  {
    echo "# written by scripts/feat10_recall/vm/run.sh on $(date -Is)"
    echo "# the previous file, if any, is at $OUT/config.toml.before"
    echo ""
    echo "cpu_embeddings_cache_size = 0"
    if [[ -d "$STORE" ]]; then
      echo "embeddings_store = \"$STORE\""
    fi
    echo 'float32_matmul_precision = "medium"'
    echo "cudnn_allow_tf32 = true"
    echo "expandable_segments = true"
    echo "tokenizers_parallelism = true"
  } > "$config"
  cp "$config" "$OUT/config.toml.used"
  if [[ -d "$STORE" ]]; then
    log "using the embeddings store at $STORE"
  else
    log "no embeddings store at $STORE — the arms will run the base model live"
    log "  (correct, but hours slower; FEAT10_STORE points at another path)"
  fi
}
stage configure configure

# --- 5. prove the labels are actually being read -----------------------------
# Exit code proves nothing here: a run whose every document misses in the label
# store still trains, with the tagger loss masked to nothing on every token,
# and looks exactly like a run that is working.
smoke () {
  local arm=${ARMS%% *}
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_$arm.toml" "$OUT/smoke.pt" --limit "$SMOKE_DOCS" \
    > "$OUT/smoke.log" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    log "the smoke training run failed (exit $status) — see $OUT/smoke.log"
    return $status
  fi
  local unlabelled
  unlabelled=$(grep -c "has no token labels" "$OUT/smoke.log")
  if [[ "$unlabelled" -gt 0 ]]; then
    log "$unlabelled of $SMOKE_DOCS smoke documents had no token labels"
    if [[ "$unlabelled" -ge "$SMOKE_DOCS" ]]; then
      log "  none of them did — the store does not cover this split"
      return 1
    fi
  fi
  rm -f "$OUT/smoke.pt"
  return 0
}
stage smoke smoke

# --- 6. the arms ------------------------------------------------------------
# Seeded through DEC-03's wrapper so initialization and batch order are shared
# across the arms, full training split, no --limit. The unweighted arm is not
# redundant with FEAT-06's published 42.7%: that was measured at `b99ade7-dirty`
# and against a token-label store this run has just replaced.
train_arm () {  # train_arm <arm>
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_$1.toml" "$OUT/model_$1.pt" \
    > "$OUT/train_$1.log" 2>&1
}

# `evaluate`, wrapped only to keep the metrics it logs. The per-type detection
# numbers — the columns this run exists for — otherwise reach MLflow and
# nowhere else.
eval_arm () {  # eval_arm <arm>
  FEAT10_METRICS_JSON="$OUT/eval_$1.json" \
    "$PDM" run python "$D/evaluate_json.py" \
      "$OUT/cfg_$1.toml" "$OUT/model_$1.pt" \
    > "$OUT/eval_$1.log" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    log "FAIL  evaluate $1 (exit $status) — see $OUT/eval_$1.log"
    return $status
  fi
  grep -iE "detection|precision" "$OUT/eval_$1.log" | tail -5 \
    | while read -r line; do log "$1: $line"; done
  return 0
}

for arm in $ARMS; do
  stage "train_$arm" train_arm "$arm"
  stage "eval_$arm" eval_arm "$arm"
done

# --- 7. the table -----------------------------------------------------------
# A table and not a verdict: the question is what the recall lever costs in
# precision, and which way `other_organisms` moves now that its surface forms
# carry abbreviated genera. Both are tradeoffs to read, not thresholds to pass.
compare () {
  local pairs=()
  local arm
  for arm in $ARMS; do
    [[ -f "$OUT/eval_$arm.json" ]] && pairs+=("$arm=$OUT/eval_$arm.json")
  done
  if [[ ${#pairs[@]} -eq 0 ]]; then
    log "no arm produced metrics — nothing to compare"
    return 1
  fi
  "$PDM" run python "$D/compare_arms.py" "${pairs[@]}" \
      --out "$OUT/arms.json" 2>&1 | tee "$OUT/arms.log"
  return "${PIPESTATUS[0]}"
}
stage compare compare

# --- 8. bundle --------------------------------------------------------------
bundle () {
  local status=0 arm
  cp "$D/cfg_base.toml" "$OUT/" || status=1
  nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  git -C "$REPO" log -1 --format="%H %ad %s" --date=iso > "$OUT/commit.txt" \
    || status=1
  git -C "$REPO" status --porcelain > "$OUT/dirty.txt" || status=1
  for arm in $ARMS; do
    "$PDM" run python scripts/benchmarks/parse_run.py \
        "$OUT/train_$arm.log" > "$OUT/epochs_$arm.txt" 2>&1 || status=1
  done
  cat "$OUT/stamps"/* > "$OUT/timings.txt" 2>/dev/null || status=1
  tar czf "$BUNDLE" -C "$(dirname "$OUT")" --exclude="*.pt" \
      "$(basename "$OUT")" || status=1
  log "bundle: $BUNDLE"
  return $status
}
stage bundle bundle

log "ALL STAGES DONE"
[[ -f "$OUT/arms.log" ]] && cat "$OUT/arms.log"
