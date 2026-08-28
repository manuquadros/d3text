#!/bin/bash
# DEC-04's falsification test on the VM, end to end.
#
#   bash scripts/dec04_full/vm/run.sh
#
# Stages run in order and each records a stamp in $OUT/stamps; a rerun skips
# the stages already stamped, so an interrupted run resumes where it stopped.
# Force one with `rm $OUT/stamps/<stage>`, or all with DEC04_FORCE=1.
#
# Everything the run produces lands in $OUT and is tarred up at the end.
set -uo pipefail

REPO="${DEC04_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

VOL="${DEC04_VOL:-/vol/storage}"
[[ -d "$VOL" ]] || VOL="$HOME"

PDM="${DEC04_PDM:-$HOME/.local/bin/pdm}"
OUT="${DEC04_OUT:-$REPO/scripts/dec04_full/vm/out}"
CORPUS="$REPO/brenda_references/src/brenda_references/data"
D="$REPO/scripts/dec04_full"
D3="$REPO/scripts/dec03_full"

# The localization probe. Under `scripts/`, not `design/`, and that is the
# whole point: `design/` is untracked, so a fresh VM checkout does not have it
# and a runner that reached in there failed at `probe_baseline` — after the
# ninety minutes `train_baseline` had already spent.
PROBE="$REPO/scripts/dec02_probe/localization_probe.py"

# The token-label store. Small next to the embeddings store — per-token int8
# codes plus mention spans, a few hundred MB over the whole corpus — but it
# goes on $VOL anyway, because `data/` is neither tracked nor ignored and a
# stray artifact there is how the last one got swept into a commit.
LABELS="${DEC04_LABELS:-$VOL/d3text-token-labels.hdf5}"

# Reused, not rebuilt. The DEC-03 run left ~101 GiB of precomputed embeddings
# on this volume and they are keyed by the same base model, so this run should
# find them and skip two hours. If the path does not exist the run still
# works — every document falls back to the live base-model forward — so this
# is a speed setting, not a correctness one, and `configure` says which way it
# went.
STORE="${DEC04_STORE:-$VOL/d3text-embeddings}"

ENCODINGS="${DEC04_ENCODINGS:-$REPO/data/biolinkbert-base-zstd-22-encodings.hdf5}"

PROBE_DOCS="${DEC04_PROBE_DOCS:-200}"
PROBE_NOISE="${DEC04_PROBE_NOISE:-50}"
AUDIT_DOCS="${DEC04_AUDIT_DOCS:-400}"

STAMP="${DEC04_STAMP:-$(date +%Y%m%d)}"
BUNDLE="${DEC04_BUNDLE:-$VOL/dec04-vm-$STAMP.tar.gz}"

# Read from the config rather than taken as a knob, for the reason the DEC-03
# runner gives: the labels are placed by re-tokenizing with this model's
# tokenizer, and a store built under one model addresses another's encodings
# nowhere at all — silently, one masked document at a time.
read_base_model () { sed -n 's/^base_model *= *"\(.*\)" *$/\1/p' "$1"; }
BASE_MODEL="$(read_base_model "$D/cfg_baseline.toml")"
if [[ -z "$BASE_MODEL" ]]; then
  echo "no base_model in $D/cfg_baseline.toml" >&2; exit 1
fi

mkdir -p "$OUT" "$OUT/stamps"
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"

log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

log "repo     $REPO"
log "labels   $LABELS"
log "store    $STORE"
log "out      $OUT"
log "bundle   $BUNDLE"

stage () {  # stage <name> <command...>
  local name=$1; shift
  if [[ -n "${DEC04_STOPPED:-}" ]]; then
    log "HOLD  $name (DEC04_UNTIL=${DEC04_UNTIL} was reached)"
    return 0
  fi
  if [[ -f "$OUT/stamps/$name" && -z "${DEC04_FORCE:-}" ]]; then
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
  if [[ "${DEC04_UNTIL:-}" == "$name" ]]; then
    DEC04_STOPPED=1
    log "HOLD  stopping after $name, as asked"
  fi
}

# --- 0. can this machine finish the run? ------------------------------------
# This run's own, reusing DEC-03's encodings-agreement check and dropping its
# ~101 GiB disk gate — that run built an embeddings store and this one never
# does. Never stamped, for the reason DEC-03's file gives: the machine is not
# what it was when a stamp was written, and that is usually why a run is being
# resumed.
preflight () {
  DEC04_LABELS="$LABELS" DEC04_STORE="$STORE" DEC04_ENCODINGS="$ENCODINGS" \
    "$PDM" run python "$D/vm/preflight.py" "$OUT/preflight.json" 2>&1 \
    | tee "$OUT/preflight.log"
  return "${PIPESTATUS[0]}"
}

log "START preflight (every run, never skipped)"
if ! preflight; then
  log "FAIL  preflight — stopping here"
  exit 1
fi
log "DONE  preflight"

# --- 1. the three-way token labels ------------------------------------------
# All four sources, the noise pool included. Its documents carry no entity
# columns, so every one of their tokens becomes a negative — which is the
# point: they are the out-of-domain negatives, and leaving them unlabelled
# would mask them out of the tagger loss entirely rather than teaching it what
# text with no mention in it looks like.
#
# Resumable on its own: a document already keyed is skipped without -f.
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
# The one check that catches a stale store, and it has to exist because the
# store records its label space but *not* the dictionary that filled it
# (BUG-60). A store built before the guard trains the tagger on `sensitive` as
# a strain in a quarter of the corpus and reports nothing at all.
audit () {
  "$PDM" run python "$D/label_audit.py" \
      "$CORPUS/documents.json" "$LABELS" \
      "$CORPUS/training_data.csv" \
      "$CORPUS/validation_data.csv" \
      "$CORPUS/test_data.csv" \
      --documents "$AUDIT_DOCS" \
      --out "$OUT/label_audit.json" \
    > "$OUT/label_audit.log" 2>&1 || {
      log "the label store was not built with the guarded dictionary"
      log "  see $OUT/label_audit.log; rm $OUT/stamps/token_labels and rerun"
      log "  with DEC04_FORCE=1 to rebuild it"
      return 1
    }
  grep -E "^(index|  )" "$OUT/label_audit.log" | head -20 \
    | while read -r line; do log "audit: $line"; done
  # Explicit, because `pipefail` is on and `head` closing the pipe early gives
  # `grep` a SIGPIPE: without this the stage's status would be that echo's,
  # and a clean audit would report itself as a failure.
  return 0
}
stage audit audit

# --- 3. the tagger arm's config ---------------------------------------------
# Generated rather than tracked: `token_labels_store` is an absolute path, so a
# committed config could only ever name one machine's. The two arms are then
# guaranteed to differ in exactly this one line, which is what makes the
# comparison attributable.
write_tagger_config () {
  {
    cat "$D/cfg_baseline.toml"
    echo ""
    echo "# written by scripts/dec04_full/vm/run.sh on $(date -Is)"
    echo "token_labels_store = \"$LABELS\""
  } > "$OUT/cfg_tagger.toml"
  local difference
  difference=$(diff <(sed '/^#/d;/^$/d' "$D/cfg_baseline.toml") \
                    <(sed '/^#/d;/^$/d' "$OUT/cfg_tagger.toml") \
                 | grep -c '^>')
  if [[ "$difference" -ne 1 ]]; then
    log "the two arms differ in $difference lines, not 1 — they would not be"
    log "comparable; see $OUT/cfg_tagger.toml"
    return 1
  fi
  log "arms differ in exactly one line: token_labels_store"
}
stage tagger_config write_tagger_config

# --- 4. point this machine's config at the embeddings store ------------------
# Only if DEC-03 left one here. Without it the run is correct and slow, so this
# says which way it went rather than insisting.
configure () {
  local config="$REPO/config.toml"
  if [[ -f "$config" && ! -f "$OUT/config.toml.before" ]]; then
    cp "$config" "$OUT/config.toml.before"
  fi
  {
    echo "# written by scripts/dec04_full/vm/run.sh on $(date -Is)"
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
    log "  (correct, but hours slower; DEC04_STORE points at another path)"
  fi
}
stage configure configure

# --- 5. prove the labels are actually being read -----------------------------
# Exit code proves nothing here, the same way it proved nothing for the
# embeddings store: a run whose every document misses in the label store still
# trains, with the tagger loss masked to nothing on every token, and looks
# exactly like a run that is working.
smoke () {
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_tagger.toml" "$OUT/smoke.pt" --limit 20 \
    > "$OUT/smoke.log" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    log "the smoke training run failed (exit $status) — see $OUT/smoke.log"
    return $status
  fi
  # The store and the encodings disagreeing about a document's token count is
  # raised by the reader, so it shows up as a crash above. What does *not*
  # raise is a document the store simply has no row for: it is warned once and
  # masked, so a store built over the wrong splits degrades into a run with no
  # token supervision at all.
  local unlabelled
  unlabelled=$(grep -c "has no token labels" "$OUT/smoke.log")
  if [[ "$unlabelled" -gt 0 ]]; then
    log "$unlabelled of 20 smoke documents had no token labels"
    if [[ "$unlabelled" -ge 20 ]]; then
      log "  none of them did — the store does not cover this split"
      return 1
    fi
  fi
  rm -f "$OUT/smoke.pt"
  return 0
}
stage smoke smoke

# --- 6. the two arms ---------------------------------------------------------
# Identical but for `token_labels_store`, seeded through DEC-03's wrapper so
# initialization and batch order are shared, full training split, no --limit.
# The baseline is not redundant with DEC-02's published numbers: those were
# taken at --limit 500, where noise=450 puts the split at 47% noise against the
# corpus's own 4.8%, and under a pooling that has since been replaced.
train_arm () {  # train_arm <arm> <config>
  "$PDM" run python "$D3/seeded_train.py" "$2" "$OUT/model_$1.pt" \
    > "$OUT/train_$1.log" 2>&1
}

probe_arm () {  # probe_arm <arm> <config>
  local encodings=()
  [[ -f "$ENCODINGS" ]] && encodings=(--encodings "$ENCODINGS")
  # `${a[@]+"${a[@]}"}` rather than `"${a[@]}"`: `set -u` is on, and expanding
  # an empty array unquoted-guarded is an error on bash before 4.4. The probe
  # runs without the cross-check when there are no encodings to check against.
  "$PDM" run python "$PROBE" \
      "$2" "$OUT/model_$1.pt" \
      --documents "$PROBE_DOCS" --noise-documents "$PROBE_NOISE" \
      ${encodings[@]+"${encodings[@]}"} \
      --out "$OUT/probe_$1.json" \
    > "$OUT/probe_$1.log" 2>&1
}

stage train_baseline train_arm baseline "$D/cfg_baseline.toml"
stage probe_baseline probe_arm baseline "$D/cfg_baseline.toml"
stage train_tagger   train_arm tagger   "$OUT/cfg_tagger.toml"
stage probe_tagger   probe_arm tagger   "$OUT/cfg_tagger.toml"

# --- 7. the verdict ----------------------------------------------------------
compare () {
  "$PDM" run python "$D/compare.py" \
      "$OUT/probe_baseline.json" "$OUT/probe_tagger.json" \
      --out "$OUT/verdict.json" 2>&1 | tee "$OUT/verdict.log"
  return "${PIPESTATUS[0]}"
}
stage compare compare

# --- 8. FEAT-06's detection recall -------------------------------------------
# The other number this run is for, and the one FEAT-01 is waiting on. Separate
# from the verdict above because it answers a different ticket: `evaluate`
# emits test/detection_* for the tagger arm, which is the measured stage-1
# recall FEAT-06 is named for. Unlike the profile stages elsewhere, this one
# *is* a deliverable rather than a measurement of one, so its failure must not
# be stamped done: a hard `stage`, not a soft one, so a rerun retries it
# instead of skipping straight to `bundle` with no detection numbers.
detection () {
  "$PDM" run evaluate "$OUT/cfg_tagger.toml" "$OUT/model_tagger.pt" \
    > "$OUT/detection.log" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    log "FAIL  evaluate (exit $status) — see $OUT/detection.log"
    return $status
  fi
  grep -iE "detection|recall|precision" "$OUT/detection.log" | head -20 \
    | while read -r line; do log "detection: $line"; done
  return 0
}
stage detection detection

# --- 9. bundle ---------------------------------------------------------------
bundle () {
  local status=0
  cp "$D/cfg_baseline.toml" "$OUT/" || status=1
  nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  git -C "$REPO" log -1 --format="%H %ad %s" --date=iso > "$OUT/commit.txt" \
    || status=1
  for arm in baseline tagger; do
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
[[ -f "$OUT/verdict.log" ]] && cat "$OUT/verdict.log"
