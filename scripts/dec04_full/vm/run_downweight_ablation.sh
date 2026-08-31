#!/bin/bash
# DEC-04 option 2: does down-weighting the class-negative false-negative mask
# beat hard abstention (run_negative_ablation.sh) or the untouched baseline?
#
#   bash scripts/dec04_full/vm/run_downweight_ablation.sh
#
# Uses ModelConfig.class_negative_downweight — a float in [0, 1] that scales
# class_negative_abstain_mask's marked pairs instead of dropping them.
# downweight=0.0 is exactly run_negative_ablation.sh's hard-abstain arm;
# downweight=1.0 would cancel the abstention, back to the untouched baseline.
# Neither endpoint is retrained here since both are already measured
# (BUG-92's run_negative_ablation.sh output) — this script only trains the
# interior grid.
#
# Requires the earlier run.sh to have completed: reuses its label store,
# its embeddings store, and its "tagger" arm's config as the base for every
# arm's config, the same way run_negative_ablation.sh does.
#
# Grid: DEC04DA_GRID, space-separated, default "0.3 0.5 0.7". Bacteria's
# separate min_chars cutoff (BUG-92) carries over unchanged —
# DEC04DA_BACTERIA_MIN_CHARS overrides it, same variable name pattern as
# run_negative_ablation.sh's DEC04NA_BACTERIA_MIN_CHARS.
#
# Stages resume the same way run.sh's and run_negative_ablation.sh's do: a
# stamp in $OUT/stamps skips a finished stage on rerun; `rm` it, or set
# DEC04DA_FORCE=1, to redo one.
set -uo pipefail

REPO="${DEC04DA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

PDM="${DEC04DA_PDM:-$HOME/.local/bin/pdm}"
BASE_OUT="${DEC04_OUT:-$REPO/scripts/dec04_full/vm/out}"
OUT="${DEC04DA_OUT:-$REPO/scripts/dec04_full/vm/out_downweight_ablation}"
D3="$REPO/scripts/dec03_full"
COMPARE="$REPO/scripts/dec04_full/negative_ablation_compare.py"

TAGGER_CONFIG="$BASE_OUT/cfg_tagger.toml"
BACTERIA_MIN_CHARS="${DEC04DA_BACTERIA_MIN_CHARS:-20}"
GRID=(${DEC04DA_GRID:-0.3 0.5 0.7})

mkdir -p "$OUT" "$OUT/stamps"
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"

log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

# --- 0a. stamp the commit every arm trains under, and reject a stale $OUT ----
# Same reasoning as run_negative_ablation.sh: a resumed run after a `git
# checkout` of a different commit must not silently mix code across arms, and
# the config diff below only checks the lines the ablation means to isolate,
# not the training/eval code each checkpoint actually ran.
COMMIT="$(git -C "$REPO" log -1 --format="%H %ad %s" --date=iso)"
if [[ -f "$OUT/commit.txt" ]]; then
  STAMPED="$(cat "$OUT/commit.txt")"
  if [[ "$STAMPED" != "$COMMIT" ]]; then
    log "FAIL  HEAD changed since this run started:"
    log "      stamped: $STAMPED"
    log "      current: $COMMIT"
    log "      a resumed run must not mix commits across arms — checkout the stamped commit, or start a fresh \$OUT"
    exit 1
  fi
  log "commit unchanged since this run started: $COMMIT"
else
  echo "$COMMIT" > "$OUT/commit.txt"
  log "commit: $COMMIT"
fi

stage () {  # stage <name> <command...>
  local name=$1; shift
  if [[ -f "$OUT/stamps/$name" && -z "${DEC04DA_FORCE:-}" ]]; then
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

# --- 0b. the earlier run's config has to exist -------------------------------
require_prior_run () {
  if [[ ! -f "$TAGGER_CONFIG" ]]; then
    log "missing $TAGGER_CONFIG — run scripts/dec04_full/vm/run.sh first"
    return 1
  fi
}
log "START preflight (every run, never skipped)"
if ! require_prior_run; then
  log "FAIL  preflight — stopping here"
  exit 1
fi
log "DONE  preflight"

# Every added line has to be one of the abstention/downweight settings this
# ablation means to isolate — same attributability argument
# run_negative_ablation.sh makes, widened to also accept
# class_negative_downweight.
assert_only_downweight_lines_differ () {  # <label> <config>
  local label=$1 config=$2
  local stray
  stray=$(diff <(sed '/^#/d;/^$/d' "$TAGGER_CONFIG") \
              <(sed '/^#/d;/^$/d' "$config") \
           | grep '^>' \
           | grep -vc -E '^> class_negative_(abstention|downweight)')
  if [[ "$stray" -ne 0 ]]; then
    log "$label differs from $TAGGER_CONFIG outside class_negative_* lines — see $config"
    return 1
  fi
  log "$label differs from $TAGGER_CONFIG only in class_negative_* settings"
}

# --- 1. the baseline arm's config --------------------------------------------
# One baseline for the whole grid, trained fresh at the commit stamped above —
# not reused from run_negative_ablation.sh's out/, since that checkpoint was
# trained at a different commit and BUG-90 is exactly the bug that comes from
# comparing a stale "before" against a fresh "after".
write_baseline_config () {
  cp "$TAGGER_CONFIG" "$OUT/cfg_baseline.toml"
  echo "class_negative_abstention = false" >> "$OUT/cfg_baseline.toml"
  assert_only_downweight_lines_differ "the baseline config" "$OUT/cfg_baseline.toml"
}
stage baseline_config write_baseline_config

train_baseline () {
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_baseline.toml" "$OUT/model_baseline.pt" \
    > "$OUT/train_baseline.log" 2>&1
}
stage train_baseline train_baseline

evaluate_arm () {  # evaluate_arm <name> <config> <checkpoint>
  "$PDM" run evaluate "$2" "$3" > "$OUT/evaluate_$1.log" 2>&1
}
stage evaluate_baseline evaluate_arm baseline "$OUT/cfg_baseline.toml" "$OUT/model_baseline.pt"

# --- 2. one arm per grid point ------------------------------------------------
# label() turns "0.3" into "03" so file/stage names stay shell- and
# filesystem-safe regardless of locale decimal points.
label () { echo "${1/./}"; }

write_downweight_config () {  # write_downweight_config <value>
  local value=$1 tag; tag=$(label "$value")
  cp "$TAGGER_CONFIG" "$OUT/cfg_downweight_$tag.toml"
  {
    echo "class_negative_abstention = true"
    echo "class_negative_abstention_min_chars_by_class = { bacteria = ${BACTERIA_MIN_CHARS} }"
    echo "class_negative_downweight = ${value}"
  } >> "$OUT/cfg_downweight_$tag.toml"
  assert_only_downweight_lines_differ \
    "the downweight=$value config" "$OUT/cfg_downweight_$tag.toml"
}

train_downweight () {  # train_downweight <value>
  local value=$1 tag; tag=$(label "$value")
  "$PDM" run python "$D3/seeded_train.py" \
      "$OUT/cfg_downweight_$tag.toml" "$OUT/model_downweight_$tag.pt" \
    > "$OUT/train_downweight_$tag.log" 2>&1
}

for value in "${GRID[@]}"; do
  tag=$(label "$value")
  stage "downweight_${tag}_config" write_downweight_config "$value"
  stage "train_downweight_${tag}" train_downweight "$value"
  stage "evaluate_downweight_${tag}" evaluate_arm \
    "downweight_$tag" "$OUT/cfg_downweight_$tag.toml" "$OUT/model_downweight_$tag.pt"
done

# --- 3. one verdict per grid point, against the shared baseline --------------
# negative_ablation_compare.py is unchanged: it already diffs any two
# `evaluate` logs' class tables, so each grid point is just another
# before/after pair rather than a reason to extend that script to N arms.
compare_one () {  # compare_one <value>
  local value=$1 tag; tag=$(label "$value")
  "$PDM" run python "$COMPARE" \
      "$OUT/evaluate_baseline.log" "$OUT/evaluate_downweight_$tag.log" \
      --out "$OUT/verdict_$tag.json" 2>&1 | tee "$OUT/verdict_$tag.log"
  return "${PIPESTATUS[0]}"
}

for value in "${GRID[@]}"; do
  tag=$(label "$value")
  stage "compare_${tag}" compare_one "$value"
done

log "ALL STAGES DONE"
for value in "${GRID[@]}"; do
  tag=$(label "$value")
  echo "--- downweight=$value ---"
  [[ -f "$OUT/verdict_$tag.log" ]] && cat "$OUT/verdict_$tag.log"
done
