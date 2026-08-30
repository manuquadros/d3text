#!/bin/bash
# The measurement DEC-04's full-split run left untaken: does removing the
# document-level false-negative label noise (class_negative_abstention) buy
# anything, once it is actually removed rather than merely carried?
#
#   bash scripts/dec04_full/vm/run_negative_ablation.sh
#
# Requires the earlier run.sh to have completed: this reuses its label store,
# its embeddings store, and its "tagger" arm's config as the base for the
# "before" side of the comparison, rather than re-deriving any of them — but
# trains and evaluates that "before" checkpoint fresh, at the same commit as
# the "after" arm, instead of reusing run.sh's own checkpoint.
#
# The abstain arm now gates bacteria's cutoff separately from the rest
# (BUG-92) — see write_abstain_config below and
# DEC04NA_BACTERIA_MIN_CHARS to change the value being tried.
#
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

mkdir -p "$OUT" "$OUT/stamps"
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"

log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

# --- 0a. stamp the commit both arms train under, and reject a stale $OUT -----
# Without this, a resumed run after a `git checkout` of a different commit
# would silently train the two arms on different code and nothing would catch
# it — the config diff a few lines down only checks the one line the ablation
# means to isolate, not the training/eval code each checkpoint actually ran.
#
# An $OUT that already carries an evaluate_tagger stamp but no commit.txt was
# left by a version of this script that re-evaluated $BASE_OUT/model_tagger.pt
# under that stage name instead of training a fresh baseline. Writing
# commit.txt fresh and moving on would leave that stale evaluate_tagger.log
# sitting there unused, which is harmless by itself — but treat it as
# untrustworthy rather than silently adopting the directory, since nothing
# else here can tell whether the rest of $OUT predates the fresh-baseline
# stages below.
COMMIT="$(git -C "$REPO" log -1 --format="%H %ad %s" --date=iso)"
if [[ -f "$OUT/commit.txt" ]]; then
  STAMPED="$(cat "$OUT/commit.txt")"
  if [[ "$STAMPED" != "$COMMIT" ]]; then
    log "FAIL  HEAD changed since this run started:"
    log "      stamped: $STAMPED"
    log "      current: $COMMIT"
    log "      a resumed run must not mix commits across its two arms — checkout the stamped commit, or start a fresh \$OUT"
    exit 1
  fi
  log "commit unchanged since this run started: $COMMIT"
elif [[ -f "$OUT/stamps/evaluate_tagger" ]]; then
  log "FAIL  $OUT/stamps/evaluate_tagger exists with no $OUT/commit.txt"
  log "      this looks like output from an older version of this script that"
  log "      re-evaluated \$BASE_OUT/model_tagger.pt under the evaluate_tagger"
  log "      stage name; that stage no longer exists, so resuming here can't"
  log "      be trusted to mean what it used to. Move $OUT aside (or remove"
  log "      $OUT/stamps/evaluate_tagger and $OUT/evaluate_tagger.log) and"
  log "      rerun so the baseline arm is trained and evaluated fresh."
  exit 1
else
  echo "$COMMIT" > "$OUT/commit.txt"
  log "commit: $COMMIT"
fi

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

# --- 0b. the earlier run's config has to exist --------------------------------
# Only the config is reused now; the "before" side trains its own checkpoint
# fresh below rather than reusing $BASE_OUT/model_tagger.pt.
require_prior_run () {
  local missing=0
  for f in "$TAGGER_CONFIG"; do
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

# Shared by both configs below: every added line has to be one of the
# abstention settings the ablation means to isolate, not something else —
# the same attributability argument run.sh makes for token_labels_store
# against the baseline. Widened from "exactly one line" (BUG-92): the
# per-class override below is a second `class_negative_abstention*` line,
# still one variable, not two.
assert_only_abstention_lines_differ () {  # <label> <config>
  local label=$1 config=$2
  local stray
  stray=$(diff <(sed '/^#/d;/^$/d' "$TAGGER_CONFIG") \
              <(sed '/^#/d;/^$/d' "$config") \
           | grep '^>' | grep -vc '^> class_negative_abstention')
  if [[ "$stray" -ne 0 ]]; then
    log "$label differs from $TAGGER_CONFIG outside class_negative_abstention* lines — see $config"
    return 1
  fi
  log "$label differs from $TAGGER_CONFIG only in class_negative_abstention settings"
}

# --- 1. the ablation arm's config --------------------------------------------
# token_labels_store is unchanged, so this reuses the same label store and
# the same base checkpoint's config. BUG-92: a uniform min_chars=8 rescues
# strains and other_organisms but not bacteria, whose lower prevalence means
# the same residual over-abstention still collapses its precision — so
# bacteria gets its own, higher cutoff here. DEC04NA_BACTERIA_MIN_CHARS
# overrides the value if a different one wants trying; 20 is untested, a
# first guess to measure rather than a derived number.
write_abstain_config () {
  cp "$TAGGER_CONFIG" "$OUT/cfg_abstain.toml"
  {
    echo "class_negative_abstention = true"
    echo "class_negative_abstention_min_chars_by_class = { bacteria = ${DEC04NA_BACTERIA_MIN_CHARS:-20} }"
  } >> "$OUT/cfg_abstain.toml"
  assert_only_abstention_lines_differ "the abstain config" "$OUT/cfg_abstain.toml"
}
stage abstain_config write_abstain_config

# --- 1b. the baseline arm's config -------------------------------------------
# Symmetric with write_abstain_config: an explicit `class_negative_abstention
# = false` line, so neither arm's behaviour depends on the option's default.
write_baseline_config () {
  cp "$TAGGER_CONFIG" "$OUT/cfg_baseline_fresh.toml"
  echo "class_negative_abstention = false" >> "$OUT/cfg_baseline_fresh.toml"
  assert_only_abstention_lines_differ "the baseline config" "$OUT/cfg_baseline_fresh.toml"
}
stage baseline_config write_baseline_config

# --- 2. train both arms fresh, at the commit stamped above -------------------
# $BASE_OUT/model_tagger.pt was trained by the earlier run.sh, potentially at
# an older commit — reusing it as the "before" side would compare a stale
# checkpoint's training code against train_abstain's current one, not just
# the one config line the comparison is meant to isolate. Training both arms
# here keeps that the only difference.
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
# The "before" side is evaluated off the freshly trained baseline checkpoint
# from train_baseline_fresh, not off $BASE_OUT/model_tagger.pt: that
# checkpoint's weights carry whatever training code was checked out when
# run.sh made it, and current evaluation code cannot retroactively correct
# what the weights learned. This stage is named evaluate_baseline_fresh,
# distinct from any evaluate_tagger stamp an older $OUT might carry, so a
# resumed run can never mistake a stale evaluation of the reused checkpoint
# for this one.
evaluate_arm () {  # evaluate_arm <name> <config> <checkpoint>
  "$PDM" run evaluate "$2" "$3" > "$OUT/evaluate_$1.log" 2>&1
}
stage evaluate_baseline_fresh evaluate_arm baseline_fresh "$OUT/cfg_baseline_fresh.toml" "$OUT/model_baseline_fresh.pt"
stage evaluate_abstain        evaluate_arm abstain        "$OUT/cfg_abstain.toml" "$OUT/model_abstain.pt"

# --- 4. the verdict -----------------------------------------------------------
compare () {
  "$PDM" run python "$COMPARE" \
      "$OUT/evaluate_baseline_fresh.log" "$OUT/evaluate_abstain.log" \
      --out "$OUT/verdict.json" 2>&1 | tee "$OUT/verdict.log"
  return "${PIPESTATUS[0]}"
}
stage compare compare

log "ALL STAGES DONE"
[[ -f "$OUT/verdict.log" ]] && cat "$OUT/verdict.log"
