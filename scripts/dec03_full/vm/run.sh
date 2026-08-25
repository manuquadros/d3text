#!/bin/bash
# The pooling experiment on the VM, end to end.
#
#   bash scripts/dec03_full/vm/run.sh
#
# Stages run in order and each records a stamp in $OUT/stamps; a rerun skips
# the stages already stamped, so an interrupted run resumes where it stopped.
# Force one with `rm $OUT/stamps/<stage>`, or all with DEC03_FORCE=1.
#
# Everything the run produces lands in $OUT and is tarred up at the end.
set -uo pipefail

# The checkout, not the home directory: on the VM this is /vol/storage/dev/
# d3text. Derived from the script's own location rather than written out, so a
# checkout that moves takes the run with it.
REPO="${DEC03_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

# Everything large lands here rather than in $HOME, which on the VM is small
# and is not where the checkout lives. Named rather than derived from $REPO:
# climbing out of the checkout gives /home on any machine that keeps its
# repositories under one, which is a default nobody can write to.
VOL="${DEC03_VOL:-/vol/storage}"
[[ -d "$VOL" ]] || VOL="$HOME"

PDM="${DEC03_PDM:-$HOME/.local/bin/pdm}"
STORE="${DEC03_STORE:-$VOL/d3text-embeddings}"   # ~101 GiB; outside the repo on purpose
OUT="${DEC03_OUT:-$REPO/scripts/dec03_full/vm/out}"
EMB_BATCH="${DEC03_EMB_BATCH:-50}"
CORPUS="$REPO/brenda_references/src/brenda_references/data"
D="$REPO/scripts/dec03_full"

# One timestamp for the whole run: computed twice, a run that crosses midnight
# names a bundle it did not write.
STAMP="${DEC03_STAMP:-$(date +%Y%m%d)}"
BUNDLE="${DEC03_BUNDLE:-$VOL/dec03-vm-$STAMP.tar.gz}"

# Every document the store must serve and every document the arms train on has
# to come from the same base model, so it is read from the config the arms
# actually use rather than taken as a knob of its own — an env var here could
# only ever build a store the training run would then miss on, silently.
read_base_model () { sed -n 's/^base_model *= *"\(.*\)" *$/\1/p' "$1"; }
BASE_MODEL="$(read_base_model "$D/cfg_logsumexp.toml")"
if [[ -z "$BASE_MODEL" ]]; then
  echo "no base_model in $D/cfg_logsumexp.toml" >&2; exit 1
fi
if [[ "$BASE_MODEL" != "$(read_base_model "$D/cfg_logmeanexp.toml")" ]]; then
  echo "the two arms name different base models; they would not be comparable" >&2
  exit 1
fi

# Below this the store stops paying for itself and the run should be rethought
# rather than spending two hours and 101 GiB to find out.
BENCH_MIN="${DEC03_BENCH_MIN:-3.0}"
# Not 1.0 because a few documents are legitimately never stored — one noise
# article has no text, and a document whose encodings hold no token is dropped.
# Measured, that is one document corpus-wide, so anything near this floor is a
# store built against the wrong corpus, the wrong key, or a build that stopped
# early and got stamped as finished.
MIN_COVERAGE="${DEC03_MIN_COVERAGE:-0.99}"

# The profile stages. They change nothing about the arms — they run beside
# them and write JSON — so this run answers the pooling question exactly as
# it would have. What they add is the evidence for the settings the *next* run
# should use: which 16-bit format this card is fast at, whether the store
# build's batching leaves the GPU idle, and where peak VRAM and the step's
# time actually go once the store removes the base-model forward.
PROFILE_DOCS="${DEC03_PROFILE_DOCS:-150}"
PROFILE_BUDGETS="${DEC03_PROFILE_BUDGETS:-64,128,256,512}"

mkdir -p "$OUT" "$OUT/stamps"
export DEC03_STORE="$STORE"

# $OUT holds two checkpoints of a few hundred MB each and now sits inside a
# *tracked* directory, so the committed `scripts/dec03_full/.gitignore` covers
# it. This stays as the belt to that pair of braces: `DEC03_OUT` can point
# anywhere, and a run that redirects it lands outside the rule the repository
# carries.
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"

log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

# Said once, up front: three of these are ~101 GiB, a few hundred MB and the
# tarball to send back, and finding out where they went by looking is worse.
log "repo   $REPO"
log "store  $STORE"
log "out    $OUT"
log "bundle $BUNDLE"

stage () {  # stage <name> <command...>
  local name=$1; shift
  if [[ -n "${DEC03_STOPPED:-}" ]]; then
    log "HOLD  $name (DEC03_UNTIL=${DEC03_UNTIL} was reached)"
    return 0
  fi
  if [[ -f "$OUT/stamps/$name" && -z "${DEC03_FORCE:-}" ]]; then
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

  # `DEC03_UNTIL=<stage>` runs up to and including that stage and holds there.
  # What it is for: the store takes two hours and depends on nothing about the
  # model, so it can be built while a decision about the arms is still being
  # made. Rerunning without it resumes at the first stage that never ran.
  if [[ "${DEC03_UNTIL:-}" == "$name" ]]; then
    DEC03_STOPPED=1
    log "HOLD  stopping after $name, as asked"
  fi
}

# Like `stage`, but a failure is recorded and the run continues. The stages
# that stop the run — bench, coverage, smoke — protect the *result*: past them
# lies five hours of measuring the wrong thing. The profiles protect nothing;
# they only inform a later decision. A typo in one of them must not cost this
# run its remaining hours, so it is stamped as failed and skipped over.
soft_stage () {  # soft_stage <name> <command...>
  local name=$1; shift
  if [[ -n "${DEC03_SKIP_PROFILE:-}" ]]; then
    log "SKIP  $name (DEC03_SKIP_PROFILE is set)"
    return 0
  fi
  if [[ -f "$OUT/stamps/$name" && -z "${DEC03_FORCE:-}" ]]; then
    log "SKIP  $name (already done; rm $OUT/stamps/$name to redo)"
    return 0
  fi
  log "START $name"
  local started=$SECONDS
  "$@"
  local status=$?
  local elapsed=$((SECONDS - started))
  if [[ $status -ne 0 ]]; then
    log "WARN  $name failed after ${elapsed}s (exit $status) — continuing;"
    log "      it measures the run, it is not part of it. See $OUT/$name.log"
    echo "$name FAILED $(date -Is) ${elapsed}s" > "$OUT/stamps/$name"
    return 0
  fi
  echo "$name $(date -Is) ${elapsed}s" > "$OUT/stamps/$name"
  log "DONE  $name in ${elapsed}s"

  # `DEC03_UNTIL=<stage>` runs up to and including that stage and holds there.
  # What it is for: the store takes two hours and depends on nothing about the
  # model, so it can be built while a decision about the arms is still being
  # made. Rerunning without it resumes at the first stage that never ran.
  if [[ "${DEC03_UNTIL:-}" == "$name" ]]; then
    DEC03_STOPPED=1
    log "HOLD  stopping after $name, as asked"
  fi
}

# --- 0. can this machine finish the run? ------------------------------------
# Through tee: a preflight failure names what is missing, and that is the one
# message an operator needs on the terminal as well as in the bundle.
preflight () {
  "$PDM" run python "$D/vm/preflight.py" "$OUT/preflight.json" 2>&1 \
    | tee "$OUT/preflight.log"
}

# Deliberately outside `stage`: preflight is a precondition, not a unit of
# work, and stamping it means a resumed run skips the checks that protect
# every stage after it. The machine is not what it was when the stamp was
# written -- that is usually *why* the run is being resumed. It costs seconds.
log "START preflight (every run, never skipped)"
if ! preflight; then
  log "FAIL  preflight — stopping here"
  exit 1
fi
log "DONE  preflight"

# --- 0b. what this card is fast at ------------------------------------------
# Before the store is built, because two of its answers would change how it is
# built. `Model.amp_dtype` takes bf16 wherever `is_bf16_supported()` says yes,
# and on a pre-Ampere card that is emulation; `precompute-embeddings` mean-
# while hardcodes fp16. And the base model's throughput against windows-per-
# forward is what says whether a batcher that crosses documents is worth
# writing — the shipped one batches within a document, and no document here
# has more than 29 windows.
profile_card () {
  "$PDM" run python "$D/vm/bench_card.py" \
      --base-model "$BASE_MODEL" \
      --out "$OUT/profile_card.json" \
    > "$OUT/profile_card.log" 2>&1
}
soft_stage profile_card profile_card

# --- 0c. where the store build's wall clock goes ----------------------------
# This stage has only ever been sized from a forward alone, and the loop
# around that forward is serial — unpinned blocking D2H, a Python-loop
# aggregation, a synchronous corpus read — so the share the GPU actually gets
# is worth knowing before committing two hours to it.
profile_build () {
  # What the volume can do, which decides whether the arms' store reads are
  # minutes or hours. `oflag=direct` to measure the disk rather than the page
  # cache; a filesystem that refuses O_DIRECT just leaves the file empty and
  # the note below says so.
  {
    echo "# $VOL, 1 GiB, O_DIRECT"
    dd if=/dev/zero of="$VOL/.dec03-io-probe" bs=1M count=1024 \
        oflag=direct 2>&1 || echo "(direct write unsupported here)"
    dd if="$VOL/.dec03-io-probe" of=/dev/null bs=1M \
        iflag=direct 2>&1 || echo "(direct read unsupported here)"
  } > "$OUT/volume_io.txt" 2>&1
  rm -f "$VOL/.dec03-io-probe"

  "$PDM" run python "$D/vm/profile_build.py" \
      --base-model "$BASE_MODEL" \
      --docs "$PROFILE_DOCS" \
      --out "$OUT/profile_build.json" \
    > "$OUT/profile_build.log" 2>&1
}
soft_stage profile_build profile_build

# --- 1. is reading cheaper than recomputing, on this GPU? -------------------
# The store's whole justification is that reading beats recomputing, and the
# margin narrows on a faster card. It was 27.8x on the laptop; measure it here
# and stop if it no longer holds, rather than recording the number in a log
# nobody reads until after the two hours have been spent.
bench () {
  # `--amp fp16 --batch-size 32` so the gate measures the forward the build
  # actually runs. Left to its defaults it takes bf16 wherever the card claims
  # support — emulation included — and 8 windows per forward, while
  # `precompute-embeddings` runs fp16 over a whole document at once. Both
  # differences inflate the forward and so inflate the store's apparent
  # benefit, which is the wrong direction for a gate to err in.
  "$PDM" run python scripts/benchmarks/bench_store.py --docs 25 \
      --base-model "$BASE_MODEL" --amp fp16 --batch-size 32 \
    > "$OUT/bench_store.log" 2>&1 || return 1
  local ratio
  ratio=$(sed -n 's|^forward / unpack  *: *\([0-9.]*\)x.*|\1|p' \
      "$OUT/bench_store.log")
  if [[ -z "$ratio" ]]; then
    log "bench_store.py printed no read/forward ratio — see $OUT/bench_store.log"
    return 1
  fi
  log "reading the store is ${ratio}x cheaper than the forward on this card"
  if ! awk -v r="$ratio" -v m="$BENCH_MIN" 'BEGIN{exit !(r+0 >= m+0)}'; then
    log "that is below the ${BENCH_MIN}x floor: the store would not pay for its"
    log "two hours and 101 GiB. Raise DEC03_BENCH_MIN to proceed anyway."
    return 1
  fi
}
stage bench bench

# --- 2. build the store -----------------------------------------------------
# No --max_length: `precompute-embeddings` then takes the base model's context
# window, which is the window `split_and_tokenize` gave the encodings. Passing
# one here is what makes the two stages disagree, and a document whose stored
# rows do not line up with its encodings is silently re-embedded live.
# Resumable on its own — documents already keyed are skipped.
build_store () {
  "$PDM" run precompute-embeddings "$BASE_MODEL" "$STORE" \
      "$CORPUS/training_data.csv" \
      "$CORPUS/validation_data.csv" \
      "$CORPUS/test_data.csv" \
      "$CORPUS/pmc_linguistics_articles.json" \
      --batch_size "$EMB_BATCH" \
    > "$OUT/precompute_embeddings.log" 2>&1
}
stage build_store build_store

# The one check that catches a store holding the wrong keys entirely: every
# `get` would miss, which is silent by design and looks exactly like having no
# store configured.
coverage () {
  du -sh "$STORE" | tee "$OUT/store_size.txt"
  "$PDM" run python "$D/vm/store_coverage.py" "$STORE" \
      --min-coverage "$MIN_COVERAGE" \
      --out "$OUT/store_coverage.json" > "$OUT/store_coverage.log" 2>&1 \
    || {
      log "the store does not hold the corpus — see $OUT/store_coverage.log"
      log "(rerun the build, or lower DEC03_MIN_COVERAGE to accept the gap)"
      return 1
    }
}
stage coverage coverage

# --- 3. point this machine's config at the store ----------------------------
# Written rather than required: the run is meant to be one command. The old
# file is kept beside it, and `config.toml` is per-machine and untracked, so
# nothing here reaches the repository.
configure () {
  local config="$REPO/config.toml"
  # Only ever the *operator's* file, never one this stage wrote: under
  # DEC03_FORCE=1 this runs a second time, and an unguarded copy would put the
  # generated file over the backup and lose the original for good.
  if [[ -f "$config" && ! -f "$OUT/config.toml.before" ]]; then
    cp "$config" "$OUT/config.toml.before"
  fi
  {
    echo "# written by scripts/dec03_full/vm/run.sh on $(date -Is)"
    echo "# the previous file, if any, is at $OUT/config.toml.before"
    echo ""
    echo "# 0, not the 6300 the earlier baseline used: with the store serving"
    echo "# every document there is nothing for the RAM cache to hold that the"
    echo "# store is not already holding, and 80 GiB of resident set buys"
    echo "# nothing."
    echo "cpu_embeddings_cache_size = 0"
    echo "embeddings_store = \"$STORE\""
    echo 'float32_matmul_precision = "medium"'
    echo "cudnn_allow_tf32 = true"
    echo "expandable_segments = true"
    echo "tokenizers_parallelism = true"
  } > "$config"
  cp "$config" "$OUT/config.toml.used"
}
stage configure configure

# --- 4. prove the store is actually being read ------------------------------
# Two epochs over 20 documents. Exit code alone proves nothing here — a run
# that reads not one document from the store still trains, just slowly — so
# each of the three ways that can happen is checked in the log.
smoke () {
  "$PDM" run python "$D/seeded_train.py" \
      "$D/cfg_logsumexp.toml" "$OUT/smoke.pt" --limit 20 \
    > "$OUT/smoke.log" 2>&1
  local status=$?
  # Before the log checks: their diagnoses all assume training ran, and a
  # crashed trainer matches none of the patterns, so it would otherwise be
  # reported as a store that was never opened.
  if [[ $status -ne 0 ]]; then
    log "the smoke training run failed (exit $status) — see $OUT/smoke.log"
    return $status
  fi
  if grep -q "holds .* tokens for document" "$OUT/smoke.log"; then
    log "the store disagrees with the encodings — see $OUT/smoke.log"
    return 1
  fi
  if ! grep -q "Reading precomputed embeddings from" "$OUT/smoke.log"; then
    log "the store was never opened — check embeddings_store in config.toml"
    return 1
  fi
  # Opening it is not reading it. A store keyed on ids this corpus does not
  # use answers every get with a miss and raises nothing, so the positive
  # confirmation is the line the reader logs on its first hit.
  if ! grep -q "served document .* from the store" "$OUT/smoke.log"; then
    log "the store was opened but served no document — see $OUT/store_coverage.json"
    return 1
  fi
  grep "served .* of .* documents" "$OUT/smoke.log" | tail -1 \
    | while read -r line; do log "smoke: $line"; done
  # The scorer runs at the very end of a five-hour run; exercising it here
  # is what keeps a typo in it from being discovered then. `--limit 20` sized
  # the checkpoint's vocabulary, so this scores a real split against a real
  # recorded vocabulary, which is the part that can break.
  "$PDM" run python "$D/score_documents.py" \
      "$D/cfg_logsumexp.toml" "$OUT/smoke.pt" \
      --split val --out "$OUT/smoke_score.json" \
    > "$OUT/smoke_score.log" 2>&1 || {
      log "the document scorer failed on the smoke checkpoint — see $OUT/smoke_score.log"
      return 1
    }
  # Same reasoning as the scorer above, one stage further on: `profile_step`
  # loads the full training split, so its own stage is minutes of work before
  # it reaches the line that could be wrong. Four batches at `--limit 20`
  # exercise every one of those lines for half a minute.
  #
  # A warning and not a `return 1`, unlike the scorer: the scorer produces the
  # run's result and the profiler only describes it, so this is the one check
  # in `smoke` whose failure is not worth the remaining four hours.
  "$PDM" run python "$D/vm/profile_step.py" \
      "$D/cfg_logsumexp.toml" --limit 20 --budgets 64 \
      --batches 4 --warmup 1 --rounds 1 \
    > "$OUT/smoke_profile_step.log" 2>&1 \
    || log "WARN the step profiler failed on the smoke split — see $OUT/smoke_profile_step.log"
  rm -f "$OUT/smoke.pt"
  return 0
}
stage smoke smoke

# --- 4b. where a step's time and VRAM go, with the store serving -------------
# After `smoke`, so the store is known to serve, and before the arms, so its
# answers arrive while they are still actionable. It sweeps `batch_max_chunks`
# and records an OOM per budget rather than raising: finding the ceiling is
# what the sweep is for, and the earlier ceiling of 512 was measured at a
# fraction of this vocabulary and before the store existed.
profile_step () {
  "$PDM" run python "$D/vm/profile_step.py" \
      "$D/cfg_logsumexp.toml" \
      --budgets "$PROFILE_BUDGETS" \
      --out "$OUT/profile_step.json" \
    > "$OUT/profile_step.log" 2>&1
}
soft_stage profile_step profile_step

# --- 5. the two arms --------------------------------------------------------
# Identical but for `entity_logits_pooling`, seeded, full training split.
train_arm () {  # train_arm <pooling>
  "$PDM" run python "$D/seeded_train.py" \
      "$D/cfg_$1.toml" "$OUT/model_$1.pt" \
    > "$OUT/train_$1.log" 2>&1
}

score_arm () {  # score_arm <pooling>
  local status=0
  for split in val test; do
    "$PDM" run python "$D/score_documents.py" \
        "$D/cfg_$1.toml" "$OUT/model_$1.pt" \
        --split $split --out "$OUT/score_$1_$split.json" \
      > "$OUT/score_$1_$split.log" 2>&1 || status=1
  done
  return $status
}

stage train_logsumexp  train_arm logsumexp
stage score_logsumexp  score_arm logsumexp
stage train_logmeanexp train_arm logmeanexp
stage score_logmeanexp score_arm logmeanexp

# --- 6. bundle --------------------------------------------------------------
bundle () {
  local status=0
  cp "$D/cfg_logsumexp.toml" "$D/cfg_logmeanexp.toml" "$OUT/" || status=1
  nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  git -C "$REPO" log -1 --format="%H %ad %s" --date=iso > "$OUT/commit.txt" \
    || status=1
  for pooling in logsumexp logmeanexp; do
    "$PDM" run python scripts/benchmarks/parse_run.py \
        "$OUT/train_$pooling.log" > "$OUT/epochs_$pooling.txt" 2>&1 || status=1
    grep "served .* of .* documents" "$OUT/train_$pooling.log" | tail -1 \
      | while read -r line; do log "$pooling: $line"; done
  done
  cat "$OUT/stamps"/* > "$OUT/timings.txt" 2>/dev/null || status=1
  # Outside the repo, for the same reason the store is: it is a large file in
  # a checkout whose stray files are neither tracked nor ignored.
  tar czf "$BUNDLE" -C "$(dirname "$OUT")" --exclude="*.pt" \
      "$(basename "$OUT")" || status=1
  log "bundle: $BUNDLE"
  return $status
}
stage bundle bundle

log "ALL STAGES DONE"
