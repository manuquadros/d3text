#!/bin/bash
# Does torch.compile pay for this model? Two arms, interleaved, in one command.
#
#   bash scripts/compile_benchmark/run.sh
#
# Requires a Triton-capable GPU — compute capability 7.0 or newer. The runner
# refuses to start on anything else, and that refusal is the point: below 7.0
# `compile_model` returns False whatever the switch says, so both arms would
# run eager and the table would be a confident reading of timing noise. The
# P100 VM is capability 6.0 and cannot host this benchmark at all.
#
# Knobs, all optional:
#   COMPILE_BENCH_CONFIG   the model config both arms train from
#   COMPILE_BENCH_EPOCHS   epochs per arm (default 3)
#   COMPILE_BENCH_LIMIT    training documents per arm (default 500)
#   COMPILE_BENCH_REPEATS  runs per arm (default 3)
#   COMPILE_BENCH_OUT      where the metrics, logs and report go
#   COMPILE_BENCH_PDM      the pdm to run through
#
# Deliberately not resumable, unlike the older run scripts here: the arms are
# interleaved and their order reversed per repeat precisely so that thermal
# drift cannot line up with the switch, and half a benchmark resumed hours
# later on a cold card is not half a measurement.
set -uo pipefail

# The checkout this script sits in, rather than one machine's home directory:
# a tracked file naming /home/<someone> runs nowhere but there.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${COMPILE_BENCH_REPO:-$HERE}"
cd "$REPO" || exit 1

PDM="${COMPILE_BENCH_PDM:-$HOME/.local/bin/pdm}"
D="$REPO/scripts/compile_benchmark"
CONFIG="${COMPILE_BENCH_CONFIG:-$D/cfg_base.toml}"
OUT="${COMPILE_BENCH_OUT:-$D/out}"
EPOCHS="${COMPILE_BENCH_EPOCHS:-3}"
LIMIT="${COMPILE_BENCH_LIMIT:-500}"
REPEATS="${COMPILE_BENCH_REPEATS:-3}"

mkdir -p "$OUT"
# Before anything is written into it: this directory holds logs, metrics and a
# scratch checkpoint, none of which belongs in the tree.
[[ -f "$OUT/.gitignore" ]] || printf '*\n' > "$OUT/.gitignore"
log () { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }

log "repo    $REPO"
log "config  $CONFIG"
log "out     $OUT"
log "arms    $((REPEATS * 2)) runs, $EPOCHS epochs, limit $LIMIT"

nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
git -C "$REPO" log -1 --format="%H %ad %s" --date=iso > "$OUT/commit.txt"
git -C "$REPO" diff --quiet HEAD || log "the tree is dirty; see commit.txt"

# The runner records a crashed arm rather than aborting, so a non-zero exit
# here means the machine was refused or the plan itself broke — not that an
# arm failed, which is a result and belongs in the report below.
"$PDM" run python "$D/run_arms.py" "$CONFIG" \
    --epochs "$EPOCHS" --limit "$LIMIT" --repeats "$REPEATS" \
    --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
status="${PIPESTATUS[0]}"
if [[ $status -ne 0 ]]; then
  log "the benchmark did not run (exit $status)"
  exit $status
fi

"$PDM" run python "$D/compare_arms.py" "$OUT/run.json" --out "$OUT/arms.json" \
    2>&1 | tee "$OUT/report.md"
report="${PIPESTATUS[0]}"
log "report $OUT/report.md (exit $report)"
exit "$report"
