#!/usr/bin/env bash
# Profile a clean --limit 200 -prof run; record epoch wall time, GPU utilization,
# and peak VRAM
#
#   scripts/benchmarks/perf_baseline.sh <outdir> [limit] [epochs]

set -euo pipefail

OUT="${1:?usage: perf_baseline.sh <outdir> [limit] [epochs]}"
LIMIT="${2:-200}"
EPOCHS="${3:-3}"
PDM="${PDM:-$HOME/.local/bin/pdm}"
CONFIG="${CONFIG:-tests/best_config_so_far.toml}"

mkdir -p "$OUT"

if [[ -n "$(git status --porcelain -uno)" && "${FORCE_DIRTY:-0}" != "1" ]]; then
  echo "refusing: tracked files are modified, so the run is not reproducible" >&2
  echo "from its commit (MLflow would stamp it -dirty). FORCE_DIRTY=1 to override." >&2
  git status --porcelain -uno >&2
  exit 1
fi

{
  echo "commit:  $(git rev-parse --short HEAD)"
  echo "config:  $CONFIG (limit=$LIMIT epochs=$EPOCHS)"
  echo "host:    $(hostname)"
  echo "date:    $(date -Is)"
  echo
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo
  free -g | head -2
  echo "cpus: $(nproc)"
  echo
  echo "--- config.toml ---"; cat config.toml 2>/dev/null || echo "(absent)"
  echo "--- $CONFIG ---";     cat "$CONFIG"
} > "$OUT/provenance.txt"

# The tuned config trains for 100 epochs with patience 10; a baseline wants a
# fixed, short count so epoch 1 and epoch 2 are comparable across arms.
sed -e "s/^num_epochs = .*/num_epochs = $EPOCHS/" \
    -e "s/^patience = .*/patience = $EPOCHS/" \
    "$CONFIG" > "$OUT/baseline.toml"

sample() {  # $1 = csv path; samples until the pid in $2 exits
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
             --format=csv,noheader,nounits -lms 200 > "$1" &
  echo $!
}

echo "==> arm A: training, --limit $LIMIT, $EPOCHS epochs"
SMI=$(sample "$OUT/gpu_train.csv")
# tqdm writes the epoch/batch bars to stderr and /usr/bin/time writes its
# report there too, so both streams go to train.log and -o keeps the timing
# report out of it. Splitting them hides the only live progress signal there is.
/usr/bin/time -v -o "$OUT/train.time" \
  "$PDM" run train "$OUT/baseline.toml" "$OUT/baseline.pt" \
  --limit "$LIMIT" > "$OUT/train.log" 2>&1 || true
kill "$SMI" 2>/dev/null || true

echo "==> arm B: -prof (single batch x25, MATH sdpa kernel)"
SMI=$(sample "$OUT/gpu_prof.csv")
"$PDM" run train "$OUT/baseline.toml" "$OUT/prof.pt" \
  --limit "$LIMIT" -prof > "$OUT/prof.log" 2>&1 || true
kill "$SMI" 2>/dev/null || true

echo "==> summary"
{
  echo "### epoch wall time (from the training log)"
  grep -aoE '(training|validation)/seconds[^,}]*' "$OUT/train.log" || \
    grep -aE 'Epoch [0-9]+' "$OUT/train.log" || echo "(parse train.log by hand)"
  echo
  echo "### peak VRAM / GPU utilisation"
  for arm in train prof; do
    f="$OUT/gpu_$arm.csv"
    [[ -s "$f" ]] || continue
    awk -F', *' -v a="$arm" '
      {u+=$2; n++; if ($2>mu) mu=$2; if ($4>mm) mm=$4}
      END {printf "%-6s peak_mem=%d MiB  mean_util=%.1f%%  peak_util=%d%%  samples=%d\n",
                  a, mm, (n?u/n:0), mu, n}' "$f"
  done
  echo
  echo "### peak RSS (arm A)"
  grep -a "Maximum resident set size" "$OUT/train.time" || true
} | tee "$OUT/summary.txt"

echo
echo "wrote $OUT/{provenance,summary}.txt, train.log, prof.log, gpu_*.csv"
