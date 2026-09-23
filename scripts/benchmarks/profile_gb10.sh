#!/usr/bin/env bash
# Profile representative training on an NVIDIA GB10 machine.
#
# Usage:
#   bash scripts/benchmarks/profile_gb10.sh
#
# Optional environment variables:
#   GB10_PROFILE_CONFIG       training config (default: best known config)
#   GB10_PROFILE_LIMIT        documents per split (default: 500)
#   GB10_PROFILE_EPOCHS       measured epochs (default: 3)
#   GB10_PROFILE_OUT          output directory
#   GB10_PROFILE_PDM          pdm executable
#   GB10_PROFILE_NSYS         run a short Nsight Systems capture (default: 1)
#   GB10_PROFILE_NSYS_LIMIT   documents per split in capture (default: 100)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${GB10_PROFILE_REPO:-$HERE}"
PDM="${GB10_PROFILE_PDM:-$HOME/.local/bin/pdm}"
CONFIG="${GB10_PROFILE_CONFIG:-$REPO/tests/best_config_so_far.toml}"
LIMIT="${GB10_PROFILE_LIMIT:-500}"
EPOCHS="${GB10_PROFILE_EPOCHS:-3}"
RUN_NSYS="${GB10_PROFILE_NSYS:-1}"
NSYS_LIMIT="${GB10_PROFILE_NSYS_LIMIT:-100}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${GB10_PROFILE_OUT:-$REPO/scripts/benchmarks/out/gb10-$STAMP}"

if [[ ! -x "$PDM" ]]; then
    echo "pdm is not executable: $PDM" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "training config does not exist: $CONFIG" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required" >&2
    exit 2
fi

mkdir -p "$OUT"
printf '*\n' > "$OUT/.gitignore"
cd "$REPO" || exit 2

log() {
    echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"
}

stop_sampler() {
    local pid="$1"
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

write_config() {
    local source="$1"
    local destination="$2"
    local epochs="$3"
    "$PDM" run python - "$source" "$destination" "$epochs" <<'PY'
import pathlib
import sys

import tomlkit

source, destination, epochs = sys.argv[1:]
document = tomlkit.parse(pathlib.Path(source).read_text())
document["num_epochs"] = int(epochs)
document["patience"] = int(epochs)
pathlib.Path(destination).write_text(tomlkit.dumps(document))
PY
}

sample_gpu() {
    local destination="$1"
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
        --format=csv,noheader,nounits \
        -lms 200 > "$destination" 2> "$destination.err" &
    echo "$!"
}

sample_host() {
    local destination="$1"
    if command -v vmstat >/dev/null 2>&1; then
        vmstat -t 1 > "$destination" 2>&1 &
        echo "$!"
    fi
}

write_config "$CONFIG" "$OUT/profile.toml" "$EPOCHS" || exit 2

{
    echo "date: $(date -Is)"
    echo "repo: $REPO"
    echo "config: $CONFIG"
    echo "limit: $LIMIT"
    echo "epochs: $EPOCHS"
    echo "commit: $(git rev-parse HEAD)"
    echo "dirty: $(git status --porcelain -uno | wc -l) tracked paths"
    echo
    uname -a
    echo
    nvidia-smi
    echo
    free -h
    echo
    lscpu
    echo
    echo "--- machine config ---"
    if [[ -f config.toml ]]; then
        cat config.toml
    else
        echo "(absent)"
    fi
    echo
    echo "--- training config ---"
    cat "$OUT/profile.toml"
    echo
    "$PDM" run python - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"cuda runtime: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"device: {properties.name}")
    print(f"capability: {properties.major}.{properties.minor}")
    print(f"device memory: {properties.total_memory}")
PY
} > "$OUT/provenance.txt" 2>&1

log "measured run: $EPOCHS epochs, limit $LIMIT"
gpu_pid="$(sample_gpu "$OUT/gpu.csv")"
host_pid="$(sample_host "$OUT/host.txt")"

# Remote tracking adds network variability and is not part of model throughput.
unset MLFLOW_TRACKING_URI
D3TEXT_LOG_LEVEL=INFO /usr/bin/time -v -o "$OUT/time.txt" \
    "$PDM" run python scripts/compile_benchmark/train_json.py \
    "$OUT/metrics.json" "$OUT/profile.toml" "$OUT/model.pt" \
    --limit "$LIMIT" > "$OUT/train.log" 2>&1
train_status=$?

stop_sampler "$gpu_pid"
stop_sampler "$host_pid"
rm -f "$OUT/model.pt"

log "measured run exit: $train_status"

if [[ "$RUN_NSYS" == "1" ]]; then
    if command -v nsys >/dev/null 2>&1; then
        write_config "$CONFIG" "$OUT/nsys.toml" 1 || exit 2
        log "Nsight Systems capture: 1 epoch, limit $NSYS_LIMIT"
        unset MLFLOW_TRACKING_URI
        nsys profile \
            --trace=cuda,nvtx,osrt,cublas,cudnn \
            --sample=none \
            --cpuctxsw=none \
            --force-overwrite=true \
            --output="$OUT/nsys" \
            "$PDM" run python scripts/compile_benchmark/train_json.py \
            "$OUT/nsys-metrics.json" "$OUT/nsys.toml" \
            "$OUT/nsys-model.pt" --limit "$NSYS_LIMIT" \
            > "$OUT/nsys.log" 2>&1
        nsys_status=$?
        rm -f "$OUT/nsys-model.pt"
        log "Nsight Systems exit: $nsys_status"
    else
        log "Nsight Systems skipped: nsys not found"
    fi
fi

{
    echo "# GB10 training profile"
    echo
    "$PDM" run python scripts/benchmarks/parse_run.py "$OUT/train.log" || true
    echo
    echo "## GPU samples"
    awk -F', *' '
        $2 ~ /^[0-9]+$/ {
            util += $2; memory_util += $3; samples++
            if ($2 > peak_util) peak_util = $2
            if ($4 > peak_memory) peak_memory = $4
        }
        END {
            printf "samples: %d\nmean GPU utilization: %.1f%%\n", \
                samples, samples ? util / samples : 0
            printf "peak GPU utilization: %d%%\n", peak_util
            printf "mean memory utilization: %.1f%%\n", \
                samples ? memory_util / samples : 0
            printf "peak reported GPU memory: %d MiB\n", peak_memory
        }
    ' "$OUT/gpu.csv"
    echo
    echo "## Process"
    grep -aE \
        'Elapsed \(wall clock\)|Maximum resident set size|File system inputs|File system outputs' \
        "$OUT/time.txt" || true
    echo
    echo "Epoch metrics: $OUT/metrics.json"
    if [[ -f "$OUT/nsys.nsys-rep" ]]; then
        echo "Nsight Systems trace: $OUT/nsys.nsys-rep"
    fi
} > "$OUT/summary.txt"

cat "$OUT/summary.txt"
log "output: $OUT"
exit "$train_status"
