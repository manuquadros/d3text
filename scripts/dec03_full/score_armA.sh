#!/bin/bash
# Arm A only: score the local logsumexp run when it finishes. Arm B has moved
# to the VM, so this no longer chains a second training run.
set -u
# The checkout this script sits in, rather than one machine's home directory:
# a tracked file that names /home/<someone> runs nowhere but there.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
PDM="${DEC03_PDM:-$HOME/.local/bin/pdm}"
D=scripts/dec03_full
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "=== arm A (logsumexp) training finished $(date -Is)"
for split in val test; do
  "$PDM" run python $D/score_documents.py $D/cfg_logsumexp.toml $D/model_logsumexp.pt \
      --split $split --out $D/score_logsumexp_$split.json > $D/score_logsumexp_$split.log 2>&1
  echo "=== scored logsumexp on $split (exit $?) $(date -Is)"
done
echo "=== ARM A DONE $(date -Is)"
