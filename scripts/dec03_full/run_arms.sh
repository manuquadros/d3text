#!/bin/bash
# The deciding pooling run: two arms at full --limit, identical but for
# `entity_logits_pooling`, scored on per-class *document* metrics.
set -u
# The checkout this script sits in, rather than one machine's home directory:
# a tracked file that names /home/<someone> runs nowhere but there.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
PDM="${DEC03_PDM:-$HOME/.local/bin/pdm}"
D=scripts/dec03_full

# arm A is already running; wait it out
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "=== arm A (logsumexp) training finished $(date -Is)"

score () {  # config checkpoint tag
  for split in val test; do
    "$PDM" run python $D/score_documents.py "$1" "$2" \
        --split $split --out $D/score_$3_$split.json \
        > $D/score_$3_$split.log 2>&1
    echo "=== scored $3 on $split (exit $?) $(date -Is)"
  done
  "$PDM" run evaluate "$1" "$2" > $D/evaluate_$3.log 2>&1
  echo "=== evaluate $3 (exit $?) $(date -Is)"
}

score $D/cfg_logsumexp.toml $D/model_logsumexp.pt logsumexp

"$PDM" run python $D/seeded_train.py \
    $D/cfg_logmeanexp.toml $D/model_logmeanexp.pt > $D/train_logmeanexp.log 2>&1
echo "=== arm B (logmeanexp) training finished (exit $?) $(date -Is)"

score $D/cfg_logmeanexp.toml $D/model_logmeanexp.pt logmeanexp
echo "=== ALL DONE $(date -Is)"
