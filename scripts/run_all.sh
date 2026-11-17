#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEED=${SEED:-1}
DATA=${DATA:-data/synthetic_link}
LINK=${LINK:-SYN-T0}
for labels in denoised ahead raw; do
  if [ -n "${CKPT:-}" ]; then
    $PY scripts/evaluate.py --checkpoint "$CKPT/$labels.pt" --data "$DATA" --link "$LINK" --out "runs/$labels" > /dev/null &
  else
    $PY scripts/run.py --data "$DATA" --link "$LINK" --labels "$labels" --seed "$SEED" --out "runs/$labels" > /dev/null &
  fi
done
$PY scripts/baselines.py --data "$DATA" --link "$LINK" --seed "$SEED" --out runs/table3 > /dev/null
wait
$PY scripts/table5.py --data "$DATA" --link "$LINK" --seed "$SEED" --runs runs --out results
$PY scripts/summarize.py --runs runs --out results
