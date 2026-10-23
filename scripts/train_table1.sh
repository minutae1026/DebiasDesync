#!/bin/bash
#
#   conda activate debiasdesync && bash scripts/train_table1.sh   # all scenes, RUNS=3 (default)
#   RUNS=4 bash scripts/train_table1.sh          # more repetitions
#   SCENES="sear_steak" bash scripts/train_table1.sh
#
#
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS="$REPO/backbones/4DGaussians"
PY=${PY:-python}
DATA_ROOT="$REPO/data/dynerf"
OUT_ROOT="$REPO/output/4DGaussians"
mkdir -p "$OUT_ROOT"
cd "$GS"
RUNS=${RUNS:-3}
SCENES=${SCENES:-"coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak"}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export PYTHONUNBUFFERED=1
LOG_FILE="$REPO/train_log_table1_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=== Table 1 queue: scenes [$SCENES], $RUNS run(s) per cell, log $LOG_FILE ==="
if ! $PY "$REPO/scripts/data/preflight_scenes.py" --scenes $SCENES; then
  echo "[ABORT] preflight failed — fix the dataset first (scripts/data/build_scenes.sh)"; exit 1
fi

OURS_ARGS="--uniform_time_sampling --pixel_weighting --pixel_weighting_ratio 30"

run_one() {
  local scene=$1 i=$2 tag=$3 cfg=$4; shift 4
  local exp="${scene}_${tag}_${i}"
  local model="$OUT_ROOT/${exp}"
  if [ -f "$model/results.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${exp}"; return
  fi
  local port=$((6100 + RANDOM % 800))
  echo ""; echo "[$(date +%H:%M:%S)] START ${exp} (port ${port})"
  local attempt
  for attempt in 1 2; do
    if $PY train.py -s "$DATA_ROOT/${scene}" --model_path "$model" --port ${port} \
         --expname "4DGaussians/${exp}" --configs "${cfg}" --resolution 2 "$@"; then
      echo "[$(date +%H:%M:%S)] train OK -> render + metrics"
      $PY render.py --model_path "$model/" --skip_train --configs "${cfg}" \
        || echo "[WARN] render failed ${exp}"
      $PY metrics.py --model_path "$model/" --dynamic_mask_root "$DATA_ROOT" --cleanup \
        || echo "[WARN] metrics failed ${exp}"
      return
    fi
    echo "[FAIL] train ${exp} (attempt ${attempt}/2)"
    rm -rf "$model"
    [ "$attempt" = "1" ] && { sleep 30; port=$((6100 + RANDOM % 800)); }
  done
  echo "[GIVEUP] ${exp}"
}

for i in $(seq 1 $RUNS); do
  echo ""; echo "########## repetition ${i} / ${RUNS} ##########"
  for S in $SCENES; do
    run_one $S $i baseline "arguments/dynerf/${S}.py"
    run_one $S $i dynamic  "arguments/dynerf/${S}.py" $OURS_ARGS
  done
done

echo ""; echo "=== queue finished: $(date) ==="
