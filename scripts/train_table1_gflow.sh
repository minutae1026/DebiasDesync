#!/bin/bash
#
#   GFLOW_PY=/path/to/envs/debiasdesync-gaussian-flow/bin/python  EVAL_PY=/path/to/envs/debiasdesync/bin/python \
#   bash scripts/train_table1_gflow.sh            # RUNS=3 by default; SCENES="sear_steak" to restrict
#
#
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GF="$REPO/backbones/Gaussian-Flow"
GFLOW_PY=${GFLOW_PY:-python}
EVAL_PY=${EVAL_PY:-python}
RUNS=${RUNS:-3}
SCENES=${SCENES:-"coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak"}
DATA_ROOT="$REPO/data/dynerf"
OUT_ROOT="$REPO/output/Gaussian-Flow"
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6}
export OURS_TORCH_VRAM_FRACTION=${OURS_TORCH_VRAM_FRACTION:-0.90}
mkdir -p "$OUT_ROOT"
LOG_FILE="$REPO/train_log_table1_gflow_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE")
exec 2>&1
echo "=== Table 1 Gaussian-Flow queue: scenes [$SCENES], $RUNS run(s) per cell, log $LOG_FILE ==="

run_one() {
  local scene=$1 cond=$2 run=$3
  local exp="${scene}_${cond}_${run}"
  local model="$OUT_ROOT/$exp"
  local data="$DATA_ROOT/$scene"
  local cfg="arguments/dynerf/gf_${cond}.py"
  if [ -f "$model/results.json" ]; then echo "[$(date +%H:%M:%S)] SKIP $exp"; return; fi
  rm -rf "$model"
  echo ""; echo "[$(date +%H:%M:%S)] START $exp"
  ( cd "$GF" && "$GFLOW_PY" train.py -s "$data" --model_path "$model" \
        --config "$cfg" --test_iterations 29999 --eval ) \
    || echo "  [WARN] train.py exited with an error (the checkpoint may still exist)"
  local ck; ck=$(ls "$model"/chkpnt*.pth 2>/dev/null | sort -V | tail -1)
  [ -z "$ck" ] && { echo "[FAIL] train $exp: no checkpoint"; return; }
  ( cd "$GF" && "$GFLOW_PY" render_dynamic.py -s "$data" --model_path "$model" \
        --configs "$cfg" --eval --skip_train --skip_video --ckpt_name "$(basename "$ck")" ) \
    || { echo "[FAIL] render $exp"; return; }
  "$EVAL_PY" "$REPO/backbones/4DGaussians/metrics.py" --model_path "$model" --dynamic_mask_root "$DATA_ROOT" --cleanup \
    || { echo "[FAIL] metrics $exp"; return; }
  rm -rf "$model/video"
  echo "[$(date +%H:%M:%S)] DONE $exp"
}

for run in $(seq 1 "$RUNS"); do
  echo ""; echo "########## repetition $run / $RUNS ##########"
  for scene in $SCENES; do
    for cond in base ours; do
      run_one "$scene" "$cond" "$run"
    done
  done
done
echo ""; echo "=== queue finished: $(date) ==="
