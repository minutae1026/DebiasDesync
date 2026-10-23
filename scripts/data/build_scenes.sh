#!/bin/bash
#
#   bash scripts/data/fetch_dynerf_src.sh                 # 1. download the original release (~40 GB)
#   python scripts/data/unpack_dynerf_src.py              # 2. extract into data/_dynerf_src/<scene>/
#   bash scripts/data/build_scenes.sh [scene ...]         # 3. this script (default: all six scenes)
#
#
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=${PY:-python}
COLMAP_ENV=${COLMAP_ENV:-}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
SCENES="${*:-coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak}"
LEGACY_SCENES=${LEGACY_SCENES:-"coffee_martini cook_spinach sear_steak"}
SRC=data/_dynerf_src
LOG=data/build_scenes.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

say "=== build_scenes: $SCENES ==="
for S in $SCENES; do
  D=data/dynerf/$S
  say "########## $S ##########"

  if [ ! -f "$D/poses_bounds.npy" ]; then
    [ -d "$SRC/$S" ] || { say "[FAIL] $SRC/$S missing — run fetch_dynerf_src.sh + unpack_dynerf_src.py"; continue; }
    mkdir -p "$D"
    cp "$SRC/$S"/poses_bounds.npy "$D/"
    for v in "$SRC/$S"/cam*.mp4; do cp "$v" "$D/"; done
    say "[a] staged $(ls $D/cam*.mp4 | wc -l) cameras"
  else
    say "[a] staged — skip"
  fi

  if [ ! -f "$D/${S}_frames.txt" ]; then
    LEGACY=""; case " $LEGACY_SCENES " in *" $S "*) LEGACY="--legacy_trim";; esac
    say "[b] extract + trim ${LEGACY}"
    $PY scripts/data/build_unsync_scene.py --scene $S $LEGACY >> $LOG 2>&1 || { say "[FAIL] build_unsync_scene $S"; continue; }
  else
    say "[b] windows exist — skip"
  fi
  $PY scripts/data/build_unsync_scene.py --scene $S --verify >> $LOG 2>&1 || { say "[FAIL] window assignment differs from the shipped config"; continue; }

  say "[c] trimmed mp4"
  $PY scripts/data/make_trimmed_videos.py --scenes $S >> $LOG 2>&1 || { say "[FAIL] make_trimmed_videos $S"; continue; }

  if [ ! -f "$D/points3D_downsample2.ply" ]; then
    if [ ! -f "$D/colmap/dense/workspace/fused.ply" ]; then
      say "[d] COLMAP"
      ( [ -n "$COLMAP_ENV" ] && export PATH=$COLMAP_ENV/bin:$PATH
        bash scripts/data/colmap.sh $D llff ) >> $LOG 2>&1 || { say "[FAIL] colmap $S"; continue; }
    fi
    say "[d] downsample"
    $PY scripts/data/downsample_point.py $D/colmap/dense/workspace/fused.ply $D/points3D_downsample2.ply >> $LOG 2>&1 \
      || { say "[FAIL] downsample $S"; continue; }
  else
    say "[d] ply exists — skip"
  fi

  say "[e] dynamic masks"
  $PY scripts/data/extract_dynamic_masks.py --scenes $S --cams cam00 --threshold 15 --color gray --skip_existing \
      >> $LOG 2>&1 || { say "[FAIL] masks $S"; continue; }

  if $PY scripts/data/preflight_scenes.py --scenes $S >> $LOG 2>&1; then
    say "[f] preflight passed"
  else
    say "[f] preflight FAILED — see $LOG"
  fi
done
say "=== done ==="
