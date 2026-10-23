#!/bin/bash
# Download the original Neural 3D Video release (v1.0, CC BY-NC 4.0) into data/_dynerf_src/.
#   bash scripts/data/fetch_dynerf_src.sh
# Archives already present are kept (wget -c resumes partial downloads).
# flame_salmon_1 is a 4-part split archive; scripts/data/unpack_dynerf_src.py joins and extracts it.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/data/_dynerf_src"
mkdir -p "$SRC"
cd "$SRC"
B=https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0
for f in coffee_martini.zip cook_spinach.zip cut_roasted_beef.zip flame_steak.zip sear_steak.zip \
         flame_salmon_1_split.z01 flame_salmon_1_split.z02 flame_salmon_1_split.z03 flame_salmon_1_split.zip; do
  echo "[$(date '+%m-%d %H:%M:%S')] $f"
  wget -c -q "$B/$f" || { echo "  download failed: $f"; exit 1; }
done
du -ch *.zip *.z0* 2>/dev/null | tail -1
