#!/bin/bash
#
#   conda env create -f backbones/4DGaussians/environment.yml
#   conda activate debiasdesync
#   bash backbones/4DGaussians/setup_env.sh
#
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "activate the environment first:  conda activate debiasdesync" >&2; exit 1
fi
PY="$CONDA_PREFIX/bin/python"
echo "env: $CONDA_PREFIX"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$("$PY" -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')")"
fi
export TORCH_CUDA_ARCH_LIST
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

CP="$CONDA_PREFIX/compiler_compat"
[ -f "$CP/ld" ] && mv "$CP/ld" "$CP/ld.bak" && echo "disabled compiler_compat/ld"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export LDSHARED="$CXX -pthread -shared"

cd "$REPO"
"$PY" -m pip install -e submodules/simple-knn --no-build-isolation
"$PY" -m pip install -e submodules/depth-diff-gaussian-rasterization --no-build-isolation

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/debiasdesync_env.sh" <<'HOOK'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME="${CONDA_PREFIX}"
HOOK

echo
echo "=== verification ==="
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" - <<'PYV'
import torch
print("torch", torch.__version__, "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
from simple_knn._C import distCUDA2
print("simple_knn: OK", tuple(distCUDA2(torch.rand(1000, 3).cuda()).shape))
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print("diff_gaussian_rasterization: OK")
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import calc_pixel_weights
from scene.sampler import UniformTimeSampler
import mmcv, lpips, cv2, open3d
print("train.py import graph: OK")
PYV
echo "done. re-activate the environment once so the activate hook takes effect:  conda activate debiasdesync"
