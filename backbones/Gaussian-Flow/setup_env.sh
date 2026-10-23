#!/bin/bash
# Build the CUDA extensions of the Gaussian-Flow backbone inside the environment created from
# Gaussian-Flow/environment.yml, and check that train.py's imports resolve.
#
#   conda env create -f backbones/Gaussian-Flow/environment.yml
#   conda activate debiasdesync-gaussian-flow
#   bash backbones/Gaussian-Flow/setup_env.sh
#
# Extensions:
#   submodules/diff-gaussian-rasterization  stock Inria rasterizer (commit 59f5f77); its glm headers
#                                           are shared with the 4DGaussians rasterizer in ../4DGaussians/submodules/
#   ../4DGaussians/submodules/simple-knn    same source as the 4DGaussians environment
set -euo pipefail

GF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$GF/.." && pwd)"
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "activate the environment first:  conda activate debiasdesync-gaussian-flow" >&2; exit 1
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
export CUDAHOSTCXX="$CXX"
export LDSHARED="$CXX -pthread -shared"

# glm headers of the stock rasterizer (same commit as the 4DGaussians variant's submodule)
GLM_SRC="$REPO/4DGaussians/submodules/depth-diff-gaussian-rasterization/third_party/glm"
GLM_DST="$GF/submodules/diff-gaussian-rasterization/third_party/glm"
if [ ! -f "$GLM_DST/glm/glm.hpp" ]; then
  cp -r "$GLM_SRC/." "$GLM_DST/"
  echo "copied glm headers"
fi

"$PY" -m pip install --no-build-isolation "$GF/submodules/diff-gaussian-rasterization"
"$PY" -m pip install --no-build-isolation "$REPO/4DGaussians/submodules/simple-knn"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/debiasdesync_gflow_env.sh" <<'HOOK'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME="${CONDA_PREFIX}"
HOOK

echo
echo "=== verification ==="
cd "$GF"
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" - <<'PYV'
import torch
print("torch", torch.__version__, "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
import taichi, pytorch3d, mmcv
print("taichi", taichi.__version__, "| pytorch3d", pytorch3d.__version__, "| mmcv", mmcv.__version__)
from simple_knn._C import distCUDA2
print("simple_knn: OK", tuple(distCUDA2(torch.rand(1000, 3).cuda()).shape))
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print("diff_gaussian_rasterization: OK")
from scene import Scene, GaussianModel, DynamicScene
from scene.uniform_t_sampler import UniformTSampler
from utils.pixel_weighting import calc_pixel_weights
print("train.py import graph: OK")
PYV
echo "done. re-activate the environment once so the activate hook takes effect:  conda activate debiasdesync-gaussian-flow"
