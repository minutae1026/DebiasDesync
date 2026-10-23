#!/bin/bash
#   bash scripts/data/colmap.sh data/dynerf/<scene> llff
# Requires the `colmap` binary on PATH (e.g. a separate conda env: conda create -n colmap -c conda-forge colmap).
# PY must be the python of the training environment (needs numpy); default: current `python`.
workdir=$1
datatype=$2 # blender, hypernerf, llff
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

rm -rf $workdir/sparse_
rm -rf $workdir/image_colmap
$PY "$REPO"/scripts/data/"$datatype"2colmap.py $workdir
rm -rf $workdir/colmap
rm -rf $workdir/colmap/sparse/0

mkdir $workdir/colmap
cp -r $workdir/image_colmap $workdir/colmap/images
cp -r $workdir/sparse_ $workdir/colmap/sparse_custom
colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
$PY "$REPO"/scripts/data/colmap_database.py --database_path $workdir/colmap/database.db --txt_path $workdir/colmap/sparse_custom/cameras.txt
colmap exhaustive_matcher --database_path $workdir/colmap/database.db
mkdir -p $workdir/colmap/sparse/0

colmap point_triangulator --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse_custom --output_path $workdir/colmap/sparse/0 --clear_points 1

mkdir -p $workdir/colmap/dense/workspace
colmap image_undistorter --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse/0 --output_path $workdir/colmap/dense/workspace
colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace
colmap stereo_fusion --workspace_path $workdir/colmap/dense/workspace --output_path $workdir/colmap/dense/workspace/fused.ply
