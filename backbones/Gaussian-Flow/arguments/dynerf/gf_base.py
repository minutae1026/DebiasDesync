# Gaussian-Flow baseline on the desynchronized Plenoptic Video setting (Table 1).
#
# Time axis: the loader reads <scene>_frames.txt next to the data and gives every training
# frame its global frame index s_k + j; the model normalizes it (t / 300 * 0.8 + 0.1).
# iterations 30k: the Gaussian-Flow paper's main setting (the released dynerf config's 10k
# overrides the repository default of 30k). densify_until_iter and the learning rates are the
# released values. batch_size 4 matches the 4DGaussians experiments.
_base_ = './default.py'

OptimizationParams = dict(
    iterations = 30_000,
    position_lr_max_steps = 30_000,
    densify_until_iter = 5_000,
    batch_size = 4,
)
