# Gaussian-Flow baseline on the desynchronized Plenoptic Video setting.
# The loader reads <scene>_frames.txt next to the data and gives every training frame its
# global frame index; the model normalizes it (t / 300 * 0.8 + 0.1).
# 30k iterations, batch size 4; everything else as released.
_base_ = './default.py'

OptimizationParams = dict(
    iterations = 30_000,
    position_lr_max_steps = 30_000,
    densify_until_iter = 5_000,
    batch_size = 4,
)
