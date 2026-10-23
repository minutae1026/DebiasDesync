_base_ = './default.py'

OptimizationParams = dict(
    iterations = 30_000,
    position_lr_max_steps = 30_000,
    densify_until_iter = 5_000,
    batch_size = 4,
)
