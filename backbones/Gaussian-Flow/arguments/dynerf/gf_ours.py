_base_ = './gf_base.py'

OptimizationParams = dict(
    uniform_t = True,
    use_pixel_weighting = True,
    pixel_weighting_ratio = 30.0,
    pw_warmup_iters = 1000,
)
