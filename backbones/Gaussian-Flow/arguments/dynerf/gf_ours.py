# Gaussian-Flow + Ours: uniform sampling across time (Sec. 3.1) and pixel-level weighing
# (Sec. 3.2, r = 30, 1000-iteration linear warm-up) on top of gf_base.py.
_base_ = './gf_base.py'

OptimizationParams = dict(
    uniform_t = True,
    use_pixel_weighting = True,
    pixel_weighting_ratio = 30.0,
    pw_warmup_iters = 1000,
)
