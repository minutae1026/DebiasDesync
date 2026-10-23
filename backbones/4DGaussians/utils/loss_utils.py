#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips

def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
    
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calc_pixel_weights(
    image_tensor,
    gt_image_tensor,
    pixel_weighting_ratio: float = 30.0,
    eps: float = 1e-8,
    soft_mining_alpha: float = 1.0,
):
    """Pixel-level weighing of the L1 reconstruction loss (Sec. 3.2, Eqs. 4-6).

    For every image in the batch, with per-pixel error err_x = mean_c |I_hat(x) - I(x)|
    and N = H * W pixels:

        w_x = N * softmax_x(lambda * err_hat_x) ** alpha,   lambda = log(r),
        err_hat_x = (err_x - min err) / (max err - min err),
        L = (1 / N) * sum_x sg(w_x) * err_x

    The weights are computed without gradient (stop-gradient) and are normalized within
    each image, so mean_x w_x = 1 and the loss scale of the plain L1 is preserved.
    r = pixel_weighting_ratio sets the ratio between the largest and smallest weight;
    alpha = 1 is the full weighing and alpha = 0 recovers the unweighted L1 (used for the
    linear warm-up in train.py).

    Returns:
        pixel_weighted_l1: [B] weighted loss of every image
        pixel_weights: [B, 1, H, W] weight maps
    """
    per_pixel_l1 = torch.abs(image_tensor - gt_image_tensor[:, :3, :, :]).mean(dim=1)  # [B,H,W]
    pixel_num = per_pixel_l1.shape[1] * per_pixel_l1.shape[2]

    with torch.no_grad():
        score = per_pixel_l1.detach()

        # lambda / (max - min) applied to the raw error equals lambda applied to the normalized error
        score_min = score.amin(dim=[1, 2], keepdim=True)
        score_max = score.amax(dim=[1, 2], keepdim=True)
        score_range = score_max - score_min
        valid_mask = score_range > 1e-6

        lam = torch.zeros_like(score_range)
        lam[valid_mask] = torch.log(torch.tensor(pixel_weighting_ratio, device=score.device)) / score_range[valid_mask]

        logits = lam * score
        logits_shifted = logits - logits.amax(dim=[1, 2], keepdim=True)
        coeffs = torch.exp(logits_shifted)
        denom = coeffs.sum(dim=[1, 2], keepdim=True)
        prob = coeffs / (denom + eps)  # [B,H,W], sums to 1 over each image

        w_full = (pixel_num * prob) ** soft_mining_alpha
        pixel_weights = w_full.unsqueeze(1)  # [B, 1, H, W]

    weighted_sum_loss = (w_full * per_pixel_l1).sum(dim=[1, 2])  # [B]
    pixel_weighted_l1 = weighted_sum_loss / pixel_num

    return pixel_weighted_l1, pixel_weights
