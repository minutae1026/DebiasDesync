"""Pixel-level weighing of the L1 loss (Sec. 3.2). Same computation as calc_pixel_weights in
the 4DGaussians code base (../../4DGaussians/utils/loss_utils.py), so both backbones use an identical loss."""
import torch

__all__ = ["calc_pixel_weights"]


def calc_pixel_weights(
    image_tensor,
    gt_image_tensor,
    pixel_weighting_ratio: float = 30.0,
    eps: float = 1e-8,
    soft_mining_alpha: float = 1.0,
):
    """For every image in the batch, with per-pixel error err_x = mean_c |I_hat(x) - I(x)|
    and N = H * W pixels:

        w_x = N * softmax_x(lambda * err_hat_x) ** alpha,   lambda = log(r),
        err_hat_x = (err_x - min err) / (max err - min err),
        L = (1 / N) * sum_x sg(w_x) * err_x

    The weights are computed without gradient and normalized within each image, so
    mean_x w_x = 1 and the loss scale of the plain L1 is preserved. r = pixel_weighting_ratio;
    alpha = 1 is the full weighing and alpha = 0 the unweighted L1 (used for the warm-up).

    Returns:
        pixel_weighted_l1: [B] weighted loss of every image
        pixel_weights: [B, 1, H, W] weight maps
    """
    per_pixel_l1 = torch.abs(image_tensor - gt_image_tensor[:, :3, :, :]).mean(dim=1)  # [B,H,W]
    pixel_num = per_pixel_l1.shape[1] * per_pixel_l1.shape[2]

    with torch.no_grad():
        score = per_pixel_l1.detach()

        score_min = score.amin(dim=[1, 2], keepdim=True)
        score_max = score.amax(dim=[1, 2], keepdim=True)
        score_range = score_max - score_min
        valid_mask = score_range > 1e-6

        lam = torch.zeros_like(score_range)
        lam[valid_mask] = torch.log(
            torch.tensor(pixel_weighting_ratio, device=score.device)
        ) / score_range[valid_mask]

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
