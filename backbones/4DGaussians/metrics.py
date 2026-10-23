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

from pathlib import Path
import os
import shutil
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
import gc
import re
import math

EDGE_FRAMES = 60


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def _infer_scene_name(model_dir: Path, mask_root: Path):
    model_name = model_dir.name
    if not mask_root.is_dir():
        return None
    candidates = sorted([p.name for p in mask_root.iterdir() if p.is_dir()], key=len, reverse=True)
    for c in candidates:
        if model_name.startswith(c):
            return c
    return None


def _load_dynamic_mask_tensor(mask_path: Path, h: int, w: int, device):
    if not mask_path.is_file():
        return None
    m = Image.open(mask_path).convert("L")
    if m.size != (w, h):
        m = m.resize((w, h), Image.NEAREST)
    mt = tf.to_tensor(m).unsqueeze(0).to(device=device)  # [1,1,H,W], 0~1
    return mt > 0.5


def _masked_psnr(render_t, gt_t, mask_bool):
    if mask_bool is None:
        return float("nan")
    m = mask_bool.expand(-1, 3, -1, -1)
    valid = m.sum().item()
    if valid <= 0:
        return float("nan")
    diff2 = (render_t - gt_t) ** 2
    mse = diff2[m].mean().item()
    return -10.0 * math.log10(max(mse, 1e-12))


def _safe_nanmean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


def _edge_mean(per_frame_dict, total_edge):
    half = total_edge // 2
    pairs = []
    for name, val in per_frame_dict.items():
        m = re.match(r"(\d+)\.png$", name)
        if m:
            pairs.append((int(m.group(1)), val))
    if not pairs:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    merged = {}
    for idx, v in pairs[:half] + (pairs[-half:] if half > 0 else []):
        merged[idx] = v
    return _safe_nanmean(list(merged.values()))


def evaluate(model_paths, dynamic_mask_root=None, cleanup=False):

    full_dict = {}
    per_view_dict = {}
    print("")
    if dynamic_mask_root:
        dm_path = Path(dynamic_mask_root)
        mask_root = dm_path if dm_path.is_absolute() else (Path(__file__).resolve().parents[2] / dm_path).resolve()
    else:
        mask_root = None
    if mask_root is not None and (not mask_root.is_dir()):
        print(f"[Dynamic-PSNR] warning: dynamic_mask_root not found: {mask_root}")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                scene_name = _infer_scene_name(Path(scene_dir), mask_root) if mask_root is not None else None

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                dyn_psnrs = []
                static_psnrs = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                    lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    Dssims.append((1-ms_ssims[-1])/2)
                    if scene_name is not None:
                        frame_idx = int(Path(image_names[idx]).stem)
                        mpath = mask_root / scene_name / "cam00" / "dynamic_mask" / f"dynamic_mask_{frame_idx}.png"
                        _, _, h, w = renders[idx].shape
                        mb = _load_dynamic_mask_tensor(mpath, h, w, renders[idx].device)
                        dyn_psnrs.append(_masked_psnr(renders[idx], gts[idx], mb))
                        static_psnrs.append(_masked_psnr(renders[idx], gts[idx], (~mb) if mb is not None else None))
                    else:
                        dyn_psnrs.append(float("nan"))
                        static_psnrs.append(float("nan"))

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))
                dyn_psnr_mean = _safe_nanmean(dyn_psnrs)
                static_psnr_mean = _safe_nanmean(static_psnrs)
                if not math.isnan(dyn_psnr_mean):
                    print("Scene: ", scene_dir, "Dynamic-PSNR : {:>12.7f}".format(dyn_psnr_mean, ".5"))
                if not math.isnan(static_psnr_mean):
                    print("Scene: ", scene_dir, "Static-PSNR : {:>12.7f}".format(static_psnr_mean, ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                        "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                        "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                        "D-SSIM": torch.tensor(Dssims).mean().item(),
                                                        "Dynamic-PSNR": dyn_psnr_mean,
                                                        "Static-PSNR": static_psnr_mean},

                                                    )
                dyn_per_view = {
                    name: (None if math.isnan(v) else float(v))
                    for v, name in zip(dyn_psnrs, image_names)
                }
                static_per_view = {
                    name: (None if math.isnan(v) else float(v))
                    for v, name in zip(static_psnrs, image_names)
                }
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                            "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                            "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},
                                                            "Dynamic-PSNR": dyn_per_view,
                                                            "Static-PSNR": static_per_view,

                                                            }
                                                        )
                full_dict[scene_dir][method].update({
                    f"Dynamic-PSNR-edge{EDGE_FRAMES}": _edge_mean(dyn_per_view, EDGE_FRAMES),
                    f"Static-PSNR-edge{EDGE_FRAMES}": _edge_mean(static_per_view, EDGE_FRAMES),
                })
                for _key in (f"Dynamic-PSNR-edge{EDGE_FRAMES}", f"Static-PSNR-edge{EDGE_FRAMES}"):
                    _val = full_dict[scene_dir][method].get(_key, float('nan'))
                    if not math.isnan(_val):
                        print("Scene: ", scene_dir, "{} : {:>12.7f}".format(_key, _val))
                del renders, gts, ssims, psnrs, lpipss, lpipsa, ms_ssims, Dssims, dyn_psnrs, static_psnrs
                torch.cuda.empty_cache()
                gc.collect()

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

            if cleanup:
                freed = 0
                for method_dir in (test_dir.iterdir() if test_dir.exists() else []):
                    for sub in ("renders", "gt"):
                        d = method_dir / sub
                        if not d.is_dir():
                            continue
                        freed += sum(p.stat().st_size for p in d.iterdir() if p.is_file())
                        shutil.rmtree(d, ignore_errors=True)
                if freed:
                    print(f"[cleanup] {scene_dir}: removed renders/gt, {freed/1048576:.0f} MB freed")
        except Exception as e:
            
            print("Unable to compute metrics for model", scene_dir)
            raise e

        

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', nargs='*', type=str, default=None,
                        help='One or more model output directories (e.g. output/4DGaussians/expname/)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Single model directory; alias for one entry in --model_paths')
    parser.add_argument('--cleanup', action='store_true',
                        help='delete test/*/renders and gt after the metrics are written (~700 MB per run)')
    parser.add_argument('--dynamic_mask_root', type=str, default='data/dynerf',
                        help='Root containing <scene>/cam00/dynamic_mask/*.png')
    args = parser.parse_args()
    if args.model_path:
        paths = [args.model_path]
    elif args.model_paths:
        paths = list(args.model_paths)
    else:
        parser.error('Provide --model_paths/-m and/or --model_path')
    evaluate(paths, dynamic_mask_root=args.dynamic_mask_root, cleanup=args.cleanup)
