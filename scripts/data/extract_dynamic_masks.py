#!/usr/bin/env python3
"""
Per-frame dynamic masks via temporal-median background subtraction.

For each (scene, cam), the per-pixel temporal median across all frames
approximates the static background; large |I_t - median| marks dynamic regions.

Setting used for the evaluation masks: --threshold 15 --color gray (no morphology).

Outputs to <out_root>/<scene>/<cam>/dynamic_mask/dynamic_mask_<frame_idx>.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def sorted_frame_paths(images_dir: Path) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for p in images_dir.glob("*.png"):
        m = re.match(r"^(\d+)\.png$", p.name)
        if m:
            paths.append((int(m.group(1)), p))
    paths.sort(key=lambda x: x[0])
    return paths


def morph_clean(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def discover_cams(scene_dir: Path) -> list[str]:
    """Return sorted list of cam names with images/ subfolder under scene_dir."""
    cams: list[str] = []
    for p in sorted(scene_dir.iterdir()):
        if p.is_dir() and re.match(r"^cam\d+$", p.name) and (p / "images").is_dir():
            cams.append(p.name)
    return cams


def process_one(
    scene: str,
    cam: str,
    dynerf_root: Path,
    out_root: Path,
    threshold: int,
    color: str,
    morph: bool,
    skip_existing: bool,
) -> None:
    img_dir = dynerf_root / scene / cam / "images"
    if not img_dir.is_dir():
        print(f"[skip] missing {img_dir}", flush=True)
        return
    paths = sorted_frame_paths(img_dir)
    if not paths:
        print(f"[skip] no png in {img_dir}", flush=True)
        return

    out_dir = out_root / scene / cam / "dynamic_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(paths)
    indices = [i for i, _ in paths]
    if skip_existing:
        existing = sum(1 for i in indices if (out_dir / f"dynamic_mask_{i}.png").is_file())
        if existing == n:
            print(f"[skip complete] {scene} {cam} ({n} masks)", flush=True)
            return

    print(
        f"[{scene}] {cam} frames={n} color={color} threshold={threshold} morph={morph}",
        flush=True,
    )

    sample = cv2.imread(str(paths[0][1]), cv2.IMREAD_COLOR)
    if sample is None:
        print(f"[skip] cannot read {paths[0][1]}", flush=True)
        return
    h, w = sample.shape[:2]

    if color == "gray":
        stack = np.empty((n, h, w), dtype=np.uint8)
        for k, (_, p) in enumerate(paths):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to read {p}")
            stack[k] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        median_bg = np.median(stack.astype(np.float32), axis=0)
        med_vis = np.clip(median_bg, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "median_background_gray.png"), med_vis)

        for k, (fi, _) in enumerate(paths):
            d = np.abs(stack[k].astype(np.float32) - median_bg)
            d_u8 = np.clip(d, 0, 255).astype(np.uint8)
            _, m = cv2.threshold(d_u8, threshold, 255, cv2.THRESH_BINARY)
            if morph:
                m = morph_clean(m)
            cv2.imwrite(str(out_dir / f"dynamic_mask_{fi}.png"), m)
    else:
        stack = np.empty((n, h, w, 3), dtype=np.uint8)
        for k, (_, p) in enumerate(paths):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to read {p}")
            stack[k] = bgr
        median_bg = np.median(stack.astype(np.float32), axis=0)
        med_vis = np.clip(median_bg, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "median_background_bgr.png"), med_vis)

        for k, (fi, _) in enumerate(paths):
            d = np.abs(stack[k].astype(np.float32) - median_bg).sum(axis=2)
            d_u8 = np.clip(d, 0, 255).astype(np.uint8)
            _, m = cv2.threshold(d_u8, threshold, 255, cv2.THRESH_BINARY)
            if morph:
                m = morph_clean(m)
            cv2.imwrite(str(out_dir / f"dynamic_mask_{fi}.png"), m)

    print(f"[done] {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dynerf_root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "dynerf",
        help="Root containing <scene>/<cam>/images/*.png",
    )
    ap.add_argument(
        "--scenes",
        nargs="+",
        default=["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"],
    )
    ap.add_argument(
        "--cams",
        nargs="+",
        default=None,
        help="cam names (e.g. cam00 cam01 ...). If omitted, all cam<NN> dirs with images/ are used.",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output root. Default: same as --dynerf_root (writes into data/dynerf/<scene>/<cam>/dynamic_mask/).",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="|I - median_bg| threshold in 0..255 (default: 15, adopted setup).",
    )
    ap.add_argument(
        "--color",
        choices=("gray", "rgb_l1"),
        default="gray",
        help="gray: median on luminance (default). rgb_l1: median per BGR channel, sum of abs diffs.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a (scene, cam) if all dynamic_mask_*.png already exist.",
    )
    ap.add_argument(
        "--morph",
        action="store_true",
        help="Apply 3x3 open+close morphology on binary masks (off by default).",
    )
    args = ap.parse_args()

    out_root = args.out_root if args.out_root is not None else args.dynerf_root

    for scene in args.scenes:
        scene_dir = args.dynerf_root / scene
        if not scene_dir.is_dir():
            print(f"[skip] missing scene dir {scene_dir}", flush=True)
            continue

        cams = args.cams if args.cams else discover_cams(scene_dir)
        if not cams:
            print(f"[skip] no cams found under {scene_dir}", flush=True)
            continue

        for cam in cams:
            process_one(
                scene=scene,
                cam=cam,
                dynerf_root=args.dynerf_root,
                out_root=out_root,
                threshold=args.threshold,
                color=args.color,
                morph=args.morph,
                skip_existing=args.skip_existing,
            )


if __name__ == "__main__":
    main()
