#!/usr/bin/env python3
"""
    python scripts/data/preflight_scenes.py
    python scripts/data/preflight_scenes.py --scenes flame_steak --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backbones" / "4DGaussians"))
from build_unsync_scene import (FULL_FRAMES, IMG_WH, KNOWN_MISALIGNED,  # noqa: E402
                                TEST_CAM, WINDOW, list_cams, plan)
from verify_frames import bad_reason  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "backbones" / "4DGaussians" / "arguments" / "dynerf"

TRAIN_T = (WINDOW - 1) / (FULL_FRAMES - 1)      # 199/299 = 0.66555...
PLY_RANGE = (20_000, 45_000)                    # the six scenes have about 36k points
DYN_RATIO = (0.5, 12.0)                         # % of dynamic pixels in cam00


class Report:
    def __init__(self, verbose): self.fail = []; self.verbose = verbose

    def check(self, ok, label, detail=""):
        if ok:
            if self.verbose:
                print(f"    PASS  {label}  {detail}")
        else:
            print(f"    FAIL  {label}  {detail}")
            self.fail.append(label)
        return ok


def check_frames(d: Path, cams, r: Report):
    for cam in cams:
        want = FULL_FRAMES if cam == TEST_CAM else WINDOW
        paths = sorted((d / cam / "images").glob("*.png"))
        if not r.check(len(paths) == want, f"A.frame count {cam}", f"{len(paths)} (expected {want})"):
            continue
        idx = [int(p.stem) for p in paths]
        r.check(idx == list(range(want)), f"A.frame indices {cam}", "0..%d contiguous" % (want - 1))
        bads = []
        for p in paths[::37]:
            why = bad_reason(p)
            if why:
                bads.append((p.name, why))
        r.check(not bads, f"A.png integrity/size {cam}", str(bads[:2]))


def check_videos(d: Path, cams, r: Report):
    for cam in cams:
        want = FULL_FRAMES if cam == TEST_CAM else WINDOW
        n = int(cv2.VideoCapture(str(d / f"{cam}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        r.check(n == want, f"B.mp4 {cam}", f"{n} frames (expected {want})")


def check_config(scene: str, d: Path, cams, r: Report):
    cfg = CFG_DIR / f"{scene}.py"
    if not r.check(cfg.is_file(), "C.config exists", str(cfg)):
        return
    txt = cfg.read_text()
    gt = re.findall(r"(-?\d+\.\d+),\s*#\s*(cam\d+)", txt)
    r.check(len(gt) == len(cams), "C.gt_delta_time count", f"{len(gt)} (cameras {len(cams)})")
    lay = plan(scene, cams)
    bad = [c for v, c in gt if c != TEST_CAM and abs(float(v) - lay.offset(c)) > 1e-9]
    bad += [c for v, c in gt if c == TEST_CAM and abs(float(v)) > 1e-12]
    r.check(not bad, "C.offset values", f"mismatch {bad[:3]}")
    try:
        import mmcv
        bs = mmcv.Config.fromfile(str(cfg)).get("OptimizationParams", {}).get("batch_size")
    except Exception:
        bs = None
    r.check(bs == 4, "C.batch_size", f"{bs} (including default.py)")

    fp = d / f"{scene}_frames.txt"
    if r.check(fp.is_file(), "C.frames.txt exists"):
        rec = {a: int(b) for a, b, _ in re.findall(r"(cam\d+): \[(\d+), (\d+)\]", fp.read_text())}
        r.check(rec == lay.starts, "C.frames.txt matches the plan",
                f"{len(rec)} entries" if rec == lay.starts else "differs from the plan")
        gstart = {a: int(b) for a, b in re.findall(r"(cam\d+):.*global_start (\d+)", fp.read_text())}
        if gstart:
            r.check(len(gstart) == len(rec) and all(gstart[c] == lay.global_start(c) for c in rec),
                    "C.frames.txt global_start", "consistent with the plan")
        else:
            print("    NOTE  C.frames.txt has no global_start field; rebuild it with "
                  "build_unsync_scene.py before training Gaussian-Flow")


def check_ply(d: Path, r: Report):
    p = d / "points3D_downsample2.ply"
    if not r.check(p.is_file(), "D.ply exists"):
        return
    n = 0
    with open(p, "rb") as fh:
        for line in fh:
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
    r.check(PLY_RANGE[0] <= n <= PLY_RANGE[1], "D.point count", f"{n} (expected {PLY_RANGE})")


def check_masks(d: Path, r: Report):
    md = d / TEST_CAM / "dynamic_mask"
    ps = sorted(md.glob("dynamic_mask_*.png"))
    if not r.check(len(ps) == FULL_FRAMES, "E.cam00 mask count", f"{len(ps)}"):
        return
    im = cv2.imread(str(ps[0]), 0)
    r.check(im.shape[::-1] == IMG_WH, "E.mask size", str(im.shape[::-1]))
    fr = [(cv2.imread(str(p), 0) > 127).mean() * 100 for p in ps[::20]]
    mu = float(np.mean(fr))
    r.check(DYN_RATIO[0] <= mu <= DYN_RATIO[1], "E.dynamic ratio", f"{mu:.2f}% (expected {DYN_RATIO})")
    r.check((md / "median_background_gray.png").is_file(), "E.median_background")


def check_loader(scene: str, d: Path, cams, r: Report):
    try:
        import mmcv
        from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    except Exception as e:
        r.check(False, "F.import", str(e)); return

    cfg = mmcv.Config.fromfile(str(CFG_DIR / f"{scene}.py"))
    gt = list(cfg["ModelParams"]["gt_delta_time"])

    ds = {}
    for split in ("train", "test"):
        try:
            ds[split] = Neural3D_NDC_Dataset(str(d), split, 1.0, time_scale=1,
                                             scene_bbox_min=[-2.5, -2.0, -1.0],
                                             scene_bbox_max=[2.5, 2.0, 1.0], eval_index=0)
        except Exception as e:
            r.check(False, f"F.{split} load", f"{type(e).__name__}: {e}"); return

    tt = np.array(ds["test"].image_times)
    r.check(abs(tt.min() + 1) < 1e-6 and abs(tt.max() - 1) < 1e-6,
            "F.test time range", f"[{tt.min():.5f}, {tt.max():.5f}] (expected [-1, 1])")
    r.check(len(tt) == FULL_FRAMES, "F.test frame count", str(len(tt)))

    tr = np.array(ds["train"].image_times)
    r.check(abs(tr.min() + TRAIN_T) < 1e-4 and abs(tr.max() - TRAIN_T) < 1e-4,
            "F.train time range",
            f"[{tr.min():.5f}, {tr.max():.5f}] (expected +-{TRAIN_T:.5f})")
    r.check(len(tr) == (len(cams) - 1) * WINDOW, "F.train sample count",
            f"{len(tr)} (expected {(len(cams)-1)*WINDOW})")

    cov = np.zeros(FULL_FRAMES, dtype=int)
    for t, ci in zip(ds["train"].image_times, ds["train"].image_cam_indices):
        g = float(t) + gt[ci]
        f = int(round((g + 1) * (FULL_FRAMES - 1) / 2))
        if 0 <= f < FULL_FRAMES:
            cov[f] += 1
    n_train = len(cams) - 1
    lay = plan(scene, cams)
    want = np.zeros(FULL_FRAMES, dtype=int)
    for cam, s in lay.starts.items():
        s2 = s + KNOWN_MISALIGNED.get((scene, cam), 0)
        lo, hi = max(0, s2), min(FULL_FRAMES, s2 + WINDOW)
        want[lo:hi] += 1
    r.check(np.array_equal(cov, want), "F.coverage profile",
            f"measured ends={cov[0]},{cov[-1]} center={cov[150]} / expected {want[0]},{want[-1]},{want[150]}")
    r.check(cov.max() == n_train, "F.center coverage", f"{cov.max()} (train cameras {n_train})")
    r.check(cov.min() >= 1, "F.no empty time step", f"min {cov.min()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"])
    ap.add_argument("--dynerf_root", type=Path, default=ROOT / "data" / "dynerf")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    total = []
    for scene in a.scenes:
        d = a.dynerf_root / scene
        print(f"\n=== {scene} ===")
        if not d.is_dir():
            print("    FAIL  missing directory"); total.append(scene); continue
        r = Report(a.verbose)
        cams = list_cams(d)
        print(f"  {len(cams)} cameras ({len(cams)-1} train)")
        check_frames(d, cams, r)
        check_videos(d, cams, r)
        check_config(scene, d, cams, r)
        check_ply(d, r)
        check_masks(d, r)
        check_loader(scene, d, cams, r)
        if r.fail:
            print(f"  -> {len(r.fail)} failure(s): {r.fail[:6]}")
            total.append(scene)
        else:
            print("  -> all checks passed")

    print("\n" + "=" * 60)
    if total:
        print(f"failed scenes: {total}")
        return 1
    print(f"all scenes passed ({', '.join(a.scenes)}) - ready for training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
