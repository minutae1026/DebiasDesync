#!/usr/bin/env python3
"""Check the extracted frames and repair broken ones from the source mp4.

Checks: the PNG opens (an interrupted write leaves a truncated file), its size is
1352x1014, and every camera has the right number of frames (cam00 = 300, train = 200).

Repair: train cameras are already trimmed to [s, s+199] and renumbered, so local frame j
is global frame s + j. The window start s comes from build_unsync_scene.plan() and the
frame is re-extracted the same way (BGR -> RGB -> LANCZOS).

    python scripts/data/verify_frames.py                       # all scenes, check + repair
    python scripts/data/verify_frames.py --scenes flame_steak
    python scripts/data/verify_frames.py --check_only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_unsync_scene import (FULL_FRAMES, IMG_WH, TEST_CAM,  # noqa: E402
                                WINDOW, list_cams, plan)

FRAME_RE = re.compile(r"(\d+)\.png$")
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def bad_reason(path: Path, deep: bool = False):
    """Reason the file is broken, or None.

    Only the 33-byte header and the 12-byte tail are read by default: truncation (the
    failure mode of an interrupted write) shows up as a missing IEND chunk. --deep also
    decodes the pixels.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(33)
            if len(head) < 33 or head[:8] != PNG_SIG:
                return "bad PNG header"
            w = int.from_bytes(head[16:20], "big")
            h = int.from_bytes(head[20:24], "big")
            if (w, h) != IMG_WH:
                return f"size {(w, h)}"
            fh.seek(-12, 2)
            if fh.read(12)[4:8] != b"IEND":
                return "no IEND chunk (truncated)"
    except OSError as e:
        return f"read failed ({type(e).__name__})"
    if deep:
        try:
            with Image.open(path) as im:
                im.load()
        except Exception as e:
            return f"decode failed ({type(e).__name__})"
    return None


def global_index(scene_dir: Path, scene: str, cam: str, local: int) -> int:
    if cam == TEST_CAM:
        return local
    cap = cv2.VideoCapture(str(scene_dir / f"{cam}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n == WINDOW:          # the mp4 is already cut to the window (legacy or re-encoded)
        return local
    lay = plan(scene, list_cams(scene_dir))
    return lay.starts[cam] + local


def repair(scene_dir: Path, scene: str, cam: str, local: int) -> bool:
    gi = global_index(scene_dir, scene, cam, local)
    cap = cv2.VideoCapture(str(scene_dir / f"{cam}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, gi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if img.size != IMG_WH:
        img = img.resize(IMG_WH, Image.LANCZOS)
    img.save(scene_dir / cam / "images" / f"{local:04d}.png")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"])
    ap.add_argument("--dynerf_root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "dynerf")
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--deep", action="store_true", help="decode the pixels too (slow)")
    a = ap.parse_args()

    total_bad = total_fixed = 0
    for scene in a.scenes:
        scene_dir = a.dynerf_root / scene
        if not scene_dir.is_dir():
            print(f"[{scene}] missing directory - skipped")
            continue
        bad = []
        for cam_dir in sorted(scene_dir.glob("cam*/images")):
            cam = cam_dir.parent.name
            paths = sorted(cam_dir.glob("*.png"))
            want = FULL_FRAMES if cam == TEST_CAM else WINDOW
            if len(paths) != want:
                print(f"[{scene}] {cam}: {len(paths)} frames (expected {want})")
            for p in paths:
                why = bad_reason(p, a.deep)
                if why:
                    bad.append((cam, int(FRAME_RE.search(p.name).group(1)), why))
        total_bad += len(bad)
        if not bad:
            print(f"[{scene}] ok ({len(list(scene_dir.glob('cam*/images/*.png')))} frames)")
            continue
        print(f"[{scene}] {len(bad)} broken frame(s)")
        for cam, local, why in bad:
            print(f"    {cam}/{local:04d}.png - {why}", end="")
            if a.check_only:
                print()
                continue
            ok = repair(scene_dir, scene, cam, local)
            total_fixed += ok
            print(f"  -> {'repaired' if ok else 'repair failed'}")

    print(f"\nbroken {total_bad}, repaired {total_fixed}")
    return 1 if (total_bad - total_fixed) else 0


if __name__ == "__main__":
    sys.exit(main())
