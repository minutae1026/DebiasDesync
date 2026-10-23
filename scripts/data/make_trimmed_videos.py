#!/usr/bin/env python3
"""Re-encode every training camera's mp4 from its trimmed frames.

The 4DGaussians loader (scene/neural_3D_dataset_NDC.py) reads the number of frames of
each mp4 and builds the time axis from it:

    image_times = (2*idx/(countss-1) - 1) * ((countss-1)/(eval_frame_count-1))

With the original 300-frame (1200 for flame_salmon_1) videos the time axis of a
200-frame window would be wrong and the loader would even extract the missing frames,
undoing the desynchronized split. The mp4 therefore defines the time axis and must hold
exactly the window's frames. cam00 keeps its 300 frames.

Encoding: libx264, preset medium, yuv420p, 30 fps, bitrate = source x (window/full).

    python scripts/data/make_trimmed_videos.py --scenes flame_steak
    python scripts/data/make_trimmed_videos.py --check_only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

TEST_CAM = "cam00"
FULL_FRAMES = 300
WINDOW = 200
FPS = "30"


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("ffmpeg not found (pip install imageio-ffmpeg)")
        return exe


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def bitrate_kbps(exe: str, path: Path) -> int:
    """Source bitrate in kbps, read with ffmpeg (the static build has no ffprobe)."""
    r = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "bitrate:" in line:
            try:
                return int(line.split("bitrate:")[1].split("kb/s")[0].strip())
            except ValueError:
                pass
    return 0


def encode(exe: str, img_dir: Path, out: Path, n: int, kbps: int) -> bool:
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as td:
        tmp_out = Path(td) / "out.mp4"
        cmd = [exe, "-y", "-framerate", FPS, "-start_number", "0",
               "-i", str(img_dir / "%04d.png"),
               "-c:v", "libx264", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-vframes", str(n)]
        if kbps > 0:
            cmd += ["-b:v", f"{kbps}k"]
        cmd.append(str(tmp_out))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp_out.is_file():
            print(f"    ffmpeg failed: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
            return False
        got = frame_count(tmp_out)
        if got != n:
            print(f"    frame count mismatch: {got} (expected {n})")
            return False
        shutil.move(str(tmp_out), str(out))   # atomic replace
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"])
    ap.add_argument("--dynerf_root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "dynerf")
    ap.add_argument("--full", type=int, default=FULL_FRAMES)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--check_only", action="store_true")
    a = ap.parse_args()

    exe = ffmpeg_exe()
    fail = 0
    for scene in a.scenes:
        d = a.dynerf_root / scene
        if not d.is_dir():
            print(f"[{scene}] missing directory - skipped"); continue
        print(f"[{scene}]")
        for mp4 in sorted(d.glob("cam*.mp4")):
            cam = mp4.stem
            want = a.full if cam == TEST_CAM else a.window
            have = frame_count(mp4)
            n_img = len(list((d / cam / "images").glob("*.png")))
            if have == want and n_img == want:
                continue
            if n_img != want:
                print(f"  {cam}: {n_img} frames on disk (expected {want}) - extract/trim the frames first")
                fail += 1
                continue
            print(f"  {cam}: mp4 {have} -> {want} frames", end="", flush=True)
            if a.check_only:
                print(" (check_only)"); fail += 1; continue
            kb = bitrate_kbps(exe, mp4)
            tgt = int(kb * want / a.full) if kb else 0
            if encode(exe, d / cam / "images", mp4, want, tgt):
                print(f"  ok ({kb}kbps -> {tgt}kbps)")
            else:
                fail += 1

    print(f"\n{fail} failure(s)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
