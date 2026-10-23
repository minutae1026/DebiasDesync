#!/usr/bin/env python3
"""Turn a Neural 3D Video scene into the desynchronized setting used in the paper.

    test view    cam00 keeps the full F-frame timeline (F = 300)
    train views  every other camera keeps a contiguous T-frame window [s, s+T-1] (T = 200)
    starts       the first camera gets s = 0 and the second s = F-T, so the whole timeline
                 is covered; the others draw s from [0, F-T-1] without repetition
    assignment   camera ids are shuffled before being paired with the starts, so cameras
                 covering the same times are not spatially adjacent
    offset       global time offset of a camera on the [-1, 1] axis:
                 (2s + T - F) / (F - 1)  ==  (s - (F-T)/2) * 2 / (F-1)
    seed         777 (random.Random)

A source video longer than F frames (flame_salmon_1 has 1200) is cut to its first F frames,
the standard 10 s x 30 fps protocol of the dataset.

Training frames. By default the PNGs are decoded from the source video and the trimmed mp4
is re-encoded afterwards from them (scripts/data/make_trimmed_videos.py). With --legacy_trim
(used for coffee_martini, cook_spinach and sear_steak) the source mp4 is cut to the window
first and the PNGs are decoded from the cut video.

    python scripts/data/build_unsync_scene.py --scene flame_steak
    python scripts/data/build_unsync_scene.py --scene cook_spinach --verify

COLMAP, the point cloud and the dynamic masks are produced afterwards by scripts/data/build_scenes.sh.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "backbones" / "4DGaussians" / "arguments" / "dynerf"

TEST_CAM = "cam00"
# The source videos are 2704x2028; the 4DGaussians loader works at 1352x1014 (downsample 2),
# which is also what llff2colmap.py and the dynamic masks assume.
IMG_WH = (1352, 1014)
FULL_FRAMES = 300      # length F of the common timeline
WINDOW = 200           # window length T of every training camera
SEED = 777

# Cameras that are desynchronized in the original capture: the global time of their window
# is shifted by this many frames.
KNOWN_MISALIGNED = {("coffee_martini", "cam13"): 45}

FRAME_RE = re.compile(r"\d+\.png$")
WINDOW_RE = re.compile(r"(cam\d+): \[(\d+), (\d+)\]")


@dataclass(frozen=True)
class Layout:
    """Window assignment of one scene."""
    scene: str
    cams: list          # all cameras including cam00, sorted by name
    starts: dict        # train camera -> window start frame
    full: int
    window: int

    def global_start(self, cam: str) -> int:
        """Global frame of the camera's first kept frame (window start + known misalignment)."""
        return self.starts[cam] + KNOWN_MISALIGNED.get((self.scene, cam), 0)

    def offset(self, cam: str) -> float:
        """Time offset of a train camera on the global [-1, 1] axis."""
        s = self.global_start(cam)
        return (2 * s + self.window - self.full) / (self.full - 1)

    def span(self, cam: str) -> tuple:
        s = self.starts[cam]
        return s, s + self.window - 1


def plan(scene: str, cams: list, full: int = FULL_FRAMES,
         window: int = WINDOW, seed: int = SEED) -> Layout:
    """Draw the window starts and assign them to the train cameras (deterministic)."""
    train = [c for c in cams if c != TEST_CAM]
    bound = full - window
    rng = random.Random(seed)

    starts: list = []
    for i in range(len(train)):
        if i == 0:
            s = 0
        elif i == 1:
            s = bound
        else:
            while True:
                s = rng.randint(0, bound - 1)
                if s not in starts:
                    break
        starts.append(s)

    shuffled = list(train)
    rng.shuffle(shuffled)
    return Layout(scene, cams, dict(zip(shuffled, starts)), full, window)


def list_cams(scene_dir: Path) -> list:
    return sorted(p.stem for p in scene_dir.glob("cam*.mp4"))


def frames_of(cam_dir: Path) -> list:
    img = cam_dir / "images"
    if not img.is_dir():
        return []
    return sorted(p.name for p in img.glob("*.png") if FRAME_RE.fullmatch(p.name))


def extract(scene_dir: Path, cam: str, n: int) -> int:
    """Write the first n frames of <cam>.mp4 to <cam>/images/0000.png ... (skipped if present)."""
    out = scene_dir / cam / "images"
    have = frames_of(scene_dir / cam)
    if len(have) >= n:
        return len(have)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(scene_dir / f"{cam}.mp4"))
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        # same path as the 4DGaussians loader: BGR -> RGB -> PIL -> LANCZOS resize
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if img.size != IMG_WH:
            img = img.resize(IMG_WH, Image.LANCZOS)
        img.save(out / f"{i:04d}.png")
        i += 1
    cap.release()
    return i


def trim(scene_dir: Path, cam: str, start: int, window: int) -> bool:
    """Keep frames [start, start+window-1] and renumber them from 0000. False if already trimmed."""
    names = frames_of(scene_dir / cam)
    if len(names) == window:
        return False
    keep = names[start:start + window]
    if len(keep) != window:
        raise SystemExit(f"{cam}: not enough frames for window [{start}, {start+window-1}] ({len(names)} frames)")
    img, tmp = scene_dir / cam / "images", scene_dir / cam / "images_trim"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for j, n in enumerate(keep):
        shutil.move(str(img / n), str(tmp / f"{j:04d}.png"))
    shutil.rmtree(img)
    tmp.rename(img)
    return True


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("ffmpeg not found (pip install imageio-ffmpeg)")
        return exe


def _video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _bitrate_kbps(exe: str, path: Path) -> int:
    r = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "bitrate:" in line:
            try:
                return int(line.split("bitrate:")[1].split("kb/s")[0].strip())
            except ValueError:
                pass
    return 0


def trim_video(scene_dir: Path, cam: str, start: int, window: int, full: int) -> bool:
    """Cut <cam>.mp4 to frames [start, start+window-1] at full resolution
    (libx264, preset medium, yuv420p, 30 fps, bitrate = source x window/full).
    False if the video already has window frames.
    """
    src = scene_dir / f"{cam}.mp4"
    if _video_frames(src) == window:
        return False
    exe = _ffmpeg()
    kbps = _bitrate_kbps(exe, src)
    with tempfile.TemporaryDirectory(dir=str(scene_dir)) as td:
        out = Path(td) / "trim.mp4"
        cmd = [exe, "-y", "-i", str(src),
               "-vf", f"select=between(n\\,{start}\\,{start + window - 1}),setpts=N/FRAME_RATE/TB",
               "-vframes", str(window),
               "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
        if kbps > 0:
            cmd += ["-b:v", f"{int(kbps * window / full)}k"]
        cmd.append(str(out))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or _video_frames(out) != window:
            raise SystemExit(f"{cam}: ffmpeg trimming failed "
                             f"({r.stderr.strip().splitlines()[-1] if r.stderr else '?'})")
        shutil.move(str(out), str(src))
    return True


def render_config(lay: Layout) -> str:
    rows = [f"        0.00000000000,  # {TEST_CAM} (test view)"]
    for cam in lay.cams:
        if cam == TEST_CAM:
            continue
        lo, hi = lay.span(cam)
        note = ""
        if (lay.scene, cam) in KNOWN_MISALIGNED:
            note = f" (misaligned +{KNOWN_MISALIGNED[(lay.scene, cam)]} frames)"
        rows.append(f"       {lay.offset(cam):+.11f},  # {cam}: [{lo}, {hi}]{note}")
    return "\n".join([
        "_base_ = './default.py'", "",
        "# Desynchronized setting: cam00 is the test view with all 300 frames; every training",
        "# camera observes a 200-frame window [s, s+199] of the common timeline. Its time offset",
        "# on the global [-1, 1] axis is (s - 50) * 2 / 299 (seed 777, scripts/data/build_unsync_scene.py).",
        "ModelParams = dict(",
        "    eval_index=0,",
        "    gt_delta_time=[", *rows, "    ],",
        ")", "",
    ])


def render_frames_txt(lay: Layout) -> str:
    """One line per train camera: the kept source frames [s, s+T-1] and the global frame of
    the first kept frame (differs from s only for known-misaligned cameras). The Gaussian-Flow
    loader reads global_start; the 4DGaussians configs carry the same information as offsets."""
    return "".join(f"{c}: [{lo}, {hi}], {lay.window} frames, global_start {lay.global_start(c)}\n"
                   for c in lay.starts for lo, hi in [lay.span(c)])


def describe(lay: Layout) -> None:
    print(f"[{lay.scene}] {len(lay.cams)} cameras ({len(lay.starts)} train)")
    for cam in sorted(lay.starts):
        lo, hi = lay.span(cam)
        print(f"    {cam}: [{lo:3d}, {hi:3d}]  offset={lay.offset(cam):+.11f}")


def verify(scene: str, root: Path) -> bool:
    """Check that <scene>_frames.txt and arguments/dynerf/<scene>.py agree with the plan."""
    txt = (root / scene / f"{scene}_frames.txt").read_text()
    recorded = {m[0]: int(m[1]) for m in WINDOW_RE.findall(txt)}
    cfg = CFG_DIR / f"{scene}.py"
    gt = dict((c, float(v)) for v, c in re.findall(r"(-?\d+\.\d+),\s*#\s*(cam\d+)", cfg.read_text()))

    lay = plan(scene, [TEST_CAM] + sorted(recorded))
    bad = [f"{c}: recorded s={recorded[c]} vs planned s={lay.starts[c]}"
           for c in recorded if lay.starts[c] != recorded[c]]
    bad += [f"{c}: config offset={gt[c]:+.11f} vs planned {lay.offset(c):+.11f}"
            for c in recorded if abs(gt.get(c, 1e9) - lay.offset(c)) > 1e-9]
    gstart = {m[0]: int(m[1]) for m in re.findall(r"(cam\d+):.*global_start (\d+)", txt)}
    bad += [f"{c}: recorded global_start={gstart[c]} vs planned {lay.global_start(c)}"
            for c in gstart if gstart[c] != lay.global_start(c)]

    print(f"[verify {scene}] {len(recorded)} train cameras -> "
          + ("match" if not bad else f"{len(bad)} mismatch(es)"))
    for b in bad[:5]:
        print("   ", b)
    return not bad


def build(scene: str, root: Path, full: int, window: int, legacy: bool = False) -> None:
    scene_dir = root / scene
    if not scene_dir.is_dir():
        raise SystemExit(f"missing: {scene_dir}")
    cams = list_cams(scene_dir)
    if TEST_CAM not in cams:
        raise SystemExit(f"{TEST_CAM} not found")
    poses = np.load(scene_dir / "poses_bounds.npy")
    if poses.shape[0] != len(cams):
        raise SystemExit(f"poses_bounds has {poses.shape[0]} rows but there are {len(cams)} cameras")

    lay = plan(scene, cams, full, window)
    describe(lay)

    if legacy:
        # cut the source video first, then decode the training frames from the cut video
        for cam in sorted(lay.starts):
            did = trim_video(scene_dir, cam, lay.starts[cam], window, full)
            print(f"  [trim mp4] {cam} -> {window} frames{'' if did else ' (already trimmed)'}", flush=True)
            n = extract(scene_dir, cam, window)
            print(f"  [extract] {cam}: {n} frames", flush=True)
            if n < window:
                raise SystemExit(f"{cam}: could not extract {window} frames ({n})")
        n = extract(scene_dir, TEST_CAM, full)
        print(f"  [extract] {TEST_CAM}: {n} frames", flush=True)
        if n < full:
            raise SystemExit(f"{TEST_CAM}: could not extract {full} frames ({n})")
    else:
        for cam in cams:
            n = extract(scene_dir, cam, full)
            print(f"  [extract] {cam}: {n} frames", flush=True)
            if n < full:
                raise SystemExit(f"{cam}: could not extract {full} frames ({n})")
        for cam in sorted(lay.starts):
            did = trim(scene_dir, cam, lay.starts[cam], window)
            print(f"  [trim] {cam} -> {window} frames{'' if did else ' (already trimmed)'}", flush=True)

    (scene_dir / f"{scene}_frames.txt").write_text(render_frames_txt(lay))
    print(f"  [write] {scene_dir / f'{scene}_frames.txt'}")
    # Shipped configs are never overwritten; an existing config must agree with the plan,
    # otherwise the trainer would use offsets that do not match the frames on disk.
    cfg = CFG_DIR / f"{scene}.py"
    if cfg.is_file():
        if not verify(scene, root):
            raise SystemExit(f"{cfg} disagrees with the window plan")
        print(f"  [keep] {cfg} (matches the plan)")
    else:
        cfg.write_text(render_config(lay))
        print(f"  [write] {cfg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dynerf_root", type=Path, default=ROOT / "data" / "dynerf")
    ap.add_argument("--full_frames", type=int, default=FULL_FRAMES)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--legacy_trim", action="store_true",
                    help="cut the source mp4 first and decode the training frames from it "
                         "(coffee_martini, cook_spinach, sear_steak)")
    ap.add_argument("--dry_run", action="store_true", help="print the assignment only")
    ap.add_argument("--verify", action="store_true", help="compare existing outputs with the plan")
    a = ap.parse_args()

    if a.verify:
        return 0 if verify(a.scene, a.dynerf_root) else 1
    if a.dry_run:
        describe(plan(a.scene, list_cams(a.dynerf_root / a.scene), a.full_frames, a.window))
        return 0
    build(a.scene, a.dynerf_root, a.full_frames, a.window, legacy=a.legacy_trim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
