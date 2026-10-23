#!/usr/bin/env python3
"""
    python scripts/data/unpack_dynerf_src.py
"""
import os
import struct
import sys
import zipfile
import zlib

SRC = "data/_dynerf_src"
SCENES = ["coffee_martini", "cook_spinach", "cut_roasted_beef",
          "flame_steak", "sear_steak", "flame_salmon_1"]


def join_split():
    parts = [os.path.join(SRC, f"flame_salmon_1_split.{e}") for e in ("z01", "z02", "z03", "zip")]
    if not all(os.path.isfile(p) for p in parts):
        print("[fail] flame_salmon_1 split archive incomplete", file=sys.stderr)
        return None
    out = os.path.join(SRC, "flame_salmon_1.zip")
    if not os.path.isfile(out):
        with open(out, "wb") as w:
            for p in parts:
                with open(p, "rb") as r:
                    while True:
                        chunk = r.read(1 << 24)
                        if not chunk:
                            break
                        w.write(chunk)
    return out


def extract_split(path, dest, part_size):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(size - min(size, 1 << 20))
        tail = f.read()
        k = tail.rfind(b"PK\x06\x06")
        if k < 0:
            print("[split] zip64 end-of-central-directory record not found", file=sys.stderr); return False
        z64 = size - len(tail) + k
        total = struct.unpack("<Q", tail[k + 32:k + 40])[0]
        cd_size = struct.unpack("<Q", tail[k + 40:k + 48])[0]
        cd_at = z64 - cd_size
        f.seek(cd_at)
        cd = f.read(cd_size)
        if cd[:4] != b"PK\x01\x02":
            print(f"[split] central directory not at {cd_at}", file=sys.stderr); return False

        n, p = 0, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method = struct.unpack("<H", cd[p + 10:p + 12])[0]
            csz = struct.unpack("<I", cd[p + 20:p + 24])[0]
            nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
            disk = struct.unpack("<H", cd[p + 34:p + 36])[0]
            rel = struct.unpack("<I", cd[p + 42:p + 46])[0] + disk * part_size
            name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
            p += 46 + nl + el + cl
            out = os.path.join(dest, name)
            if name.endswith("/"):
                os.makedirs(out, exist_ok=True); continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            f.seek(rel)
            if f.read(4) != b"PK\x03\x04":
                print(f"[split] {name}: no local header at {rel} (disk={disk})", file=sys.stderr)
                return False
            hdr = f.read(26)
            lnl, lel = struct.unpack("<HH", hdr[22:26])
            f.seek(rel + 30 + lnl + lel)
            d = zlib.decompressobj(-15) if method == 8 else None
            left = csz
            with open(out, "wb") as w:
                while left > 0:
                    chunk = f.read(min(1 << 22, left))
                    if not chunk:
                        break
                    left -= len(chunk)
                    w.write(d.decompress(chunk) if d else chunk)
                if d:
                    w.write(d.flush())
            n += 1
            print(f"  [split] {name} ({os.path.getsize(out)/1e6:.0f} MB)", flush=True)
        print(f"[split] extracted {n}/{total} entries")
        return n > 1


def unpack(scene):
    dst = os.path.join(SRC, scene)
    z = os.path.join(SRC, f"{scene}.zip")
    if os.path.isdir(dst) and len([f for f in os.listdir(dst) if f.endswith(".mp4")]) > 1:
        print(f"[skip] {scene} already extracted")
        return True
    if scene == "flame_salmon_1" and not os.path.isfile(z):
        z = join_split()
        if z is None:
            return False
    if not os.path.isfile(z):
        print(f"[fail] {z} missing", file=sys.stderr)
        return False
    if scene == "flame_salmon_1":
        part = os.path.getsize(os.path.join(SRC, "flame_salmon_1_split.z01"))
        if not extract_split(z, SRC, part):
            return False
    else:
        with zipfile.ZipFile(z) as f:
            f.extractall(SRC)
    if not os.path.isdir(dst):
        os.makedirs(dst, exist_ok=True)
    n = len([f for f in os.listdir(dst) if f.endswith(".mp4")]) if os.path.isdir(dst) else 0
    print(f"[unpack] {scene}: {n} mp4 files")
    return n > 1


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    bad = [s for s in SCENES if not unpack(s)]
    if bad:
        print(f"failed: {bad}", file=sys.stderr)
    sys.exit(1 if bad else 0)
