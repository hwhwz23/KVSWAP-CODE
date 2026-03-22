#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

DEFAULT_VIDEO_ROOT = (
    "./bench/MLVU/MVLU_DATA/MLVU/video"
)


def parse_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    line = line.strip('"').strip("'").strip()
    return line if line else None


def find_file(root: str, basename: str) -> str | None:
    direct = os.path.join(root, basename)
    if os.path.isfile(direct):
        return direct
    for dp, _, fns in os.walk(root):
        if basename in fns:
            return os.path.join(dp, basename)
    return None


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    x = float(n)
    units = ("KiB", "MiB", "GiB", "TiB", "PiB")
    i = 0
    x /= 1024.0
    while i < len(units) - 1 and x >= 1024:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def main() -> None:
    root = os.environ.get("MLVU_VIDEO_ROOT", DEFAULT_VIDEO_ROOT)
    if not os.path.isdir(root):
        print(f"Error: video root not found: {root}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    names: list[str] = []
    for line in lines:
        n = parse_line(line)
        if n:
            names.append(n)

    total = 0
    missing: list[str] = []
    for name in names:
        path = find_file(root, name)
        if path is None:
            missing.append(name)
            continue
        total += os.path.getsize(path)

    print(f"Video root: {root}")
    print(f"Files listed: {len(names)}")
    print(f"Found:       {len(names) - len(missing)}")
    print(f"Missing:     {len(missing)}")
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  {m}")
    print(f"Total size:  {total} bytes ({human(total)})")


if __name__ == "__main__":
    main()

# python ./scripts/sum_mlvu_video_sizes.py ./scripts/mlvu_subset.txt
# 
# Video root: ./bench/MLVU/MVLU_DATA/MLVU/video
# Files listed: 100
# Found:       100
# Missing:     0
# Total size:  26989349406 bytes (25.14 GiB)

