#!/usr/bin/env python3
"""Extract iPhone frames from ScanNet++ ``iphone/rgb.mkv``.

Writes only those frames that appear in ``iphone/colmap/images.txt`` so
that every extracted JPEG has a corresponding camera pose.

Usage::

    python scripts/extract_scannetpp_iphone_frames.py \\
        --data_root ++data --scene 0d2ee665be
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _frame_index_from_name(image_name: str) -> int:
    """Extract the frame index from a name like ``frame_000010.jpg``."""
    stem = Path(image_name).stem
    return int(stem.split("_")[1])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract ScanNet++ iPhone frames from rgb.mkv"
    )
    p.add_argument("--data_root", default="/home/sujinyue/datasets/scannetpp")
    p.add_argument("--scene", default="0d2ee665be")
    p.add_argument("--output_root", default="output/scannetpp_iphone_frames")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from src.datasets.scannetpp import _parse_colmap_images

    scene_dir = Path(args.data_root) / args.scene
    video_path = scene_dir / "iphone" / "rgb.mkv"
    images_txt = scene_dir / "iphone" / "colmap" / "images.txt"

    if not video_path.is_file():
        print(f"ERROR: rgb.mkv not found: {video_path}")
        sys.exit(1)
    if not images_txt.is_file():
        print(f"ERROR: images.txt not found: {images_txt}")
        sys.exit(1)

    # 1. Get list of required frame names + indices
    colmap_images = _parse_colmap_images(images_txt)
    required: list[tuple[str, int]] = []
    for entry in colmap_images:
        name = entry["name"]
        idx = _frame_index_from_name(name)
        required.append((name, idx))

    print(f"Found {len(required)} frames in images.txt")

    # 2. Open video
    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_video_frames} total frames")

    # 3. Output directory
    output_dir = Path(args.output_root) / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Extract each required frame
    extracted = 0
    missing = 0
    skipped = 0

    for name, idx in required:
        out_path = output_dir / name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  MISSING: {name} (video frame {idx})")
            missing += 1
            continue

        _write_image_unicode_safe(out_path, frame)
        extracted += 1

    cap.release()

    # 5. Summary
    print(f"\nDone.")
    print(f"  requested: {len(required)}")
    print(f"  extracted: {extracted}")
    print(f"  skipped (exists): {skipped}")
    print(f"  missing:   {missing}")
    print(f"  output:    {output_dir}")


if __name__ == "__main__":
    main()
