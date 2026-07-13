#!/usr/bin/env python3
"""Check which ScanNet++ scenes in a scene-list are missing extracted iPhone frames.

For each scene id in --scene_list, compares the frame names required by
``iphone/colmap/images.txt`` against what actually exists under
--frame_root/<scene_id>/, and reports scenes that are fully missing,
partially missing, or fully extracted.

Usage::

    python scripts/check_scannetpp_iphone_extraction_coverage.py \\
        --data_root /data/zju-151/scannet/data \\
        --frame_root ~/datasets/scannetpp/train/iphone_frames \\
        --scene_list ~/datasets/scannetpp/train/selected_train_100.txt \\
        --missing_output ~/datasets/scannetpp/train/scenes_needing_extraction.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True,
                    help="Root dir containing one subdirectory per ScanNet++ scene id (has iphone/colmap/images.txt)")
    p.add_argument("--frame_root", type=Path, required=True,
                    help="Root dir where extracted frames live: <frame_root>/<scene_id>/<image_name>")
    p.add_argument("--scene_list", type=Path, required=True,
                    help="One scene id per line, e.g. selected_train_100.txt")
    p.add_argument("--missing_output", type=Path, default=None,
                    help="Optional: write scene ids that need (re-)extraction, one per line")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from src.datasets.scannetpp import _parse_colmap_images

    scene_ids = [
        line.strip()
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(scene_ids)} scene ids from {args.scene_list}")

    fully_missing: list[str] = []
    partially_missing: list[tuple[str, int, int]] = []
    complete = 0
    errored: list[tuple[str, str]] = []

    for scene_id in scene_ids:
        images_txt = args.data_root / scene_id / "iphone" / "colmap" / "images.txt"
        if not images_txt.is_file():
            errored.append((scene_id, f"missing {images_txt}"))
            continue

        try:
            required = _parse_colmap_images(images_txt)
        except Exception as exc:  # noqa: BLE001
            errored.append((scene_id, f"{type(exc).__name__}: {exc}"))
            continue

        total = len(required)
        if total == 0:
            errored.append((scene_id, "images.txt has 0 entries"))
            continue

        scene_frame_dir = args.frame_root / scene_id
        found = 0
        for entry in required:
            if (scene_frame_dir / entry["name"]).is_file():
                found += 1

        if found == 0:
            fully_missing.append(scene_id)
        elif found < total:
            partially_missing.append((scene_id, found, total))
        else:
            complete += 1

    print()
    print(f"Complete:          {complete}/{len(scene_ids)}")
    print(f"Fully missing:     {len(fully_missing)}")
    print(f"Partially missing: {len(partially_missing)}")
    print(f"Errored/unreadable:{len(errored)}")

    if fully_missing:
        print("\n--- Fully missing (0 frames extracted) ---")
        for s in fully_missing:
            print(s)

    if partially_missing:
        print("\n--- Partially missing ---")
        for s, found, total in partially_missing:
            print(f"{s}: {found}/{total}")

    if errored:
        print("\n--- Errored ---")
        for s, msg in errored:
            print(f"{s}: {msg}")

    if args.missing_output:
        needs_extraction = fully_missing + [s for s, _, _ in partially_missing]
        args.missing_output.parent.mkdir(parents=True, exist_ok=True)
        args.missing_output.write_text("\n".join(needs_extraction) + ("\n" if needs_extraction else ""), encoding="utf-8")
        print(f"\nWrote {len(needs_extraction)} scene ids needing (re-)extraction to {args.missing_output}")


if __name__ == "__main__":
    main()
