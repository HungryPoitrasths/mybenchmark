#!/usr/bin/env python3
"""Pilot study script for CausalSpatial-Bench.

Processes a small set of scenes (default 30), generates questions, and
prints summary statistics to validate the pipeline before full-scale run.

Usage:
    python scripts/pilot_study.py --data_root data/scannet/scans --n_scenes 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import run_pipeline
from src.quality_control import compute_statistics, sample_for_human_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pilot_study")


def main():
    parser = argparse.ArgumentParser(description="CausalSpatial-Bench pilot study")
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
    )
    parser.add_argument("--output_dir", type=str, default="output/pilot")
    parser.add_argument("--n_scenes", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline on a small subset
    questions = run_pipeline(
        data_root=Path(args.data_root),
        output_dir=output_dir,
        max_scenes=args.n_scenes,
        max_frames=args.max_frames,
        use_ray_casting=True,
    )

    # Print detailed statistics
    stats = compute_statistics(questions)
    print("\n" + "=" * 60)
    print("PILOT STUDY RESULTS")
    print("=" * 60)
    print(f"Total questions generated: {stats['total']}")
    print(f"\nBy level: {json.dumps(stats['by_level'], indent=2)}")
    print(f"\nBy type: {json.dumps(stats['by_type'], indent=2)}")

    for level in ("L1", "L2", "L3"):
        key = f"{level}_answer_dist"
        if key in stats:
            print(f"\n{level} answer distribution: {stats[key]}")

    # Generate human validation sample
    sample = sample_for_human_validation(questions, n_per_level=50)
    sample_path = output_dir / "human_validation_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"\nHuman validation sample ({len(sample)} questions) saved to: {sample_path}")

    # Print expected thresholds
    print("\n" + "-" * 60)
    print("VALIDATION THRESHOLDS:")
    print("  L1 model accuracy should be > 65%")
    print("  L2 model accuracy should be < 45%")
    print("  L1 - L2 gap should be > 20%")
    print("-" * 60)


if __name__ == "__main__":
    main()
