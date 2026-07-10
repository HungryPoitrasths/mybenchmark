#!/usr/bin/env python3
"""Validation stats for the v2 route-continuity occlusion auxiliary-frame algorithm.

NOT part of the production pipeline. Run against a handful of scenes to check
find rate, chain length distribution, empty-original-coverage rate, failure
reason breakdown, and timing before wiring
find_auxiliary_frames_for_occlusion_question_v2 into
scripts/find_occlusion_auxiliary_frames.py / scripts/run_pipeline.py (which is
already done — this script is for tuning threshold_angle_deg / min_overlap_frac
/ max_backtrack / max_primary_candidates and spot-checking behavior).

Usage:
    python scripts/aux_route_coverage_stats.py \
        --benchmark output/benchmark_subset.json \
        --scannet_image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --scannetpp_image_root /home/sujinyue/mybenchmark/output/scannetpp_iphone_frames \
        --max_scenes 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.find_occlusion_auxiliary_frames import _obj_center, load_scene_resources
from src.qa_generator import (
    MIN_PROJECTED_AREA_PX,
    _route_runs_for_pose,
    _route_sample_points,
    find_auxiliary_frames_for_occlusion_question_v2,
    quick_moved_bbox_projection,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _diagnose(
    *,
    original_center: np.ndarray,
    moved_center: np.ndarray,
    moved_target: dict,
    orig_camera_pose,
    all_poses: dict,
    color_intrinsics,
) -> tuple[bool, int]:
    """Cheaply re-derive (orig_frame_covers_route_start, primary_candidate_count) for diagnostics.

    Duplicates a slice of find_auxiliary_frames_for_occlusion_question_v2's internal
    logic on purpose — this is a read-only diagnostic, not the selection path itself.
    """
    ts, route_points = _route_sample_points(
        np.asarray(original_center, dtype=np.float64), np.asarray(moved_center, dtype=np.float64)
    )
    orig_runs = _route_runs_for_pose(route_points, ts, orig_camera_pose, color_intrinsics)
    orig_covers_start = bool(orig_runs) and orig_runs[0][0] <= 1e-6

    primary_count = 0
    for name, pose in all_poses.items():
        if name == orig_camera_pose.image_name:
            continue
        area, in_frame_ratio = quick_moved_bbox_projection(moved_target, pose, color_intrinsics)
        if in_frame_ratio < 1.0 or area < MIN_PROJECTED_AREA_PX:
            continue
        runs = _route_runs_for_pose(route_points, ts, pose, color_intrinsics)
        if runs and runs[-1][1] >= 1.0 - 1e-6:
            primary_count += 1

    return orig_covers_start, primary_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="output/benchmark_subset.json")
    ap.add_argument("--scannet_image_root", default=None)
    ap.add_argument("--scannetpp_image_root", default=None)
    ap.add_argument("--scannetpp_data_root", default=None)
    ap.add_argument("--scannetpp_sensor", default="iphone")
    ap.add_argument("--scenes", nargs="*", help="Scene ids to sample; default = first --max_scenes scenes with questions")
    ap.add_argument("--max_scenes", type=int, default=8)
    ap.add_argument("--threshold_angle_deg", type=float, default=60.0)
    ap.add_argument("--min_overlap_frac", type=float, default=0.15)
    ap.add_argument("--max_backtrack", type=int, default=20)
    ap.add_argument("--max_primary_candidates", type=int, default=8)
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = [q for q in benchmark["questions"] if q.get("type") == "object_move_occlusion"]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_scene[q["scene_id"]].append(q)

    scene_ids = args.scenes if args.scenes else sorted(by_scene)[: args.max_scenes]

    n_total = 0
    n_found = 0
    n_no_primary_candidate = 0
    n_primary_no_chain = 0
    n_orig_coverage_empty = 0
    chain_lengths: Counter = Counter()
    timings: list[float] = []

    for scene_id in scene_ids:
        scene_questions = by_scene.get(scene_id, [])
        if not scene_questions:
            logger.warning("No object_move_occlusion questions for scene %s, skipping", scene_id)
            continue

        resources = load_scene_resources(scene_id, args)
        if resources is None:
            logger.warning("Skipping %s: could not load scene resources", scene_id)
            continue
        full_poses, color_intrinsics, scene, _ray_caster, _instance_mesh_data = resources
        objects_by_id = {int(o["id"]): o for o in scene["objects"]}

        for q in scene_questions:
            orig_pose = full_poses.get(q.get("image_name", ""))
            if orig_pose is None:
                continue
            orig_obj = objects_by_id.get(int(q["target_obj_id"]))
            if orig_obj is None:
                continue

            delta = np.asarray(q["delta"], dtype=np.float64)
            moved_target = dict(orig_obj)
            moved_target["bbox_min"] = (np.asarray(orig_obj["bbox_min"]) + delta).tolist()
            moved_target["bbox_max"] = (np.asarray(orig_obj["bbox_max"]) + delta).tolist()
            if "center" in orig_obj:
                moved_target["center"] = (np.asarray(orig_obj["center"]) + delta).tolist()

            original_center = _obj_center(orig_obj)
            moved_center = _obj_center(moved_target)

            n_total += 1
            t0 = time.perf_counter()
            aux = find_auxiliary_frames_for_occlusion_question_v2(
                original_center=original_center,
                moved_center=moved_center,
                moved_target=moved_target,
                orig_camera_pose=orig_pose,
                all_poses=full_poses,
                color_intrinsics=color_intrinsics,
                orientation_threshold_deg=args.threshold_angle_deg,
                min_overlap_frac=args.min_overlap_frac,
                max_backtrack=args.max_backtrack,
                max_primary_candidates=args.max_primary_candidates,
            )
            timings.append(time.perf_counter() - t0)

            orig_covers_start, primary_count = _diagnose(
                original_center=original_center,
                moved_center=moved_center,
                moved_target=moved_target,
                orig_camera_pose=orig_pose,
                all_poses=full_poses,
                color_intrinsics=color_intrinsics,
            )
            if not orig_covers_start:
                n_orig_coverage_empty += 1

            if aux:
                n_found += 1
                chain_lengths[len(aux)] += 1
            elif primary_count == 0:
                n_no_primary_candidate += 1
            else:
                n_primary_no_chain += 1

    logger.info("=== aux_route_coverage_stats ===")
    logger.info("scenes sampled: %s", scene_ids)
    logger.info("total questions: %d", n_total)
    if n_total:
        logger.info("found auxiliary chain: %d (%.1f%%)", n_found, 100 * n_found / n_total)
        logger.info(
            "  failure - no primary candidate: %d (%.1f%%)",
            n_no_primary_candidate, 100 * n_no_primary_candidate / n_total,
        )
        logger.info(
            "  failure - primary found, chain bridging failed: %d (%.1f%%)",
            n_primary_no_chain, 100 * n_primary_no_chain / n_total,
        )
        logger.info(
            "original frame's own route coverage empty at route start: %d (%.1f%%)",
            n_orig_coverage_empty, 100 * n_orig_coverage_empty / n_total,
        )
        logger.info("chain length distribution (1 = primary only, no secondaries): %s", dict(sorted(chain_lengths.items())))
        if timings:
            arr = np.asarray(timings)
            logger.info(
                "per-question timing: mean=%.1fms p50=%.1fms p95=%.1fms max=%.1fms",
                1000 * arr.mean(), 1000 * np.median(arr), 1000 * np.percentile(arr, 95), 1000 * arr.max(),
            )
    else:
        logger.info("No questions processed — check --scenes / dataset roots.")


if __name__ == "__main__":
    main()
