#!/usr/bin/env python3
"""One-click pipeline runner for CausalSpatial-Bench.

Usage:
    python scripts/run_pipeline.py --data_root data/scannet/scans \\
                                   --output_dir output \\
                                   --max_scenes 300 \\
                                   --max_frames 5
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

from src.scene_parser import parse_scene, load_scannet_label_map
from src.support_graph import enrich_scene_with_support, has_nontrivial_support
from src.frame_selector import (
    select_frames,
    compute_frame_object_visibility,
)
from src.qa_generator import generate_all_questions
from src.quality_control import full_quality_pipeline, compute_statistics
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from src.utils.depth_occlusion import load_depth_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def _load_referability_cache(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("Referability cache not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded referability cache from %s", path)
    return data


def _get_referability_entry(cache: dict | None, scene_id: str, image_name: str) -> dict | None:
    if not cache:
        return None
    frames = cache.get("frames", cache)
    scene_frames = frames.get(scene_id)
    if isinstance(scene_frames, dict):
        return scene_frames.get(image_name)
    return frames.get(f"{scene_id}/{image_name}")


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    max_scenes: int = 300,
    max_frames: int = 5,
    use_occlusion: bool = True,
    strict_mode: bool = False,
    referability_cache: dict | None = None,
):
    """Execute the full CausalSpatial-Bench data generation pipeline."""

    meta_dir      = output_dir / "scene_metadata"
    questions_dir = output_dir / "questions"
    meta_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)

    # Discover scene directories (ScanNet scenes have a pose/ subdir)
    scene_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and (p / "pose").exists()
    )
    logger.info("Found %d candidate scenes", len(scene_dirs))

    all_questions: list[dict] = []
    processed = 0

    for scene_dir in scene_dirs:
        if processed >= max_scenes:
            break

        scene_id = scene_dir.name
        logger.info(
            "=== Processing scene %s (%d/%d) ===",
            scene_id, processed + 1, max_scenes,
        )

        # ---- Stage 1: Parse ----
        scene = parse_scene(scene_dir)
        if scene is None:
            continue

        # ---- Stage 2: Support graph ----
        enrich_scene_with_support(scene)
        support_graph = {int(k): v for k, v in scene["support_graph"].items()}
        supported_by  = {int(k): v for k, v in scene["supported_by"].items()}

        if not has_nontrivial_support(support_graph):
            logger.info("Scene %s has no support relations — skipping", scene_id)
            continue

        with open(meta_dir / f"{scene_id}.json", "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)

        # ---- Stage 3: Frame selection ----
        frames = select_frames(scene_dir, scene["objects"], support_graph, max_frames)
        if not frames:
            logger.info("No valid frames for scene %s — skipping", scene_id)
            continue

        # Load camera poses (with axis alignment so coords match the mesh)
        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)

        # Load depth intrinsics once per scene (shared across all frames)
        depth_intrinsics = None
        if use_occlusion:
            try:
                depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
            except Exception as e:
                logger.warning("Depth intrinsics load failed for %s: %s", scene_id, e)

        # Load colour intrinsics for local ROI blur check
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception as e:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, e)
            color_intrinsics = None

        # ---- Stages 4-6: Relations + Virtual ops + QA ----
        for frame in frames:
            image_name = frame["image_name"]
            if image_name not in poses:
                continue
            camera_pose = poses[image_name]

            # Load depth map for this frame
            depth_image = None
            if use_occlusion and depth_intrinsics is not None:
                frame_id = image_name.replace(".jpg", "")
                depth_path = scene_dir / "depth" / f"{frame_id}.png"
                if depth_path.exists():
                    try:
                        depth_image = load_depth_image(depth_path)
                    except Exception as e:
                        logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, e)

            # In normal mode, keep projection-based visible IDs unchanged.
            # Depth is only used downstream for occlusion-question generation.
            visible_ids = frame["visible_object_ids"]
            visibility_table = None
            if strict_mode:
                image_path = scene_dir / "color" / image_name
                if depth_image is None or depth_intrinsics is None:
                    logger.debug(
                        "Frame %s/%s: strict mode requires depth; skipping",
                        scene_id, image_name,
                    )
                    continue
                if color_intrinsics is None or not image_path.exists():
                    logger.debug(
                        "Frame %s/%s: strict mode requires readable color image; skipping",
                        scene_id, image_name,
                    )
                    continue
                visibility_table = compute_frame_object_visibility(
                    scene["objects"],
                    camera_pose,
                    color_intrinsics,
                    image_path=image_path,
                    depth_image=depth_image,
                    depth_intrinsics=depth_intrinsics,
                    strict_mode=True,
                )
                visible_ids = [
                    obj_id for obj_id, meta in visibility_table.items()
                    if meta.get("eligible_as_reference", False)
                ]
                if len(visible_ids) < 3:
                    logger.debug(
                        "Frame %s/%s: only %d strict-eligible objects; skipping",
                        scene_id, image_name, len(visible_ids),
                    )
                    continue

            referable_ids = None
            referability_entry = _get_referability_entry(
                referability_cache, scene_id, image_name,
            )
            if referability_entry is not None:
                if not referability_entry.get("frame_usable", True):
                    logger.debug(
                        "Frame %s/%s rejected by referability cache: %s",
                        scene_id, image_name,
                        referability_entry.get("frame_reject_reason", "frame_not_usable"),
                    )
                    continue
                referable_ids = [
                    int(obj_id) for obj_id in referability_entry.get("referable_object_ids", [])
                    if int(obj_id) in visible_ids
                ]
                if not referable_ids:
                    logger.debug(
                        "Frame %s/%s has no referable objects after visibility intersection",
                        scene_id, image_name,
                    )
                    continue

            questions = generate_all_questions(
                objects=scene["objects"],
                support_graph=support_graph,
                supported_by=supported_by,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                visible_object_ids=visible_ids,
                referable_object_ids=referable_ids,
                object_visibility=visibility_table,
                strict_mode=strict_mode,
                room_bounds=scene.get("room_bounds"),
                wall_objects=scene.get("wall_objects"),
            )

            for q in questions:
                q["scene_id"]   = scene_id
                q["image_name"] = image_name

            all_questions.extend(questions)

        processed += 1
        logger.info(
            "Scene %s: %d questions accumulated", scene_id, len(all_questions),
        )

    # ---- Stage 7: Quality control ----
    logger.info("Running quality control on %d raw questions…", len(all_questions))
    final_questions = full_quality_pipeline(all_questions)

    from collections import defaultdict
    by_scene: dict[str, list] = defaultdict(list)
    for q in final_questions:
        by_scene[q["scene_id"]].append(q)

    for sid, qs in by_scene.items():
        with open(questions_dir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(qs, f, indent=2, ensure_ascii=False)

    benchmark = {
        "name":       "CausalSpatial-Bench",
        "version":    "1.0",
        "statistics": compute_statistics(final_questions),
        "questions":  final_questions,
    }
    benchmark_path = output_dir / "benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    logger.info(
        "Pipeline complete! %d questions saved to %s",
        len(final_questions), benchmark_path,
    )
    return final_questions


def main():
    parser = argparse.ArgumentParser(
        description="CausalSpatial-Bench data generation pipeline"
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="Root directory of ScanNet scans (contains scene subdirectories)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=300,
        help="Maximum number of scenes to process",
    )
    parser.add_argument(
        "--max_frames", type=int, default=5,
        help="Maximum frames per scene",
    )
    parser.add_argument(
        "--no_occlusion", action="store_true",
        help="Disable depth-map occlusion (faster but no occlusion questions)",
    )
    parser.add_argument(
        "--strict_mode", action="store_true",
        help="Require every mentioned object to pass strict per-frame visibility checks",
    )
    parser.add_argument(
        "--referability_cache", type=str, default=None,
        help="Optional JSON cache of VLM frame/object referability decisions",
    )
    parser.add_argument(
        "--label_map", type=str, default=None,
        help="Path to scannetv2-labels.combined.tsv for raw_category→nyu40class normalization",
    )
    args = parser.parse_args()

    if args.label_map:
        load_scannet_label_map(args.label_map)

    referability_cache = None
    if args.referability_cache:
        referability_cache = _load_referability_cache(Path(args.referability_cache))

    run_pipeline(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
        use_occlusion=not args.no_occlusion,
        strict_mode=args.strict_mode,
        referability_cache=referability_cache,
    )


if __name__ == "__main__":
    main()
