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

from src.scene_parser import parse_scene
from src.support_graph import enrich_scene_with_support, has_nontrivial_support
from src.frame_selector import select_frames, refine_visible_ids_with_raycasting
from src.qa_generator import generate_all_questions
from src.quality_control import full_quality_pipeline, compute_statistics
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_poses,
)
from src.utils.ray_casting import RayCaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    max_scenes: int = 300,
    max_frames: int = 5,
    use_ray_casting: bool = True,
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

        # Load axis alignment and camera poses.  Must happen before ray-caster
        # init so the mesh can be transformed into the same coordinate frame.
        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)

        # Load ray caster once per scene — apply axis alignment so the mesh
        # lives in the same coordinate frame as object centres and poses.
        ray_caster = None
        if use_ray_casting:
            mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
            if not mesh_path.exists():
                mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
            if mesh_path.exists():
                try:
                    ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_align)
                    logger.info("Ray caster ready for %s", scene_id)
                except Exception as e:
                    logger.warning("Ray caster init failed for %s: %s", scene_id, e)

        # Refine visible_object_ids for each selected frame using ray casting.
        # This removes objects that project into the 2D frame but are physically
        # occluded by walls or other furniture (e.g. objects in adjacent rooms).
        if ray_caster is not None:
            for frame in frames:
                image_name = frame["image_name"]
                if image_name not in poses:
                    continue
                frame["visible_object_ids"] = refine_visible_ids_with_raycasting(
                    visible_object_ids=frame["visible_object_ids"],
                    objects=scene["objects"],
                    pose=poses[image_name],
                    ray_caster=ray_caster,
                )
            logger.info(
                "Scene %s: visible objects after ray-cast refinement: %s",
                scene_id,
                [len(f["visible_object_ids"]) for f in frames],
            )

        # ---- Stages 4-6: Relations + Virtual ops + QA ----
        for frame in frames:
            image_name = frame["image_name"]
            if image_name not in poses:
                continue
            camera_pose = poses[image_name]

            questions = generate_all_questions(
                objects=scene["objects"],
                support_graph=support_graph,
                supported_by=supported_by,
                camera_pose=camera_pose,
                ray_caster=ray_caster,
                visible_object_ids=frame["visible_object_ids"],
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
        "--no_ray_casting", action="store_true",
        help="Disable ray casting (faster but no occlusion questions)",
    )
    args = parser.parse_args()

    run_pipeline(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
        use_ray_casting=not args.no_ray_casting,
    )


if __name__ == "__main__":
    main()
