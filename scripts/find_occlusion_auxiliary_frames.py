#!/usr/bin/env python3
"""Find auxiliary frames for existing object_move_occlusion questions.

Usage:
    python scripts/find_occlusion_auxiliary_frames.py \
        --benchmark output/benchmark_subset.json \
        --scannet_image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --scannetpp_image_root /home/sujinyue/mybenchmark/output/scannetpp_iphone_frames \
        --scannetpp_sensor iphone \
        --output output/occlusion_with_auxiliary.json

For ScanNet++ the raw scene data (mesh, annotations) is expected at
--scannetpp_data_root (defaults to --scannetpp_image_root if omitted).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qa_generator import find_auxiliary_frame_for_occlusion_question, quick_moved_bbox_projection
from src.scene_parser import load_instance_mesh_data, parse_scene
from src.utils import RayCaster
from src.utils.colmap_loader import load_axis_alignment, load_scannet_intrinsics, load_scannet_poses

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_SCANNET_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
MIN_ORIG_VISIBLE_IN_FRAME_RATIO = 0.3
MAX_CANDIDATE_POSES = 400


def _subsample_poses(poses: dict, max_n: int) -> dict:
    if len(poses) <= max_n:
        return poses
    items = list(poses.items())
    step = len(items) / max_n
    return dict(items[int(i * step)] for i in range(max_n))


def _is_scannetpp(scene_id: str) -> bool:
    return not _SCANNET_SCENE_RE.match(scene_id)


def _approx_visible_ids(scene_objects: list[dict], camera_pose, color_intrinsics) -> set[int]:
    return {
        int(o["id"])
        for o in scene_objects
        if quick_moved_bbox_projection(o, camera_pose, color_intrinsics)[1] >= MIN_ORIG_VISIBLE_IN_FRAME_RATIO
    }


def _load_scene_resources(scene_id: str, args: argparse.Namespace):
    """Load (poses, intrinsics, scene, ray_caster, instance_mesh_data) for a scene.
    Returns None on any failure.
    """
    if _is_scannetpp(scene_id):
        if not args.scannetpp_image_root:
            logger.warning("No --scannetpp_image_root provided, skipping ScanNet++ scene %s", scene_id)
            return None
        data_root = Path(args.scannetpp_data_root or args.scannetpp_image_root)
        frame_root = Path(args.scannetpp_image_root)
        scene_dir = data_root / scene_id
        if not scene_dir.exists():
            logger.warning("ScanNet++ scene dir not found: %s", scene_dir)
            return None
        try:
            from src.datasets import make_data_source
            ds = make_data_source("scannetpp", scene_dir, sensor=args.scannetpp_sensor,
                                  frame_root=frame_root / scene_id)
            axis_align = ds.load_axis_alignment()
            poses = ds.load_poses()
            color_intrinsics = ds.load_intrinsics()
            scene = ds.load_scene()
            mesh_path = ds.mesh_path()
        except Exception as e:
            logger.warning("Failed to load ScanNet++ scene %s: %s", scene_id, e)
            return None
    else:
        if not args.scannet_image_root:
            logger.warning("No --scannet_image_root provided, skipping ScanNet scene %s", scene_id)
            return None
        scene_dir = Path(args.scannet_image_root) / scene_id
        if not scene_dir.exists():
            logger.warning("ScanNet scene dir not found: %s", scene_dir)
            return None
        try:
            axis_align = load_axis_alignment(scene_dir)
            poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
            color_intrinsics = load_scannet_intrinsics(scene_dir)
            scene = parse_scene(scene_dir, dataset="scannet")
        except Exception as e:
            logger.warning("Failed to load ScanNet scene %s: %s", scene_id, e)
            return None
        mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
        if not mesh_path.exists():
            mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
        if not mesh_path.exists():
            logger.warning("Mesh not found for %s", scene_id)
            return None

    try:
        ray_caster = RayCaster.from_ply(str(mesh_path), axis_alignment=axis_align)
        instance_mesh_data = load_instance_mesh_data(
            scene_dir,
            instance_ids=[int(o["id"]) for o in scene["objects"]],
            n_surface_samples=512,
            **({} if not _is_scannetpp(scene_id) else {"dataset": "scannetpp"}),
        )
    except Exception as e:
        logger.warning("Failed to load mesh resources for %s: %s", scene_id, e)
        return None

    return poses, color_intrinsics, scene, ray_caster, instance_mesh_data


def _process_scene(scene_id: str, questions: list[dict], args: argparse.Namespace) -> None:
    resources = _load_scene_resources(scene_id, args)
    if resources is None:
        return
    full_poses, color_intrinsics, scene, ray_caster, instance_mesh_data = resources

    candidate_poses = _subsample_poses(full_poses, MAX_CANDIDATE_POSES)
    logger.info("%s: %d questions, %d poses (sampled %d)", scene_id, len(questions), len(full_poses), len(candidate_poses))

    objects_by_id = {int(o["id"]): o for o in scene["objects"]}

    # Precompute visible object ids per candidate pose (amortized across all questions in scene).
    frame_visible: dict[str, set[int]] = {
        img_name: {
            int(o["id"]) for o in scene["objects"]
            if quick_moved_bbox_projection(o, pose, color_intrinsics)[1] >= 0.3
        }
        for img_name, pose in candidate_poses.items()
    }

    for q in questions:
        q["auxiliary_image_names"] = None
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

        orig_visible = _approx_visible_ids(scene["objects"], orig_pose, color_intrinsics)

        # Pre-filter to frames that share at least one visible object (condition 2),
        # excluding the original frame. This avoids redundant raycasting in the inner loop.
        filtered_poses = {
            img_name: pose
            for img_name, pose in candidate_poses.items()
            if img_name != orig_pose.image_name and (orig_visible & frame_visible[img_name])
        }

        try:
            aux = find_auxiliary_frame_for_occlusion_question(
                moved_target=moved_target,
                gt_new_status=str(q.get("new_visibility", "")),
                occluder_id=None,
                orig_camera_pose=orig_pose,
                orig_visible_ids=orig_visible,
                all_poses=filtered_poses,
                scene_objects=scene["objects"],
                color_intrinsics=color_intrinsics,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            )
            q["auxiliary_image_names"] = aux
        except Exception as e:
            logger.warning("Error on %s/%s: %s", scene_id, q.get("image_name"), e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="output/benchmark_subset.json")
    ap.add_argument("--scannet_image_root", default=None)
    ap.add_argument("--scannetpp_image_root", default=None)
    ap.add_argument("--scannetpp_data_root", default=None,
                    help="ScanNet++ raw scene data root (mesh/anno). Defaults to --scannetpp_image_root.")
    ap.add_argument("--scannetpp_sensor", default="iphone")
    ap.add_argument("--output", default="output/occlusion_with_auxiliary.json")
    ap.add_argument("--scenes", nargs="*", help="Only process these scene_ids")
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = [q for q in benchmark["questions"] if q.get("type") == "object_move_occlusion"]
    logger.info("Found %d object_move_occlusion questions", len(questions))

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_scene[q["scene_id"]].append(q)

    scene_ids = args.scenes if args.scenes else sorted(by_scene)
    for scene_id in scene_ids:
        _process_scene(scene_id, by_scene[scene_id], args)

    found = sum(1 for q in questions if q.get("auxiliary_image_names"))
    logger.info("Done: %d/%d questions got an auxiliary frame", found, len(questions))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)
    logger.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
