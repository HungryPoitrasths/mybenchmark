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

Auxiliary frames are picked via route-continuity: sample the straight-line
path the moved object travels (original position -> moved position) and
chain together frames whose orientation stays roughly parallel and whose
in-frame coverage of that path overlaps enough to read as "the same route",
ending in a frame that fully frames the moved target. See
find_auxiliary_frames_for_occlusion_question_v2 in src/qa_generator.py.
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

from src.qa_generator import find_auxiliary_frames_for_occlusion_question_v2
from src.scene_parser import load_instance_mesh_data, parse_scene
from src.utils import RayCaster
from src.utils.colmap_loader import load_axis_alignment, load_scannet_intrinsics, load_scannet_poses

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_SCANNET_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")


def _is_scannetpp(scene_id: str) -> bool:
    return not _SCANNET_SCENE_RE.match(scene_id)


def _obj_center(obj: dict) -> np.ndarray:
    if "center" in obj:
        return np.asarray(obj["center"], dtype=np.float64)
    bbox_min = np.asarray(obj["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(obj["bbox_max"], dtype=np.float64)
    return (bbox_min + bbox_max) / 2.0


def load_scene_resources(scene_id: str, args: argparse.Namespace):
    """Load (poses, intrinsics, scene, ray_caster, instance_mesh_data) for a scene.
    Returns None on any failure. Shared with scripts/aux_route_coverage_stats.py.
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
    resources = load_scene_resources(scene_id, args)
    if resources is None:
        return
    full_poses, color_intrinsics, scene, _ray_caster, _instance_mesh_data = resources

    logger.info("%s: %d questions, %d poses", scene_id, len(questions), len(full_poses))

    objects_by_id = {int(o["id"]): o for o in scene["objects"]}

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

        original_center = _obj_center(orig_obj)
        moved_center = _obj_center(moved_target)

        try:
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
            q["auxiliary_image_names"] = aux or None
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
    ap.add_argument("--threshold_angle_deg", type=float, default=60.0,
                    help="Max forward-direction angle (degrees) between chain-adjacent frames.")
    ap.add_argument("--min_overlap_frac", type=float, default=0.15,
                    help="Min route-coverage overlap (fraction of route length in [0,1]) between chain-adjacent frames.")
    ap.add_argument("--max_backtrack", type=int, default=20,
                    help="Max alternate-candidate retries in the chain search before giving up.")
    ap.add_argument("--max_primary_candidates", type=int, default=8,
                    help="Max primary-frame candidates to try chain-bridging for, ordered by smallest bridge gap first.")
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
