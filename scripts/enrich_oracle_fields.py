#!/usr/bin/env python3
"""Add _oracle_info fields to benchmark JSON for oracle VLM experiments.

The legacy oracle writes world-frame object centers.  The task-frame oracle
instead writes coordinates in the reference frame used by each question type:
agent/camera, allocentric/world, or object-centric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scene_parser import parse_scene
from src.utils.colmap_loader import CameraPose, load_axis_alignment, load_scannet_poses


AGENT_FRAME_TYPES = {
    "direction_agent",
    "object_move_agent",
    "attachment_move_agent",
    "coordinate_rotation_agent",
    "distance",
    "occlusion",
    "object_move_distance",
    "object_move_occlusion",
    "object_remove",
}

ALLOCENTRIC_FRAME_TYPES = {
    "direction_allocentric",
    "object_move_allocentric",
    "attachment_move_allocentric",
    "coordinate_rotation_allocentric",
    "attachment_chain",
}

OBJECT_CENTRIC_FRAME_TYPES = {
    "direction_object_centric",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "attachment_move_object_centric",
    "coordinate_rotation_object_centric",
}


@dataclass
class SceneCacheEntry:
    scene_path: Path
    objects: dict[int, dict[str, Any]]
    poses: dict[str, CameraPose] | None = None


def _scene_path(scene_id: str, dataset: str, scannet_root: str, scannetpp_root: str) -> Path:
    if scene_id.startswith("scene"):
        return Path(scannet_root) / scene_id

    root = Path(scannetpp_root)
    by_scene = root / scene_id
    if by_scene.exists():
        return by_scene

    # Backward compatibility with older subset metadata where _dataset held a
    # shard directory instead of a dataset name.
    by_dataset = root / dataset
    if by_dataset.exists():
        return by_dataset
    return by_scene


def _has_scannetpp_geometry(scene_path: Path) -> bool:
    return (
        (scene_path / "scans" / "mesh_aligned_0.05.ply").is_file()
        and (scene_path / "scans" / "segments.json").is_file()
        and (scene_path / "scans" / "segments_anno.json").is_file()
    )


def _has_scannetpp_pose_files(scene_path: Path, sensor: str) -> bool:
    if sensor == "iphone":
        return (scene_path / "iphone" / "colmap" / "images.txt").is_file()
    if sensor == "dslr":
        return (scene_path / "dslr" / "nerfstudio" / "transforms.json").is_file()
    return False


def _dataset_kind(scene_id: str, dataset: str) -> str:
    text = f"{scene_id} {dataset}".lower()
    if scene_id.startswith("scene") and "scannetpp" not in text:
        return "scannet"
    return "scannetpp"


def _load_poses(scene_path: Path, dataset_kind: str, scannetpp_sensor: str) -> dict[str, CameraPose]:
    if dataset_kind == "scannet":
        axis_align = load_axis_alignment(scene_path)
        return load_scannet_poses(scene_path, axis_alignment=axis_align)

    if scannetpp_sensor == "iphone":
        from src.datasets.scannetpp import load_scannetpp_iphone_poses

        return load_scannetpp_iphone_poses(scene_path)
    if scannetpp_sensor == "dslr":
        from src.datasets.scannetpp import compute_scannetpp_dslr_z_alignment, load_scannetpp_dslr_poses

        z_offset = compute_scannetpp_dslr_z_alignment(scene_path)
        return load_scannetpp_dslr_poses(scene_path, z_offset=z_offset)
    raise ValueError(f"Unsupported ScanNet++ sensor: {scannetpp_sensor!r}")


def _object_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mentioned_objects(question: dict[str, Any], scene_objects: dict[int, dict[str, Any]]) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for mention in question.get("mentioned_objects") or []:
        if not isinstance(mention, dict):
            continue
        obj_id = _object_id(mention.get("obj_id"))
        if obj_id is None or obj_id in seen or obj_id not in scene_objects:
            continue
        seen.add(obj_id)
        obj = scene_objects[obj_id]
        label = str(mention.get("label") or obj.get("label") or f"object {obj_id}")
        role = str(mention.get("role") or "object")
        rows.append((obj_id, label, role))
    return rows


def _center(obj: dict[str, Any]) -> np.ndarray:
    return np.asarray(obj["center"], dtype=np.float64)


def agent_oracle_coords(point_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    """Return semantic agent coords: +x forward, +y image-right, +z up."""
    cam = pose.world_to_camera_point(np.asarray(point_world, dtype=np.float64))
    return np.array([cam[2], cam[0], -cam[1]], dtype=np.float64)


def object_centric_coords(
    point_world: np.ndarray,
    ref_world: np.ndarray,
    face_world: np.ndarray,
) -> np.ndarray | None:
    """Return object-centric coords: +x forward, +y right, +z up."""
    ref = np.asarray(ref_world, dtype=np.float64)
    face = np.asarray(face_world, dtype=np.float64)
    point = np.asarray(point_world, dtype=np.float64)

    forward = np.array([face[0] - ref[0], face[1] - ref[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        return None
    forward /= norm
    right = np.array([forward[1], -forward[0], 0.0], dtype=np.float64)
    delta = point - ref
    return np.array(
        [
            float(np.dot(delta, forward)),
            float(np.dot(delta, right)),
            float(delta[2]),
        ],
        dtype=np.float64,
    )


def _fmt_vec(vec: np.ndarray, labels: tuple[str, str, str]) -> str:
    return (
        f"[{labels[0]}={float(vec[0]):.3f}, "
        f"{labels[1]}={float(vec[1]):.3f}, "
        f"{labels[2]}={float(vec[2]):.3f}]"
    )


def _line(label: str, role: str, vec: np.ndarray, labels: tuple[str, str, str]) -> str:
    return f"  {label} ({role}): {_fmt_vec(vec, labels)}"


def _make_world_oracle_prefix(question: dict[str, Any], scene_objects: dict[int, dict[str, Any]]) -> str | None:
    lines = ["[3D Oracle: object positions in world coordinates (meters)]"]
    for obj_id, label, role in _mentioned_objects(question, scene_objects):
        obj = scene_objects[obj_id]
        lines.append(_line(label, role, _center(obj), ("x", "y", "z")))
    return "\n".join(lines) if len(lines) > 1 else None


def _frame_kind_for_question(question: dict[str, Any]) -> str:
    question_type = str(question.get("type") or "")
    if question_type == "attachment_move":
        text = str(question.get("question") or "").lower()
        if question.get("camera_cardinal") or "cardinal direction" in text or "floor plan" in text:
            return "allocentric"
        if "imagine you are" in text or "from your perspective" in text:
            return "object_centric"
        return "agent"
    if question_type in OBJECT_CENTRIC_FRAME_TYPES:
        return "object_centric"
    if question_type in ALLOCENTRIC_FRAME_TYPES:
        return "allocentric"
    return "agent"


def _reference_ids_for_object_frame(question: dict[str, Any]) -> tuple[int | None, int | None]:
    qtype = str(question.get("type") or "")
    text = str(question.get("question") or "").lower()
    query_id = _object_id(question.get("query_obj_id"))

    if qtype in {"object_move_object_centric", "object_rotate_object_centric", "attachment_move"} and query_id is not None:
        ref_id = query_id
        face_id = _object_id(question.get("obj_face_id"))
        return ref_id, face_id

    ref_id = _object_id(question.get("obj_ref_id"))
    face_id = _object_id(question.get("obj_face_id"))

    if ("imagine you are" in text or "from your perspective" in text) and query_id is not None:
        ref_id = query_id
    return ref_id, face_id


def _camera_facing_point(ref_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    from src.utils.coordinate_transform import get_camera_forward

    # "Facing the camera" is the opposite of camera forward, projected to floor.
    toward_camera = -np.asarray(get_camera_forward(pose), dtype=np.float64)
    toward_camera[2] = 0.0
    norm = float(np.linalg.norm(toward_camera))
    if norm < 1e-8:
        toward_camera = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        toward_camera /= norm
    return np.asarray(ref_world, dtype=np.float64) + toward_camera


def _make_agent_oracle_prefix(
    question: dict[str, Any],
    scene_objects: dict[int, dict[str, Any]],
    pose: CameraPose | None,
) -> str | None:
    if pose is None:
        return None
    rows = _mentioned_objects(question, scene_objects)
    if not rows:
        return None
    lines = [
        "[3D Oracle: agent-frame object centers (meters)]",
        "Coordinate frame: origin = current camera; +x = forward away from camera; +y = image-right; +z = up.",
        "For horizontal direction questions, use x/y floor-plane components; use z for above/below.",
    ]
    for obj_id, label, role in rows:
        lines.append(_line(label, role, agent_oracle_coords(_center(scene_objects[obj_id]), pose), ("x", "y", "z")))
    return "\n".join(lines)


def _make_allocentric_oracle_prefix(
    question: dict[str, Any],
    scene_objects: dict[int, dict[str, Any]],
) -> str | None:
    rows = _mentioned_objects(question, scene_objects)
    if not rows:
        return None
    lines = [
        "[3D Oracle: allocentric/world object centers (meters)]",
        "Coordinate frame: origin = axis-aligned scene origin; +X = east; +Y = north; +Z = up.",
        "For horizontal cardinal directions, use X/Y floor-plane components; use Z for above/below.",
    ]
    camera_cardinal = question.get("camera_cardinal")
    if camera_cardinal:
        lines.append(f"Camera cardinal direction in this frame: {camera_cardinal}.")
    for obj_id, label, role in rows:
        lines.append(_line(label, role, _center(scene_objects[obj_id]), ("X", "Y", "Z")))
    return "\n".join(lines)


def _make_object_centric_oracle_prefix(
    question: dict[str, Any],
    scene_objects: dict[int, dict[str, Any]],
    pose: CameraPose | None,
) -> str | None:
    rows = _mentioned_objects(question, scene_objects)
    if not rows:
        return None

    ref_id, face_id = _reference_ids_for_object_frame(question)
    if ref_id is None or ref_id not in scene_objects:
        return None

    ref_center = np.asarray(
        question.get("facing_anchor_center") or _center(scene_objects[ref_id]),
        dtype=np.float64,
    )
    face_center: np.ndarray | None = None
    facing_label: str
    if question.get("facing_target_center") is not None:
        face_center = np.asarray(question["facing_target_center"], dtype=np.float64)
        facing_label = (
            str(scene_objects[face_id].get("label") or f"object {face_id}")
            if face_id is not None and face_id in scene_objects
            else "the facing target"
        )
    elif face_id is not None and face_id in scene_objects:
        face_center = _center(scene_objects[face_id])
        facing_label = str(scene_objects[face_id].get("label") or f"object {face_id}")
    elif pose is not None:
        face_center = _camera_facing_point(ref_center, pose)
        facing_label = "the camera"
    else:
        return None

    ref_label = str(scene_objects[ref_id].get("label") or f"object {ref_id}")
    lines = [
        "[3D Oracle: object-centric object centers (meters)]",
        f"Coordinate frame: origin = {ref_label}; +x = forward toward {facing_label}; +y = right; +z = up.",
        "For horizontal direction questions, use x/y floor-plane components; use z for above/below.",
    ]
    for obj_id, label, role in rows:
        coords = object_centric_coords(_center(scene_objects[obj_id]), ref_center, face_center)
        if coords is None:
            continue
        lines.append(_line(label, role, coords, ("x", "y", "z")))
    return "\n".join(lines) if len(lines) > 3 else None


def _make_task_frame_oracle_prefix(
    question: dict[str, Any],
    scene_objects: dict[int, dict[str, Any]],
    pose: CameraPose | None,
) -> str | None:
    frame = _frame_kind_for_question(question)
    if frame == "object_centric":
        return _make_object_centric_oracle_prefix(question, scene_objects, pose)
    if frame == "allocentric":
        return _make_allocentric_oracle_prefix(question, scene_objects)
    return _make_agent_oracle_prefix(question, scene_objects, pose)


def _load_scene_cache_entry(
    scene_id: str,
    dataset: str,
    *,
    scannet_root: str,
    scannetpp_root: str,
    scannetpp_sensor: str,
    load_poses_for_task_frame: bool,
) -> SceneCacheEntry:
    scene_path = _scene_path(scene_id, dataset, scannet_root, scannetpp_root)
    dataset_kind = _dataset_kind(scene_id, dataset)
    if dataset_kind == "scannetpp":
        if not _has_scannetpp_geometry(scene_path):
            raise FileNotFoundError(
                f"{scene_path} is not a raw ScanNet++ scene directory with scans/mesh_aligned_0.05.ply"
            )
        if load_poses_for_task_frame and not _has_scannetpp_pose_files(scene_path, scannetpp_sensor):
            raise FileNotFoundError(
                f"{scene_path} is missing {scannetpp_sensor} pose files required for task-frame oracle"
            )
    parsed = parse_scene(scene_path, dataset=dataset_kind)
    objects = {int(o["id"]): o for o in (parsed or {}).get("objects", [])}
    poses = _load_poses(scene_path, dataset_kind, scannetpp_sensor) if load_poses_for_task_frame else None
    return SceneCacheEntry(scene_path=scene_path, objects=objects, poses=poses)


def _questions(data: Any) -> list[dict[str, Any]]:
    questions = data if isinstance(data, list) else data.get("questions", data)
    if not isinstance(questions, list):
        raise ValueError("Unsupported benchmark structure: expected list or dict with questions")
    return [q for q in questions if isinstance(q, dict)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="output/benchmark_subset.json")
    ap.add_argument("--scannet_root", default="data/scannet/scans")
    ap.add_argument("--scannetpp_root", default="data/scannetpp")
    ap.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    ap.add_argument("--oracle_mode", choices=("world", "task_frame"), default="task_frame")
    ap.add_argument("--out", default="output/benchmark_subset.oracle.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    questions = _questions(data)

    scene_cache: dict[str, SceneCacheEntry] = {}
    skipped = 0
    pose_missing = 0
    for q in questions:
        scene_id = str(q.get("scene_id") or "")
        dataset = str(q.get("_dataset") or q.get("dataset") or "")
        if scene_id not in scene_cache:
            try:
                scene_cache[scene_id] = _load_scene_cache_entry(
                    scene_id,
                    dataset,
                    scannet_root=args.scannet_root,
                    scannetpp_root=args.scannetpp_root,
                    scannetpp_sensor=args.scannetpp_sensor,
                    load_poses_for_task_frame=args.oracle_mode == "task_frame",
                )
            except Exception as exc:
                print(f"  warn: {scene_id}: {exc}", file=sys.stderr)
                scene_cache[scene_id] = SceneCacheEntry(
                    scene_path=_scene_path(scene_id, dataset, args.scannet_root, args.scannetpp_root),
                    objects={},
                    poses={},
                )

        entry = scene_cache[scene_id]
        pose = entry.poses.get(str(q.get("image_name") or "")) if entry.poses is not None else None
        if args.oracle_mode == "world":
            prefix = _make_world_oracle_prefix(q, entry.objects)
        else:
            if _frame_kind_for_question(q) in {"agent", "object_centric"} and pose is None:
                pose_missing += 1
            prefix = _make_task_frame_oracle_prefix(q, entry.objects, pose)

        if prefix:
            q["_oracle_info"] = prefix
            q["_oracle_mode"] = args.oracle_mode
        else:
            skipped += 1

    if isinstance(data, dict):
        metadata = data.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["oracle_mode"] = args.oracle_mode
            metadata["oracle_generated_by"] = "scripts/enrich_oracle_fields.py"
            metadata["oracle_skipped_count"] = skipped
            metadata["oracle_pose_missing_count"] = pose_missing

    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Written {args.out}  "
        f"(mode={args.oracle_mode}, skipped={skipped}, pose_missing={pose_missing})"
    )


if __name__ == "__main__":
    main()
