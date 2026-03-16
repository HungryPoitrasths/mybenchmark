"""Stage 3: Representative frame selection.

For each scene, select 3-5 frames that collectively cover as many objects
(and support relationships) as possible while maintaining viewpoint diversity.

ScanNet stores one colour image per frame in ``color/<frame_id>.jpg`` and the
corresponding camera pose in ``pose/<frame_id>.txt``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .utils.colmap_loader import (
    CameraIntrinsics,
    CameraPose,
    load_axis_alignment,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from .utils.coordinate_transform import is_in_image, project_to_image

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAMES = 5


def refine_visible_ids_with_raycasting(
    visible_object_ids: list[int],
    objects: list[dict],
    pose: CameraPose,
    ray_caster,
) -> list[int]:
    """Remove fully-occluded objects from *visible_object_ids*.

    Called AFTER projection-based frame selection so that ray casting only
    runs on the 3-5 selected frames rather than every frame in the scene.

    An object is dropped when ``multi_ray_occlusion`` (8 sample rays toward
    the bbox) classifies it as ``"fully_occluded"``.  Partially-occluded
    objects are kept — they are still meaningful as question subjects.
    """
    obj_map = {o["id"]: o for o in objects}
    refined: list[int] = []
    for obj_id in visible_object_ids:
        obj = obj_map.get(obj_id)
        if obj is None:
            continue
        status = ray_caster.multi_ray_occlusion(
            camera_pos=pose.position,
            target_bbox_min=np.array(obj["bbox_min"]),
            target_bbox_max=np.array(obj["bbox_max"]),
        )
        if status != "fully_occluded":
            refined.append(obj_id)
    return refined


MIN_VISIBLE_OBJECTS = 3
VIEWPOINT_DIVERSITY_MIN_ANGLE = 20  # degrees


def get_visible_objects(
    objects: list[dict],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    margin: int = 80,
    min_depth: float = 0.3,
    max_depth: float = 6.0,
) -> list[dict]:
    """Return objects whose centre projects into the image frame.

    margin:    pixels from image edge (larger = more conservative)
    min_depth: minimum distance from camera in metres (filters objects
               clipped right against the camera plane)
    max_depth: maximum distance from camera in metres. Objects beyond this
               threshold are filtered out even if their centre projects into
               the frame — this prevents objects in adjacent rooms (visible
               through a doorway as a tiny projection) from appearing in
               questions. Typical ScanNet indoor scenes are ≤8 m across;
               6 m is conservative enough to keep far-wall objects while
               excluding the next room.
    """
    visible = []
    for obj in objects:
        center = np.array(obj["center"])
        uv, depth = project_to_image(center, pose, intrinsics)
        if min_depth < depth <= max_depth and is_in_image(uv, intrinsics, margin=margin):
            visible.append(obj)
    return visible


def _angular_distance(pose_a: CameraPose, pose_b: CameraPose) -> float:
    """Approximate angular difference between two viewing directions (degrees)."""
    fwd_a = pose_a.rotation.T[:, 2]
    fwd_b = pose_b.rotation.T[:, 2]
    cos_angle = np.clip(np.dot(fwd_a, fwd_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _count_support_objects(visible: list[dict], support_ids: set[int]) -> int:
    return sum(1 for o in visible if o["id"] in support_ids)


def select_frames(
    scene_path: str | Path,
    objects: list[dict],
    support_graph: dict[int, list[int]] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[dict[str, Any]]:
    """Select representative frames for a ScanNet scene.

    Algorithm:
        1. Score each frame = #visible_objects × (1 + #support_objects).
        2. Greedy selection: pick the highest-scoring frame, then iteratively
           pick the next that is at least VIEWPOINT_DIVERSITY_MIN_ANGLE away
           from all already-selected frames.

    Returns:
        List of dicts: ``{image_name, camera_position, visible_object_ids,
        n_visible, score}``.
    """
    scene_path = Path(scene_path)

    # Support both intrinsic/intrinsic_color.txt and intrinsic_color.txt (at root)
    intr_path = scene_path / "intrinsic" / "intrinsic_color.txt"
    if not intr_path.exists():
        intr_path = scene_path / "intrinsic_color.txt"
    pose_dir  = scene_path / "pose"

    if not intr_path.exists() or not pose_dir.exists():
        logger.warning("Intrinsic or pose directory missing for %s", scene_path.name)
        return []

    intrinsics = load_scannet_intrinsics(scene_path)
    axis_align = load_axis_alignment(scene_path)
    poses      = load_scannet_poses(scene_path, axis_alignment=axis_align)

    if not poses:
        logger.warning("No valid poses found for %s", scene_path.name)
        return []

    # Collect support-participating object IDs
    support_ids: set[int] = set()
    if support_graph:
        for parent_id, children in support_graph.items():
            support_ids.add(int(parent_id))
            support_ids.update(children)

    # Score every frame
    frame_entries: list[dict[str, Any]] = []
    for image_name, pose in poses.items():
        visible = get_visible_objects(objects, pose, intrinsics)
        if len(visible) < MIN_VISIBLE_OBJECTS:
            continue
        n_support = _count_support_objects(visible, support_ids)
        score = len(visible) * (1 + n_support)
        frame_entries.append(
            {
                "image_name":        image_name,
                "pose":              pose,
                "visible_object_ids": [o["id"] for o in visible],
                "n_visible":         len(visible),
                "score":             score,
            }
        )

    if not frame_entries:
        logger.warning(
            "No frames with >= %d visible objects in %s",
            MIN_VISIBLE_OBJECTS, scene_path.name,
        )
        return []

    # Greedy diverse selection
    frame_entries.sort(key=lambda e: e["score"], reverse=True)
    selected: list[dict[str, Any]] = [frame_entries[0]]

    for entry in frame_entries[1:]:
        if len(selected) >= max_frames:
            break
        too_close = any(
            _angular_distance(entry["pose"], sel["pose"]) < VIEWPOINT_DIVERSITY_MIN_ANGLE
            for sel in selected
        )
        if not too_close:
            selected.append(entry)

    results = []
    for s in selected:
        results.append(
            {
                "image_name":        s["image_name"],
                "camera_position":   s["pose"].position.tolist(),
                "visible_object_ids": s["visible_object_ids"],
                "n_visible":         s["n_visible"],
                "score":             s["score"],
            }
        )

    logger.info(
        "Selected %d frames for %s (top score=%d)",
        len(results), scene_path.name,
        results[0]["score"] if results else 0,
    )
    return results
