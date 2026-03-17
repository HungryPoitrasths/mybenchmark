"""Stage 4: Spatial relation computation engine.

Computes three types of spatial relations between object pairs:
  - Direction (egocentric, relative to camera viewpoint)
  - Distance (Euclidean, with categorical binning)
  - Occlusion (ray-casting against the scene mesh)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .utils.colmap_loader import CameraPose
from .utils.coordinate_transform import world_to_camera
from .utils.ray_casting import RayCaster

logger = logging.getLogger(__name__)

# ---- Direction relation ----

DIRECTION_THRESHOLD_XY = 0.3  # metres — minimum offset to declare a direction
DIRECTION_THRESHOLD_Z = 0.5   # metres — depth axis needs larger margin


def compute_direction(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
    camera_pose: CameraPose,
) -> list[str]:
    """Compute the directions of obj_b relative to obj_a in camera coordinates.

    Camera convention (OpenCV): x→right, y→down, z→forward.

    Returns a list of direction labels (may contain 0-3 elements).
    """
    a_cam = world_to_camera(obj_a_center, camera_pose)
    b_cam = world_to_camera(obj_b_center, camera_pose)
    delta = b_cam - a_cam

    relations: list[str] = []

    # Horizontal (camera x-axis)
    if abs(delta[0]) > DIRECTION_THRESHOLD_XY:
        relations.append("right" if delta[0] > 0 else "left")

    # Vertical (camera y-axis, positive = down in OpenCV)
    if abs(delta[1]) > DIRECTION_THRESHOLD_XY:
        relations.append("below" if delta[1] > 0 else "above")

    # Depth (camera z-axis)
    if abs(delta[2]) > DIRECTION_THRESHOLD_Z:
        relations.append("behind" if delta[2] > 0 else "in front")

    return relations


def primary_direction(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
    camera_pose: CameraPose,
) -> tuple[str, float]:
    """Return the single most dominant direction of B relative to A.

    Returns (direction_label, ambiguity_score) where ambiguity_score ∈ [0,1].
    Higher ambiguity means the direction is close to a boundary.
    """
    a_cam = world_to_camera(obj_a_center, camera_pose)
    b_cam = world_to_camera(obj_b_center, camera_pose)
    delta = b_cam - a_cam

    candidates = [
        (abs(delta[0]), "right" if delta[0] > 0 else "left"),
        (abs(delta[1]), "below" if delta[1] > 0 else "above"),
        (abs(delta[2]), "behind" if delta[2] > 0 else "in front"),
    ]
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_mag, best_dir = candidates[0]
    second_mag = candidates[1][0]

    # Ambiguity: how close are the two largest components
    if best_mag < 1e-6:
        return best_dir, 1.0
    ambiguity = second_mag / best_mag  # 0 = unambiguous, 1 = totally ambiguous
    return best_dir, float(ambiguity)


# ---- Distance relation ----

DISTANCE_BINS = [
    (0.5, "touching (<0.5m)"),
    (1.5, "very close (0.5-1.5m)"),
    (3.0, "close (1.5-3m)"),
    (float("inf"), "far (>3m)"),
]

DISTANCE_BIN_BOUNDARIES = [b[0] for b in DISTANCE_BINS[:-1]]  # [0.5, 1.5, 3.0]


def compute_distance(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
) -> tuple[str, float, bool]:
    """Compute Euclidean distance and categorical bin.

    Returns (bin_label, raw_distance, near_boundary).
    near_boundary is True if the distance is within 0.2 m of a bin edge.
    """
    dist = float(np.linalg.norm(obj_a_center - obj_b_center))
    near_boundary = any(abs(dist - b) < 0.2 for b in DISTANCE_BIN_BOUNDARIES)

    for threshold, label in DISTANCE_BINS:
        if dist < threshold:
            return label, dist, near_boundary

    return DISTANCE_BINS[-1][1], dist, near_boundary


# ---- Occlusion relation ----

def compute_occlusion(
    obj_a: dict,
    obj_b: dict,
    camera_pos: np.ndarray,
    ray_caster: RayCaster,
) -> str:
    """Determine if obj_a is occluded by obj_b from the camera position.

    Uses multi-ray sampling for a more nuanced assessment.

    Returns: "fully_visible", "partially_occluded", or "fully_occluded".
    """
    return ray_caster.multi_ray_occlusion(
        camera_pos=camera_pos,
        target_bbox_min=np.array(obj_a["bbox_min"]),
        target_bbox_max=np.array(obj_a["bbox_max"]),
    )


# ---- Batch computation ----

def compute_all_relations(
    objects: list[dict],
    camera_pose: CameraPose,
    ray_caster: RayCaster | None = None,
) -> list[dict[str, Any]]:
    """Compute pairwise spatial relations for all object pairs.

    Returns a list of relation dicts, each containing:
        obj_a_id, obj_b_id, direction, distance_bin, distance_m, occlusion
    """
    relations: list[dict[str, Any]] = []
    camera_pos = camera_pose.position

    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            a_center = np.array(a["center"])
            b_center = np.array(b["center"])

            # Direction: B relative to A
            dir_label, ambiguity = primary_direction(a_center, b_center, camera_pose)
            all_dirs = compute_direction(a_center, b_center, camera_pose)

            # Distance
            dist_bin, dist_m, near_bound = compute_distance(a_center, b_center)

            # Occlusion (only if ray caster is available)
            occ = "unknown"
            if ray_caster is not None:
                occ = compute_occlusion(a, b, camera_pos, ray_caster)

            relations.append(
                {
                    "obj_a_id": a["id"],
                    "obj_a_label": a["label"],
                    "obj_b_id": b["id"],
                    "obj_b_label": b["label"],
                    "direction_b_rel_a": dir_label,
                    "all_directions": all_dirs,
                    "ambiguity_score": ambiguity,
                    "distance_bin": dist_bin,
                    "distance_m": round(dist_m, 2),
                    "near_boundary": near_bound,
                    "occlusion": occ,
                }
            )

    return relations


def find_changed_relations(
    old_relations: list[dict],
    new_relations: list[dict],
) -> list[dict]:
    """Compare two relation sets and return entries that changed.

    Returns a list of dicts with old and new values for changed pairs.
    """
    old_map = {(r["obj_a_id"], r["obj_b_id"]): r for r in old_relations}
    new_map = {(r["obj_a_id"], r["obj_b_id"]): r for r in new_relations}

    changed: list[dict] = []
    for key in old_map:
        if key not in new_map:
            continue
        o = old_map[key]
        n = new_map[key]
        diffs = {}
        for field in ("direction_b_rel_a", "distance_bin", "occlusion"):
            if o.get(field) != n.get(field):
                diffs[field] = {"old": o[field], "new": n[field]}
        if diffs:
            changed.append(
                {
                    "obj_a_id": key[0],
                    "obj_b_id": key[1],
                    "changes": diffs,
                    "old": o,
                    "new": n,
                }
            )

    return changed
