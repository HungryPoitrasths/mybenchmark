"""Stage 4: Spatial relation computation engine.

Computes three types of spatial relations between object pairs:
  - Direction (egocentric, relative to camera viewpoint — 10 directions)
  - Distance (Euclidean, with categorical binning)
  - Occlusion (depth-map based per-object visibility)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .utils.colmap_loader import CameraPose
from .utils.coordinate_transform import world_to_camera

logger = logging.getLogger(__name__)

# ---- Direction relation (10-direction system) ----

# 8 horizontal directions (egocentric, camera-frame) + 2 vertical
HORIZONTAL_DIRECTIONS = [
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
]
VERTICAL_DIRECTIONS = ["above", "below"]
ALL_DIRECTIONS_10 = HORIZONTAL_DIRECTIONS + VERTICAL_DIRECTIONS


def primary_direction(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
    camera_pose: CameraPose,
) -> tuple[str, float]:
    """Return the single most dominant direction of B relative to A.

    Uses a 10-direction system: 8 horizontal bins (45° each, centred at
    0°/45°/90°/…/315°) plus "above"/"below" for vertical dominance.

    Camera convention (OpenCV): x→right, y→down, z→forward.

    Returns (direction_label, ambiguity_score) where ambiguity_score ∈ [0,1].
    Higher ambiguity means the direction is close to a bin boundary.
    """
    a_cam = world_to_camera(obj_a_center, camera_pose)
    b_cam = world_to_camera(obj_b_center, camera_pose)
    delta = b_cam - a_cam  # x=right, y=down, z=forward

    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])

    horizontal_mag = math.sqrt(dx * dx + dz * dz)
    vertical_mag = abs(dy)

    # If vertical component dominates → above/below
    if vertical_mag > horizontal_mag:
        direction = "below" if dy > 0 else "above"  # OpenCV y-down
        if horizontal_mag < 1e-6:
            ambiguity = 0.0
        else:
            ambiguity = horizontal_mag / vertical_mag  # how close to switching
        return direction, float(min(ambiguity, 1.0))

    if horizontal_mag < 1e-6:
        return "front", 1.0

    # Compute angle in horizontal plane: atan2(dx, dz)
    # dz = forward direction (0°), dx = right (90°)
    angle = math.degrees(math.atan2(dx, dz))  # range [-180, 180]
    if angle < 0:
        angle += 360  # → [0, 360)

    # Bin into 8 sectors (each 45°, centred at 0°, 45°, …, 315°)
    bin_idx = int((angle + 22.5) % 360 / 45)
    direction = HORIZONTAL_DIRECTIONS[bin_idx]

    # Ambiguity: distance from bin centre, normalised to [0,1]
    bin_centre = bin_idx * 45.0
    offset = abs(angle - bin_centre)
    if offset > 180:
        offset = 360 - offset
    horiz_ambiguity = offset / 22.5  # 0 at centre, 1 at boundary

    # Also factor in vertical component ratio
    vert_ratio = vertical_mag / horizontal_mag if horizontal_mag > 1e-6 else 0
    ambiguity = max(horiz_ambiguity, vert_ratio)

    return direction, float(min(ambiguity, 1.0))


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


# ---- Occlusion relation (per-object, depth-map based) ----

def compute_occlusion_per_object(
    objects: list[dict],
    camera_pose: CameraPose,
    depth_image: np.ndarray | None = None,
    depth_intrinsics=None,
) -> dict[int, tuple[str, float]]:
    """Compute per-object occlusion status using the depth map.

    Returns a dict mapping obj_id → (status, visibility_ratio).
    Status is one of: "fully visible", "partially occluded", "fully occluded".

    If depth_image is None, returns "unknown" for all objects.
    """
    if depth_image is None or depth_intrinsics is None:
        return {o["id"]: ("unknown", 0.0) for o in objects}

    from .utils.depth_occlusion import compute_depth_occlusion

    cache: dict[int, tuple[str, float]] = {}
    for obj in objects:
        status, ratio = compute_depth_occlusion(
            bbox_min=np.array(obj["bbox_min"]),
            bbox_max=np.array(obj["bbox_max"]),
            camera_pose=camera_pose,
            intrinsics=depth_intrinsics,
            depth_image=depth_image,
        )
        cache[obj["id"]] = (status, ratio)
    return cache


# ---- Batch computation ----

def compute_all_relations(
    objects: list[dict],
    camera_pose: CameraPose,
    depth_image: np.ndarray | None = None,
    depth_intrinsics=None,
) -> list[dict[str, Any]]:
    """Compute pairwise spatial relations for all object pairs.

    Occlusion is computed per-object (not pairwise) using depth maps.

    Returns a list of relation dicts, each containing:
        obj_a_id, obj_b_id, direction, distance_bin, distance_m,
        occlusion_a, occlusion_b
    """
    # Pre-compute per-object occlusion (each object checked once, not per pair)
    occ_cache = compute_occlusion_per_object(
        objects, camera_pose, depth_image, depth_intrinsics
    )

    relations: list[dict[str, Any]] = []

    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            a_center = np.array(a["center"])
            b_center = np.array(b["center"])

            # Direction: B relative to A
            dir_label, ambiguity = primary_direction(a_center, b_center, camera_pose)

            # Distance
            dist_bin, dist_m, near_bound = compute_distance(a_center, b_center)

            relations.append(
                {
                    "obj_a_id": a["id"],
                    "obj_a_label": a["label"],
                    "obj_b_id": b["id"],
                    "obj_b_label": b["label"],
                    "direction_b_rel_a": dir_label,
                    "ambiguity_score": ambiguity,
                    "distance_bin": dist_bin,
                    "distance_m": round(dist_m, 2),
                    "near_boundary": near_bound,
                    "occlusion_a": occ_cache.get(a["id"], ("unknown", 0.0))[0],
                    "occlusion_b": occ_cache.get(b["id"], ("unknown", 0.0))[0],
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
        for field in ("direction_b_rel_a", "distance_bin", "occlusion_a", "occlusion_b"):
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
