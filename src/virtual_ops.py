"""Stage 5: Virtual operation engine.

Implements four types of spatial interventions:
  L2.1 — Object movement (with support-chain propagation)
  L2.2 — Viewpoint movement
  L2.3 — Object removal
  L3.2 — Coordinate-system rotation (counterfactual)
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

import numpy as np

from .utils.colmap_loader import CameraPose
from .utils.coordinate_transform import (
    get_camera_forward,
    get_camera_right,
    get_camera_up,
    rotation_matrix_z,
)
from .relation_engine import compute_all_relations, find_changed_relations
from .support_graph import get_support_chain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L2.1  Object movement (with support-chain propagation)
# ---------------------------------------------------------------------------

MOVEMENT_CANDIDATES = [
    # Varied distances across all horizontal axes for movement diversity.
    # Order is shuffled per-call in find_meaningful_movement() to avoid
    # systematic bias toward any one direction/distance.
    np.array([0.5, 0.0, 0.0]),
    np.array([-0.5, 0.0, 0.0]),
    np.array([0.0, 0.5, 0.0]),
    np.array([0.0, -0.5, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([1.5, 0.0, 0.0]),
    np.array([-1.5, 0.0, 0.0]),
    np.array([0.0, 1.5, 0.0]),
    np.array([0.0, -1.5, 0.0]),
    np.array([2.0, 0.0, 0.0]),
    np.array([-2.0, 0.0, 0.0]),
    np.array([0.0, 2.0, 0.0]),
    np.array([0.0, -2.0, 0.0]),
    np.array([2.5, 0.0, 0.0]),
    np.array([-2.5, 0.0, 0.0]),
    np.array([0.0, 2.5, 0.0]),
    np.array([0.0, -2.5, 0.0]),
]


def apply_movement(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    target_obj_id: int,
    delta_position: np.ndarray,
) -> list[dict]:
    """Move *target_obj_id* by *delta_position* and propagate to dependents.

    Returns a deep-copied list of objects with updated coordinates.
    Does NOT modify the input list.
    """
    updated = copy.deepcopy(objects)

    # Collect all IDs that must move together
    dependents = get_support_chain(target_obj_id, support_graph)
    to_move = set(dependents) | {target_obj_id}

    for obj in updated:
        if obj["id"] in to_move:
            obj["center"] = (np.array(obj["center"]) + delta_position).tolist()
            obj["bbox_min"] = (np.array(obj["bbox_min"]) + delta_position).tolist()
            obj["bbox_max"] = (np.array(obj["bbox_max"]) + delta_position).tolist()

    return updated


def is_within_room(
    objects: list[dict],
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray,
) -> bool:
    """Check that every object centre lies inside the room bounding box."""
    for obj in objects:
        c = np.array(obj["center"])
        if np.any(c < room_bbox_min) or np.any(c > room_bbox_max):
            return False
    return True


def compute_room_bounds(objects: list[dict], margin: float = 0.5, room_bounds: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute an axis-aligned bounding box for the room.

    If *room_bounds* is provided (from wall/floor mesh annotations), use it
    directly — no margin added, because these represent the actual physical
    walls.  Otherwise, fall back to computing the bbox from all objects with
    no extra margin (margin=0) to avoid creating room bounds that extend
    beyond physical walls.
    """
    if room_bounds is not None:
        return np.array(room_bounds["bbox_min"]), np.array(room_bounds["bbox_max"])

    all_mins = np.array([o["bbox_min"] for o in objects])
    all_maxs = np.array([o["bbox_max"] for o in objects])
    return all_mins.min(axis=0), all_maxs.max(axis=0)


def find_meaningful_movement(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    target_id: int,
    camera_pose: CameraPose,
    room_bounds: dict | None = None,
) -> tuple[np.ndarray | None, list[dict]]:
    """Search for a movement vector that changes at least one spatial relation.

    Returns (delta_vector, list_of_changed_relations) or (None, []).
    """
    # No depth/occlusion needed — we only need direction/distance changes.
    original_relations = compute_all_relations(objects, camera_pose, None, None)
    room_min, room_max = compute_room_bounds(objects, room_bounds=room_bounds)

    # Shuffle candidates to avoid systematic bias toward the first entry
    candidates = list(MOVEMENT_CANDIDATES)
    random.shuffle(candidates)

    for delta in candidates:
        new_objects = apply_movement(objects, support_graph, target_id, delta)
        if not is_within_room(new_objects, room_min, room_max):
            continue
        new_relations = compute_all_relations(new_objects, camera_pose, None, None)
        changed = find_changed_relations(original_relations, new_relations)
        if changed:
            return delta, changed

    return None, []


# ---------------------------------------------------------------------------
# L2.2  Viewpoint movement
# ---------------------------------------------------------------------------

VIEWPOINT_MOVEMENTS = {
    "right":   lambda pose: get_camera_right(pose),
    "left":    lambda pose: -get_camera_right(pose),
    "forward": lambda pose: get_camera_forward(pose),
    "back":    lambda pose: -get_camera_forward(pose),
    "up":      lambda pose: get_camera_up(pose),
    "down":    lambda pose: -get_camera_up(pose),
}


def apply_viewpoint_change(
    camera_pose: CameraPose,
    direction: str,
    distance: float = 3.0,
) -> CameraPose:
    """Create a new camera pose moved along the specified direction.

    Args:
        camera_pose: Original camera pose.
        direction:   One of "right", "left", "forward", "back", "up", "down".
        distance:    Movement distance in metres.

    Returns:
        New CameraPose with updated position, same rotation.
    """
    dir_fn = VIEWPOINT_MOVEMENTS.get(direction)
    if dir_fn is None:
        raise ValueError(f"Unknown direction '{direction}'. Choose from {list(VIEWPOINT_MOVEMENTS)}")

    dir_vec = dir_fn(camera_pose)
    new_pos = camera_pose.position + dir_vec * distance

    new_pose = CameraPose(
        image_name=f"{camera_pose.image_name}_moved_{direction}_{distance}m",
        rotation=camera_pose.rotation.copy(),
        translation=camera_pose.translation.copy(),
    )
    new_pose.position = new_pos
    return new_pose


# ---------------------------------------------------------------------------
# L2.3  Object removal
# ---------------------------------------------------------------------------

def apply_removal(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    target_id: int,
    cascade: bool = True,
) -> list[dict]:
    """Remove an object from the scene.

    If *cascade* is True, objects transitively supported by the removed object
    are also removed (they would fall / disappear).  Otherwise only the target
    is removed.

    Returns a new (deep-copied) object list.
    """
    to_remove = {target_id}
    if cascade:
        to_remove.update(get_support_chain(target_id, support_graph))

    return [copy.deepcopy(o) for o in objects if o["id"] not in to_remove]


# ---------------------------------------------------------------------------
# L3.1  Support-chain counterfactual (object repositioning)
# ---------------------------------------------------------------------------

def apply_counterfactual_placement(
    objects: list[dict],
    support_graph: dict[int, list[int]],
    target_id: int,
    new_position: np.ndarray,
) -> list[dict]:
    """Counterfactual: 'suppose *target_id* had been placed at *new_position*'.

    Moves the target and all its dependents by the same delta.
    """
    obj_map = {o["id"]: o for o in objects}
    old_center = np.array(obj_map[target_id]["center"])
    delta = new_position - old_center
    return apply_movement(objects, support_graph, target_id, delta)


# ---------------------------------------------------------------------------
# L3.2  Coordinate-system rotation (counterfactual)
# ---------------------------------------------------------------------------

def apply_coordinate_rotation(
    objects: list[dict],
    rotation_angle_deg: float,
) -> list[dict]:
    """Rotate all objects around the room centre by *rotation_angle_deg* about z.

    Simulates: "if the room had been oriented differently…".
    All relative positions are preserved; absolute coordinates change.

    Returns a new (deep-copied) object list.
    """
    room_center = np.mean([np.array(o["center"]) for o in objects], axis=0)
    R = rotation_matrix_z(rotation_angle_deg)

    rotated = copy.deepcopy(objects)
    for obj in rotated:
        for key in ("center", "bbox_min", "bbox_max"):
            vec = np.array(obj[key]) - room_center
            obj[key] = (R @ vec + room_center).tolist()

    return rotated


def angle_to_compass(angle_deg: float) -> str:
    """Convert a rotation angle to a compass-style description."""
    angle_deg = angle_deg % 360
    compass = [
        (0, "north"), (45, "northeast"), (90, "east"), (135, "southeast"),
        (180, "south"), (225, "southwest"), (270, "west"), (315, "northwest"),
    ]
    closest = min(compass, key=lambda c: min(abs(c[0] - angle_deg), 360 - abs(c[0] - angle_deg)))
    return closest[1]
