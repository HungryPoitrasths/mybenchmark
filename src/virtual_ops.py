"""Stage 5: Virtual operation engine.

Implements four types of spatial interventions:
  L2.1 — Object movement (with support-chain propagation)
  L2.2 — Viewpoint movement
  L2.3 — Object removal
  L3.2 — Coordinate-system rotation (counterfactual)
"""

from __future__ import annotations

import copy
import itertools
import logging
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
from .support_graph import get_attachment_chain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L2.1  Object movement (with attachment-chain propagation)
# ---------------------------------------------------------------------------

MOVEMENT_CANDIDATES = [
    # Varied distances across all horizontal axes for movement diversity.
    # Order is fixed so virtual operations remain reproducible.
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

ORBIT_ROTATION_CANDIDATES = [
    # Canonical representatives of the 3 unique horizontal orbit geometries.
    (90, "clockwise", -90.0),
    (90, "counterclockwise", 90.0),
    (180, "clockwise", -180.0),
]

ROOM_BBOX_TOL = 1e-6


def get_moved_object_ids(
    target_obj_id: int,
    attachment_graph: dict[int, list[int]],
) -> set[int]:
    """Return all object IDs that move together with the target."""
    dependents = get_attachment_chain(target_obj_id, attachment_graph)
    return set(dependents) | {target_obj_id}


def _translate_support_geom_in_place(obj: dict, delta_position: np.ndarray) -> None:
    support_geom = obj.get("support_geom")
    if not isinstance(support_geom, dict):
        return

    delta_xy = np.asarray(delta_position, dtype=float)[:2]
    for key in ("bottom_hull_xy", "top_hull_xy"):
        points = np.asarray(support_geom.get(key, []), dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) > 0:
            support_geom[key] = (points + delta_xy).tolist()

    candidates = support_geom.get("top_surface_candidates", [])
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        hull_xy = np.asarray(candidate.get("hull_xy", []), dtype=float)
        if hull_xy.ndim == 2 and hull_xy.shape[1] == 2 and len(hull_xy) > 0:
            candidate["hull_xy"] = (hull_xy + delta_xy).tolist()


def _translate_object_in_place(obj: dict, delta_position: np.ndarray) -> None:
    delta = np.asarray(delta_position, dtype=float)
    obj["center"] = (np.asarray(obj["center"], dtype=float) + delta).tolist()
    obj["bbox_min"] = (np.asarray(obj["bbox_min"], dtype=float) + delta).tolist()
    obj["bbox_max"] = (np.asarray(obj["bbox_max"], dtype=float) + delta).tolist()
    _translate_support_geom_in_place(obj, delta)


def _rotate_points(points: np.ndarray, rotation: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=float) - np.asarray(pivot, dtype=float)
    return (rotation @ centered.T).T + np.asarray(pivot, dtype=float)


def _rotate_points_xy(points_xy: np.ndarray, rotation: np.ndarray, pivot_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.asarray(points_xy, dtype=float)
    points_xyz = np.column_stack([
        np.asarray(points_xy, dtype=float),
        np.zeros(len(points_xy), dtype=float),
    ])
    pivot_xyz = np.array([pivot_xy[0], pivot_xy[1], 0.0], dtype=float)
    rotated_xyz = _rotate_points(points_xyz, rotation, pivot_xyz)
    return rotated_xyz[:, :2]


def _rotate_support_geom_in_place(obj: dict, rotation: np.ndarray, pivot: np.ndarray) -> None:
    support_geom = obj.get("support_geom")
    if not isinstance(support_geom, dict):
        return

    pivot_xy = np.asarray(pivot, dtype=float)[:2]
    for key in ("bottom_hull_xy", "top_hull_xy"):
        points = np.asarray(support_geom.get(key, []), dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) > 0:
            support_geom[key] = _rotate_points_xy(points, rotation, pivot_xy).tolist()

    candidates = support_geom.get("top_surface_candidates", [])
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        hull_xy = np.asarray(candidate.get("hull_xy", []), dtype=float)
        if hull_xy.ndim == 2 and hull_xy.shape[1] == 2 and len(hull_xy) > 0:
            candidate["hull_xy"] = _rotate_points_xy(hull_xy, rotation, pivot_xy).tolist()


def _rotate_aabb(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    rotation: np.ndarray,
    pivot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        list(itertools.product(*zip(np.asarray(bbox_min, dtype=float), np.asarray(bbox_max, dtype=float)))),
        dtype=float,
    )
    rotated_corners = _rotate_points(corners, rotation, pivot)
    return rotated_corners.min(axis=0), rotated_corners.max(axis=0)


def apply_movement(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_obj_id: int,
    delta_position: np.ndarray,
) -> list[dict]:
    """Move *target_obj_id* by *delta_position* and propagate to dependents.

    Returns a deep-copied list of objects with updated coordinates.
    Does NOT modify the input list.
    """
    updated = copy.deepcopy(objects)

    # Collect all IDs that must move together
    to_move = get_moved_object_ids(target_obj_id, attachment_graph)

    for obj in updated:
        if obj["id"] in to_move:
            _translate_object_in_place(obj, delta_position)

    return updated


def is_within_room(
    objects: list[dict],
    room_bbox_min: np.ndarray,
    room_bbox_max: np.ndarray,
) -> bool:
    """Check that every object's full bbox stays inside the room bounds."""
    for obj in objects:
        obj_bbox_min = np.array(obj["bbox_min"], dtype=float)
        obj_bbox_max = np.array(obj["bbox_max"], dtype=float)
        if np.any(obj_bbox_min < (room_bbox_min - ROOM_BBOX_TOL)):
            return False
        if np.any(obj_bbox_max > (room_bbox_max + ROOM_BBOX_TOL)):
            return False
    return True


def compute_room_bounds(objects: list[dict], margin: float = 0.0, room_bounds: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute an axis-aligned bounding box for the room.

    If *room_bounds* is provided (from wall/floor mesh annotations), use it
    directly — no margin added, because these represent the actual physical
    walls.  Otherwise, fall back to computing the bbox from all objects with
    no extra margin (margin=0) to avoid creating room bounds that extend
    beyond physical walls.
    """
    # room_bounds are authoritative; margin only applies to object-derived bounds.
    if room_bounds is not None:
        return np.array(room_bounds["bbox_min"]), np.array(room_bounds["bbox_max"])

    all_mins = np.array([o["bbox_min"] for o in objects])
    all_maxs = np.array([o["bbox_max"] for o in objects])
    padding = np.full(3, float(margin), dtype=float)
    return all_mins.min(axis=0) - padding, all_maxs.max(axis=0) + padding


def _bboxes_intersect_strict(obj_a: dict, obj_b: dict) -> bool:
    """Whether two axis-aligned boxes overlap with positive volume.

    Face-touching is allowed; only strict interior overlap counts as collision.
    """
    a_min = np.array(obj_a["bbox_min"], dtype=float)
    a_max = np.array(obj_a["bbox_max"], dtype=float)
    b_min = np.array(obj_b["bbox_min"], dtype=float)
    b_max = np.array(obj_b["bbox_max"], dtype=float)
    return bool(np.all(a_min < b_max) and np.all(b_min < a_max))


def has_terminal_bbox_collision(
    original_objects: list[dict],
    moved_objects: list[dict],
    moved_ids: set[int],
    collision_objects: list[dict] | None = None,
) -> bool:
    """Reject movements whose final boxes intersect any static collision object."""
    moved_map = {obj["id"]: obj for obj in moved_objects if obj["id"] in moved_ids}
    collision_source = original_objects if collision_objects is None else collision_objects
    static_objects = [obj for obj in collision_source if obj["id"] not in moved_ids]

    for moved_obj in moved_map.values():
        for static_obj in static_objects:
            if _bboxes_intersect_strict(moved_obj, static_obj):
                return True
    return False


def find_meaningful_movement(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    camera_pose: CameraPose,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
) -> tuple[np.ndarray | None, list[dict]]:
    """Search for a movement vector that changes at least one spatial relation.

    Returns (delta_vector, list_of_changed_relations) or (None, []).
    """
    # No depth/occlusion needed — we only need direction/distance changes.
    original_relations = compute_all_relations(objects, camera_pose, None, None)
    room_min, room_max = compute_room_bounds(objects, room_bounds=room_bounds)
    moved_ids = get_moved_object_ids(target_id, attachment_graph)

    for delta in MOVEMENT_CANDIDATES:
        new_objects = apply_movement(objects, attachment_graph, target_id, delta)
        if not is_within_room(new_objects, room_min, room_max):
            continue
        if has_terminal_bbox_collision(
            objects,
            new_objects,
            moved_ids,
            collision_objects=collision_objects,
        ):
            continue
        new_relations = compute_all_relations(new_objects, camera_pose, None, None)
        changed = find_changed_relations(original_relations, new_relations)
        if changed:
            return delta, changed

    return None, []


def apply_orbit_rotation(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    pivot_id: int,
    angle_deg: float,
) -> list[dict]:
    """Orbit a moved attachment chain around a static pivot in the horizontal plane.

    This rotates the target chain's position around *pivot_id* as seen from
    above. It does not rotate any object's intrinsic orientation.
    """
    moved_ids = get_moved_object_ids(target_id, attachment_graph)
    if pivot_id in moved_ids:
        raise ValueError("Pivot object must stay outside the moved attachment chain")

    obj_map = {obj["id"]: obj for obj in objects}
    target = obj_map.get(target_id)
    pivot = obj_map.get(pivot_id)
    if target is None or pivot is None:
        raise ValueError("Target and pivot objects must exist")

    pivot_center = np.array(pivot["center"], dtype=float)
    rotation = rotation_matrix_z(angle_deg)

    updated = copy.deepcopy(objects)
    for obj in updated:
        if obj["id"] not in moved_ids:
            continue
        obj_center = np.array(obj["center"], dtype=float)
        rotated_center = _rotate_points(np.array([obj_center]), rotation, pivot_center)[0]
        _translate_object_in_place(obj, rotated_center - obj_center)
    return updated


def find_meaningful_orbit_rotation(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    pivot_id: int,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate physically valid orbit rotations around a static pivot."""
    room_min, room_max = compute_room_bounds(objects, room_bounds=room_bounds)
    moved_ids = get_moved_object_ids(target_id, attachment_graph)
    if pivot_id in moved_ids:
        return []

    valid_rotations: list[dict[str, Any]] = []
    for angle, rotation_direction, signed_angle in ORBIT_ROTATION_CANDIDATES:
        rotated_objects = apply_orbit_rotation(
            objects,
            attachment_graph,
            target_id,
            pivot_id,
            signed_angle,
        )
        if not is_within_room(rotated_objects, room_min, room_max):
            continue
        if has_terminal_bbox_collision(
            objects,
            rotated_objects,
            moved_ids,
            collision_objects=collision_objects,
        ):
            continue
        valid_rotations.append({
            "angle": angle,
            "rotation_direction": rotation_direction,
            "signed_angle": signed_angle,
            "objects": rotated_objects,
        })

    return valid_rotations


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
    attachment_graph: dict[int, list[int]],
    target_id: int,
    cascade: bool = False,
) -> list[dict]:
    """Remove an object from the scene.

    If *cascade* is True, objects transitively attached to the removed object
    are also removed. Otherwise only the target is removed.

    Returns a new (deep-copied) object list.
    """
    to_remove = {target_id}
    if cascade:
        to_remove.update(get_attachment_chain(target_id, attachment_graph))

    return [copy.deepcopy(o) for o in objects if o["id"] not in to_remove]


# ---------------------------------------------------------------------------
# L3.1  Support-chain counterfactual (object repositioning)
# ---------------------------------------------------------------------------

def apply_counterfactual_placement(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    new_position: np.ndarray,
) -> list[dict]:
    """Counterfactual: 'suppose *target_id* had been placed at *new_position*'.

    Moves the target and all its dependents by the same delta.
    """
    obj_map = {o["id"]: o for o in objects}
    old_center = np.array(obj_map[target_id]["center"])
    delta = new_position - old_center
    return apply_movement(objects, attachment_graph, target_id, delta)


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
        obj["center"] = _rotate_points(
            np.array([obj["center"]], dtype=float),
            R,
            room_center,
        )[0].tolist()
        bbox_min, bbox_max = _rotate_aabb(
            np.array(obj["bbox_min"], dtype=float),
            np.array(obj["bbox_max"], dtype=float),
            R,
            room_center,
        )
        obj["bbox_min"] = bbox_min.tolist()
        obj["bbox_max"] = bbox_max.tolist()
        _rotate_support_geom_in_place(obj, R, room_center)

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
