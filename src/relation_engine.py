"""Stage 4: Spatial relation computation engine.

Computes three types of spatial relations between object pairs:
  - Direction (egocentric, relative to camera viewpoint, 10 directions)
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

# 8 cardinal directions (allocentric, world-frame) + 2 vertical
CARDINAL_DIRECTIONS_8 = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]
ALL_ALLOCENTRIC_10 = CARDINAL_DIRECTIONS_8 + VERTICAL_DIRECTIONS

_GEOM_EPS = 1e-8
_VERTICAL_CLEARANCE_TOL = 0.02  # metres
_SPINE_ELONGATION_RATIO_MIN = 1.8
_SPINE_ELONGATION_DELTA_MIN = 0.35  # metres


def _rectangle_from_bbox_xy(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.array([
        [bbox_min[0], bbox_min[1]],
        [bbox_max[0], bbox_min[1]],
        [bbox_max[0], bbox_max[1]],
        [bbox_min[0], bbox_max[1]],
    ], dtype=float)


def _object_bottom_hull_xy(obj: dict) -> np.ndarray:
    support_geom = obj.get("support_geom", {})
    hull = np.asarray(support_geom.get("bottom_hull_xy", []), dtype=float)
    if hull.ndim == 2 and hull.shape[0] >= 3 and hull.shape[1] == 2:
        return hull
    bbox_min = np.asarray(obj.get("bbox_min", []), dtype=float)
    bbox_max = np.asarray(obj.get("bbox_max", []), dtype=float)
    if bbox_min.shape == (3,) and bbox_max.shape == (3,):
        return _rectangle_from_bbox_xy(bbox_min, bbox_max)
    center = np.asarray(obj.get("center", [0.0, 0.0, 0.0]), dtype=float)
    return np.array([center[:2]], dtype=float)


def _point_on_segment_2d(a: np.ndarray, b: np.ndarray, p: np.ndarray, tol: float = _GEOM_EPS) -> bool:
    ab = b - a
    ap = p - a
    cross = float(ab[0] * ap[1] - ab[1] * ap[0])
    if abs(cross) > tol:
        return False
    dot = float(np.dot(ap, ab))
    if dot < -tol:
        return False
    if dot > float(np.dot(ab, ab)) + tol:
        return False
    return True


def _orientation_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _segments_intersect_2d(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, tol: float = _GEOM_EPS) -> bool:
    o1 = _orientation_2d(a1, a2, b1)
    o2 = _orientation_2d(a1, a2, b2)
    o3 = _orientation_2d(b1, b2, a1)
    o4 = _orientation_2d(b1, b2, a2)

    if (
        ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol))
        and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol))
    ):
        return True

    return any((
        abs(o1) <= tol and _point_on_segment_2d(a1, a2, b1, tol),
        abs(o2) <= tol and _point_on_segment_2d(a1, a2, b2, tol),
        abs(o3) <= tol and _point_on_segment_2d(b1, b2, a1, tol),
        abs(o4) <= tol and _point_on_segment_2d(b1, b2, a2, tol),
    ))


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray, tol: float = _GEOM_EPS) -> bool:
    n = len(polygon)
    if n == 0:
        return False
    if n == 1:
        return bool(np.linalg.norm(point - polygon[0]) <= tol)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if _point_on_segment_2d(a, b, point, tol):
            return True

    x, y = float(point[0]), float(point[1])
    inside = False
    for i in range(n):
        x1, y1 = float(polygon[i][0]), float(polygon[i][1])
        x2, y2 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        intersects = ((y1 > y) != (y2 > y))
        if not intersects:
            continue
        x_at_y = x1 + (y - y1) * (x2 - x1) / max(y2 - y1, _GEOM_EPS)
        if x_at_y >= x - tol:
            inside = not inside
    return inside


def _polygons_overlap_xy(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    if len(poly_a) == 0 or len(poly_b) == 0:
        return False

    for i in range(len(poly_a)):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % len(poly_b)]
            if _segments_intersect_2d(a1, a2, b1, b2):
                return True

    return _point_in_polygon_2d(poly_a[0], poly_b) or _point_in_polygon_2d(poly_b[0], poly_a)


def _nearest_point_on_segment(p1: np.ndarray, p2: np.ndarray, query: np.ndarray) -> np.ndarray:
    d = p2 - p1
    denom = float(np.dot(d, d))
    if denom < _GEOM_EPS:
        return p1.copy()
    t = float(np.clip(np.dot(query - p1, d) / denom, 0.0, 1.0))
    return p1 + t * d


def nearest_point_on_hull(hull_xy: np.ndarray | list[list[float]], query_2d: np.ndarray) -> np.ndarray:
    """Nearest point on a polygon boundary to a query point."""
    hull = np.asarray(hull_xy, dtype=float)
    query = np.asarray(query_2d, dtype=float)
    if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) == 0:
        return query.copy()
    if len(hull) == 1:
        return hull[0].copy()

    best_point = hull[0].copy()
    best_dist = float("inf")
    for i in range(len(hull)):
        point = _nearest_point_on_segment(hull[i], hull[(i + 1) % len(hull)], query)
        dist = float(np.linalg.norm(point - query))
        if dist < best_dist:
            best_dist = dist
            best_point = point
    return best_point


def _polygon_centroid_xy(hull_xy: np.ndarray, fallback_xy: np.ndarray) -> np.ndarray:
    if hull_xy.ndim == 2 and hull_xy.shape[0] >= 1 and hull_xy.shape[1] == 2:
        return np.mean(hull_xy, axis=0)
    return np.asarray(fallback_xy, dtype=float).copy()


def _compute_min_area_obb_2d(hull_xy: np.ndarray | list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    poly = np.asarray(hull_xy, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        return None

    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None
    best_area = float("inf")

    for i in range(len(poly)):
        edge = poly[(i + 1) % len(poly)] - poly[i]
        edge_len = float(np.linalg.norm(edge))
        if edge_len < _GEOM_EPS:
            continue
        axis0 = edge / edge_len
        axis1 = np.array([-axis0[1], axis0[0]], dtype=float)

        proj0 = poly @ axis0
        proj1 = poly @ axis1
        min0, max0 = float(np.min(proj0)), float(np.max(proj0))
        min1, max1 = float(np.min(proj1)), float(np.max(proj1))
        len0 = max0 - min0
        len1 = max1 - min1
        area = len0 * len1
        if area >= best_area:
            continue

        center = axis0 * (0.5 * (min0 + max0)) + axis1 * (0.5 * (min1 + max1))
        if len0 >= len1:
            major_dir = axis0
            minor_dir = axis1
            length = len0
            width = len1
        else:
            major_dir = axis1
            minor_dir = axis0
            length = len1
            width = len0

        best_area = area
        best = (
            np.asarray(center, dtype=float),
            np.asarray(major_dir, dtype=float),
            np.asarray(minor_dir, dtype=float),
            float(length),
            float(width),
        )

    return best


def _spine_endpoints(center_xy: np.ndarray, major_dir: np.ndarray, length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    half_len = max(0.5 * (length - width), 0.0)
    return (
        np.asarray(center_xy, dtype=float) + half_len * np.asarray(major_dir, dtype=float),
        np.asarray(center_xy, dtype=float) - half_len * np.asarray(major_dir, dtype=float),
    )


def _project_point_to_segment(query_xy: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    return _nearest_point_on_segment(
        np.asarray(seg_start, dtype=float),
        np.asarray(seg_end, dtype=float),
        np.asarray(query_xy, dtype=float),
    )


def _should_use_spine_override(length: float, width: float) -> bool:
    safe_width = max(float(width), _GEOM_EPS)
    return (
        float(length) > float(width)
        and (float(length) / safe_width) >= _SPINE_ELONGATION_RATIO_MIN
        and (float(length) - float(width)) >= _SPINE_ELONGATION_DELTA_MIN
    )


def footprint_nearest_pair(
    hull_a: np.ndarray | list[list[float]],
    hull_b: np.ndarray | list[list[float]],
    fallback_a_xy: np.ndarray,
    fallback_b_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable nearest horizontal reference pair between two footprints."""
    poly_a = np.asarray(hull_a, dtype=float)
    poly_b = np.asarray(hull_b, dtype=float)
    fallback_a = np.asarray(fallback_a_xy, dtype=float)
    fallback_b = np.asarray(fallback_b_xy, dtype=float)

    if poly_a.ndim != 2 or poly_a.shape[1] != 2 or len(poly_a) < 2:
        return fallback_a.copy(), fallback_b.copy()
    if poly_b.ndim != 2 or poly_b.shape[1] != 2 or len(poly_b) < 2:
        return fallback_a.copy(), fallback_b.copy()

    if _polygons_overlap_xy(poly_a, poly_b):
        return _polygon_centroid_xy(poly_a, fallback_a), _polygon_centroid_xy(poly_b, fallback_b)

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []

    for a_point in poly_a:
        b_point = nearest_point_on_hull(poly_b, a_point)
        dist = float(np.linalg.norm(b_point - a_point))
        candidates.append((a_point.copy(), b_point, dist))

    for b_point in poly_b:
        a_point = nearest_point_on_hull(poly_a, b_point)
        dist = float(np.linalg.norm(a_point - b_point))
        candidates.append((a_point, b_point.copy(), dist))

    if not candidates:
        return fallback_a.copy(), fallback_b.copy()

    best_dist = min(item[2] for item in candidates)
    near_best = [
        (a_point, b_point)
        for a_point, b_point, dist in candidates
        if dist <= best_dist + 1e-6
    ]
    if not near_best:
        return fallback_a.copy(), fallback_b.copy()

    best_a = np.mean([item[0] for item in near_best], axis=0)
    best_b = np.mean([item[1] for item in near_best], axis=0)
    return np.asarray(best_a, dtype=float), np.asarray(best_b, dtype=float)


def _horizontal_direction_from_components(
    primary_comp: float,
    secondary_comp: float,
    labels: list[str],
) -> tuple[str, float]:
    horizontal_mag = math.sqrt(primary_comp * primary_comp + secondary_comp * secondary_comp)
    if horizontal_mag < 1e-6:
        return labels[0], 1.0

    angle = math.degrees(math.atan2(secondary_comp, primary_comp))
    if angle < 0:
        angle += 360

    bin_idx = int((angle + 22.5) % 360 / 45)
    direction = labels[bin_idx]
    bin_centre = bin_idx * 45.0
    offset = abs(angle - bin_centre)
    if offset > 180:
        offset = 360 - offset
    ambiguity = offset / 22.5
    return direction, float(min(max(ambiguity, 0.0), 1.0))


def _vertical_interval_direction(
    a_bbox_min: np.ndarray | None,
    a_bbox_max: np.ndarray | None,
    b_bbox_min: np.ndarray | None,
    b_bbox_max: np.ndarray | None,
) -> tuple[str | None, float]:
    if any(item is None for item in (a_bbox_min, a_bbox_max, b_bbox_min, b_bbox_max)):
        return None, 0.0

    a_min = np.asarray(a_bbox_min, dtype=float)
    a_max = np.asarray(a_bbox_max, dtype=float)
    b_min = np.asarray(b_bbox_min, dtype=float)
    b_max = np.asarray(b_bbox_max, dtype=float)
    if a_min.shape != (3,) or a_max.shape != (3,) or b_min.shape != (3,) or b_max.shape != (3,):
        return None, 0.0

    gap_up = float(b_min[2] - a_max[2])
    gap_down = float(a_min[2] - b_max[2])
    extent = max(
        float(a_max[2] - a_min[2]),
        float(b_max[2] - b_min[2]),
        _VERTICAL_CLEARANCE_TOL,
    )

    if gap_up > _VERTICAL_CLEARANCE_TOL:
        ambiguity = 1.0 - min(gap_up / extent, 1.0)
        return "above", float(max(ambiguity, 0.0))
    if gap_down > _VERTICAL_CLEARANCE_TOL:
        ambiguity = 1.0 - min(gap_down / extent, 1.0)
        return "below", float(max(ambiguity, 0.0))
    return None, 0.0


def _horizontal_reference_points_with_spine_override(
    anchor_center_xy: np.ndarray,
    target_center_xy: np.ndarray,
    anchor_hull_xy: np.ndarray | list[list[float]],
    target_hull_xy: np.ndarray | list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    anchor_hull = np.asarray(anchor_hull_xy, dtype=float)
    target_hull = np.asarray(target_hull_xy, dtype=float)
    anchor_center_xy = np.asarray(anchor_center_xy, dtype=float)
    target_center_xy = np.asarray(target_center_xy, dtype=float)
    target_hull_valid = (
        target_hull.ndim == 2
        and target_hull.shape[1] == 2
        and len(target_hull) >= 3
    )

    anchor_ref_xy, target_ref_xy = footprint_nearest_pair(
        anchor_hull,
        target_hull,
        anchor_center_xy,
        target_center_xy,
    )

    if (
        anchor_hull.ndim != 2
        or anchor_hull.shape[1] != 2
        or len(anchor_hull) < 3
        or (target_hull_valid and _polygons_overlap_xy(anchor_hull, target_hull))
    ):
        return anchor_ref_xy, target_ref_xy

    obb = _compute_min_area_obb_2d(anchor_hull)
    if obb is None:
        return anchor_ref_xy, target_ref_xy
    center_xy, major_dir, _minor_dir, length, width = obb
    if not _should_use_spine_override(length, width):
        return anchor_ref_xy, target_ref_xy

    spine_start, spine_end = _spine_endpoints(center_xy, major_dir, length, width)
    anchor_spine_xy = _project_point_to_segment(target_ref_xy, spine_start, spine_end)
    return anchor_spine_xy, target_ref_xy


def _pairwise_horizontal_reference_points(obj_a: dict, obj_b: dict) -> tuple[np.ndarray, np.ndarray]:
    a_center = np.asarray(obj_a.get("center", [0.0, 0.0, 0.0]), dtype=float)
    b_center = np.asarray(obj_b.get("center", [0.0, 0.0, 0.0]), dtype=float)
    hull_a = _object_bottom_hull_xy(obj_a)
    hull_b = _object_bottom_hull_xy(obj_b)
    return _horizontal_reference_points_with_spine_override(
        a_center[:2],
        b_center[:2],
        hull_a,
        hull_b,
    )


def compute_pairwise_direction(
    obj_a: dict,
    obj_b: dict,
    camera_pose: CameraPose,
) -> tuple[str, float]:
    """Direction of B relative to A using footprint geometry where available."""
    vertical_label, vertical_ambiguity = _vertical_interval_direction(
        np.asarray(obj_a.get("bbox_min", []), dtype=float) if "bbox_min" in obj_a else None,
        np.asarray(obj_a.get("bbox_max", []), dtype=float) if "bbox_max" in obj_a else None,
        np.asarray(obj_b.get("bbox_min", []), dtype=float) if "bbox_min" in obj_b else None,
        np.asarray(obj_b.get("bbox_max", []), dtype=float) if "bbox_max" in obj_b else None,
    )
    if vertical_label is not None:
        return vertical_label, vertical_ambiguity

    a_ref_xy, b_ref_xy = _pairwise_horizontal_reference_points(obj_a, obj_b)
    a_ref = np.array([a_ref_xy[0], a_ref_xy[1], 0.0], dtype=float)
    b_ref = np.array([b_ref_xy[0], b_ref_xy[1], 0.0], dtype=float)
    return primary_direction(a_ref, b_ref, camera_pose, horizontal_only=True)


def primary_direction(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
    camera_pose: CameraPose,
    *,
    horizontal_only: bool = False,
) -> tuple[str, float]:
    """Return the single most dominant direction of B relative to A."""
    a_cam = world_to_camera(np.asarray(obj_a_center, dtype=float), camera_pose)
    b_cam = world_to_camera(np.asarray(obj_b_center, dtype=float), camera_pose)
    delta = b_cam - a_cam  # x=right, y=down, z=forward

    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    horizontal_mag = math.sqrt(dx * dx + dz * dz)
    vertical_mag = abs(dy)

    if not horizontal_only and vertical_mag > horizontal_mag:
        direction = "below" if dy > 0 else "above"  # OpenCV y-down
        if horizontal_mag < 1e-6:
            ambiguity = 0.0
        else:
            ambiguity = horizontal_mag / vertical_mag
        return direction, float(min(ambiguity, 1.0))

    direction, ambiguity = _horizontal_direction_from_components(dz, dx, HORIZONTAL_DIRECTIONS)
    if horizontal_only:
        return direction, ambiguity

    vert_ratio = vertical_mag / horizontal_mag if horizontal_mag > 1e-6 else 0.0
    return direction, float(min(max(ambiguity, vert_ratio), 1.0))


# ---- Object-centric direction (reference-pair frame) ----


def primary_direction_object_centric(
    anchor_center: np.ndarray,
    facing_center: np.ndarray,
    target_center: np.ndarray,
    *,
    anchor_hull_xy: np.ndarray | list[list[float]] | None = None,
    target_hull_xy: np.ndarray | list[list[float]] | None = None,
    anchor_bbox_min: np.ndarray | list[float] | None = None,
    anchor_bbox_max: np.ndarray | list[float] | None = None,
    target_bbox_min: np.ndarray | list[float] | None = None,
    target_bbox_max: np.ndarray | list[float] | None = None,
) -> tuple[str, float]:
    """Direction of *target* as seen from *anchor* facing toward *facing*."""
    anchor_center = np.asarray(anchor_center, dtype=float)
    facing_center = np.asarray(facing_center, dtype=float)
    target_center = np.asarray(target_center, dtype=float)

    fwd_3d = facing_center - anchor_center
    fwd_horiz = np.array([fwd_3d[0], fwd_3d[1], 0.0], dtype=float)
    horiz_len = np.linalg.norm(fwd_horiz)
    if horiz_len < 1e-6:
        return "front", 1.0
    fwd_horiz /= horiz_len

    vertical_label, vertical_ambiguity = _vertical_interval_direction(
        np.asarray(anchor_bbox_min, dtype=float) if anchor_bbox_min is not None else None,
        np.asarray(anchor_bbox_max, dtype=float) if anchor_bbox_max is not None else None,
        np.asarray(target_bbox_min, dtype=float) if target_bbox_min is not None else None,
        np.asarray(target_bbox_max, dtype=float) if target_bbox_max is not None else None,
    )
    if vertical_label is not None:
        return vertical_label, vertical_ambiguity

    right_horiz = np.array([fwd_horiz[1], -fwd_horiz[0], 0.0], dtype=float)

    if anchor_hull_xy is not None and target_hull_xy is not None:
        anchor_ref_xy, target_ref_xy = footprint_nearest_pair(
            anchor_hull_xy,
            target_hull_xy,
            anchor_center[:2],
            target_center[:2],
        )
        delta = np.array([
            target_ref_xy[0] - anchor_ref_xy[0],
            target_ref_xy[1] - anchor_ref_xy[1],
            0.0,
        ], dtype=float)
    else:
        delta = target_center - anchor_center
        delta[2] = 0.0

    fwd_comp = float(np.dot(delta, fwd_horiz))
    right_comp = float(np.dot(delta, right_horiz))
    return _horizontal_direction_from_components(fwd_comp, right_comp, HORIZONTAL_DIRECTIONS)


# ---- Allocentric direction (world-frame, axis-aligned) ----


def primary_direction_allocentric(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
    *,
    obj_a_hull_xy: np.ndarray | list[list[float]] | None = None,
    obj_b_hull_xy: np.ndarray | list[list[float]] | None = None,
    obj_a_bbox_min: np.ndarray | list[float] | None = None,
    obj_a_bbox_max: np.ndarray | list[float] | None = None,
    obj_b_bbox_min: np.ndarray | list[float] | None = None,
    obj_b_bbox_max: np.ndarray | list[float] | None = None,
) -> tuple[str, float]:
    """Cardinal direction of A as seen from B in axis-aligned world coords."""
    obj_a_center = np.asarray(obj_a_center, dtype=float)
    obj_b_center = np.asarray(obj_b_center, dtype=float)

    vertical_label, vertical_ambiguity = _vertical_interval_direction(
        np.asarray(obj_b_bbox_min, dtype=float) if obj_b_bbox_min is not None else None,
        np.asarray(obj_b_bbox_max, dtype=float) if obj_b_bbox_max is not None else None,
        np.asarray(obj_a_bbox_min, dtype=float) if obj_a_bbox_min is not None else None,
        np.asarray(obj_a_bbox_max, dtype=float) if obj_a_bbox_max is not None else None,
    )
    if vertical_label is not None:
        return vertical_label, vertical_ambiguity

    if obj_a_hull_xy is not None and obj_b_hull_xy is not None:
        a_ref_xy, b_ref_xy = _horizontal_reference_points_with_spine_override(
            obj_a_center[:2],
            obj_b_center[:2],
            obj_a_hull_xy,
            obj_b_hull_xy,
        )
        dx = float(a_ref_xy[0] - b_ref_xy[0])
        dy = float(a_ref_xy[1] - b_ref_xy[1])
    else:
        delta = obj_a_center - obj_b_center
        dx, dy = float(delta[0]), float(delta[1])

    return _horizontal_direction_from_components(dy, dx, CARDINAL_DIRECTIONS_8)


def camera_cardinal_direction(camera_pose: CameraPose) -> str:
    """Return the cardinal direction the camera is facing."""
    from .utils.coordinate_transform import get_camera_forward
    fwd = get_camera_forward(camera_pose)
    fwd_horiz = np.array([fwd[0], fwd[1]])
    if np.linalg.norm(fwd_horiz) < 1e-6:
        return "north"
    angle = math.degrees(math.atan2(float(fwd_horiz[0]), float(fwd_horiz[1])))
    if angle < 0:
        angle += 360
    bin_idx = int((angle + 22.5) % 360 / 45)
    return CARDINAL_DIRECTIONS_8[bin_idx]


# ---- Distance relation ----

DISTANCE_BINS = [
    (1.0, "very close (<1.0m)"),
    (2.0, "close (1.0-2.0m)"),
    (3.3, "moderate (2.0-3.3m)"),
    (float("inf"), "far (>3.3m)"),
]

# Minimum centre-to-centre distance for direction questions.
# Below this threshold, bbox annotation errors (~0.1-0.2 m) make direction
# judgements unreliable.
MIN_DIRECTION_DISTANCE = 0.5  # metres

DISTANCE_BIN_BOUNDARIES = [b[0] for b in DISTANCE_BINS[:-1]]  # [1.0, 2.0, 3.3]
DISTANCE_BOUNDARY_MARGIN = 0.1


def compute_distance(
    obj_a_center: np.ndarray,
    obj_b_center: np.ndarray,
) -> tuple[str, float, bool]:
    """Compute Euclidean distance and categorical bin."""
    dist = float(np.linalg.norm(np.asarray(obj_a_center, dtype=float) - np.asarray(obj_b_center, dtype=float)))
    near_boundary = any(abs(dist - b) < DISTANCE_BOUNDARY_MARGIN for b in DISTANCE_BIN_BOUNDARIES)

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
    """Compute per-object occlusion status using the depth map."""
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
    """Compute pairwise spatial relations for all object pairs."""
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

            dir_label, ambiguity = compute_pairwise_direction(a, b, camera_pose)
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
    """Compare two relation sets and return entries that changed."""
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
