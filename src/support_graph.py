"""Stage 2: Support relationship graph construction.

Detects which objects rest on top of other objects using layered geometric
heuristics: AABB overlap for coarse recall, then support-face footprint
overlap for confirmation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Floor z-threshold: objects whose bottom is below this are considered
# resting on the floor (not an interesting support relationship).
FLOOR_Z_MAX = 0.10  # metres

# Two-stage support thresholds.
AABB_CHILD_COVERAGE_MIN = 0.12
FALLBACK_CHILD_COVERAGE_MIN = 0.25
HULL_CHILD_COVERAGE_MIN = 0.20
GEOM_EPS = 1e-8


def _vertical_tolerance(obj_a: dict, obj_b: dict, z_threshold: float | None = None) -> float:
    a_height = float(obj_a["bbox_max"][2] - obj_a["bbox_min"][2])
    b_height = float(obj_b["bbox_max"][2] - obj_b["bbox_min"][2])
    adaptive = float(np.clip(0.02 + 0.05 * min(a_height, b_height), 0.03, 0.08))
    if z_threshold is None:
        return adaptive
    return float(np.clip(z_threshold, 0.03, 0.08))


def _is_on_floor(obj: dict) -> bool:
    """Check if the object is resting directly on the floor."""
    return obj["bbox_min"][2] < FLOOR_Z_MAX


def _bbox_xy_overlap_area(obj_a: dict, obj_b: dict) -> float:
    a_min = np.asarray(obj_a["bbox_min"][:2], dtype=float)
    a_max = np.asarray(obj_a["bbox_max"][:2], dtype=float)
    b_min = np.asarray(obj_b["bbox_min"][:2], dtype=float)
    b_max = np.asarray(obj_b["bbox_max"][:2], dtype=float)
    overlap = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    return float(overlap[0] * overlap[1])


def _bbox_xy_area(obj: dict) -> float:
    dims = np.asarray(obj["bbox_max"][:2], dtype=float) - np.asarray(obj["bbox_min"][:2], dtype=float)
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))


def _cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if len(poly) < 3:
        return poly
    signed = 0.5 * (np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    return poly if signed >= 0 else poly[::-1].copy()


def _is_inside_edge(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    return _cross_2d(edge_start, edge_end, point) >= -GEOM_EPS


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) <= GEOM_EPS:
        return p2.copy()
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    return p1 + t * r


def _clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject.copy()
    clipper = _ensure_ccw(clipper)
    if len(output) < 3 or len(clipper) < 3:
        return np.empty((0, 2), dtype=float)

    for i in range(len(clipper)):
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]
        input_list = output
        output_list: list[np.ndarray] = []
        if len(input_list) == 0:
            break
        s = input_list[-1]
        for e in input_list:
            e_inside = _is_inside_edge(e, edge_start, edge_end)
            s_inside = _is_inside_edge(s, edge_start, edge_end)
            if e_inside:
                if not s_inside:
                    output_list.append(_line_intersection(s, e, edge_start, edge_end))
                output_list.append(e)
            elif s_inside:
                output_list.append(_line_intersection(s, e, edge_start, edge_end))
            s = e
        output = np.array(output_list, dtype=float) if output_list else np.empty((0, 2), dtype=float)
    return output


def _convex_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    clipped = _clip_polygon(_ensure_ccw(poly_a), _ensure_ccw(poly_b))
    return _polygon_area(clipped)


def _support_face_polygon(obj: dict, face_key: str) -> np.ndarray:
    support_geom = obj.get("support_geom") or {}
    poly = support_geom.get(face_key)
    if not poly:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(poly, dtype=float)
    return arr if len(arr) >= 3 else np.empty((0, 2), dtype=float)


def _support_metrics(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> dict[str, float | bool] | None:
    """Return support metrics if A is plausibly supported by B, else None."""
    gap = abs(float(obj_a["bbox_min"][2]) - float(obj_b["bbox_max"][2]))
    z_tol = _vertical_tolerance(obj_a, obj_b, z_threshold)
    if gap > z_tol:
        return None

    child_xy_area = _bbox_xy_area(obj_a)
    if child_xy_area <= GEOM_EPS:
        return None

    overlap_area = _bbox_xy_overlap_area(obj_a, obj_b)
    if overlap_area <= GEOM_EPS:
        return None

    aabb_child_coverage = overlap_area / child_xy_area
    if aabb_child_coverage < AABB_CHILD_COVERAGE_MIN:
        return None

    child_hull = _support_face_polygon(obj_a, "bottom_hull_xy")
    parent_hull = _support_face_polygon(obj_b, "top_hull_xy")
    child_hull_area = _polygon_area(child_hull)
    parent_hull_area = _polygon_area(parent_hull)

    hull_overlap_area = 0.0
    hull_child_coverage = 0.0
    used_hull = False
    confirmed = False

    if child_hull_area > GEOM_EPS and parent_hull_area > GEOM_EPS:
        used_hull = True
        hull_overlap_area = _convex_intersection_area(child_hull, parent_hull)
        hull_child_coverage = hull_overlap_area / child_hull_area if child_hull_area > GEOM_EPS else 0.0
        confirmed = hull_child_coverage >= HULL_CHILD_COVERAGE_MIN
    else:
        confirmed = aabb_child_coverage >= FALLBACK_CHILD_COVERAGE_MIN

    if not confirmed:
        return None

    return {
        "gap": gap,
        "z_tol": z_tol,
        "aabb_overlap_area": overlap_area,
        "aabb_child_coverage": aabb_child_coverage,
        "hull_overlap_area": hull_overlap_area,
        "hull_child_coverage": hull_child_coverage,
        "coverage_score": hull_child_coverage if used_hull else aabb_child_coverage,
        "overlap_score": hull_overlap_area if used_hull else overlap_area,
        "used_hull": used_hull,
    }


def detect_support(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> bool:
    """Return True if obj_a is supported by obj_b (A sits on top of B)."""
    return _support_metrics(obj_a, obj_b, z_threshold) is not None


def build_support_graph(
    objects: list[dict],
    z_threshold: float | None = None,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Build support relationships for a set of objects.

    Args:
        objects: List of object dicts (id, label, center, bbox_min, bbox_max).
        z_threshold: Optional override for vertical contact tolerance.

    Returns:
        support_graph: {supporter_id: [list of supported obj ids]}
        supported_by:  {obj_id: supporter_id}  (each child has at most one parent)
    """
    support_graph: dict[int, list[int]] = {}
    supported_by: dict[int, int] = {}
    candidates: list[tuple[int, int, dict[str, float | bool]]] = []

    for a in objects:
        if _is_on_floor(a):
            continue
        for b in objects:
            if a["id"] == b["id"]:
                continue
            metrics = _support_metrics(a, b, z_threshold)
            if metrics is not None:
                candidates.append((int(a["id"]), int(b["id"]), metrics))

    best: dict[int, tuple[int, dict[str, float | bool]]] = {}
    for aid, bid, metrics in candidates:
        if aid not in best:
            best[aid] = (bid, metrics)
            continue

        _, current = best[aid]
        current_key = (
            float(current["coverage_score"]),
            -float(current["gap"]),
            float(current["overlap_score"]),
        )
        new_key = (
            float(metrics["coverage_score"]),
            -float(metrics["gap"]),
            float(metrics["overlap_score"]),
        )
        if new_key > current_key:
            best[aid] = (bid, metrics)

    for aid, (bid, _) in best.items():
        supported_by[aid] = bid
        support_graph.setdefault(bid, []).append(aid)

    logger.info(
        "Support graph: %d support edges among %d objects",
        len(supported_by),
        len(objects),
    )
    return support_graph, supported_by


def get_support_chain(
    obj_id: int,
    support_graph: dict[int, list[int]],
) -> list[int]:
    """Return all transitive dependents of *obj_id* (depth-first).

    If you move obj_id, all returned objects must move with it.
    """
    dependents: list[int] = []

    def _dfs(oid: int):
        for child in support_graph.get(oid, []):
            dependents.append(child)
            _dfs(child)

    _dfs(obj_id)
    return dependents


def has_nontrivial_support(
    support_graph: dict[int, list[int]],
) -> bool:
    """Return True if the scene has at least one object-on-object support."""
    return len(support_graph) > 0


def enrich_scene_with_support(
    scene: dict[str, Any],
    z_threshold: float | None = None,
) -> dict[str, Any]:
    """Add support_graph and supported_by fields to a scene dict (in-place)."""
    objects = scene["objects"]
    sg, sb = build_support_graph(objects, z_threshold)
    scene["support_graph"] = {str(k): v for k, v in sg.items()}
    scene["supported_by"] = {str(k): v for k, v in sb.items()}
    return scene
