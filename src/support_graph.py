"""Stage 2: attachment/support graph construction.

This module keeps the historical ``support_graph`` API for compatibility, while
internally introducing a more general attachment relation used for movement
propagation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Floor z-threshold: objects whose bottom is below this are usually resting on
# the floor. We keep the constant for compatibility even though attachment
# detection no longer uses it as a hard global filter.
FLOOR_Z_MAX = 0.10  # metres

# Surface-contact thresholds.
AABB_CHILD_COVERAGE_MIN = 0.12
FALLBACK_CHILD_COVERAGE_MIN = 0.25
HULL_CHILD_COVERAGE_MIN = 0.20
SOFT_HULL_CHILD_COVERAGE_MIN = 0.12
SOFT_FALLBACK_CHILD_COVERAGE_MIN = 0.18
GEOM_EPS = 1e-8

SUPPORT_LIKE_TYPES = {
    "supported_by",
    "resting_on_soft_surface",
}

ATTACHMENT_TYPE_PRIORITY = {
    "contained_in": 4,
    "affixed_to": 3,
    "resting_on_soft_surface": 2,
    "supported_by": 1,
}

SOFT_PARENT_LABELS = {
    "bed",
    "sofa",
    "couch",
    "chair",
    "bench",
    "ottoman",
}
SOFT_CHILD_LABELS = {
    "pillow",
    "blanket",
    "cushion",
    "clothing",
    "towel",
    "sheet",
}
SOFT_SURFACE_PRIORS = {
    ("pillow", "bed"),
    ("blanket", "bed"),
    ("cushion", "sofa"),
    ("cushion", "chair"),
    ("clothing", "bed"),
    ("towel", "bed"),
}

CONTAINER_PARENT_LABELS = {
    "drawer",
    "cabinet",
    "box",
    "storage container",
    "basket",
    "bowl",
    "bin",
    "trash can",
    "refrigerator",
    "sink",
}
CONTAINMENT_PRIORS = {
    ("apple", "bowl"),
    ("fruit", "bowl"),
    ("clothing", "drawer"),
    ("book", "drawer"),
    ("toy", "box"),
}

AFFIXED_PRIORS = {
    ("monitor", "monitor stand"),
    ("monitor", "desk"),
    ("television", "tv stand"),
    ("tv", "tv stand"),
    ("handle", "cabinet"),
    ("knob", "cabinet"),
    ("lamp", "wall"),
}


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


def _label(obj: dict) -> str:
    return str(obj.get("label", "")).strip().lower()


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
    # Sutherland-Hodgman clipping expects convex polygons. Current callers feed
    # convex hulls for both polygons; keep this assumption explicit here.
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


def _top_surface_candidates(obj: dict) -> list[dict[str, Any]]:
    support_geom = obj.get("support_geom") or {}
    raw_candidates = support_geom.get("top_surface_candidates") or []
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        hull = np.asarray(raw.get("hull_xy") or [], dtype=float)
        if len(hull) < 3:
            continue
        area = float(raw.get("area", _polygon_area(hull)))
        if area <= GEOM_EPS:
            continue
        candidates.append({
            "z": float(raw.get("z", obj["bbox_max"][2])),
            "hull_xy": hull,
            "area": area,
            "score": float(raw.get("score", 0.0)),
        })

    if candidates:
        return candidates

    legacy_hull = _support_face_polygon(obj, "top_hull_xy")
    if len(legacy_hull) >= 3:
        return [{
            "z": float(obj["bbox_max"][2]),
            "hull_xy": legacy_hull,
            "area": _polygon_area(legacy_hull),
            "score": 0.0,
        }]
    return []


def _bbox_axis_gaps(obj_a: dict, obj_b: dict) -> np.ndarray:
    a_min = np.asarray(obj_a["bbox_min"], dtype=float)
    a_max = np.asarray(obj_a["bbox_max"], dtype=float)
    b_min = np.asarray(obj_b["bbox_min"], dtype=float)
    b_max = np.asarray(obj_b["bbox_max"], dtype=float)
    return np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))


def _surface_attachment_metrics(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
    *,
    gap_scale: float = 1.0,
    hull_coverage_min: float = HULL_CHILD_COVERAGE_MIN,
    fallback_coverage_min: float = FALLBACK_CHILD_COVERAGE_MIN,
) -> dict[str, float | bool] | None:
    """Return the best parent surface contact metrics if plausible, else None."""
    child_xy_area = _bbox_xy_area(obj_a)
    if child_xy_area <= GEOM_EPS:
        return None

    overlap_area = _bbox_xy_overlap_area(obj_a, obj_b)
    if overlap_area <= GEOM_EPS:
        return None

    # Cheap coarse gate to reject obvious non-contacts before more expensive
    # hull processing. This is not the final confirmation threshold.
    aabb_child_coverage = overlap_area / child_xy_area
    if aabb_child_coverage < AABB_CHILD_COVERAGE_MIN:
        return None

    child_hull = _support_face_polygon(obj_a, "bottom_hull_xy")
    child_hull_area = _polygon_area(child_hull)
    z_tol = _vertical_tolerance(obj_a, obj_b, z_threshold) * gap_scale
    best: dict[str, float | bool] | None = None

    for surface in _top_surface_candidates(obj_b):
        gap = abs(float(obj_a["bbox_min"][2]) - float(surface["z"]))
        if gap > z_tol:
            continue

        parent_hull = np.asarray(surface["hull_xy"], dtype=float)
        parent_hull_area = _polygon_area(parent_hull)
        hull_overlap_area = 0.0
        hull_child_coverage = 0.0
        used_hull = False

        if child_hull_area > GEOM_EPS and parent_hull_area > GEOM_EPS:
            used_hull = True
            hull_overlap_area = _convex_intersection_area(child_hull, parent_hull)
            hull_child_coverage = (
                hull_overlap_area / child_hull_area
                if child_hull_area > GEOM_EPS else 0.0
            )
            confirmed = hull_child_coverage >= hull_coverage_min
        else:
            # Without usable hull geometry we require a stricter AABB overlap,
            # because the approximation is less precise than polygon contact.
            confirmed = aabb_child_coverage >= fallback_coverage_min

        if not confirmed:
            continue

        metrics = {
            "gap": float(gap),
            "z_tol": float(z_tol),
            "contact_z_parent": float(surface["z"]),
            "contact_z_child": float(obj_a["bbox_min"][2]),
            "aabb_overlap_area": float(overlap_area),
            "aabb_child_coverage": float(aabb_child_coverage),
            "hull_overlap_area": float(hull_overlap_area),
            "hull_child_coverage": float(hull_child_coverage),
            "coverage_score": float(hull_child_coverage if used_hull else aabb_child_coverage),
            "overlap_score": float(hull_overlap_area if used_hull else overlap_area),
            "used_hull": used_hull,
            "surface_area": float(surface["area"]),
            "surface_score": float(surface["score"]),
        }
        if best is None:
            best = metrics
            continue

        current_key = (
            float(best["coverage_score"]),
            -float(best["gap"]),
            float(best["overlap_score"]),
            float(best["surface_area"]),
            float(best["surface_score"]),
        )
        new_key = (
            float(metrics["coverage_score"]),
            -float(metrics["gap"]),
            float(metrics["overlap_score"]),
            float(metrics["surface_area"]),
            float(metrics["surface_score"]),
        )
        if new_key > current_key:
            best = metrics

    return best


def _support_metrics(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> dict[str, float | bool] | None:
    """Return strict support metrics if A is plausibly supported by B."""
    return _surface_attachment_metrics(obj_a, obj_b, z_threshold)


def _soft_surface_prior(parent_label: str, child_label: str) -> float:
    if (child_label, parent_label) in SOFT_SURFACE_PRIORS:
        return 1.0
    if parent_label in SOFT_PARENT_LABELS and child_label in SOFT_CHILD_LABELS:
        return 0.75
    return 0.0


def _containment_prior(parent_label: str, child_label: str) -> float:
    if (child_label, parent_label) in CONTAINMENT_PRIORS:
        return 1.0
    if parent_label in CONTAINER_PARENT_LABELS:
        return 0.7
    return 0.0


def _affixed_prior(parent_label: str, child_label: str) -> float:
    if (child_label, parent_label) in AFFIXED_PRIORS:
        return 1.0
    if child_label in {"monitor", "television", "tv"} and parent_label in {"monitor stand", "tv stand", "desk"}:
        return 0.8
    return 0.0


def _surface_evidence(
    obj_a: dict,
    obj_b: dict,
    metrics: dict[str, float | bool],
    *,
    semantic_score: float = 0.0,
) -> dict[str, Any]:
    return {
        "geometry_contact": {
            "z_gap": float(metrics["gap"]),
            "contact_z_parent": float(metrics["contact_z_parent"]),
            "contact_z_child": float(metrics["contact_z_child"]),
            "z_tolerance": float(metrics["z_tol"]),
        },
        "xy_overlap": {
            "child_coverage": float(metrics["coverage_score"]),
            "aabb_child_coverage": float(metrics["aabb_child_coverage"]),
            "overlap_area": float(metrics["overlap_score"]),
            "used_hull": bool(metrics["used_hull"]),
        },
        "containment": None,
        "semantic_prior": {
            "parent_label": _label(obj_b),
            "child_label": _label(obj_a),
            "score": float(semantic_score),
        },
    }


def _supported_by_metrics(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> dict[str, Any] | None:
    metrics = _support_metrics(obj_a, obj_b, z_threshold)
    if metrics is None:
        return None
    gap_score = float(np.clip(1.0 - float(metrics["gap"]) / max(float(metrics["z_tol"]), GEOM_EPS), 0.0, 1.0))
    confidence = float(np.clip(
        0.60 * float(metrics["coverage_score"]) +
        0.25 * gap_score +
        0.15 * float(metrics["aabb_child_coverage"]),
        0.0,
        1.0,
    ))
    return {
        "type": "supported_by",
        "confidence": confidence,
        "evidence": _surface_evidence(obj_a, obj_b, metrics),
    }


def _resting_on_soft_surface_metrics(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> dict[str, Any] | None:
    child_label = _label(obj_a)
    parent_label = _label(obj_b)
    prior = _soft_surface_prior(parent_label, child_label)
    if prior <= 0.0:
        return None

    metrics = _surface_attachment_metrics(
        obj_a,
        obj_b,
        z_threshold,
        gap_scale=1.75,
        hull_coverage_min=SOFT_HULL_CHILD_COVERAGE_MIN,
        fallback_coverage_min=SOFT_FALLBACK_CHILD_COVERAGE_MIN,
    )
    if metrics is None:
        return None

    gap_score = float(np.clip(1.0 - float(metrics["gap"]) / max(float(metrics["z_tol"]), GEOM_EPS), 0.0, 1.0))
    confidence = float(np.clip(
        0.45 * float(metrics["coverage_score"]) +
        0.25 * gap_score +
        0.30 * prior,
        0.0,
        1.0,
    ))
    return {
        "type": "resting_on_soft_surface",
        "confidence": confidence,
        "evidence": _surface_evidence(obj_a, obj_b, metrics, semantic_score=prior),
    }


def _contained_in_metrics(obj_a: dict, obj_b: dict) -> dict[str, Any] | None:
    child_label = _label(obj_a)
    parent_label = _label(obj_b)
    prior = _containment_prior(parent_label, child_label)
    if prior <= 0.0:
        return None

    child_area = _bbox_xy_area(obj_a)
    parent_area = _bbox_xy_area(obj_b)
    if child_area <= GEOM_EPS or parent_area <= GEOM_EPS or parent_area <= child_area * 1.05:
        return None

    overlap_area = _bbox_xy_overlap_area(obj_a, obj_b)
    xy_coverage = overlap_area / max(child_area, GEOM_EPS)

    child_min = np.asarray(obj_a["bbox_min"], dtype=float)
    child_max = np.asarray(obj_a["bbox_max"], dtype=float)
    parent_min = np.asarray(obj_b["bbox_min"], dtype=float)
    parent_max = np.asarray(obj_b["bbox_max"], dtype=float)
    child_center = np.asarray(obj_a["center"], dtype=float)
    z_tol = 0.05 + 0.05 * float(obj_b["bbox_max"][2] - obj_b["bbox_min"][2])

    center_inside_xy = bool(np.all(child_center[:2] >= parent_min[:2]) and np.all(child_center[:2] <= parent_max[:2]))
    center_inside_xyz = bool(np.all(child_center >= (parent_min - z_tol)) and np.all(child_center <= (parent_max + z_tol)))
    z_inside = bool(child_min[2] >= parent_min[2] - z_tol and child_center[2] <= parent_max[2] + z_tol)
    size_score = float(np.clip((parent_area - child_area) / max(parent_area, GEOM_EPS), 0.0, 1.0))

    strong_geometry = xy_coverage >= 0.80 and center_inside_xy and z_inside
    if not strong_geometry and not (prior > 0.0 and xy_coverage >= 0.65 and center_inside_xy):
        return None

    containment_score = float(np.clip(
        0.55 * xy_coverage +
        0.15 * float(center_inside_xy) +
        0.15 * float(center_inside_xyz) +
        0.15 * size_score,
        0.0,
        1.0,
    ))
    confidence = float(np.clip(0.80 * containment_score + 0.20 * prior, 0.0, 1.0))
    return {
        "type": "contained_in",
        "confidence": confidence,
        "evidence": {
            "geometry_contact": {
                "axis_gaps": [float(v) for v in _bbox_axis_gaps(obj_a, obj_b)],
            },
            "xy_overlap": {
                "child_coverage": float(xy_coverage),
                "overlap_area": float(overlap_area),
            },
            "containment": {
                "score": containment_score,
                "center_inside_xy": center_inside_xy,
                "center_inside_xyz": center_inside_xyz,
                "z_inside": z_inside,
                "size_score": size_score,
            },
            "semantic_prior": {
                "parent_label": parent_label,
                "child_label": child_label,
                "score": float(prior),
            },
        },
    }


def _affixed_to_metrics(obj_a: dict, obj_b: dict) -> dict[str, Any] | None:
    child_label = _label(obj_a)
    parent_label = _label(obj_b)
    prior = _affixed_prior(parent_label, child_label)
    if prior <= 0.0:
        return None

    axis_gaps = _bbox_axis_gaps(obj_a, obj_b)
    total_gap = float(np.linalg.norm(axis_gaps))
    min_gap = float(axis_gaps.min())
    overlap_area = _bbox_xy_overlap_area(obj_a, obj_b)
    xy_coverage = overlap_area / max(_bbox_xy_area(obj_a), GEOM_EPS)

    if total_gap > 0.12:
        return None
    if min_gap > 0.06 and xy_coverage < 0.20:
        return None

    proximity_score = float(np.clip(1.0 - total_gap / 0.12, 0.0, 1.0))
    confidence = float(np.clip(0.55 * prior + 0.30 * proximity_score + 0.15 * xy_coverage, 0.0, 1.0))
    return {
        "type": "affixed_to",
        "confidence": confidence,
        "evidence": {
            "geometry_contact": {
                "axis_gaps": [float(v) for v in axis_gaps],
                "distance": total_gap,
                "min_axis_gap": min_gap,
            },
            "xy_overlap": {
                "child_coverage": float(xy_coverage),
                "overlap_area": float(overlap_area),
            },
            "containment": None,
            "semantic_prior": {
                "parent_label": parent_label,
                "child_label": child_label,
                "score": float(prior),
            },
        },
    }


def _attachment_candidate(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> dict[str, Any] | None:
    # Type order is semantic, not purely confidence-based: e.g. "contained_in"
    # should win over a weaker "supported_by" interpretation for the same pair.
    for builder in (
        lambda: _contained_in_metrics(obj_a, obj_b),
        lambda: _affixed_to_metrics(obj_a, obj_b),
        lambda: _resting_on_soft_surface_metrics(obj_a, obj_b, z_threshold),
        lambda: _supported_by_metrics(obj_a, obj_b, z_threshold),
    ):
        candidate = builder()
        if candidate is None:
            continue
        return {
            "parent_id": int(obj_b["id"]),
            "child_id": int(obj_a["id"]),
            "type": str(candidate["type"]),
            "confidence": float(candidate["confidence"]),
            "evidence": candidate["evidence"],
            "move_with_parent": True,
            "remove_with_parent": False,
        }
    return None


def _edge_sort_key(edge: dict[str, Any]) -> tuple[float, int, float, float, float]:
    evidence = edge.get("evidence") or {}
    geometry = evidence.get("geometry_contact") or {}
    overlap = evidence.get("xy_overlap") or {}
    containment = evidence.get("containment") or {}
    gap = float(geometry.get("z_gap", geometry.get("distance", 1e6)))
    return (
        float(edge.get("confidence", 0.0)),
        ATTACHMENT_TYPE_PRIORITY.get(str(edge.get("type", "")), 0),
        float(containment.get("score", 0.0)),
        float(overlap.get("child_coverage", 0.0)),
        -gap,
    )


def _derive_graph_from_edges(
    edges: list[dict[str, Any]],
    *,
    allowed_types: set[str] | None = None,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    graph: dict[int, list[int]] = {}
    reverse: dict[int, int] = {}
    for edge in edges:
        edge_type = str(edge.get("type", ""))
        if allowed_types is not None and edge_type not in allowed_types:
            continue
        parent_id = int(edge["parent_id"])
        child_id = int(edge["child_id"])
        graph.setdefault(parent_id, []).append(child_id)
        reverse[child_id] = parent_id
    return graph, reverse


def _would_create_cycle(parent_id: int, child_id: int, reverse: dict[int, int]) -> bool:
    current = parent_id
    seen: set[int] = set()
    while current in reverse and current not in seen:
        if current == child_id:
            return True
        seen.add(current)
        current = reverse[current]
    return current == child_id


def detect_support(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float | None = None,
) -> bool:
    """Return True if obj_a is rigidly supported by obj_b."""
    return _support_metrics(obj_a, obj_b, z_threshold) is not None


def build_attachment_graph(
    objects: list[dict],
    z_threshold: float | None = None,
) -> tuple[dict[int, list[int]], dict[int, int], list[dict[str, Any]]]:
    """Build attachment relations used for movement propagation."""
    candidates: dict[int, dict[str, Any]] = {}

    for obj_a in objects:
        for obj_b in objects:
            if obj_a["id"] == obj_b["id"]:
                continue
            edge = _attachment_candidate(obj_a, obj_b, z_threshold)
            if edge is None:
                continue

            child_id = int(edge["child_id"])
            current = candidates.get(child_id)
            if current is None or _edge_sort_key(edge) > _edge_sort_key(current):
                candidates[child_id] = edge

    final_edges: list[dict[str, Any]] = []
    attachment_graph: dict[int, list[int]] = {}
    attached_by: dict[int, int] = {}
    for edge in sorted(candidates.values(), key=_edge_sort_key, reverse=True):
        parent_id = int(edge["parent_id"])
        child_id = int(edge["child_id"])
        if _would_create_cycle(parent_id, child_id, attached_by):
            continue
        attached_by[child_id] = parent_id
        attachment_graph.setdefault(parent_id, []).append(child_id)
        final_edges.append(edge)

    logger.info(
        "Attachment graph: %d edges among %d objects",
        len(attached_by),
        len(objects),
    )
    return attachment_graph, attached_by, final_edges


def build_support_graph(
    objects: list[dict],
    z_threshold: float | None = None,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Build legacy support relationships as a subset of attachment edges."""
    _attachment_graph, _attached_by, attachment_edges = build_attachment_graph(objects, z_threshold)
    support_graph, supported_by = _derive_graph_from_edges(
        attachment_edges,
        allowed_types=SUPPORT_LIKE_TYPES,
    )
    logger.info(
        "Support graph: %d support-like edges among %d objects",
        len(supported_by),
        len(objects),
    )
    return support_graph, supported_by


def get_support_chain(
    obj_id: int,
    support_graph: dict[int, list[int]] | dict[str, list[int]],
) -> list[int]:
    """Return all transitive dependents of *obj_id* (depth-first)."""
    dependents: list[int] = []
    visited: set[int] = set()

    def _dfs(oid: int):
        children = support_graph.get(oid) or support_graph.get(str(oid)) or []
        for child in children:
            child_id = int(child)
            if child_id in visited:
                continue
            visited.add(child_id)
            dependents.append(child_id)
            _dfs(child_id)

    _dfs(int(obj_id))
    return dependents


def get_attachment_chain(
    obj_id: int,
    attachment_graph: dict[int, list[int]] | dict[str, list[int]],
) -> list[int]:
    """Alias for movement-dependency traversal."""
    return get_support_chain(obj_id, attachment_graph)


def has_nontrivial_support(
    support_graph: dict[int, list[int]] | dict[str, list[int]],
) -> bool:
    """Return True if the graph has at least one support/dependency edge."""
    return len(support_graph) > 0


def has_nontrivial_attachment(
    attachment_graph: dict[int, list[int]] | dict[str, list[int]],
) -> bool:
    return len(attachment_graph) > 0


def get_scene_attachment_graph(
    scene: dict[str, Any],
    scene_id: str | None = None,
) -> dict[int, list[int]]:
    """Return the scene attachment graph with int-normalized IDs.

    Falls back to legacy ``support_graph`` with a warning if needed.
    Raises ``KeyError`` if neither field is present.
    """
    scene_label = scene_id or str(scene.get("scene_id", "<unknown>"))
    raw_graph = scene.get("attachment_graph")
    if raw_graph is None:
        raw_graph = scene.get("support_graph")
        if raw_graph is None:
            raise KeyError(
                f"Scene {scene_label} is missing both 'attachment_graph' and legacy 'support_graph'"
            )
        logger.warning(
            "Scene %s is missing 'attachment_graph'; falling back to legacy 'support_graph'",
            scene_label,
        )

    normalized: dict[int, list[int]] = {}
    for parent_id, child_ids in raw_graph.items():
        normalized[int(parent_id)] = [int(child_id) for child_id in child_ids]
    return normalized


def get_scene_attached_by(
    scene: dict[str, Any],
    scene_id: str | None = None,
) -> dict[int, int]:
    """Return the scene reverse attachment map with int-normalized IDs.

    Falls back to legacy ``supported_by`` with a warning if needed.
    Raises ``KeyError`` if neither field is present.
    """
    scene_label = scene_id or str(scene.get("scene_id", "<unknown>"))
    raw_map = scene.get("attached_by")
    if raw_map is None:
        raw_map = scene.get("supported_by")
        if raw_map is None:
            raise KeyError(
                f"Scene {scene_label} is missing both 'attached_by' and legacy 'supported_by'"
            )
        logger.warning(
            "Scene %s is missing 'attached_by'; falling back to legacy 'supported_by'",
            scene_label,
        )

    return {int(child_id): int(parent_id) for child_id, parent_id in raw_map.items()}


def enrich_scene_with_attachment(
    scene: dict[str, Any],
    z_threshold: float | None = None,
) -> dict[str, Any]:
    """Add attachment/support graph fields to a scene dict (in-place)."""
    objects = scene["objects"]
    attachment_graph, attached_by, attachment_edges = build_attachment_graph(objects, z_threshold)
    support_graph, supported_by = _derive_graph_from_edges(
        attachment_edges,
        allowed_types=SUPPORT_LIKE_TYPES,
    )
    scene["attachment_graph"] = {str(k): v for k, v in attachment_graph.items()}
    scene["attached_by"] = {str(k): v for k, v in attached_by.items()}
    scene["attachment_edges"] = attachment_edges
    scene["support_graph"] = {str(k): v for k, v in support_graph.items()}
    scene["supported_by"] = {str(k): v for k, v in supported_by.items()}
    return scene


def enrich_scene_with_support(
    scene: dict[str, Any],
    z_threshold: float | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper that now also writes attachment fields."""
    return enrich_scene_with_attachment(scene, z_threshold)
