"""Stage 2: Support relationship graph construction.

Detects which objects rest on top of other objects using geometric heuristics
(vertical contact + horizontal overlap).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Floor z-threshold: objects whose bottom is below this are considered
# resting on the floor (not an interesting support relationship).
FLOOR_Z_MAX = 0.10  # metres


def detect_support(
    obj_a: dict,
    obj_b: dict,
    z_threshold: float = 0.05,
) -> bool:
    """Return True if obj_a is supported by obj_b (A sits on top of B).

    Conditions:
        1. A's bottom z ≈ B's top z  (vertical contact within *z_threshold*).
        2. A's bottom-face centre falls inside B's top-face XY projection.
    """
    a_bottom_z = obj_a["bbox_min"][2]
    b_top_z = obj_b["bbox_max"][2]

    # Condition 1: vertical contact
    if abs(a_bottom_z - b_top_z) > z_threshold:
        return False

    # Condition 2: horizontal overlap (A's centre inside B's XY footprint)
    a_cx, a_cy = obj_a["center"][0], obj_a["center"][1]
    b_xmin, b_ymin = obj_b["bbox_min"][0], obj_b["bbox_min"][1]
    b_xmax, b_ymax = obj_b["bbox_max"][0], obj_b["bbox_max"][1]

    return b_xmin <= a_cx <= b_xmax and b_ymin <= a_cy <= b_ymax


def _is_on_floor(obj: dict) -> bool:
    """Check if the object is resting directly on the floor."""
    return obj["bbox_min"][2] < FLOOR_Z_MAX


def build_support_graph(
    objects: list[dict],
    z_threshold: float = 0.05,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Build support relationships for a set of objects.

    Args:
        objects: List of object dicts (id, label, center, bbox_min, bbox_max).
        z_threshold: Maximum vertical gap to count as "contact".

    Returns:
        support_graph: {supporter_id: [list of supported obj ids]}
        supported_by:  {obj_id: supporter_id}  (each child has at most one parent)
    """
    support_graph: dict[int, list[int]] = {}
    supported_by: dict[int, int] = {}

    # Build an index for quick look-up
    obj_by_id = {o["id"]: o for o in objects}

    candidates: list[tuple[dict, dict, float]] = []

    for a in objects:
        if _is_on_floor(a):
            # Object on the floor: it *may* support others but isn't supported
            # by a non-floor object.
            continue
        for b in objects:
            if a["id"] == b["id"]:
                continue
            if detect_support(a, b, z_threshold):
                gap = abs(a["bbox_min"][2] - b["bbox_max"][2])
                candidates.append((a, b, gap))

    # Resolve conflicts: if A is detected as supported by multiple objects,
    # keep the one with the smallest vertical gap.
    best: dict[int, tuple[int, float]] = {}
    for a, b, gap in candidates:
        aid = a["id"]
        bid = b["id"]
        if aid not in best or gap < best[aid][1]:
            best[aid] = (bid, gap)

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
    z_threshold: float = 0.05,
) -> dict[str, Any]:
    """Add support_graph and supported_by fields to a scene dict (in-place)."""
    objects = scene["objects"]
    sg, sb = build_support_graph(objects, z_threshold)
    scene["support_graph"] = {str(k): v for k, v in sg.items()}
    scene["supported_by"] = {str(k): v for k, v in sb.items()}
    return scene
