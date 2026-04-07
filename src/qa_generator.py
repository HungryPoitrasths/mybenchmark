"""Stage 6: QA generation.

Generates multiple-choice questions from computed spatial relations and
virtual operation results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import inspect
import json
import math
import random
import logging
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .relation_engine import (
    ALL_DIRECTIONS_10,
    CARDINAL_DIRECTIONS_8,
    DISTANCE_BINS,
    DISTANCE_BIN_IDS,
    DISTANCE_SURFACE_BARYCENTRICS_KEY,
    DISTANCE_SURFACE_POINTS_KEY,
    DISTANCE_SURFACE_TRIANGLE_IDS_KEY,
    DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY,
    HORIZONTAL_DIRECTIONS,
    MIN_DIRECTION_DISTANCE,
    compute_all_relations,
    find_changed_relations,
    primary_direction,
    primary_direction_object_centric,
    primary_direction_allocentric,
    camera_cardinal_direction,
    compute_distance_details,
)
from .support_graph import compute_bottom_footprint_overlap_metrics
from .virtual_ops import (
    MOVEMENT_CANDIDATES,
    apply_movement,
    apply_viewpoint_change,
    apply_removal,
    apply_coordinate_rotation,
    compute_room_bounds,
    find_meaningful_movement,
    find_meaningful_orbit_rotation,
    get_moved_object_ids,
    has_terminal_bbox_collision,
    is_within_room,
)
from .utils.colmap_loader import CameraPose
from .utils.colmap_loader import CameraIntrinsics
from .utils.coordinate_transform import is_in_image, project_to_image
from .utils.depth_occlusion import (
    MIN_IN_FRAME_RATIO,
    MIN_PROJECTED_AREA_PX,
    FULLY_VISIBLE_RATIO_MIN,
    PARTIALLY_VISIBLE_RATIO_MIN,
    compute_mesh_depth_occlusion,
    compute_mesh_depth_occlusion_metrics,
)
try:
    from .utils.ray_casting import (
        _LOCAL_BOUNDARY_RESAMPLE_COUNT,
        _classify_hit_path,
        _local_triangle_resamples,
    )
except ImportError:
    _HIT_PATH_MERGE_EPS = 1e-3
    _LOCAL_BOUNDARY_RESAMPLE_COUNT = 12
    _LOCAL_BOUNDARY_BLEND = 0.2

    def _compress_hit_path(
        hits: list[tuple[int, float]],
        target_tri_ids: set[int],
    ) -> list[tuple[bool, float]]:
        compressed: list[tuple[bool, float]] = []
        for tri_id, dist in hits:
            is_target = tri_id in target_tri_ids
            if (
                compressed
                and compressed[-1][0] == is_target
                and abs(compressed[-1][1] - float(dist)) <= _HIT_PATH_MERGE_EPS
            ):
                continue
            compressed.append((is_target, float(dist)))
        return compressed

    def _classify_hit_path(
        hits: list[tuple[int, float]],
        expected_dist: float,
        target_tri_ids: set[int],
        hit_epsilon: float,
    ) -> str:
        path = _compress_hit_path(hits, target_tri_ids)
        sample_hit_idx = next(
            (
                idx for idx, (is_target, dist) in enumerate(path)
                if is_target and abs(dist - float(expected_dist)) <= hit_epsilon
            ),
            None,
        )
        if sample_hit_idx is None:
            return "invalid"

        prior_hits = path[:sample_hit_idx]
        if not prior_hits:
            return "visible"
        if not prior_hits[0][0]:
            return "externally_occluded"
        if all(is_target for is_target, _ in prior_hits):
            return "self_occluded"
        return "mixed_boundary"

    def _local_triangle_resamples(
        triangle_vertices: np.ndarray,
        barycentric: np.ndarray,
        triangle_id: int,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        tri_vertices = np.asarray(triangle_vertices, dtype=np.float64)
        bary = np.asarray(barycentric, dtype=np.float64)
        if tri_vertices.shape != (3, 3) or bary.shape != (3,) or n_samples <= 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        bary_sum = float(np.sum(bary))
        if not np.isfinite(bary_sum) or bary_sum <= 1e-12:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        bary = np.clip(bary / bary_sum, 0.0, 1.0)
        bary = bary / max(float(np.sum(bary)), 1e-12)

        bary_seed = np.round(bary * 1_000_000.0).astype(np.int64)
        seed = zlib.crc32(
            np.asarray([int(triangle_id), *bary_seed.tolist()], dtype=np.int64).tobytes(),
        ) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        random_barys = rng.dirichlet(np.ones(3, dtype=np.float64), size=n_samples)
        local_barys = (
            (1.0 - _LOCAL_BOUNDARY_BLEND) * bary[None, :]
            + _LOCAL_BOUNDARY_BLEND * random_barys
        )
        local_barys = local_barys / np.maximum(local_barys.sum(axis=1, keepdims=True), 1e-12)
        local_points = local_barys @ tri_vertices
        return np.asarray(local_points, dtype=np.float64), np.asarray(local_barys, dtype=np.float64)
from .scene_parser import EXCLUDED_LABELS, InstanceMeshData

logger = logging.getLogger(__name__)

# Default template file; can be overridden
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

ALL_DIRECTIONS = ALL_DIRECTIONS_10
ALL_DIRECTIONS_ALLOCENTRIC = list(CARDINAL_DIRECTIONS_8)
ALL_DISTANCES = [label for _, label in DISTANCE_BINS]
L1_OCCLUSION_STATES = ["not occluded", "occluded", "not visible"]
OCCLUSION_DEFINITION_NOTE = (
    "Here, 'occluded' means blocked by another object; being partly outside "
    "the image frame does not count as occlusion."
)
YES_NO = ["Yes", "No"]
DISTANCE_MOVE_SEARCH_STEP_M = 0.1
DISTANCE_MOVE_SEARCH_MAX_M = 3.0
MIN_DISTANCE_QUESTION_DISTANCE_M = 0.2
DISTANCE_MOVE_DIRECTIONS = [
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
    np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
]

# Object-centric questions need a stable horizontal facing direction.
MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_DISTANCE = 0.3
MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_RATIO = 0.5
MIN_WALL_HEIGHT = 1.5
MIN_WALL_MAJOR_AXIS = 1.5
MAX_WALL_MINOR_AXIS = 1.0
MIN_WALL_AXIS_RATIO = 2.0
L1_OCCLUSION_MIN_IN_FRAME_RATIO = 0.05
L1_OCCLUSION_NOT_OCCLUDED_MAX = 0.02
L1_OCCLUSION_OCCLUDED_MIN = 0.10
L1_OCCLUSION_SAMPLE_COUNT = 512
L1_OCCLUSION_MIN_EFFECTIVE_COUNT = 64
L1_OCCLUSION_MIN_EFFECTIVE_RATIO = 0.25
HORIZONTAL_DIRECTION_OVERLAP_SUPPRESS_MIN = 0.10
VERTICAL_DIRECTIONS = {"above", "below"}


def _bbox_rect_xy(obj: dict) -> np.ndarray:
    bbox_min = np.asarray(obj.get("bbox_min", [0.0, 0.0, 0.0]), dtype=float)
    bbox_max = np.asarray(obj.get("bbox_max", [0.0, 0.0, 0.0]), dtype=float)
    return np.array([
        [bbox_min[0], bbox_min[1]],
        [bbox_max[0], bbox_min[1]],
        [bbox_max[0], bbox_max[1]],
        [bbox_min[0], bbox_max[1]],
    ], dtype=float)


def _object_bottom_hull_xy(obj: dict) -> np.ndarray:
    hull = np.asarray(obj.get("support_geom", {}).get("bottom_hull_xy", []), dtype=float)
    if hull.ndim == 2 and hull.shape[0] >= 3 and hull.shape[1] == 2:
        return hull
    return _bbox_rect_xy(obj)


def _translated_bottom_hull_xy(obj: dict, delta: np.ndarray) -> np.ndarray:
    return _object_bottom_hull_xy(obj) + np.asarray(delta, dtype=float)[:2]


def _translated_bbox(obj: dict, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offset = np.asarray(delta, dtype=float)
    return (
        np.asarray(obj["bbox_min"], dtype=float) + offset,
        np.asarray(obj["bbox_max"], dtype=float) + offset,
    )


def _build_attachment_edge_lookup(
    attachment_edges: list[dict[str, Any]] | None,
) -> dict[frozenset[int], dict[str, Any]]:
    lookup: dict[frozenset[int], dict[str, Any]] = {}
    for edge in attachment_edges or []:
        try:
            pair_key = frozenset((int(edge["parent_id"]), int(edge["child_id"])))
        except (KeyError, TypeError, ValueError):
            continue
        lookup[pair_key] = edge
    return lookup


def _direction_suppression_reason(
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    answer_direction: str,
    attachment_edge_lookup: dict[frozenset[int], dict[str, Any]] | None,
) -> tuple[str, str, dict[str, Any]] | None:
    if answer_direction in VERTICAL_DIRECTIONS:
        return None

    edge = None
    if attachment_edge_lookup:
        pair_key = frozenset((int(obj_a["id"]), int(obj_b["id"])))
        edge = attachment_edge_lookup.get(pair_key)
    if edge is not None:
        return (
            "attached_pair_non_vertical_direction",
            "directly attached object pairs only allow vertical direction questions",
            {
                "direction": answer_direction,
                "attachment_edge_type": str(edge.get("type", "unknown")),
                "attachment_parent_id": int(edge["parent_id"]),
                "attachment_child_id": int(edge["child_id"]),
            },
        )

    overlap_metrics = compute_bottom_footprint_overlap_metrics(obj_a, obj_b)
    if overlap_metrics["coverage_small"] >= HORIZONTAL_DIRECTION_OVERLAP_SUPPRESS_MIN:
        return (
            "horizontal_projection_overlap",
            "horizontal-answer direction questions require sufficiently separated floor-plane footprints",
            {
                "direction": answer_direction,
                "overlap_area": float(overlap_metrics["overlap_area"]),
                "coverage_a": float(overlap_metrics["coverage_a"]),
                "coverage_b": float(overlap_metrics["coverage_b"]),
                "coverage_small": float(overlap_metrics["coverage_small"]),
                "footprint_source_a": str(overlap_metrics["source_a"]),
                "footprint_source_b": str(overlap_metrics["source_b"]),
                "threshold": HORIZONTAL_DIRECTION_OVERLAP_SUPPRESS_MIN,
                "attachment_edge_type": None,
            },
        )
    return None


@dataclass(frozen=True)
class _ModifiedSceneContext:
    ray_caster: Any
    ignored_tri_ids: frozenset[int]


@dataclass(frozen=True)
class _L1OcclusionMetrics:
    projected_area: float
    in_frame_ratio: float
    occlusion_ratio_in_frame: float
    valid_in_frame_count: int
    sampled_point_count: int
    in_frame_sample_count: int
    effective_ratio: float
    sufficient_evidence: bool
    decision: str
    backend: str


@dataclass
class _SelectedObjectMoveState:
    delta: np.ndarray
    moved_objects: list[dict[str, Any]]
    moved_ids: set[int]
    changed_relations: list[dict[str, Any]]
    used_changed_delta: bool


def _instance_triangle_id_set(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> set[int]:
    if instance_mesh_data is None:
        return set()

    triangle_ids_by_instance = getattr(instance_mesh_data, "triangle_ids_by_instance", {}) or {}
    boundary_triangle_ids_by_instance = getattr(
        instance_mesh_data,
        "boundary_triangle_ids_by_instance",
        {},
    ) or {}
    tri_parts = [
        arr for arr in (
            triangle_ids_by_instance.get(int(obj_id)),
            boundary_triangle_ids_by_instance.get(int(obj_id)),
        )
        if arr is not None and len(arr) > 0
    ]
    if not tri_parts:
        return set()
    tri_ids = np.unique(np.concatenate(tri_parts).astype(np.int64))
    return {int(tid) for tid in tri_ids.tolist()}


def _instance_surface_samples(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> np.ndarray:
    if instance_mesh_data is None:
        return np.empty((0, 3), dtype=np.float64)
    surface_points_by_instance = getattr(instance_mesh_data, "surface_points_by_instance", {}) or {}
    samples = surface_points_by_instance.get(int(obj_id))
    if samples is None:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def _get_instance_intersector(
    instance_mesh_data: InstanceMeshData,
    obj_id: int,
) -> Any:
    """Build (or retrieve from cache) a per-instance RayCaster for *obj_id*.

    The sub-mesh is built from the instance's solid + boundary triangles using
    the global vertices/faces stored in *instance_mesh_data*.  The intersector
    operates in the original (un-translated) coordinate frame; callers must
    shift ray origins by ``-delta`` when querying translated objects.

    Returns ``None`` when the instance has no triangles.
    """
    cache: dict[int, Any] = instance_mesh_data.__dict__.setdefault(
        "_intersector_cache", {}
    )
    if obj_id in cache:
        return cache[obj_id]

    tri_ids_set = _instance_triangle_id_set(instance_mesh_data, obj_id)
    if not tri_ids_set:
        cache[obj_id] = None
        return None

    import trimesh
    from .utils.ray_casting import RayCaster

    tri_ids = np.array(sorted(tri_ids_set), dtype=np.int64)
    vertices = np.asarray(instance_mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)

    # Extract compact sub-mesh with remapped local face indices.
    sub_faces_global = faces[tri_ids]
    unique_vert_ids, remapped = np.unique(sub_faces_global, return_inverse=True)
    sub_vertices = vertices[unique_vert_ids]
    sub_faces_local = remapped.reshape(-1, 3).astype(np.int64)

    sub_mesh = trimesh.Trimesh(
        vertices=sub_vertices, faces=sub_faces_local, process=False
    )
    caster = RayCaster(sub_mesh)
    cache[obj_id] = caster
    return caster


def _instance_surface_sample_metadata(
    instance_mesh_data: InstanceMeshData | None,
    obj_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if instance_mesh_data is None:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    surface_triangle_ids_by_instance = getattr(
        instance_mesh_data,
        "surface_triangle_ids_by_instance",
        {},
    ) or {}
    surface_barycentrics_by_instance = getattr(
        instance_mesh_data,
        "surface_barycentrics_by_instance",
        {},
    ) or {}
    triangle_ids = surface_triangle_ids_by_instance.get(int(obj_id))
    barycentrics = surface_barycentrics_by_instance.get(int(obj_id))
    if triangle_ids is None or barycentrics is None:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(triangle_ids, dtype=np.int64),
        np.asarray(barycentrics, dtype=np.float64),
    )


def _clear_distance_geometry_fields(obj: dict[str, Any]) -> None:
    for key in (
        DISTANCE_SURFACE_POINTS_KEY,
        DISTANCE_SURFACE_TRIANGLE_IDS_KEY,
        DISTANCE_SURFACE_BARYCENTRICS_KEY,
        DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY,
    ):
        obj.pop(key, None)


def enrich_objects_with_distance_geometry(
    objects: list[dict[str, Any]],
    instance_mesh_data: InstanceMeshData | None,
) -> None:
    """Attach runtime-only surface samples used by closest-point distance GT."""
    if instance_mesh_data is None:
        for obj in objects:
            _clear_distance_geometry_fields(obj)
        return

    vertices = np.asarray(instance_mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)
    for obj in objects:
        _clear_distance_geometry_fields(obj)
        obj_id = int(obj["id"])
        surface_points = _instance_surface_samples(instance_mesh_data, obj_id)
        if len(surface_points) == 0:
            continue
        triangle_ids, barycentrics = _instance_surface_sample_metadata(instance_mesh_data, obj_id)
        obj[DISTANCE_SURFACE_POINTS_KEY] = np.asarray(surface_points, dtype=np.float64).copy()
        if len(triangle_ids) != len(surface_points) or len(barycentrics) != len(surface_points):
            continue
        valid_triangle_mask = (
            triangle_ids >= 0
        ) & (
            triangle_ids < len(faces)
        )
        if not np.all(valid_triangle_mask):
            continue
        triangle_vertices = vertices[faces[triangle_ids]]
        obj[DISTANCE_SURFACE_TRIANGLE_IDS_KEY] = np.asarray(triangle_ids, dtype=np.int64).copy()
        obj[DISTANCE_SURFACE_BARYCENTRICS_KEY] = np.asarray(barycentrics, dtype=np.float64).copy()
        obj[DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY] = np.asarray(triangle_vertices, dtype=np.float64).copy()


def _invoke_method_with_supported_kwargs(
    method: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Call *method* after dropping keyword args unsupported by older backends."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return method(**kwargs)

    supported = {
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in supported
    }
    return method(**filtered_kwargs)


def _mesh_visibility_stats_compat(
    ray_caster: Any,
    **kwargs: Any,
) -> tuple[int, int]:
    return _invoke_method_with_supported_kwargs(
        ray_caster.mesh_visibility_stats,
        **kwargs,
    )


def _mesh_visibility_ratio_compat(
    ray_caster: Any,
    **kwargs: Any,
) -> float:
    return float(
        _invoke_method_with_supported_kwargs(
            ray_caster.mesh_visibility_ratio,
            **kwargs,
        )
    )


def _make_l1_occlusion_metrics(
    projected_area: float,
    in_frame_ratio: float,
    occlusion_ratio_in_frame: float,
    valid_in_frame_count: int,
    sampled_point_count: int,
    in_frame_sample_count: int,
    backend: str,
) -> _L1OcclusionMetrics:
    effective_ratio = (
        float(valid_in_frame_count / in_frame_sample_count)
        if in_frame_sample_count > 0 else 0.0
    )
    sufficient_evidence = (
        valid_in_frame_count >= L1_OCCLUSION_MIN_EFFECTIVE_COUNT
        and effective_ratio >= L1_OCCLUSION_MIN_EFFECTIVE_RATIO
    )
    metrics = _L1OcclusionMetrics(
        projected_area=float(projected_area),
        in_frame_ratio=float(in_frame_ratio),
        occlusion_ratio_in_frame=float(occlusion_ratio_in_frame),
        valid_in_frame_count=int(valid_in_frame_count),
        sampled_point_count=int(sampled_point_count),
        in_frame_sample_count=int(in_frame_sample_count),
        effective_ratio=effective_ratio,
        sufficient_evidence=bool(sufficient_evidence),
        decision="skip",
        backend=str(backend),
    )
    return _L1OcclusionMetrics(
        projected_area=metrics.projected_area,
        in_frame_ratio=metrics.in_frame_ratio,
        occlusion_ratio_in_frame=metrics.occlusion_ratio_in_frame,
        valid_in_frame_count=metrics.valid_in_frame_count,
        sampled_point_count=metrics.sampled_point_count,
        in_frame_sample_count=metrics.in_frame_sample_count,
        effective_ratio=metrics.effective_ratio,
        sufficient_evidence=metrics.sufficient_evidence,
        decision=_classify_l1_occlusion_metrics(metrics),
        backend=metrics.backend,
    )


def _depth_backend_inputs_ready(
    *,
    depth_image,
    depth_intrinsics,
    ray_caster,
    context: str,
) -> bool:
    """Validate optional depth-evaluation inputs.

    Returning ``False`` means the caller intentionally has no depth evidence
    available and should skip the depth-based check. Partial configuration is
    treated as an error because it usually means the caller requested the
    depth backend but forgot one of the required resources.
    """
    has_depth_image = depth_image is not None
    has_depth_intrinsics = depth_intrinsics is not None
    if has_depth_image != has_depth_intrinsics:
        missing = "depth_image" if not has_depth_image else "depth_intrinsics"
        raise RuntimeError(
            f"Depth backend for {context} requires both depth_image and depth_intrinsics; missing {missing}.",
        )
    if not has_depth_image:
        return False
    if ray_caster is None:
        raise RuntimeError(
            f"Depth backend for {context} requires ray_caster to compute target-mesh entry depth.",
        )
    return True


def _the(label: str) -> str:
    """Prepend 'the' to an object label for natural English grammar.

    'shoes' → 'the shoes', 'table' → 'the table'.
    Avoids 'where is shoes positioned' type errors.
    """
    return f"the {label}"


def _mention(role: str, label: str, obj_id: int | None = None) -> dict[str, Any]:
    """Create a normalised mentioned-object record for a question."""
    return {"role": role, "obj_id": obj_id, "label": label}


def _has_stable_object_centric_facing(
    anchor_center: np.ndarray,
    facing_center: np.ndarray,
) -> bool:
    """Whether anchor->facing defines a reliable horizontal heading.

    Reject pairs that are almost vertically aligned, because "face toward X"
    then fails to define an unambiguous left/right/front/back frame.
    """
    delta = facing_center - anchor_center
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    total_distance = float(np.linalg.norm(delta))
    if horizontal_distance < MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_DISTANCE:
        return False
    if total_distance < 1e-6:
        return False
    if horizontal_distance / total_distance < MIN_OBJECT_CENTRIC_FACING_HORIZONTAL_RATIO:
        return False
    return True


def _invert_direction(direction: str) -> str:
    """Convert a direction of B relative to A into A relative to B."""
    opposites = {
        "front": "back",
        "front-right": "back-left",
        "right": "left",
        "back-right": "front-left",
        "back": "front",
        "back-left": "front-right",
        "left": "right",
        "front-left": "back-right",
        "above": "below",
        "below": "above",
        "north": "south",
        "northeast": "southwest",
        "east": "west",
        "southeast": "northwest",
        "south": "north",
        "southwest": "northeast",
        "west": "east",
        "northwest": "southeast",
    }
    return opposites.get(direction, direction)


def _direction_with_camera_hint(
    direction: str,
    moving_subject: str = "object",
) -> str:
    """Clarify forward/backward wording for camera-centric question text."""
    if moving_subject == "camera":
        if direction == "forward":
            return "forward (along its viewing direction)"
        if direction == "backward":
            return "backward (opposite its viewing direction)"
        return direction

    if direction == "forward":
        return "forward (away from the camera)"
    if direction == "backward":
        return "backward (toward the camera)"
    return direction


def _wall_image_side_phrase(side: str) -> str:
    if side == "left":
        return "on the left side of the image"
    if side == "right":
        return "on the right side of the image"
    if side == "top":
        return "near the top of the image"
    if side == "bottom":
        return "near the bottom of the image"
    return "in the image"


def _build_visible_wall_anchor(
    wall_objects: list[dict] | None,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
) -> dict[str, Any] | None:
    """Select one visible, axis-aligned wall to anchor allocentric wording."""
    if not wall_objects or color_intrinsics is None:
        return None

    candidates: list[dict[str, Any]] = []
    width = float(color_intrinsics.width)
    height = float(color_intrinsics.height)
    if width <= 0 or height <= 0:
        return None

    for wall in wall_objects:
        dims = np.array(wall.get("dimensions", [0.0, 0.0, 0.0]), dtype=float)
        if dims.shape[0] < 3:
            continue
        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
        if dz < MIN_WALL_HEIGHT:
            continue

        major_axis = max(dx, dy)
        minor_axis = min(dx, dy)
        if major_axis < MIN_WALL_MAJOR_AXIS or minor_axis > MAX_WALL_MINOR_AXIS:
            continue
        if minor_axis <= 1e-6:
            axis_ratio = float("inf")
        else:
            axis_ratio = major_axis / minor_axis
        if axis_ratio < MIN_WALL_AXIS_RATIO:
            continue

        if dy >= dx:
            wall_axis = "north_south"
            wall_axis_text = "north-south"
        else:
            wall_axis = "east_west"
            wall_axis_text = "east-west"

        center = np.array(wall["center"], dtype=float)
        uv, depth = project_to_image(center, camera_pose, color_intrinsics)
        if not (0.3 < depth <= 6.0):
            continue
        if not is_in_image(uv, color_intrinsics, margin=80):
            continue

        u = float(uv[0]) / width
        v = float(uv[1]) / height
        dx_img = u - 0.5
        dy_img = v - 0.5
        if abs(dx_img) >= abs(dy_img):
            image_side = "right" if dx_img >= 0 else "left"
            side_separation = abs(dx_img)
        else:
            image_side = "bottom" if dy_img >= 0 else "top"
            side_separation = abs(dy_img)

        side_phrase = _wall_image_side_phrase(image_side)
        note = (
            f"The wall {side_phrase} runs {wall_axis_text} on the floor plan."
        )
        candidates.append({
            "visible_wall_anchor_note": note,
            "anchor_wall_id": wall["id"],
            "anchor_wall_image_side": image_side,
            "anchor_wall_axis": wall_axis,
            "axis_ratio": float(axis_ratio),
            "side_separation": float(side_separation),
            "major_axis": float(major_axis),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (
            c["side_separation"],
            c["axis_ratio"],
            c["major_axis"],
        ),
        reverse=True,
    )
    best = candidates[0].copy()
    best.pop("axis_ratio", None)
    best.pop("side_separation", None)
    best.pop("major_axis", None)
    return best

_TEMPLATE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "L2_object_rotate_object_centric": ("L2_object_move_object_centric",),
}


def _normalize_template_aliases(templates: dict) -> dict:
    """Backfill template aliases so old and new names both work."""
    normalized = dict(templates)
    for canonical_key, alias_keys in _TEMPLATE_KEY_ALIASES.items():
        source_value = normalized.get(canonical_key)
        if source_value is None:
            for alias_key in alias_keys:
                if alias_key in normalized:
                    source_value = normalized[alias_key]
                    normalized[canonical_key] = source_value
                    break
        if source_value is None:
            continue
        for alias_key in alias_keys:
            normalized.setdefault(alias_key, source_value)
    return normalized


def _load_templates() -> dict:
    """Load question templates from the JSON file."""
    tpl_path = _TEMPLATE_DIR / "question_templates.json"
    if tpl_path.exists():
        with open(tpl_path, "r", encoding="utf-8") as f:
            return _normalize_template_aliases(json.load(f))
    # Fallback inline templates
    return _normalize_template_aliases(_default_templates())


def _default_templates() -> dict:
    return {
        # ==== L1 — Static perception ====

        # --- Ego-centric ---
        "L1_direction_agent": [
            "From the camera's viewpoint, {obj_a} is in which direction relative to {obj_b}?",
            "Looking at the scene from the camera's perspective, where is {obj_a} positioned relative to {obj_b}?",
            "From the current camera perspective, what is the spatial relationship of {obj_a} to {obj_b}?",
        ],
        "L1_distance": [
            "What is the approximate shortest distance between {obj_a} and {obj_b}, measured from their closest points?",
            "Measured from the closest points of each object, what is the approximate shortest distance between {obj_a} and {obj_b}?",
        ],
        "L1_occlusion": [
            f"What is the occlusion status of {{obj_a}} in the current view? {OCCLUSION_DEFINITION_NOTE}",
            f"From the current viewpoint, which best describes {{obj_a}}: not occluded, occluded, or not visible? {OCCLUSION_DEFINITION_NOTE}",
            f"In the current image, is {{obj_a}} unoccluded, occluded by another object, or not visible? {OCCLUSION_DEFINITION_NOTE}",
        ],

        # --- Object-centric ---
        "L1_direction_object_centric": [
            "Imagine you are {obj_ref} and facing toward {obj_face}. From your perspective, in which direction is {obj_target}?",
            "If you were {obj_ref}, looking toward {obj_face}, where would {obj_target} be?",
        ],

        # --- Allocentric ---
        "L1_direction_allocentric": [
            "The camera is facing {camera_cardinal} in this scene. On the room's floor plan, in which cardinal direction is {obj_a} from {obj_b}?",
            "In this image the camera faces {camera_cardinal}. Viewed from above on the room's layout, {obj_a} is in which cardinal direction relative to {obj_b}?",
        ],

        # ==== L2 — Intervention ====

        # --- Ego-centric (existing) ---
        "L2_object_move_agent": [
            "From the camera's perspective, imagine moving {obj_a} {direction_with_camera_hint} by {distance}. After this change, what is the relative position of {obj_b} to {obj_c}?",
            "From the camera's perspective, if we move {obj_a} {direction_with_camera_hint} by {distance}, where is {obj_b} relative to {obj_c}?",
        ],
        "L2_object_move_distance": [
            "From the camera's perspective, if {obj_a} is moved {direction_with_camera_hint} by {distance}, what is the approximate shortest distance between {obj_b} and {obj_c}, measured from their closest points?",
            "From the camera's perspective, imagine moving {obj_a} {direction_with_camera_hint} by {distance}. After this change, what is the approximate shortest distance between {obj_b} and {obj_c}, measured from their closest points?",
        ],
        "L2_object_move_occlusion": [
            f"From the camera's perspective, imagine moving {{obj_a}} {{direction_with_camera_hint}} by {{distance}}. After this change, what is the occlusion status of {{obj_b}}? {OCCLUSION_DEFINITION_NOTE}",
            f"From the camera's perspective, if {{obj_a}} is moved {{direction_with_camera_hint}} by {{distance}}, which best describes {{obj_b}}: not occluded, occluded, or not visible? {OCCLUSION_DEFINITION_NOTE}",
        ],
        "L2_viewpoint_move": [
            f"If the camera translates {{direction_with_camera_hint}} by {{distance}} while keeping its intrinsics and orientation unchanged, what is the occlusion status of {{obj_a}}? {OCCLUSION_DEFINITION_NOTE}",
            f"After the camera moves {{direction_with_camera_hint}} by {{distance}} without changing its viewing direction, which best describes {{obj_a}}: not occluded, occluded, or not visible? {OCCLUSION_DEFINITION_NOTE}",
            f"If the camera shifts {{direction_with_camera_hint}} by {{distance}} with no change in intrinsics or orientation, is {{obj_a}} not occluded, occluded, or not visible? {OCCLUSION_DEFINITION_NOTE}",
        ],
        "L2_object_remove": [
            f"If {{obj_a}} were removed from the scene, what would be the occlusion status of {{obj_b}} from the current viewpoint? {OCCLUSION_DEFINITION_NOTE}",
            f"After removing {{obj_a}}, which best describes {{obj_b}}: not occluded, occluded, or not visible? {OCCLUSION_DEFINITION_NOTE}",
        ],

        # --- Object-centric ---
        "L2_object_rotate_object_centric": [
            "Imagine you are {obj_query} and facing toward {obj_face}. If {obj_move_source} were moved along a {angle}-degree {rotation_direction} (viewed from above) orbit around the center of {obj_face}, without changing its own facing direction, from your perspective, in which direction would {obj_ref} be?",
        ],
        "L2_object_move_object_centric": [
            "Imagine you are {obj_query} and facing toward {obj_face}. If {obj_move_source} were moved along a {angle}-degree {rotation_direction} (viewed from above) orbit around the center of {obj_face}, without changing its own facing direction, from your perspective, in which direction would {obj_ref} be?",
        ],

        # --- Allocentric ---
        "L2_object_move_allocentric": [
            "If {obj_move_source} is moved {distance} to the {direction}, the camera faces {camera_cardinal}. On the floor plan, in which cardinal direction would {obj_query} be from {obj_ref}?",
            "After moving {obj_move_source} {distance} to the {direction}, on the room's layout (camera facing {camera_cardinal}), in which cardinal direction is {obj_query} from {obj_ref}?",
        ],

        # ==== L3 — Counterfactual / multi-hop ====

        "L3_attachment_chain": [
            "Suppose {obj_a} were moved to a different location. Which of the following objects would also be displaced from their current positions?",
            "If {obj_a} were relocated elsewhere in the room, which of the following objects would also change position?",
            "Imagine {obj_a} is moved to a new spot. Which of the following objects would also be displaced as a result?",
        ],

        # --- Ego-centric (rewritten — 方案B) ---
        "L3_coordinate_rotation_agent": [
            "Suppose this room had originally been designed with its orientation rotated {angle} degrees clockwise (viewed from above), with all objects keeping their relative positions. Observed from the original camera position and viewing direction (unchanged), in which direction is {obj_a} relative to {obj_b}?",
            "If the room layout had been rotated {angle} degrees clockwise (top-down view) from the start, with all relative object positions preserved and camera position and orientation unchanged, from the camera's perspective, where would {obj_a} be relative to {obj_b}?",
            "Imagine the room was originally built rotated {angle} degrees clockwise (as seen from above). With all inter-object relationships intact and the camera at its original pose, from the camera's perspective, what is the direction of {obj_a} from {obj_b}?",
        ],

        # --- Object-centric ---
        "L3_coordinate_rotation_object_centric": [
            "Suppose this room had originally been oriented {angle} degrees clockwise (viewed from above), with all objects keeping their relative positions. If you were {obj_ref} at its rotated position and faced toward {obj_face}'s rotated position, in which direction would {obj_target} be?",
        ],

        # --- Allocentric ---
        "L3_coordinate_rotation_allocentric": [
            "Imagine all furniture is rotated {angle} degrees clockwise around the room center (viewed from above). The camera, facing {camera_cardinal}, remains in place. On the floor plan, in which cardinal direction is {obj_a} from {obj_b}?",
        ],
    }


def generate_options(
    correct_answer: str,
    answer_pool: list[str],
    n_options: int = 4,
) -> tuple[list[str], str]:
    """Generate MCQ options from an answer pool.

    When the correct answer is a vertical direction (above/below), horizontal
    directions are excluded from distractors and vice versa — this prevents
    distractors that are also plausibly correct (e.g. an object that is
    "above" is also "front" if it happens to be slightly forward).

    Returns (shuffled_options, correct_letter).
    """
    VERTICAL = {"above", "below"}
    HORIZONTAL = set(HORIZONTAL_DIRECTIONS)  # front/back/left/right/…

    HORIZONTAL.update(CARDINAL_DIRECTIONS_8)
    if correct_answer in VERTICAL:
        exclude = HORIZONTAL
    elif correct_answer in HORIZONTAL:
        exclude = VERTICAL
    else:
        exclude = set()

    options = [correct_answer]
    distractors = [a for a in answer_pool if a != correct_answer and a not in exclude]
    random.shuffle(distractors)
    options.extend(distractors[: n_options - 1])

    # If the strict pool is too small, fall back to the excluded pool before
    # introducing a single "None of the above" option.
    if len(options) < n_options:
        fallback = [
            a for a in answer_pool
            if a != correct_answer
            and a not in options
            and (not exclude or a in exclude)
        ]
        random.shuffle(fallback)
        options.extend(fallback[: n_options - len(options)])

    if len(options) < n_options and "None of the above" not in options:
        options.append("None of the above")

    has_none_of_above = "None of the above" in options
    shuffled = [opt for opt in options if opt != "None of the above"]
    random.shuffle(shuffled)
    if has_none_of_above:
        shuffled.append("None of the above")

    options = shuffled
    correct_idx = options.index(correct_answer)
    correct_letter = chr(65 + correct_idx)  # A, B, C, D
    return options, correct_letter


def _adjacent_directions(direction: str, ordered_directions: list[str]) -> set[str]:
    if direction not in ordered_directions:
        return set()

    idx = ordered_directions.index(direction)
    return {
        ordered_directions[(idx - 1) % len(ordered_directions)],
        ordered_directions[(idx + 1) % len(ordered_directions)],
    }


def _direction_distractor_exclusions(
    correct_answer: str,
    answer_pool: list[str],
    *,
    horizontal_context: str | None = None,
) -> set[str]:
    horizontal_ring: list[str] = []
    pool_set = set(answer_pool)

    if correct_answer in HORIZONTAL_DIRECTIONS or horizontal_context in HORIZONTAL_DIRECTIONS:
        horizontal_ring = list(HORIZONTAL_DIRECTIONS)
    elif correct_answer in CARDINAL_DIRECTIONS_8 or horizontal_context in CARDINAL_DIRECTIONS_8:
        horizontal_ring = list(CARDINAL_DIRECTIONS_8)
    elif pool_set & set(HORIZONTAL_DIRECTIONS):
        horizontal_ring = list(HORIZONTAL_DIRECTIONS)
    elif pool_set & set(CARDINAL_DIRECTIONS_8):
        horizontal_ring = list(CARDINAL_DIRECTIONS_8)

    if correct_answer in horizontal_ring:
        return _adjacent_directions(correct_answer, horizontal_ring)

    if correct_answer in {"above", "below"} and horizontal_context in horizontal_ring:
        return {horizontal_context} | _adjacent_directions(horizontal_context, horizontal_ring)

    return set()


def generate_direction_options(
    correct_answer: str,
    answer_pool: list[str],
    *,
    horizontal_context: str | None = None,
    n_options: int = 4,
) -> tuple[list[str], str]:
    """Generate direction options while excluding adjacent/confusable directions."""
    exclude = _direction_distractor_exclusions(
        correct_answer,
        answer_pool,
        horizontal_context=horizontal_context,
    )
    options = [correct_answer]
    distractors = [a for a in answer_pool if a != correct_answer and a not in exclude]
    random.shuffle(distractors)
    options.extend(distractors[: n_options - 1])

    if len(options) < n_options and "None of the above" not in options:
        options.append("None of the above")

    has_none_of_above = "None of the above" in options
    shuffled = [opt for opt in options if opt != "None of the above"]
    random.shuffle(shuffled)
    if has_none_of_above:
        shuffled.append("None of the above")

    options = shuffled
    correct_idx = options.index(correct_answer)
    correct_letter = chr(65 + correct_idx)  # A, B, C, D
    return options, correct_letter


# ---------------------------------------------------------------------------
#  L1 generators
# ---------------------------------------------------------------------------

def generate_l1_direction(
    relation: dict,
    templates: dict,
    *,
    obj_a: dict[str, Any] | None = None,
    obj_b: dict[str, Any] | None = None,
    attachment_edge_lookup: dict[frozenset[int], dict[str, Any]] | None = None,
) -> dict | None:
    """Generate an L1-direction question from a precomputed relation."""
    if relation["ambiguity_score"] > 0.7:
        return None  # too ambiguous
    if relation["distance_m"] < MIN_DIRECTION_DISTANCE:
        return None  # too close — annotation errors dominate
    if relation["obj_a_label"] == relation["obj_b_label"]:
        return None  # same label → "chair relative to chair" is meaningless

    correct = relation["direction_b_rel_a"]
    if obj_a is not None and obj_b is not None:
        suppression = _direction_suppression_reason(
            obj_a,
            obj_b,
            str(correct),
            attachment_edge_lookup,
        )
        if suppression is not None:
            return None
    tpl_list = templates.get(
        "L1_direction_agent",
        templates.get("L1_direction", _default_templates()["L1_direction_agent"]),
    )
    tpl = random.choice(tpl_list)
    question_text = tpl.format(
        obj_a=_the(relation["obj_b_label"]),  # "where is B relative to A?"
        obj_b=_the(relation["obj_a_label"]),
    )
    options, answer = generate_direction_options(
        correct,
        ALL_DIRECTIONS,
        horizontal_context=relation.get("horizontal_direction_b_rel_a"),
    )

    return {
        "level": "L1",
        "type": "direction_agent",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": relation["obj_a_id"],
        "obj_b_id": relation["obj_b_id"],
        "obj_a_label": relation["obj_a_label"],
        "obj_b_label": relation["obj_b_label"],
        "mentioned_objects": [
            _mention("reference", relation["obj_a_label"], relation["obj_a_id"]),
            _mention("target", relation["obj_b_label"], relation["obj_b_id"]),
        ],
        "ambiguity_score": relation["ambiguity_score"],
        "relation_unchanged": False,
    }


def generate_l1_distance(
    relation: dict,
    templates: dict,
) -> dict | None:
    """Generate an L1-distance question."""
    if _relation_distance_for_distance_questions(relation) < MIN_DISTANCE_QUESTION_DISTANCE_M:
        return None
    if relation["near_boundary"]:
        return None
    if relation["obj_a_label"] == relation["obj_b_label"]:
        return None  # same label → ambiguous question

    correct = relation["distance_bin"]
    tpl = random.choice(templates.get("L1_distance", _default_templates()["L1_distance"]))
    question_text = tpl.format(
        obj_a=_the(relation["obj_a_label"]),
        obj_b=_the(relation["obj_b_label"]),
    )
    options, answer = generate_options(correct, ALL_DISTANCES)

    return {
        "level": "L1",
        "type": "distance",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "distance_m": float(relation.get("distance_m", 0.0)),
        "distance_bin_id": relation.get("distance_bin_id"),
        "distance_definition": relation.get("distance_definition"),
        "obj_a_id": relation["obj_a_id"],
        "obj_b_id": relation["obj_b_id"],
        "obj_a_label": relation["obj_a_label"],
        "obj_b_label": relation["obj_b_label"],
        "mentioned_objects": [
            _mention("obj_a", relation["obj_a_label"], relation["obj_a_id"]),
            _mention("obj_b", relation["obj_b_label"], relation["obj_b_id"]),
        ],
        "ambiguity_score": 0.0,
        "near_boundary": relation["near_boundary"],
        "relation_unchanged": False,
    }


def _normalize_label_counts(label_counts: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(label_counts, dict):
        return {}

    normalized: dict[str, int] = {}
    for label, count in label_counts.items():
        if not isinstance(label, str):
            continue
        label_text = label.strip().lower()
        if not label_text or label_text in EXCLUDED_LABELS:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int < 0:
            continue
        normalized[label_text] = count_int
    return normalized


def _normalize_label_statuses(label_statuses: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(label_statuses, dict):
        return {}

    normalized: dict[str, str] = {}
    for label, status in label_statuses.items():
        if not isinstance(label, str):
            continue
        label_text = label.strip().lower()
        if not label_text or label_text in EXCLUDED_LABELS:
            continue
        status_text = str(status or "").strip().lower()
        if status_text not in {"absent", "unique", "multiple", "unsure"}:
            continue
        normalized[label_text] = status_text
    return normalized


def _l1_occlusion_question(
    label: str,
    correct: str,
    templates: dict,
    obj_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tpl = random.choice(templates.get("L1_occlusion", _default_templates()["L1_occlusion"]))
    question_text = _with_occlusion_definition(tpl.format(obj_a=_the(label)))
    options, answer = generate_options(correct, L1_OCCLUSION_STATES, n_options=3)
    question = {
        "level": "L1",
        "type": "occlusion",
        "question": question_text,
        "options": options,
        "answer": answer,
        "correct_value": correct,
        "obj_a_id": obj_id,
        "obj_a_label": label,
        "mentioned_objects": [_mention("target", label, obj_id)],
        "ambiguity_score": 0.0,
        "relation_unchanged": False,
    }
    if extra:
        question.update(extra)
    return question


def _with_occlusion_definition(question_text: str) -> str:
    if "does not count as occlusion" in question_text.lower():
        return question_text
    return f"{question_text} {OCCLUSION_DEFINITION_NOTE}"


def _project_sample_point_records(
    sample_points: np.ndarray,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
) -> list[dict[str, float | int | bool]]:
    records: list[dict[str, float | int | bool]] = []
    for idx, pt in enumerate(sample_points):
        uv, depth = project_to_image(pt, camera_pose, color_intrinsics)
        if uv is None or depth <= 0:
            continue
        u = float(uv[0])
        v = float(uv[1])
        records.append({
            "index": idx,
            "u": u,
            "v": v,
            "depth": float(depth),
            "in_frame": (
                0 <= u < color_intrinsics.width
                and 0 <= v < color_intrinsics.height
            ),
        })
    return records


def _projected_area_from_records(
    projected_records: list[dict[str, float | int | bool]],
    color_intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    if not projected_records:
        return 0.0, 0.0

    us = [float(rec["u"]) for rec in projected_records]
    vs = [float(rec["v"]) for rec in projected_records]
    u_min = max(0.0, min(us))
    v_min = max(0.0, min(vs))
    u_max = min(float(color_intrinsics.width), max(us))
    v_max = min(float(color_intrinsics.height), max(vs))
    area = max(0.0, u_max - u_min) * max(0.0, v_max - v_min)
    in_frame_count = sum(1 for rec in projected_records if bool(rec["in_frame"]))
    return float(area), float(in_frame_count / len(projected_records))


def _in_frame_surface_sample_subset(
    sample_points: np.ndarray,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    sample_triangle_ids: np.ndarray | None = None,
    sample_barycentrics: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(sample_points, dtype=np.float64)
    if len(points) == 0:
        return (
            0.0,
            0.0,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    projected_records = _project_sample_point_records(
        points,
        camera_pose,
        color_intrinsics,
    )
    projected_area, in_frame_ratio = _projected_area_from_records(
        projected_records,
        color_intrinsics,
    )
    in_frame_records = [rec for rec in projected_records if bool(rec["in_frame"])]
    if not in_frame_records:
        return (
            projected_area,
            in_frame_ratio,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    in_frame_indices = np.asarray(
        [int(rec["index"]) for rec in in_frame_records],
        dtype=np.int64,
    )
    in_frame_points = points[in_frame_indices]
    in_frame_triangle_ids = (
        np.asarray(sample_triangle_ids, dtype=np.int64)[in_frame_indices]
        if sample_triangle_ids is not None and len(sample_triangle_ids) == len(points)
        else np.empty((0,), dtype=np.int64)
    )
    in_frame_barycentrics = (
        np.asarray(sample_barycentrics, dtype=np.float64)[in_frame_indices]
        if sample_barycentrics is not None and len(sample_barycentrics) == len(points)
        else np.empty((0, 3), dtype=np.float64)
    )
    return (
        projected_area,
        in_frame_ratio,
        in_frame_points,
        in_frame_triangle_ids,
        in_frame_barycentrics,
    )


def _classify_l1_occlusion_metrics(metrics: _L1OcclusionMetrics) -> str:
    if (
        metrics.projected_area < MIN_PROJECTED_AREA_PX
        or metrics.in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or metrics.in_frame_sample_count <= 0
    ):
        return "skip"
    if (
        metrics.valid_in_frame_count <= 0
        or not metrics.sufficient_evidence
    ):
        return "skip"
    if metrics.occlusion_ratio_in_frame < L1_OCCLUSION_NOT_OCCLUDED_MAX:
        return "not occluded"
    if metrics.occlusion_ratio_in_frame > L1_OCCLUSION_OCCLUDED_MIN:
        return "occluded"
    return "grayzone"


def _is_l1_occlusion_frame_skip(metrics: _L1OcclusionMetrics) -> bool:
    return (
        metrics.projected_area < MIN_PROJECTED_AREA_PX
        or metrics.in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or metrics.in_frame_sample_count <= 0
    )


def _resolve_l1_occlusion_decision(
    *,
    metrics: _L1OcclusionMetrics,
    source_used: str,
    grayzone_fallback_source: str,
) -> tuple[str | None, str, bool]:
    decision = metrics.decision
    overlay_available = False
    if decision == "skip":
        return None, source_used, overlay_available

    if decision == "grayzone":
        return None, grayzone_fallback_source, overlay_available

    return decision, source_used, overlay_available


def _compute_l1_occlusion_metrics(
    obj: dict[str, Any],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> _L1OcclusionMetrics:
    backend = str(occlusion_backend)
    if color_intrinsics is None:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=0,
            in_frame_sample_count=0,
            backend=backend,
        )

    obj_id = int(obj["id"])
    sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
    sample_triangle_ids, sample_barycentrics = _instance_surface_sample_metadata(
        instance_mesh_data,
        obj_id,
    )
    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, obj_id)
    sampled_point_count = int(len(sample_points))
    if sampled_point_count <= 0 or not target_tri_ids:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=0,
            backend=backend,
        )

    projected_records = _project_sample_point_records(
        sample_points, camera_pose, color_intrinsics,
    )
    projected_area, in_frame_ratio = _projected_area_from_records(
        projected_records, color_intrinsics,
    )
    in_frame_records = [rec for rec in projected_records if bool(rec["in_frame"])]
    in_frame_sample_count = len(in_frame_records)
    if (
        projected_area < MIN_PROJECTED_AREA_PX
        or in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or in_frame_sample_count <= 0
    ):
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    in_frame_indices = np.asarray(
        [int(rec["index"]) for rec in in_frame_records],
        dtype=np.int64,
    )
    in_frame_points = sample_points[in_frame_indices]
    in_frame_triangle_ids = (
        sample_triangle_ids[in_frame_indices]
        if len(sample_triangle_ids) == sampled_point_count
        else np.empty((0,), dtype=np.int64)
    )
    in_frame_barycentrics = (
        sample_barycentrics[in_frame_indices]
        if len(sample_barycentrics) == sampled_point_count
        else np.empty((0, 3), dtype=np.float64)
    )

    if backend == "depth":
        if not _depth_backend_inputs_ready(
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            ray_caster=ray_caster,
            context="L1 occlusion",
        ):
            return _make_l1_occlusion_metrics(
                projected_area=projected_area,
                in_frame_ratio=in_frame_ratio,
                occlusion_ratio_in_frame=1.0,
                valid_in_frame_count=0,
                sampled_point_count=sampled_point_count,
                in_frame_sample_count=in_frame_sample_count,
                backend=backend,
            )
        depth_metrics = compute_mesh_depth_occlusion_metrics(
            # Share the same color-in-frame sample pool with mesh_ray so the
            # evidence threshold uses the same denominator across backends.
            target_points=in_frame_points,
            target_tri_ids=target_tri_ids,
            camera_pose=camera_pose,
            intrinsics=depth_intrinsics,
            depth_image=depth_image,
            ray_caster=ray_caster,
        )
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=float(depth_metrics["occlusion_ratio_in_frame"]),
            valid_in_frame_count=int(depth_metrics["valid_in_frame_count"]),
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    if backend == "mesh_ray" and ray_caster is not None:
        camera_pos = np.asarray(camera_pose.position, dtype=np.float64)
        visible_count, valid_count = _mesh_visibility_stats_compat(
            ray_caster,
            camera_pos=camera_pos,
            target_points=in_frame_points,
            target_tri_ids=target_tri_ids,
            sample_triangle_ids=in_frame_triangle_ids,
            sample_barycentrics=in_frame_barycentrics,
            vertices=(
                np.asarray(instance_mesh_data.vertices, dtype=np.float64)
                if instance_mesh_data is not None else None
            ),
            faces=(
                np.asarray(instance_mesh_data.faces, dtype=np.int64)
                if instance_mesh_data is not None else None
            ),
        )
        occlusion_ratio = 1.0
        if valid_count > 0:
            occlusion_ratio = float(1.0 - (visible_count / valid_count))
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=occlusion_ratio,
            valid_in_frame_count=valid_count,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    return _make_l1_occlusion_metrics(
        projected_area=projected_area,
        in_frame_ratio=in_frame_ratio,
        occlusion_ratio_in_frame=1.0,
        valid_in_frame_count=0,
        sampled_point_count=sampled_point_count,
        in_frame_sample_count=in_frame_sample_count,
        backend=backend,
    )


def _l1_occlusion_metrics_payload(metrics: _L1OcclusionMetrics) -> dict[str, Any]:
    return {
        "projected_area": float(metrics.projected_area),
        "in_frame_ratio": float(metrics.in_frame_ratio),
        "occlusion_ratio_in_frame": float(metrics.occlusion_ratio_in_frame),
        "valid_in_frame_count": int(metrics.valid_in_frame_count),
        "sampled_point_count": int(metrics.sampled_point_count),
        "in_frame_sample_count": int(metrics.in_frame_sample_count),
        "effective_ratio": float(metrics.effective_ratio),
        "sufficient_evidence": bool(metrics.sufficient_evidence),
        "decision": str(metrics.decision),
        "backend": str(metrics.backend),
    }


def _compute_mesh_ray_l1_occlusion_metrics_for_static_target(
    *,
    obj: dict[str, Any],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    modified_scene: _ModifiedSceneContext | None = None,
) -> _L1OcclusionMetrics:
    backend = "mesh_ray"
    if (
        color_intrinsics is None
        or ray_caster is None
        or instance_mesh_data is None
    ):
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=0,
            in_frame_sample_count=0,
            backend=backend,
        )

    obj_id = int(obj["id"])
    sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
    sample_triangle_ids, sample_barycentrics = _instance_surface_sample_metadata(
        instance_mesh_data,
        obj_id,
    )
    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, obj_id)
    sampled_point_count = int(len(sample_points))
    if sampled_point_count <= 0 or not target_tri_ids:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=0,
            backend=backend,
        )

    (
        projected_area,
        in_frame_ratio,
        in_frame_points,
        in_frame_triangle_ids,
        in_frame_barycentrics,
    ) = _in_frame_surface_sample_subset(
        sample_points,
        camera_pose,
        color_intrinsics,
        sample_triangle_ids=sample_triangle_ids,
        sample_barycentrics=sample_barycentrics,
    )
    in_frame_sample_count = int(len(in_frame_points))
    if (
        projected_area < MIN_PROJECTED_AREA_PX
        or in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or in_frame_sample_count <= 0
    ):
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    if modified_scene is None:
        modified_scene = _build_modified_scene(
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            removed_ids=set(),
        )
    if modified_scene is None or modified_scene.ray_caster is None:
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    visible_count, valid_count = _mesh_visibility_stats_compat(
        modified_scene.ray_caster,
        camera_pos=np.asarray(camera_pose.position, dtype=np.float64),
        target_points=in_frame_points,
        target_tri_ids=target_tri_ids,
        ignored_tri_ids=set(modified_scene.ignored_tri_ids),
        sample_triangle_ids=in_frame_triangle_ids,
        sample_barycentrics=in_frame_barycentrics,
        vertices=np.asarray(instance_mesh_data.vertices, dtype=np.float64),
        faces=np.asarray(instance_mesh_data.faces, dtype=np.int64),
    )
    occlusion_ratio = 1.0
    if valid_count > 0:
        occlusion_ratio = float(1.0 - (visible_count / valid_count))
    return _make_l1_occlusion_metrics(
        projected_area=projected_area,
        in_frame_ratio=in_frame_ratio,
        occlusion_ratio_in_frame=occlusion_ratio,
        valid_in_frame_count=int(valid_count),
        sampled_point_count=sampled_point_count,
        in_frame_sample_count=in_frame_sample_count,
        backend=backend,
    )


def _compute_mesh_ray_l1_occlusion_metrics_for_moved_target(
    *,
    target_obj_id: int,
    original_objects: list[dict[str, Any]],
    moved_objects: list[dict[str, Any]],
    moved_ids: set[int],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> _L1OcclusionMetrics:
    backend = "mesh_ray"
    if (
        color_intrinsics is None
        or ray_caster is None
        or instance_mesh_data is None
    ):
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=0,
            in_frame_sample_count=0,
            backend=backend,
        )

    target_obj_id = int(target_obj_id)
    original_map = {int(obj["id"]): obj for obj in original_objects}
    moved_map = {int(obj["id"]): obj for obj in moved_objects}
    moved_target = moved_map.get(target_obj_id)
    if moved_target is None:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=0,
            in_frame_sample_count=0,
            backend=backend,
        )

    sample_points = _instance_surface_samples(instance_mesh_data, target_obj_id)
    sample_triangle_ids, sample_barycentrics = _instance_surface_sample_metadata(
        instance_mesh_data,
        target_obj_id,
    )
    target_tri_ids = _instance_triangle_id_set(instance_mesh_data, target_obj_id)
    sampled_point_count = int(len(sample_points))
    if sampled_point_count <= 0 or not target_tri_ids:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=0,
            backend=backend,
        )

    moved_scene_context = _build_modified_scene(
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
        removed_ids=set(moved_ids),
    )
    if moved_scene_context is None or moved_scene_context.ray_caster is None:
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=0,
            backend=backend,
        )

    moved_deltas: dict[int, np.ndarray] = {}
    for moved_obj in moved_objects:
        moved_id = int(moved_obj["id"])
        if moved_id not in moved_ids:
            continue
        original_obj = original_map.get(moved_id)
        if original_obj is None:
            continue
        moved_deltas[moved_id] = (
            np.asarray(moved_obj["center"], dtype=np.float64)
            - np.asarray(original_obj["center"], dtype=np.float64)
        )

    target_delta = moved_deltas.get(target_obj_id)
    adjusted_sample_points = np.asarray(sample_points, dtype=np.float64)
    if target_delta is not None:
        adjusted_sample_points = adjusted_sample_points + target_delta

    (
        projected_area,
        in_frame_ratio,
        in_frame_points,
        in_frame_triangle_ids,
        in_frame_barycentrics,
    ) = _in_frame_surface_sample_subset(
        adjusted_sample_points,
        camera_pose,
        color_intrinsics,
        sample_triangle_ids=sample_triangle_ids,
        sample_barycentrics=sample_barycentrics,
    )
    in_frame_sample_count = int(len(in_frame_points))
    if (
        projected_area < MIN_PROJECTED_AREA_PX
        or in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO
        or in_frame_sample_count <= 0
    ):
        return _make_l1_occlusion_metrics(
            projected_area=projected_area,
            in_frame_ratio=in_frame_ratio,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=sampled_point_count,
            in_frame_sample_count=in_frame_sample_count,
            backend=backend,
        )

    visible_count, valid_count = _compute_counterfactual_target_visibility_stats(
        modified_scene=moved_scene_context,
        target_surface_points=in_frame_points,
        target_triangle_ids=target_tri_ids,
        camera_pos=np.asarray(camera_pose.position, dtype=np.float64),
        instance_mesh_data=instance_mesh_data,
        target_obj_id=target_obj_id,
        target_delta=target_delta,
        moved_blocker_deltas={
            moved_id: delta
            for moved_id, delta in moved_deltas.items()
            if moved_id != target_obj_id
        },
        sample_triangle_ids=in_frame_triangle_ids,
        sample_barycentrics=in_frame_barycentrics,
        vertices=np.asarray(instance_mesh_data.vertices, dtype=np.float64),
        faces=np.asarray(instance_mesh_data.faces, dtype=np.int64),
    )
    occlusion_ratio = 1.0
    if valid_count > 0:
        occlusion_ratio = float(1.0 - (visible_count / valid_count))
    return _make_l1_occlusion_metrics(
        projected_area=projected_area,
        in_frame_ratio=in_frame_ratio,
        occlusion_ratio_in_frame=occlusion_ratio,
        valid_in_frame_count=int(valid_count),
        sampled_point_count=sampled_point_count,
        in_frame_sample_count=in_frame_sample_count,
        backend=backend,
    )


def _compute_l1_style_visibility_metrics_for_static_target(
    *,
    obj: dict[str, Any],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    modified_scene: _ModifiedSceneContext | None = None,
) -> tuple[_L1OcclusionMetrics, str]:
    backend = str(occlusion_backend)
    if backend not in {"depth", "mesh_ray"}:
        raise ValueError(f"Unsupported occlusion backend: {backend}")

    if backend == "depth" and modified_scene is None:
        return (
            _compute_l1_occlusion_metrics(
                obj=obj,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                occlusion_backend="depth",
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            ),
            "depth",
        )

    return (
        _compute_mesh_ray_l1_occlusion_metrics_for_static_target(
            obj=obj,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            modified_scene=modified_scene,
        ),
        "mesh_ray" if backend == "mesh_ray" else backend,
    )


def _compute_l1_style_visibility_metrics_for_moved_target(
    *,
    target_obj_id: int,
    original_objects: list[dict[str, Any]],
    moved_objects: list[dict[str, Any]],
    moved_ids: set[int],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> tuple[_L1OcclusionMetrics, str]:
    _ = str(occlusion_backend)
    return (
        _compute_mesh_ray_l1_occlusion_metrics_for_moved_target(
            target_obj_id=target_obj_id,
            original_objects=original_objects,
            moved_objects=moved_objects,
            moved_ids=moved_ids,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        ),
        "mesh_ray",
    )


def _resolve_counterfactual_l1_visibility_status(
    metrics: _L1OcclusionMetrics,
) -> tuple[str | None, str, str]:
    if (
        metrics.projected_area <= 0.0
        or metrics.in_frame_sample_count <= 0
        or metrics.in_frame_ratio <= 0.0
    ):
        return (
            "not visible",
            "not_visible_out_of_frame",
            "object falls outside the image after applying L1 occlusion framing",
        )
    if metrics.projected_area < MIN_PROJECTED_AREA_PX:
        return (
            None,
            "projected_area_too_small",
            "projected footprint is too small for reliable L1-style occlusion judgement",
        )
    if metrics.in_frame_ratio < L1_OCCLUSION_MIN_IN_FRAME_RATIO:
        return (
            None,
            "in_frame_ratio_too_small",
            "too little of the sampled surface remains in-frame for L1-style occlusion judgement",
        )
    if metrics.valid_in_frame_count <= 0:
        return (
            None,
            "no_valid_visibility_samples",
            "no valid in-frame mesh samples survived occlusion evaluation",
        )
    if not metrics.sufficient_evidence:
        return (
            None,
            "insufficient_visibility_evidence",
            "effective in-frame evidence is below the L1 occlusion threshold",
        )
    if metrics.decision == "grayzone":
        return (
            None,
            "grayzone_visibility",
            "occlusion ratio falls into the L1 grayzone and is intentionally not trusted",
        )
    if metrics.decision in {"not occluded", "occluded"}:
        return (
            metrics.decision,
            "resolved_visibility",
            "visibility state is resolved under L1 occlusion thresholds",
        )
    return (
        None,
        "unresolved_visibility_decision",
        f"unexpected L1-style visibility decision: {metrics.decision}",
    )


def _render_instance_projection_mask(
    obj_id: int,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
    instance_mesh_data: InstanceMeshData | None,
) -> np.ndarray | None:
    if instance_mesh_data is None:
        return None
    tri_ids = _instance_triangle_id_set(instance_mesh_data, int(obj_id))
    if not tri_ids:
        return None

    import cv2

    mask = np.zeros(
        (int(color_intrinsics.height), int(color_intrinsics.width)),
        dtype=np.uint8,
    )
    vertices = np.asarray(instance_mesh_data.vertices, dtype=np.float64)
    faces = np.asarray(instance_mesh_data.faces, dtype=np.int64)
    for tri_id in sorted(tri_ids):
        tri_vertices = vertices[faces[int(tri_id)]]
        projected: list[list[int]] = []
        invalid = False
        for vertex in tri_vertices:
            uv, depth = project_to_image(vertex, camera_pose, color_intrinsics)
            if uv is None or depth <= 0:
                invalid = True
                break
            projected.append([int(round(float(uv[0]))), int(round(float(uv[1])))])
        if invalid or len(projected) != 3:
            continue
        xs = [pt[0] for pt in projected]
        ys = [pt[1] for pt in projected]
        if max(xs) < 0 or max(ys) < 0:
            continue
        if min(xs) >= color_intrinsics.width or min(ys) >= color_intrinsics.height:
            continue
        cv2.fillConvexPoly(mask, np.asarray(projected, dtype=np.int32), 255)

    if not np.any(mask):
        return None
    return mask


def generate_l1_occlusion_questions(
    objects: list[dict[str, Any]],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    templates: dict,
    label_statuses: dict[str, Any] | None = None,
    label_counts: dict[str, Any] | None = None,
    referable_object_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    normalized_statuses = _normalize_label_statuses(label_statuses)
    normalized_counts = _normalize_label_counts(label_counts)
    if not normalized_statuses and normalized_counts:
        for label, count in normalized_counts.items():
            if count == 0:
                normalized_statuses[label] = "absent"
            elif count == 1:
                normalized_statuses[label] = "unique"
            elif count > 1:
                normalized_statuses[label] = "multiple"
    referable_id_set = _normalize_object_id_set(
        referable_object_ids,
        "referable_object_ids_for_l1_occlusion",
    )
    label_to_objects: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        label_to_objects.setdefault(label, []).append(obj)

    questions: list[dict[str, Any]] = []

    def _append_geometry_question(
        obj: dict[str, Any],
        label: str,
        source: str,
        vlm_status: str | None,
        vlm_count: int | None,
    ) -> None:
        obj_id = int(obj["id"])
        metrics = _compute_l1_occlusion_metrics(
            obj=obj,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            occlusion_backend=occlusion_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        source_used = source
        grayzone_fallback_source = "geometry_grayzone_fallback"

        decision, source_used, overlay_available = _resolve_l1_occlusion_decision(
            metrics=metrics,
            source_used=source_used,
            grayzone_fallback_source=grayzone_fallback_source,
        )
        if decision is None:
            return

        questions.append(
            _l1_occlusion_question(
                label=label,
                correct=decision,
                templates=templates,
                obj_id=obj_id,
                extra={
                    "occlusion_decision_source": source_used,
                    "vlm_label_status": vlm_status,
                    "vlm_label_count": vlm_count,
                    "geometry_projected_area": metrics.projected_area,
                    "geometry_in_frame_ratio": metrics.in_frame_ratio,
                    "geometry_occlusion_ratio_in_frame": metrics.occlusion_ratio_in_frame,
                    "geometry_valid_in_frame_count": metrics.valid_in_frame_count,
                    "geometry_sampled_point_count": metrics.sampled_point_count,
                    "geometry_in_frame_sample_count": metrics.in_frame_sample_count,
                    "geometry_effective_ratio": metrics.effective_ratio,
                    "geometry_sufficient_evidence": metrics.sufficient_evidence,
                    "geometry_backend": metrics.backend,
                    "mask_overlay_available": overlay_available,
                },
            )
        )

    if normalized_statuses:
        for label, status in sorted(normalized_statuses.items()):
            count = normalized_counts.get(label)
            if status == "absent":
                questions.append(
                    _l1_occlusion_question(
                        label=label,
                        correct="not visible",
                        templates=templates,
                        obj_id=None,
                        extra={
                            "occlusion_decision_source": "vlm_status_absent",
                            "vlm_label_status": status,
                            "vlm_label_count": 0 if count is None else count,
                        },
                    )
                )
                continue
            if status != "unique":
                continue

            candidates = label_to_objects.get(label, [])
            if len(candidates) == 1:
                _append_geometry_question(
                    obj=candidates[0],
                    label=label,
                    source="geometry_from_vlm_unique",
                    vlm_status=status,
                    vlm_count=count,
                )
                continue

            referable_candidates = [
                obj for obj in candidates
                if int(obj["id"]) in referable_id_set
            ]
            if len(referable_candidates) == 1:
                _append_geometry_question(
                    obj=referable_candidates[0],
                    label=label,
                    source="geometry_from_vlm_unique",
                    vlm_status=status,
                    vlm_count=count,
                )
        return questions

    for obj in objects:
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        _append_geometry_question(
            obj=obj,
            label=label,
            source="geometry_only_fallback",
            vlm_status=None,
            vlm_count=None,
        )
    return questions


# ---------------------------------------------------------------------------
#  L1 generators – new reference frames
# ---------------------------------------------------------------------------

def generate_l1_direction_object_centric(
    objects: list[dict],
    templates: dict,
    max_questions: int = 20,
    attachment_edge_lookup: dict[frozenset[int], dict[str, Any]] | None = None,
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
    trace_detail: str = "light",
) -> list[dict]:
    """Generate L1 object-centric direction questions.

    Uses triples (ref, face, target) to define a reference frame at *ref*
    facing *face*, then asks for the direction of *target*.
    """
    n = len(objects)
    if n < 3:
        _emit_generator_summary(
            trace_recorder,
            "generate_l1_direction_object_centric",
            generated_count=0,
            candidate_count=0,
            generated_candidate_count=0,
            skipped_candidate_count=0,
            reason_counts={"insufficient_objects": 1},
            details={"object_count": n, "max_questions": int(max_questions)},
        )
        return []

    tpl_list = templates.get(
        "L1_direction_object_centric",
        _default_templates()["L1_direction_object_centric"],
    )

    candidates: list[dict] = []
    reason_counts: Counter[str] = Counter()
    generated_candidate_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                ref = objects[i]
                face = objects[j]
                target = objects[k]
                candidate_id = _candidate_key(ref["id"], face["id"], target["id"])
                object_ids = [int(ref["id"]), int(face["id"]), int(target["id"])]
                # All three labels must be distinct for unambiguous reference
                if len({ref["label"], face["label"], target["label"]}) < 3:
                    reason_counts["duplicate_labels"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l1_direction_object_centric",
                        candidate_kind="object_triplet",
                        candidate_key=candidate_id,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="duplicate_labels",
                        reason_detail="reference, facing, and target labels are not all distinct",
                    )
                    continue

                ref_c = np.array(ref["center"])
                face_c = np.array(face["center"])
                target_c = np.array(target["center"])

                # Require minimum distances to avoid annotation-error-dominated results
                if np.linalg.norm(face_c - ref_c) < MIN_DIRECTION_DISTANCE:
                    reason_counts["face_too_close"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l1_direction_object_centric",
                        candidate_kind="object_triplet",
                        candidate_key=candidate_id,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="face_too_close",
                        reason_detail="facing anchor is too close to the reference object",
                        evidence={"min_distance": MIN_DIRECTION_DISTANCE},
                    )
                    continue
                if np.linalg.norm(target_c - ref_c) < MIN_DIRECTION_DISTANCE:
                    reason_counts["target_too_close"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l1_direction_object_centric",
                        candidate_kind="object_triplet",
                        candidate_key=candidate_id,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="target_too_close",
                        reason_detail="target object is too close to the reference object",
                        evidence={"min_distance": MIN_DIRECTION_DISTANCE},
                    )
                    continue
                if not _has_stable_object_centric_facing(ref_c, face_c):
                    reason_counts["unstable_facing"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l1_direction_object_centric",
                        candidate_kind="object_triplet",
                        candidate_key=candidate_id,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="unstable_facing",
                        reason_detail="reference-to-facing-object geometry does not define a stable object-centric frame",
                    )
                    continue

                direction, ambiguity = primary_direction_object_centric(
                    ref_c,
                    face_c,
                    target_c,
                    anchor_hull_xy=_object_bottom_hull_xy(ref),
                    target_hull_xy=_object_bottom_hull_xy(target),
                    anchor_bbox_min=np.array(ref["bbox_min"], dtype=float),
                    anchor_bbox_max=np.array(ref["bbox_max"], dtype=float),
                    target_bbox_min=np.array(target["bbox_min"], dtype=float),
                    target_bbox_max=np.array(target["bbox_max"], dtype=float),
                )
                if ambiguity > 0.7:
                    reason_counts["ambiguous_direction"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l1_direction_object_centric",
                        candidate_kind="object_triplet",
                        candidate_key=candidate_id,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="ambiguous_direction",
                        reason_detail="object-centric direction falls too close to an ambiguity boundary",
                        evidence={"ambiguity_score": float(ambiguity), "threshold": 0.7},
                    )
                    continue
                suppression = _direction_suppression_reason(
                    ref,
                    target,
                    direction,
                    attachment_edge_lookup,
                )
                if suppression is not None:
                    reason_code, reason_detail, evidence = suppression
                    reason_counts[reason_code] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l1_direction_object_centric",
                        candidate_kind="object_triplet",
                        candidate_key=candidate_id,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code=reason_code,
                        reason_detail=reason_detail,
                        evidence=evidence,
                    )
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    obj_ref=_the(ref["label"]),
                    obj_face=_the(face["label"]),
                    obj_target=_the(target["label"]),
                )
                horizontal_context = None
                if direction in {"above", "below"}:
                    horizontal_context, _ = primary_direction_object_centric(
                        ref_c,
                        face_c,
                        target_c,
                        horizontal_only=True,
                        anchor_hull_xy=_object_bottom_hull_xy(ref),
                        target_hull_xy=_object_bottom_hull_xy(target),
                        anchor_bbox_min=np.array(ref["bbox_min"], dtype=float),
                        anchor_bbox_max=np.array(ref["bbox_max"], dtype=float),
                        target_bbox_min=np.array(target["bbox_min"], dtype=float),
                        target_bbox_max=np.array(target["bbox_max"], dtype=float),
                    )
                options, answer = generate_direction_options(
                    direction,
                    ALL_DIRECTIONS,
                    horizontal_context=horizontal_context,
                )
                candidates.append({
                    "level": "L1",
                    "type": "direction_object_centric",
                    "reference_frame": "object_centric",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": direction,
                    "obj_ref_id": ref["id"],
                    "obj_ref_label": ref["label"],
                    "obj_face_id": face["id"],
                    "obj_face_label": face["label"],
                    "facing_anchor_center": ref_c.tolist(),
                    "facing_target_center": face_c.tolist(),
                    "obj_target_id": target["id"],
                    "obj_target_label": target["label"],
                    "mentioned_objects": [
                        _mention("reference_origin", ref["label"], ref["id"]),
                        _mention("reference_facing", face["label"], face["id"]),
                        _mention("target", target["label"], target["id"]),
                    ],
                    "ambiguity_score": ambiguity,
                    "relation_unchanged": False,
                })
                generated_candidate_count += 1
                reason_counts["generated"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction_object_centric",
                    candidate_kind="object_triplet",
                    candidate_key=candidate_id,
                    object_ids=object_ids,
                    status="generated",
                    reason_code="generated",
                    reason_detail="triplet defines a stable object-centric frame and yields an unambiguous direction",
                    evidence={"ambiguity_score": float(ambiguity)},
                    question_preview=_question_preview_payload(candidates[-1]),
                )

    if len(candidates) > max_questions:
        candidates = random.sample(candidates, max_questions)
        reason_counts["sampled_out_by_max_questions"] += max(0, generated_candidate_count - len(candidates))
    total_candidate_count = n * (n - 1) * (n - 2)
    _emit_generator_summary(
        trace_recorder,
        "generate_l1_direction_object_centric",
        generated_count=len(candidates),
        candidate_count=total_candidate_count,
        generated_candidate_count=generated_candidate_count,
        skipped_candidate_count=max(total_candidate_count - generated_candidate_count, 0),
        reason_counts=dict(reason_counts),
        details={"object_count": n, "max_questions": int(max_questions)},
    )
    return candidates


def generate_l1_direction_allocentric(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_questions: int = 20,
    attachment_edge_lookup: dict[frozenset[int], dict[str, Any]] | None = None,
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
    trace_detail: str = "light",
) -> list[dict]:
    """Generate L1 allocentric (cardinal) direction questions.

    Provides the camera's cardinal facing direction so the model can anchor
    absolute directions from the image.
    """
    cam_cardinal = camera_cardinal_direction(camera_pose)
    n = len(objects)

    tpl_list = templates.get(
        "L1_direction_allocentric",
        _default_templates()["L1_direction_allocentric"],
    )

    candidates: list[dict] = []
    reason_counts: Counter[str] = Counter()
    generated_candidate_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objects[i], objects[j]
            candidate_id = _candidate_key(a["id"], b["id"])
            object_ids = [int(a["id"]), int(b["id"])]
            if a["label"] == b["label"]:
                reason_counts["duplicate_labels"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction_allocentric",
                    candidate_kind="object_pair",
                    candidate_key=candidate_id,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="duplicate_labels",
                    reason_detail="allocentric direction questions require distinct object labels",
                )
                continue

            a_c = np.array(a["center"])
            b_c = np.array(b["center"])
            if np.linalg.norm(b_c - a_c) < MIN_DIRECTION_DISTANCE:
                reason_counts["pair_too_close"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction_allocentric",
                    candidate_kind="object_pair",
                    candidate_key=candidate_id,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="pair_too_close",
                    reason_detail="object centers are too close for a reliable allocentric direction question",
                    evidence={"min_distance": MIN_DIRECTION_DISTANCE},
                )
                continue
            direction, ambiguity = primary_direction_allocentric(
                a_c,
                b_c,
                obj_a_hull_xy=_object_bottom_hull_xy(a),
                obj_b_hull_xy=_object_bottom_hull_xy(b),
                obj_a_bbox_min=np.array(a["bbox_min"], dtype=float),
                obj_a_bbox_max=np.array(a["bbox_max"], dtype=float),
                obj_b_bbox_min=np.array(b["bbox_min"], dtype=float),
                obj_b_bbox_max=np.array(b["bbox_max"], dtype=float),
            )
            if ambiguity > 0.7:
                reason_counts["ambiguous_direction"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction_allocentric",
                    candidate_kind="object_pair",
                    candidate_key=candidate_id,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="ambiguous_direction",
                    reason_detail="allocentric direction falls too close to an ambiguity boundary",
                    evidence={"ambiguity_score": float(ambiguity), "threshold": 0.7},
                )
                continue
            if direction not in CARDINAL_DIRECTIONS_8:
                reason_counts["non_cardinal_direction"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction_allocentric",
                    candidate_kind="object_pair",
                    candidate_key=candidate_id,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="non_cardinal_direction",
                    reason_detail="computed allocentric direction is not one of the supported cardinal answers",
                    evidence={"direction": direction},
                )
                continue
            suppression = _direction_suppression_reason(
                a,
                b,
                direction,
                attachment_edge_lookup,
            )
            if suppression is not None:
                reason_code, reason_detail, evidence = suppression
                reason_counts[reason_code] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction_allocentric",
                    candidate_kind="object_pair",
                    candidate_key=candidate_id,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    evidence=evidence,
                )
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                camera_cardinal=cam_cardinal,
                obj_a=_the(a["label"]),
                obj_b=_the(b["label"]),
            )
            options, answer = generate_direction_options(
                direction,
                ALL_DIRECTIONS_ALLOCENTRIC,
            )
            candidates.append({
                "level": "L1",
                "type": "direction_allocentric",
                "reference_frame": "allocentric",
                "question": question_text,
                "options": options,
                "answer": answer,
                "correct_value": direction,
                "camera_cardinal": cam_cardinal,
                "obj_a_id": a["id"],
                "obj_a_label": a["label"],
                "obj_b_id": b["id"],
                "obj_b_label": b["label"],
                "mentioned_objects": [
                    _mention("obj_a", a["label"], a["id"]),
                    _mention("obj_b", b["label"], b["id"]),
                ],
                "ambiguity_score": ambiguity,
                "relation_unchanged": False,
            })
            generated_candidate_count += 1
            reason_counts["generated"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_direction_allocentric",
                candidate_kind="object_pair",
                candidate_key=candidate_id,
                object_ids=object_ids,
                status="generated",
                reason_code="generated",
                reason_detail="pair yields a valid allocentric cardinal-direction relation",
                evidence={"ambiguity_score": float(ambiguity), "direction": direction},
                question_preview=_question_preview_payload(candidates[-1]),
            )

    if len(candidates) > max_questions:
        candidates = random.sample(candidates, max_questions)
        reason_counts["sampled_out_by_max_questions"] += max(0, generated_candidate_count - len(candidates))
    total_candidate_count = int((n * max(n - 1, 0)) / 2)
    _emit_generator_summary(
        trace_recorder,
        "generate_l1_direction_allocentric",
        generated_count=len(candidates),
        candidate_count=total_candidate_count,
        generated_candidate_count=generated_candidate_count,
        skipped_candidate_count=max(total_candidate_count - generated_candidate_count, 0),
        reason_counts=dict(reason_counts),
        details={"object_count": n, "max_questions": int(max_questions)},
    )
    return candidates


# ---------------------------------------------------------------------------
#  L2 generators
# ---------------------------------------------------------------------------

def _visibility_status_from_ratio(visible_ratio: float) -> str:
    if visible_ratio >= FULLY_VISIBLE_RATIO_MIN:
        return "fully visible"
    if visible_ratio >= PARTIALLY_VISIBLE_RATIO_MIN:
        return "partially occluded"
    return "not visible"


def _projected_area_summary(
    sample_points: np.ndarray,
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    projected_records = _project_sample_point_records(
        sample_points,
        camera_pose,
        color_intrinsics,
    )
    return _projected_area_from_records(projected_records, color_intrinsics)


_COUNTERFACTUAL_TARGET_TRI_ID = 1
_COUNTERFACTUAL_OTHER_TRI_ID = 0


def _hits_up_to_distance_from_caster(
    caster,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    ignored_tri_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    if caster is None or not np.isfinite(max_distance) or max_distance <= 1e-12:
        return []

    hits_helper = getattr(caster, "_hits_up_to_distance", None)
    if callable(hits_helper):
        return hits_helper(
            origin=origin,
            direction=direction,
            max_distance=float(max_distance),
            ignored_tri_ids=ignored_tri_ids,
        )

    cast_ray = getattr(caster, "cast_ray", None)
    if not callable(cast_ray):
        return []

    filtered: list[tuple[int, float]] = []
    for _hit_point, tri_id, dist in cast_ray(origin, direction):
        if dist > float(max_distance):
            break
        tri_id_int = int(tri_id)
        if ignored_tri_ids and tri_id_int in ignored_tri_ids:
            continue
        filtered.append((tri_id_int, float(dist)))
    return filtered


def _counterfactual_hit_path(
    modified_scene: _ModifiedSceneContext,
    camera_pos: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    target_triangle_ids: set[int],
    target_caster,
    target_delta: np.ndarray,
    blocker_casters: dict[int, Any],
    blocker_deltas: dict[int, np.ndarray],
) -> list[tuple[int, float]]:
    static_ignored_tri_ids = set(modified_scene.ignored_tri_ids)
    use_static_target_fallback = target_caster is None
    if not use_static_target_fallback:
        static_ignored_tri_ids.update(int(tid) for tid in target_triangle_ids)

    merged_hits: list[tuple[int, float]] = []
    static_hits = _hits_up_to_distance_from_caster(
        modified_scene.ray_caster,
        origin=camera_pos,
        direction=direction,
        max_distance=max_distance,
        ignored_tri_ids=static_ignored_tri_ids,
    )
    for tri_id, dist in static_hits:
        if use_static_target_fallback and int(tri_id) in target_triangle_ids:
            merged_hits.append((_COUNTERFACTUAL_TARGET_TRI_ID, float(dist)))
        else:
            merged_hits.append((_COUNTERFACTUAL_OTHER_TRI_ID, float(dist)))

    target_hits = _hits_up_to_distance_from_caster(
        target_caster,
        origin=camera_pos - target_delta,
        direction=direction,
        max_distance=max_distance,
    )
    merged_hits.extend(
        (_COUNTERFACTUAL_TARGET_TRI_ID, float(dist))
        for _tri_id, dist in target_hits
    )

    for blocker_id, blocker_caster in blocker_casters.items():
        blocker_delta = blocker_deltas.get(blocker_id)
        if blocker_delta is None:
            continue
        blocker_hits = _hits_up_to_distance_from_caster(
            blocker_caster,
            origin=camera_pos - blocker_delta,
            direction=direction,
            max_distance=max_distance,
        )
        merged_hits.extend(
            (_COUNTERFACTUAL_OTHER_TRI_ID, float(dist))
            for _tri_id, dist in blocker_hits
        )

    merged_hits.sort(key=lambda item: item[1])
    return merged_hits


def _build_modified_scene(
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    removed_ids: set[int],
) -> _ModifiedSceneContext | None:
    """Prepare a lightweight counterfactual scene query context."""
    if ray_caster is None:
        return None

    ignored_tri_ids: set[int] = set()
    if instance_mesh_data is not None:
        for obj_id in removed_ids:
            ignored_tri_ids.update(
                _instance_triangle_id_set(instance_mesh_data, int(obj_id))
            )

    return _ModifiedSceneContext(
        ray_caster=ray_caster,
        ignored_tri_ids=frozenset(ignored_tri_ids),
    )


def _compute_target_visibility(
    modified_scene: _ModifiedSceneContext | None,
    target_surface_points: np.ndarray,
    target_triangle_ids: set[int],
    camera_pos: np.ndarray,
    target_sample_triangle_ids: np.ndarray | None = None,
    target_sample_barycentrics: np.ndarray | None = None,
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> tuple[str, float]:
    """Evaluate target visibility against a possibly modified scene."""
    if (
        modified_scene is None
        or modified_scene.ray_caster is None
        or len(target_surface_points) == 0
        or not target_triangle_ids
    ):
        return "not visible", 0.0

    visible_ratio = _mesh_visibility_ratio_compat(
        modified_scene.ray_caster,
        camera_pos=camera_pos,
        target_points=target_surface_points,
        target_tri_ids=target_triangle_ids,
        ignored_tri_ids=set(modified_scene.ignored_tri_ids),
        sample_triangle_ids=target_sample_triangle_ids,
        sample_barycentrics=target_sample_barycentrics,
        vertices=mesh_vertices,
        faces=mesh_faces,
    )
    return _visibility_status_from_ratio(visible_ratio), float(visible_ratio)


def _compute_counterfactual_target_visibility_stats(
    modified_scene: _ModifiedSceneContext | None,
    target_surface_points: np.ndarray,
    target_triangle_ids: set[int],
    camera_pos: np.ndarray,
    hit_epsilon: float = 0.05,
    instance_mesh_data: InstanceMeshData | None = None,
    target_obj_id: int | None = None,
    target_delta: np.ndarray | None = None,
    moved_blocker_deltas: dict[int, np.ndarray] | None = None,
    sample_triangle_ids: np.ndarray | None = None,
    sample_barycentrics: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    local_resample_count: int = _LOCAL_BOUNDARY_RESAMPLE_COUNT,
) -> tuple[int, int]:
    """Return visible/valid sample counts for a translated target.

    The hit path is merged from three sources:

    1. the static scene with all moved objects ignored,
    2. the target's own per-instance mesh, queried with origin shifted by
       ``-target_delta``, and
    3. any other moved blockers, queried the same way.

    The merged path is then classified with the same ordered
    visible / externally_occluded / self_occluded / mixed_boundary logic used
    by static ``mesh_ray`` visibility. Mixed boundary samples trigger local
    same-triangle refinement when sample metadata is available.
    """
    if (
        modified_scene is None
        or modified_scene.ray_caster is None
        or len(target_surface_points) == 0
        or not target_triangle_ids
    ):
        return 0, 0

    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    sampled_points = np.asarray(target_surface_points, dtype=np.float64)
    directions = sampled_points - camera_pos
    expected_dists = np.linalg.norm(directions, axis=1)
    valid_mask = np.isfinite(expected_dists) & (expected_dists > 1e-6)
    if not np.any(valid_mask):
        return 0, 0

    sampled_points = sampled_points[valid_mask]
    directions = directions[valid_mask]
    expected_dists = expected_dists[valid_mask]
    triangle_meta = None
    barycentric_meta = None
    if sample_triangle_ids is not None and len(sample_triangle_ids) == len(target_surface_points):
        triangle_meta = np.asarray(sample_triangle_ids, dtype=np.int64)[valid_mask]
    if sample_barycentrics is not None and len(sample_barycentrics) == len(target_surface_points):
        barycentric_meta = np.asarray(sample_barycentrics, dtype=np.float64)[valid_mask]
    vertices_arr = np.asarray(vertices, dtype=np.float64) if vertices is not None else None
    faces_arr = np.asarray(faces, dtype=np.int64) if faces is not None else None

    # Build / retrieve per-instance intersectors.
    target_caster = None
    t_delta = np.zeros(3, dtype=np.float64)
    if instance_mesh_data is not None and target_obj_id is not None:
        target_caster = _get_instance_intersector(instance_mesh_data, int(target_obj_id))
        if target_delta is not None:
            t_delta = np.asarray(target_delta, dtype=np.float64)

    blocker_casters: dict[int, Any] = {}
    b_deltas: dict[int, np.ndarray] = {}
    if instance_mesh_data is not None and moved_blocker_deltas:
        for b_id, b_delta in moved_blocker_deltas.items():
            b_caster = _get_instance_intersector(instance_mesh_data, int(b_id))
            if b_caster is not None:
                blocker_casters[b_id] = b_caster
                b_deltas[b_id] = np.asarray(b_delta, dtype=np.float64)

    visible_count = 0
    valid_count = 0
    mixed_records: list[int] = []
    counterfactual_target_tri_ids = {_COUNTERFACTUAL_TARGET_TRI_ID}

    for ray_idx, (direction, expected_dist) in enumerate(zip(directions, expected_dists)):
        hit_path = _counterfactual_hit_path(
            modified_scene=modified_scene,
            camera_pos=camera_pos,
            direction=direction,
            max_distance=float(expected_dist) + hit_epsilon,
            target_triangle_ids=target_triangle_ids,
            target_caster=target_caster,
            target_delta=t_delta,
            blocker_casters=blocker_casters,
            blocker_deltas=b_deltas,
        )
        classification = _classify_hit_path(
            hit_path,
            expected_dist=float(expected_dist),
            target_tri_ids=counterfactual_target_tri_ids,
            hit_epsilon=hit_epsilon,
        )
        if classification == "visible":
            visible_count += 1
            valid_count += 1
        elif classification == "externally_occluded":
            valid_count += 1
        elif classification == "mixed_boundary":
            mixed_records.append(ray_idx)

    can_refine_mixed = (
        triangle_meta is not None
        and barycentric_meta is not None
        and vertices_arr is not None
        and faces_arr is not None
        and int(local_resample_count) > 0
    )
    if can_refine_mixed:
        for ray_idx in mixed_records:
            tri_id = int(triangle_meta[ray_idx])
            if tri_id < 0 or tri_id >= len(faces_arr):
                continue
            tri_vertices = vertices_arr[faces_arr[tri_id]]
            local_points, _local_barys = _local_triangle_resamples(
                triangle_vertices=tri_vertices,
                barycentric=barycentric_meta[ray_idx],
                triangle_id=tri_id,
                n_samples=int(local_resample_count),
            )
            if len(local_points) == 0:
                continue

            local_points = np.asarray(local_points, dtype=np.float64) + t_delta
            local_visible = 0
            local_valid = 0
            for point in local_points:
                direction = np.asarray(point, dtype=np.float64) - camera_pos
                expected_dist = float(np.linalg.norm(direction))
                if not np.isfinite(expected_dist) or expected_dist <= 1e-6:
                    continue
                hit_path = _counterfactual_hit_path(
                    modified_scene=modified_scene,
                    camera_pos=camera_pos,
                    direction=direction,
                    max_distance=expected_dist + hit_epsilon,
                    target_triangle_ids=target_triangle_ids,
                    target_caster=target_caster,
                    target_delta=t_delta,
                    blocker_casters=blocker_casters,
                    blocker_deltas=b_deltas,
                )
                classification = _classify_hit_path(
                    hit_path,
                    expected_dist=expected_dist,
                    target_tri_ids=counterfactual_target_tri_ids,
                    hit_epsilon=hit_epsilon,
                )
                if classification == "visible":
                    local_visible += 1
                    local_valid += 1
                elif classification == "externally_occluded":
                    local_valid += 1

            if local_valid >= 2:
                visible_count += local_visible
                valid_count += local_valid

    return int(visible_count), int(valid_count)


def _compute_counterfactual_target_visibility(
    modified_scene: _ModifiedSceneContext | None,
    target_surface_points: np.ndarray,
    target_triangle_ids: set[int],
    camera_pos: np.ndarray,
    hit_epsilon: float = 0.05,
    instance_mesh_data: InstanceMeshData | None = None,
    target_obj_id: int | None = None,
    target_delta: np.ndarray | None = None,
    moved_blocker_deltas: dict[int, np.ndarray] | None = None,
    sample_triangle_ids: np.ndarray | None = None,
    sample_barycentrics: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    local_resample_count: int = _LOCAL_BOUNDARY_RESAMPLE_COUNT,
) -> tuple[str, float]:
    """Estimate visibility for a target that may have been translated."""
    visible_count, valid_count = _compute_counterfactual_target_visibility_stats(
        modified_scene=modified_scene,
        target_surface_points=target_surface_points,
        target_triangle_ids=target_triangle_ids,
        camera_pos=camera_pos,
        hit_epsilon=hit_epsilon,
        instance_mesh_data=instance_mesh_data,
        target_obj_id=target_obj_id,
        target_delta=target_delta,
        moved_blocker_deltas=moved_blocker_deltas,
        sample_triangle_ids=sample_triangle_ids,
        sample_barycentrics=sample_barycentrics,
        vertices=vertices,
        faces=faces,
        local_resample_count=local_resample_count,
    )
    if valid_count <= 0:
        return "not visible", 0.0
    visible_ratio = float(visible_count / valid_count)
    return _visibility_status_from_ratio(visible_ratio), visible_ratio


def _compute_visibility_status_per_object(
    objects: list[dict],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image=None,
    depth_intrinsics=None,
    occlusion_backend: str = "depth",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
    modified_scene: _ModifiedSceneContext | None = None,
) -> dict[int, tuple[str, float]]:
    if color_intrinsics is None:
        return {int(obj["id"]): ("not visible", 0.0) for obj in objects}

    backend = str(occlusion_backend)
    camera_pos = np.array(camera_pose.position, dtype=np.float64)
    if backend == "mesh_ray" and modified_scene is None:
        modified_scene = _build_modified_scene(
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            removed_ids=set(),
        )

    visibility: dict[int, tuple[str, float]] = {}
    mesh_vertices = (
        np.asarray(instance_mesh_data.vertices, dtype=np.float64)
        if instance_mesh_data is not None else None
    )
    mesh_faces = (
        np.asarray(instance_mesh_data.faces, dtype=np.int64)
        if instance_mesh_data is not None else None
    )
    for obj in objects:
        obj_id = int(obj["id"])
        sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
        sample_triangle_ids, sample_barycentrics = _instance_surface_sample_metadata(
            instance_mesh_data,
            obj_id,
        )
        target_tri_ids = _instance_triangle_id_set(instance_mesh_data, obj_id)
        (
            projected_area,
            in_frame_ratio,
            in_frame_points,
            in_frame_triangle_ids,
            in_frame_barycentrics,
        ) = _in_frame_surface_sample_subset(
            sample_points,
            camera_pose,
            color_intrinsics,
            sample_triangle_ids=sample_triangle_ids,
            sample_barycentrics=sample_barycentrics,
        )
        if (
            len(sample_points) == 0
            or not target_tri_ids
            or projected_area < MIN_PROJECTED_AREA_PX
            or in_frame_ratio < MIN_IN_FRAME_RATIO
            or len(in_frame_points) == 0
        ):
            visibility[obj_id] = ("not visible", 0.0)
            continue

        if backend == "mesh_ray":
            status, visible_ratio = _compute_target_visibility(
                modified_scene=modified_scene,
                target_surface_points=in_frame_points,
                target_triangle_ids=target_tri_ids,
                camera_pos=camera_pos,
                target_sample_triangle_ids=in_frame_triangle_ids,
                target_sample_barycentrics=in_frame_barycentrics,
                mesh_vertices=mesh_vertices,
                mesh_faces=mesh_faces,
            )
            visibility[obj_id] = (status, float(visible_ratio))
            continue

        if backend == "depth":
            if not _depth_backend_inputs_ready(
                depth_image=depth_image,
                depth_intrinsics=depth_intrinsics,
                ray_caster=ray_caster,
                context="counterfactual visibility",
            ):
                visibility[obj_id] = ("not visible", 0.0)
                continue
            status, visible_ratio = compute_mesh_depth_occlusion(
                target_points=sample_points,
                target_tri_ids=target_tri_ids,
                camera_pose=camera_pose,
                intrinsics=depth_intrinsics,
                depth_image=depth_image,
                ray_caster=ray_caster,
            )
            visibility[obj_id] = (status, float(visible_ratio))
            continue

        raise ValueError(f"Unsupported occlusion backend: {backend}")

    return visibility


def _compute_movement_visibility_status_per_object(
    original_objects: list[dict],
    moved_objects: list[dict],
    moved_ids: set[int],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> dict[int, tuple[str, float]]:
    """Compute post-move visibility using per-instance mesh intersectors."""
    if color_intrinsics is None:
        return {int(obj["id"]): ("not visible", 0.0) for obj in moved_objects}

    original_map = {int(obj["id"]): obj for obj in original_objects}
    moved_scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set(moved_ids))
    camera_pos = np.array(camera_pose.position, dtype=np.float64)
    mesh_vertices = (
        np.asarray(instance_mesh_data.vertices, dtype=np.float64)
        if instance_mesh_data is not None else None
    )
    mesh_faces = (
        np.asarray(instance_mesh_data.faces, dtype=np.int64)
        if instance_mesh_data is not None else None
    )

    # Translation delta for every moved object.
    moved_deltas: dict[int, np.ndarray] = {}
    for obj in moved_objects:
        mid = int(obj["id"])
        if mid not in moved_ids:
            continue
        orig = original_map.get(mid)
        if orig is not None:
            moved_deltas[mid] = (
                np.asarray(obj["center"], dtype=np.float64)
                - np.asarray(orig["center"], dtype=np.float64)
            )

    visibility: dict[int, tuple[str, float]] = {}
    for obj in moved_objects:
        obj_id = int(obj["id"])
        sample_points = _instance_surface_samples(instance_mesh_data, obj_id)
        sample_triangle_ids, sample_barycentrics = _instance_surface_sample_metadata(
            instance_mesh_data,
            obj_id,
        )
        target_tri_ids = _instance_triangle_id_set(instance_mesh_data, obj_id)
        target_delta = moved_deltas.get(obj_id)
        if target_delta is not None:
            sample_points = sample_points + target_delta

        (
            projected_area,
            in_frame_ratio,
            in_frame_points,
            in_frame_triangle_ids,
            in_frame_barycentrics,
        ) = _in_frame_surface_sample_subset(
            sample_points,
            camera_pose,
            color_intrinsics,
            sample_triangle_ids=sample_triangle_ids,
            sample_barycentrics=sample_barycentrics,
        )
        if (
            len(sample_points) == 0
            or not target_tri_ids
            or projected_area < MIN_PROJECTED_AREA_PX
            or in_frame_ratio < MIN_IN_FRAME_RATIO
            or len(in_frame_points) == 0
        ):
            visibility[obj_id] = ("not visible", 0.0)
            continue

        moved_blocker_deltas = {
            bid: delta for bid, delta in moved_deltas.items() if bid != obj_id
        }
        status, visible_ratio = _compute_counterfactual_target_visibility(
            modified_scene=moved_scene_context,
            target_surface_points=in_frame_points,
            target_triangle_ids=target_tri_ids,
            camera_pos=camera_pos,
            instance_mesh_data=instance_mesh_data,
            target_obj_id=obj_id,
            target_delta=target_delta,
            moved_blocker_deltas=moved_blocker_deltas,
            sample_triangle_ids=in_frame_triangle_ids,
            sample_barycentrics=in_frame_barycentrics,
            vertices=mesh_vertices,
            faces=mesh_faces,
        )
        visibility[obj_id] = (status, float(visible_ratio))

    return visibility


def _counterfactual_occlusion_backend(
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> str:
    """Return the backend used for L2 counterfactual visibility comparisons.

    Even when the primary per-frame backend is ``depth``, L2 comparisons still
    use ``mesh_ray`` because a counterfactual camera/object edit does not come
    with a newly rendered depth map.
    """
    requested_backend = str(occlusion_backend)
    if requested_backend not in {"depth", "mesh_ray"}:
        raise ValueError(f"Unsupported occlusion backend: {requested_backend}")
    if ray_caster is None or instance_mesh_data is None:
        raise RuntimeError(
            "Counterfactual visibility requires mesh geometry for depth and mesh_ray backends",
        )
    if requested_backend == "depth":
        logger.debug(
            "Counterfactual visibility requested with %s backend; falling back to mesh_ray because no counterfactual depth map exists.",
            requested_backend,
        )
    return "mesh_ray"


def _merge_changed_relations(
    primary_changed: list[dict],
    extra_changed: list[dict],
) -> list[dict]:
    merged: dict[tuple[int, int], dict] = {}
    for ch in primary_changed:
        key = (int(ch["obj_a_id"]), int(ch["obj_b_id"]))
        merged[key] = {
            **ch,
            "changes": dict(ch.get("changes", {})),
        }

    for ch in extra_changed:
        key = (int(ch["obj_a_id"]), int(ch["obj_b_id"]))
        if key not in merged:
            merged[key] = {
                **ch,
                "changes": dict(ch.get("changes", {})),
            }
            continue
        merged[key]["changes"].update(ch.get("changes", {}))
    return list(merged.values())


def _iter_valid_object_move_states(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
):
    """Yield physically valid movement candidates in the canonical search order."""
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
        yield delta, new_objects, moved_ids


def _relation_map_by_pair(relations: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(relation["obj_a_id"]), int(relation["obj_b_id"])): relation
        for relation in relations
    }


def _first_valid_object_move_state(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    *,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
) -> _SelectedObjectMoveState | None:
    for delta, moved_objects, moved_ids in _iter_valid_object_move_states(
        objects,
        attachment_graph,
        target_id,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
    ):
        return _SelectedObjectMoveState(
            delta=np.asarray(delta, dtype=np.float64),
            moved_objects=moved_objects,
            moved_ids=set(moved_ids),
            changed_relations=[],
            used_changed_delta=False,
        )
    return None


def _select_object_move_state(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    camera_pose: CameraPose,
    *,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    allow_unchanged_attachment: bool = False,
    color_intrinsics: CameraIntrinsics | None = None,
    occlusion_backend: str = "depth",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
) -> _SelectedObjectMoveState | None:
    delta, changed = _find_object_move_delta_and_changes(
        objects,
        attachment_graph,
        target_id,
        camera_pose,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
        color_intrinsics=color_intrinsics,
        occlusion_backend=occlusion_backend,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
    )
    if delta is not None:
        moved_objects = apply_movement(objects, attachment_graph, target_id, delta)
        moved_ids = get_moved_object_ids(target_id, attachment_graph)
        return _SelectedObjectMoveState(
            delta=np.asarray(delta, dtype=np.float64),
            moved_objects=moved_objects,
            moved_ids=set(moved_ids),
            changed_relations=changed,
            used_changed_delta=True,
        )
    if not allow_unchanged_attachment:
        return None
    return _first_valid_object_move_state(
        objects,
        attachment_graph,
        target_id,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
    )


def _other_obj_id_for_query(query_obj_id: int, relation: dict[str, Any]) -> int | None:
    obj_a_id = int(relation["obj_a_id"])
    obj_b_id = int(relation["obj_b_id"])
    if int(query_obj_id) == obj_a_id:
        return obj_b_id
    if int(query_obj_id) == obj_b_id:
        return obj_a_id
    return None


def _direction_values_for_query_object(
    query_obj_id: int,
    old_relation: dict[str, Any],
    new_relation: dict[str, Any],
) -> tuple[str, str] | None:
    obj_a_id = int(old_relation["obj_a_id"])
    obj_b_id = int(old_relation["obj_b_id"])
    if int(query_obj_id) == obj_b_id:
        return (
            str(old_relation["direction_b_rel_a"]),
            str(new_relation["direction_b_rel_a"]),
        )
    if int(query_obj_id) == obj_a_id:
        return (
            _invert_direction(str(old_relation["direction_b_rel_a"])),
            _invert_direction(str(new_relation["direction_b_rel_a"])),
        )
    return None


def _find_object_move_occlusion_changes(
    original_objects: list[dict],
    moved_objects: list[dict],
    moved_ids: set[int],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
) -> list[dict]:
    """Return L1-style visibility changes for moved targets."""
    if color_intrinsics is None:
        return []

    compare_backend = _counterfactual_occlusion_backend(
        occlusion_backend, ray_caster, instance_mesh_data,
    )
    original_scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set())
    original_map = {int(obj["id"]): obj for obj in original_objects}
    occlusion_changes: list[dict] = []
    for target_obj_id in sorted(int(obj_id) for obj_id in moved_ids):
        target_obj = original_map.get(target_obj_id)
        if target_obj is None:
            continue

        old_metrics, old_source = _compute_l1_style_visibility_metrics_for_static_target(
            obj=target_obj,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend=compare_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            modified_scene=original_scene_context,
        )
        old_status, old_reason_code, _old_reason_detail = _resolve_counterfactual_l1_visibility_status(
            old_metrics
        )
        if old_status is None:
            continue

        new_metrics, new_source = _compute_l1_style_visibility_metrics_for_moved_target(
            target_obj_id=target_obj_id,
            original_objects=original_objects,
            moved_objects=moved_objects,
            moved_ids=moved_ids,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            occlusion_backend=compare_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        new_status, new_reason_code, _new_reason_detail = _resolve_counterfactual_l1_visibility_status(
            new_metrics
        )
        if new_status is None or old_status == new_status:
            continue

        occlusion_changes.append({
            "obj_a_id": target_obj_id,
            "obj_b_id": target_obj_id,
            "target_obj_id": target_obj_id,
            "target_obj_label": target_obj.get("label", "object"),
            "changes": {
                "visibility_status": {"old": old_status, "new": new_status},
            },
            "old": {
                "visibility_status": old_status,
                "visibility_source": old_source,
                "visibility_resolution": old_reason_code,
                "visibility_metrics": _l1_occlusion_metrics_payload(old_metrics),
            },
            "new": {
                "visibility_status": new_status,
                "visibility_source": new_source,
                "visibility_resolution": new_reason_code,
                "visibility_metrics": _l1_occlusion_metrics_payload(new_metrics),
            },
        })

    return occlusion_changes


def _find_object_move_delta_and_changes(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    camera_pose: CameraPose,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    color_intrinsics: CameraIntrinsics | None = None,
    occlusion_backend: str = "depth",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
) -> tuple[np.ndarray | None, list[dict]]:
    """Return the first valid movement delta that yields relation or visibility changes."""
    delta, changed = find_meaningful_movement(
        objects,
        attachment_graph,
        target_id,
        camera_pose,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
    )
    occlusion_enabled = (
        color_intrinsics is not None
        and ray_caster is not None
        and instance_mesh_data is not None
    )
    if delta is not None:
        if not occlusion_enabled:
            return delta, changed
        moved_scene_objects = apply_movement(
            objects,
            attachment_graph,
            target_id,
            delta,
        )
        moved_ids = get_moved_object_ids(target_id, attachment_graph)
        occlusion_changed = _find_object_move_occlusion_changes(
            original_objects=objects,
            moved_objects=moved_scene_objects,
            moved_ids=moved_ids,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            occlusion_backend=occlusion_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        return delta, _merge_changed_relations(changed, occlusion_changed)

    if not occlusion_enabled:
        return None, []

    for candidate_delta, moved_scene_objects, moved_ids in _iter_valid_object_move_states(
        objects,
        attachment_graph,
        target_id,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
    ):
        occlusion_changed = _find_object_move_occlusion_changes(
            original_objects=objects,
            moved_objects=moved_scene_objects,
            moved_ids=moved_ids,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            occlusion_backend=occlusion_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        if occlusion_changed:
            return candidate_delta, occlusion_changed

    return None, []


def _distance_bin_index(
    label: str | None = None,
    *,
    bin_id: str | None = None,
) -> int | None:
    if bin_id:
        for idx, candidate_id in enumerate(DISTANCE_BIN_IDS):
            if candidate_id == bin_id:
                return idx
    for idx, (_, candidate_label) in enumerate(DISTANCE_BINS):
        if candidate_label == label:
            return idx
    return None


def _relation_distance_for_distance_questions(relation: dict[str, Any]) -> float:
    raw_distance = relation.get("distance_m_raw")
    if raw_distance is not None:
        try:
            return float(raw_distance)
        except (TypeError, ValueError):
            pass

    rounded_distance = relation.get("distance_m")
    if rounded_distance is not None:
        try:
            return float(rounded_distance)
        except (TypeError, ValueError):
            pass

    return float("inf")


def _iter_distance_move_deltas():
    max_steps = int(round(DISTANCE_MOVE_SEARCH_MAX_M / DISTANCE_MOVE_SEARCH_STEP_M))
    for step_idx in range(1, max_steps + 1):
        magnitude = round(step_idx * DISTANCE_MOVE_SEARCH_STEP_M, 1)
        for direction in DISTANCE_MOVE_DIRECTIONS:
            yield np.round(direction * magnitude, 1)


def _find_stable_distance_move_for_relation(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    target_id: int,
    relation: dict[str, Any],
    *,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    allow_unchanged_fallback: bool = False,
) -> tuple[np.ndarray | None, str | None, str | None, bool]:
    """Find a valid move for distance questions.

    Prefer the smallest stable one-bin crossing. If that fails and the relation
    is attachment-mediated, fall back to the first physically valid move that
    keeps the distance bin unchanged so the question can still be retained and
    explicitly marked as unchanged.
    """
    old_label = str(relation.get("distance_bin", "")).strip()
    old_bin_id = str(relation.get("distance_bin_id", "")).strip() or None
    old_idx = _distance_bin_index(old_label, bin_id=old_bin_id)
    if old_idx is None:
        return None, None, None, False

    obj_a_id = int(relation["obj_a_id"])
    obj_b_id = int(relation["obj_b_id"])
    moved_ids = get_moved_object_ids(target_id, attachment_graph)
    affects_a = obj_a_id in moved_ids
    affects_b = obj_b_id in moved_ids
    if affects_a == affects_b:
        return None, None, None, False

    room_min, room_max = compute_room_bounds(objects, room_bounds=room_bounds)
    obj_map = {int(obj["id"]): obj for obj in objects}
    if obj_a_id not in obj_map or obj_b_id not in obj_map:
        return None, None, None, False
    if _relation_distance_for_distance_questions(relation) < MIN_DISTANCE_QUESTION_DISTANCE_M:
        return None, old_label, None, False

    for delta in _iter_distance_move_deltas():
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

        new_map = {int(obj["id"]): obj for obj in new_objects}
        approx_details = compute_distance_details(
            new_map[obj_a_id],
            new_map[obj_b_id],
            force_aabb=True,
        )
        approx_idx = _distance_bin_index(
            str(approx_details.get("distance_bin", "")),
            bin_id=str(approx_details.get("distance_bin_id", "")).strip() or None,
        )
        if approx_idx is None:
            continue
        if abs(approx_idx - old_idx) != 1 and not bool(approx_details.get("near_boundary", False)):
            continue

        exact_details = compute_distance_details(
            new_map[obj_a_id],
            new_map[obj_b_id],
        )
        if float(exact_details.get("distance_m", 0.0)) < MIN_DISTANCE_QUESTION_DISTANCE_M:
            continue
        new_label = str(exact_details["distance_bin"])
        new_near_boundary = bool(exact_details["near_boundary"])
        new_idx = _distance_bin_index(
            new_label,
            bin_id=str(exact_details.get("distance_bin_id", "")).strip() or None,
        )
        if new_idx is None or abs(new_idx - old_idx) != 1:
            continue
        if new_near_boundary:
            continue
        return np.asarray(delta, dtype=np.float64), old_label, new_label, False

    if allow_unchanged_fallback:
        for delta, new_objects, _moved_ids in _iter_valid_object_move_states(
            objects,
            attachment_graph,
            target_id,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
        ):
            new_map = {int(obj["id"]): obj for obj in new_objects}
            exact_details = compute_distance_details(
                new_map[obj_a_id],
                new_map[obj_b_id],
            )
            if float(exact_details.get("distance_m", 0.0)) < MIN_DISTANCE_QUESTION_DISTANCE_M:
                continue
            new_label = str(exact_details["distance_bin"])
            if new_label != old_label:
                continue
            return np.asarray(delta, dtype=np.float64), old_label, new_label, True

    return None, old_label, None, False


def _generate_l2_distance_questions_for_object(
    *,
    query_obj: dict[str, Any],
    move_source: dict[str, Any],
    move_source_id: int,
    attachment_remapped: bool,
    relations: list[dict[str, Any]],
    movement_scene_objects: list[dict],
    attachment_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    templates: dict[str, Any],
    obj_map: dict[int, dict[str, Any]],
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Generate L2 distance questions using pair-specific stable cross-bin moves."""
    tpl_list = templates.get(
        "L2_object_move_distance",
        _default_templates()["L2_object_move_distance"],
    )
    has_attachment_chain = len(get_attachment_chain_ids(move_source_id, attachment_graph)) > 0
    questions: list[dict[str, Any]] = []

    for relation in relations:
        if query_obj["id"] not in (relation["obj_a_id"], relation["obj_b_id"]):
            continue

        delta, old_value, answer_value, relation_unchanged = _find_stable_distance_move_for_relation(
            movement_scene_objects,
            attachment_graph,
            move_source_id,
            relation,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
            allow_unchanged_fallback=attachment_remapped,
        )
        if delta is None or answer_value is None or old_value is None:
            continue

        if query_obj["id"] == relation["obj_b_id"]:
            relation_obj_b_id = relation["obj_b_id"]
            relation_obj_c_id = relation["obj_a_id"]
        else:
            relation_obj_b_id = relation["obj_a_id"]
            relation_obj_c_id = relation["obj_b_id"]

        moved_state = apply_movement(movement_scene_objects, attachment_graph, move_source_id, delta)
        moved_map = {int(obj["id"]): obj for obj in moved_state}
        new_distance = compute_distance_details(
            moved_map[int(relation["obj_a_id"])],
            moved_map[int(relation["obj_b_id"])],
        )
        if float(new_distance.get("distance_m", 0.0)) < MIN_DISTANCE_QUESTION_DISTANCE_M:
            continue
        answer_value = str(new_distance.get("distance_bin", answer_value))
        obj_b_label = obj_map.get(relation_obj_b_id, {}).get("label", "object")
        obj_c_label = obj_map.get(relation_obj_c_id, {}).get("label", "object")
        direction_desc = _delta_to_description(delta, camera_pose)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"
        tpl = random.choice(tpl_list)
        question_text = tpl.format(
            obj_a=_the(move_source["label"]),
            direction=direction_desc,
            direction_with_camera_hint=_direction_with_camera_hint(direction_desc),
            distance=distance_desc,
            obj_b=_the(obj_b_label),
            obj_c=_the(obj_c_label),
        )
        options, answer = generate_options(answer_value, ALL_DISTANCES)
        questions.append({
            "level": "L2",
            "type": "object_move_distance",
            "question": question_text,
            "options": options,
            "answer": answer,
            "correct_value": answer_value,
            "old_correct_value": old_value,
            "new_correct_value": answer_value,
            "old_distance_m": float(relation.get("distance_m", 0.0)),
            "new_distance_m": float(new_distance.get("distance_m", 0.0)),
            "old_distance_bin_id": relation.get("distance_bin_id"),
            "new_distance_bin_id": new_distance.get("distance_bin_id"),
            "distance_definition": new_distance.get("distance_definition"),
            "old_distance_definition": relation.get("distance_definition"),
            "new_distance_definition": new_distance.get("distance_definition"),
            "moved_obj_id": move_source_id,
            "moved_obj_label": move_source["label"],
            "query_obj_id": query_obj["id"],
            "query_obj_label": query_obj["label"],
            "attachment_remapped": attachment_remapped,
            "obj_b_id": relation_obj_b_id,
            "obj_b_label": obj_b_label,
            "obj_c_id": relation_obj_c_id,
            "obj_c_label": obj_c_label,
            "mentioned_objects": [
                _mention("moved_object", move_source["label"], move_source_id),
                _mention("query_object", query_obj["label"], query_obj["id"]),
                _mention("relation_obj_b", obj_b_label, relation_obj_b_id),
                _mention("relation_obj_c", obj_c_label, relation_obj_c_id),
            ],
            "delta": delta.tolist(),
            "relation_unchanged": relation_unchanged,
            "has_attachment_chain": has_attachment_chain,
        })

    return questions


def _cap_question_groups(
    questions_by_key: dict[Any, list[dict]],
    max_per_group: int | None,
) -> list[dict]:
    """Downsample question pools while prioritizing changed attachment questions.

    Attachment-mediated questions are harder to obtain, but unchanged
    attachment questions should not crowd out changed attachment questions from
    the same group.
    """
    groups = list(questions_by_key.values())
    if max_per_group is None or max_per_group <= 0:
        return [q for group in groups for q in group]

    capped: list[dict] = []
    for group in groups:
        if len(group) <= max_per_group:
            capped.extend(group)
            continue

        attachment_changed = [
            question for question in group
            if bool(question.get("attachment_remapped", False))
            and not bool(question.get("relation_unchanged", False))
        ]
        if len(attachment_changed) >= max_per_group:
            sampled_ids = {
                id(question)
                for question in random.sample(attachment_changed, max_per_group)
            }
            capped.extend([
                question for question in attachment_changed
                if id(question) in sampled_ids
            ])
            continue

        attachment_unchanged = [
            question for question in group
            if bool(question.get("attachment_remapped", False))
            and bool(question.get("relation_unchanged", False))
        ]
        capped.extend(attachment_changed)
        remaining_slots = max(0, max_per_group - len(attachment_changed))
        if remaining_slots == 0:
            continue

        if len(attachment_unchanged) <= remaining_slots:
            kept_attachment_unchanged = attachment_unchanged
        else:
            sampled_ids = {
                id(question)
                for question in random.sample(attachment_unchanged, remaining_slots)
            }
            kept_attachment_unchanged = [
                question for question in attachment_unchanged
                if id(question) in sampled_ids
            ]
        capped.extend(kept_attachment_unchanged)
        remaining_slots = max(0, max_per_group - len(attachment_changed) - len(kept_attachment_unchanged))
        if remaining_slots == 0:
            continue

        unprotected = [
            question for question in group
            if not bool(question.get("attachment_remapped", False))
        ]
        if len(unprotected) <= remaining_slots:
            kept_unprotected = unprotected
        else:
            sampled_ids = {id(question) for question in random.sample(unprotected, remaining_slots)}
            kept_unprotected = [
                question for question in unprotected
                if id(question) in sampled_ids
            ]
        capped.extend(kept_unprotected)
    return capped


def _balance_l2_object_move_attachment_counts(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cap per-type unattached L2 object-move questions at 2x attached count."""
    keep_mask = [True] * len(questions)
    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        if str(question.get("level", "")).strip() != "L2":
            continue
        qtype = str(question.get("type", "")).strip()
        if not qtype.startswith("object_move_"):
            continue
        grouped_indices.setdefault(qtype, []).append(idx)

    for qtype, indices in grouped_indices.items():
        attached = [
            idx for idx in indices
            if bool(questions[idx].get("attachment_remapped", False))
        ]
        unattached = [
            idx for idx in indices
            if not bool(questions[idx].get("attachment_remapped", False))
        ]
        allowed_unattached = (2 * len(attached)) if attached else 3
        if len(unattached) <= allowed_unattached:
            continue
        for idx in unattached[allowed_unattached:]:
            keep_mask[idx] = False
        logger.info(
            "Balanced %s questions for one frame: kept %d attached and %d/%d unattached",
            qtype,
            len(attached),
            allowed_unattached,
            len(unattached),
        )

    return [
        question for idx, question in enumerate(questions)
        if keep_mask[idx]
    ]

def generate_l2_object_move(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
    color_intrinsics: CameraIntrinsics | None = None,
    occlusion_backend: str = "depth",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
) -> list[dict]:
    """Generate L2.1 object-movement questions for a scene."""
    questions_by_object: dict[int, list[dict]] = {}
    referable_object_ids = {int(o["id"]) for o in objects}
    movement_scene_objects = movement_objects if movement_objects is not None else objects
    obj_map = object_map if object_map is not None else {
        int(o["id"]): o for o in movement_scene_objects
    }
    base_relations = compute_all_relations(movement_scene_objects, camera_pose, None, None)
    base_relation_map = _relation_map_by_pair(base_relations)
    occlusion_enabled = (
        color_intrinsics is not None
        and ray_caster is not None
        and instance_mesh_data is not None
    )
    compare_backend = None
    original_visibility: dict[int, tuple[str | None, str, str, _L1OcclusionMetrics]] = {}
    if occlusion_enabled:
        compare_backend = _counterfactual_occlusion_backend(
            occlusion_backend,
            ray_caster,
            instance_mesh_data,
        )
        original_scene_context = _build_modified_scene(
            ray_caster,
            instance_mesh_data,
            set(),
        )
        for visibility_obj in movement_scene_objects:
            metrics, source_used = _compute_l1_style_visibility_metrics_for_static_target(
                obj=visibility_obj,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend=compare_backend,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
                modified_scene=original_scene_context,
            )
            status, reason_code, _reason_detail = _resolve_counterfactual_l1_visibility_status(
                metrics
            )
            original_visibility[int(visibility_obj["id"])] = (
                status,
                source_used,
                reason_code,
                metrics,
            )

    for obj in objects:
        # Skip structural room elements — they cannot be "moved" in any
        # meaningful physical sense and confuse human annotators.
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        move_source_id = _resolve_attachment_root_id(obj["id"], attached_by)
        move_source = obj_map.get(move_source_id)
        if move_source is None:
            continue
        attachment_remapped = move_source_id != obj["id"]
        if attachment_remapped and move_source_id not in referable_object_ids:
            continue
        has_attachment_chain = len(get_attachment_chain_ids(move_source_id, attachment_graph)) > 0
        selected_state = _select_object_move_state(
            movement_scene_objects,
            attachment_graph,
            move_source_id,
            camera_pose,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
            allow_unchanged_attachment=attachment_remapped,
            color_intrinsics=color_intrinsics,
            occlusion_backend=occlusion_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
        )
        obj_questions: list[dict] = []
        if selected_state is not None:
            delta = selected_state.delta
            direction_desc = _delta_to_description(delta, camera_pose)
            distance_desc = f"{np.linalg.norm(delta):.1f}m"
            moved_relation_map = _relation_map_by_pair(
                compute_all_relations(selected_state.moved_objects, camera_pose, None, None)
            )
            query_old_status: str | None = None
            query_old_source = "mesh_ray"
            query_old_reason_code = "missing_original_visibility"
            query_old_metrics = _make_l1_occlusion_metrics(0.0, 0.0, 1.0, 0, 0, 0, "mesh_ray")
            query_new_status: str | None = None
            query_new_source = "mesh_ray"
            query_new_reason_code = "counterfactual_visibility_unresolved"
            query_new_metrics = _make_l1_occlusion_metrics(0.0, 0.0, 1.0, 0, 0, 0, "mesh_ray")
            if occlusion_enabled and compare_backend is not None:
                (
                    query_old_status,
                    query_old_source,
                    query_old_reason_code,
                    query_old_metrics,
                ) = original_visibility.get(
                    int(obj["id"]),
                    (
                        None,
                        "mesh_ray",
                        "missing_original_visibility",
                        _make_l1_occlusion_metrics(0.0, 0.0, 1.0, 0, 0, 0, "mesh_ray"),
                    ),
                )
                query_new_metrics, query_new_source = _compute_l1_style_visibility_metrics_for_moved_target(
                    target_obj_id=int(obj["id"]),
                    original_objects=movement_scene_objects,
                    moved_objects=selected_state.moved_objects,
                    moved_ids=selected_state.moved_ids,
                    camera_pose=camera_pose,
                    color_intrinsics=color_intrinsics,
                    occlusion_backend=compare_backend,
                    ray_caster=ray_caster,
                    instance_mesh_data=instance_mesh_data,
                )
                (
                    query_new_status,
                    query_new_reason_code,
                    _query_new_reason_detail,
                ) = _resolve_counterfactual_l1_visibility_status(query_new_metrics)

            for key, old_relation in base_relation_map.items():
                if obj["id"] not in key:
                    continue
                new_relation = moved_relation_map.get(key)
                if new_relation is None:
                    continue

                relation_obj_b_id = int(obj["id"])
                relation_obj_c_id = _other_obj_id_for_query(obj["id"], old_relation)
                if relation_obj_c_id is None:
                    continue
                obj_b_label = obj_map.get(relation_obj_b_id, {}).get("label", "object")
                obj_c_label = obj_map.get(relation_obj_c_id, {}).get("label", "object")

                direction_values = _direction_values_for_query_object(
                    obj["id"],
                    old_relation,
                    new_relation,
                )
                if direction_values is not None:
                    old_value, new_value = direction_values
                    relation_unchanged = old_value == new_value
                    if attachment_remapped or not relation_unchanged:
                        tpl_list = templates.get(
                            "L2_object_move_agent",
                            _default_templates()["L2_object_move_agent"],
                        )
                        tpl = random.choice(tpl_list)
                        question_text = tpl.format(
                            obj_a=_the(move_source["label"]),
                            direction=direction_desc,
                            direction_with_camera_hint=_direction_with_camera_hint(direction_desc),
                            distance=distance_desc,
                            obj_b=_the(obj_b_label),
                            obj_c=_the(obj_c_label),
                        )
                        options, answer = generate_options(new_value, ALL_DIRECTIONS)
                        obj_questions.append({
                            "level": "L2",
                            "type": "object_move_agent",
                            "question": question_text,
                            "options": options,
                            "answer": answer,
                            "correct_value": new_value,
                            "old_correct_value": old_value,
                            "new_correct_value": new_value,
                            "moved_obj_id": move_source_id,
                            "moved_obj_label": move_source["label"],
                            "query_obj_id": obj["id"],
                            "query_obj_label": obj["label"],
                            "attachment_remapped": attachment_remapped,
                            "obj_b_id": relation_obj_b_id,
                            "obj_b_label": obj_b_label,
                            "obj_c_id": relation_obj_c_id,
                            "obj_c_label": obj_c_label,
                            "mentioned_objects": [
                                _mention("moved_object", move_source["label"], move_source_id),
                                _mention("query_object", obj["label"], obj["id"]),
                                _mention("relation_obj_b", obj_b_label, relation_obj_b_id),
                                _mention("relation_obj_c", obj_c_label, relation_obj_c_id),
                            ],
                            "delta": delta.tolist(),
                            "relation_unchanged": relation_unchanged,
                            "has_attachment_chain": has_attachment_chain,
                        })

            if (
                occlusion_enabled
                and compare_backend is not None
                and query_old_status is not None
                and query_new_status is not None
                and query_old_status != query_new_status
            ):
                tpl_list = templates.get(
                    "L2_object_move_occlusion",
                    _default_templates()["L2_object_move_occlusion"],
                )
                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    obj_a=_the(move_source["label"]),
                    direction=direction_desc,
                    direction_with_camera_hint=_direction_with_camera_hint(direction_desc),
                    distance=distance_desc,
                    obj_b=_the(obj["label"]),
                    obj_target=_the(obj["label"]),
                )
                question_text = _with_occlusion_definition(question_text)
                options, answer = generate_options(
                    query_new_status,
                    L1_OCCLUSION_STATES,
                    n_options=3,
                )
                obj_questions.append({
                    "level": "L2",
                    "type": "object_move_occlusion",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": query_new_status,
                    "old_correct_value": query_old_status,
                    "new_correct_value": query_new_status,
                    "old_visibility": query_old_status,
                    "new_visibility": query_new_status,
                    "old_visibility_source": query_old_source,
                    "new_visibility_source": query_new_source,
                    "old_visibility_resolution": query_old_reason_code,
                    "new_visibility_resolution": query_new_reason_code,
                    "old_visibility_metrics": _l1_occlusion_metrics_payload(query_old_metrics),
                    "new_visibility_metrics": _l1_occlusion_metrics_payload(query_new_metrics),
                    "moved_obj_id": move_source_id,
                    "moved_obj_label": move_source["label"],
                    "target_obj_id": obj["id"],
                    "target_obj_label": obj["label"],
                    "query_obj_id": obj["id"],
                    "query_obj_label": obj["label"],
                    "attachment_remapped": attachment_remapped,
                    "obj_b_id": obj["id"],
                    "obj_b_label": obj["label"],
                    "mentioned_objects": [
                        _mention("moved_object", move_source["label"], move_source_id),
                        _mention("target_object", obj["label"], obj["id"]),
                    ],
                    "delta": delta.tolist(),
                    "relation_unchanged": False,
                    "has_attachment_chain": has_attachment_chain,
                })

        obj_questions.extend(
            _generate_l2_distance_questions_for_object(
                query_obj=obj,
                move_source=move_source,
                move_source_id=move_source_id,
                attachment_remapped=attachment_remapped,
                relations=base_relations,
                movement_scene_objects=movement_scene_objects,
                attachment_graph=attachment_graph,
                camera_pose=camera_pose,
                templates=templates,
                obj_map=obj_map,
                room_bounds=room_bounds,
                collision_objects=collision_objects,
            )
        )

        if obj_questions:
            object_key = int(obj["id"])
            questions_by_object.setdefault(object_key, []).extend(obj_questions)

    # Cap per query-object instance so repeated labels can still contribute.
    return _cap_question_groups(questions_by_object, max_per_object)


def generate_l2_viewpoint_move(
    objects: list[dict],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    templates: dict,
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
    trace_detail: str = "light",
) -> list[dict]:
    """Generate L2.2 viewpoint-movement questions.

    Compares target-object visibility before/after moving the observer.
    """
    questions: list[dict] = []
    if color_intrinsics is None:
        _emit_generator_summary(
            trace_recorder,
            "generate_l2_viewpoint_move",
            generated_count=0,
            candidate_count=0,
            generated_candidate_count=0,
            skipped_candidate_count=0,
            reason_counts={"missing_color_intrinsics": 1},
            details={"occlusion_mode": "l1_style"},
        )
        return questions
    tpl_list = templates.get("L2_viewpoint_move", _default_templates()["L2_viewpoint_move"])
    compare_backend = _counterfactual_occlusion_backend(
        occlusion_backend, ray_caster, instance_mesh_data,
    )
    scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set())
    reason_counts: Counter[str] = Counter()
    candidate_count = int(len(objects) * 4 * 3)
    generated_candidate_count = 0
    original_visibility: dict[int, tuple[str | None, str, str, _L1OcclusionMetrics]] = {}
    for obj in objects:
        metrics, source_used = _compute_l1_style_visibility_metrics_for_static_target(
            obj=obj,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            occlusion_backend=compare_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            modified_scene=scene_context,
        )
        status, reason_code, reason_detail = _resolve_counterfactual_l1_visibility_status(metrics)
        original_visibility[int(obj["id"])] = (status, source_used, reason_code, metrics)

    for direction, prompt_direction in (
        ("right", "right"),
        ("left", "left"),
        ("forward", "forward"),
        ("back", "backward"),
    ):
        for dist in (1.0, 2.0, 3.0):
            new_pose = apply_viewpoint_change(camera_pose, direction, dist)
            for obj in objects:
                candidate_key = _candidate_key(direction, f"{dist:.1f}", obj["id"])
                object_ids = [int(obj["id"])]
                old_status, old_source, old_reason_code, old_metrics = original_visibility.get(
                    int(obj["id"]),
                    (None, "mesh_ray", "missing_original_visibility", _make_l1_occlusion_metrics(0.0, 0.0, 1.0, 0, 0, 0, "mesh_ray")),
                )
                if old_status is None:
                    reason_counts["original_visibility_unresolved"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l2_viewpoint_move",
                        candidate_kind="viewpoint_target",
                        candidate_key=candidate_key,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="original_visibility_unresolved",
                        reason_detail=f"original visibility could not be resolved: {old_reason_code}",
                        evidence={
                            "camera_translation": prompt_direction,
                            "distance_m": float(dist),
                            "original_source": old_source,
                            "original_resolution": old_reason_code,
                            "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                        },
                    )
                    continue

                new_metrics, new_source = _compute_l1_style_visibility_metrics_for_static_target(
                    obj=obj,
                    camera_pose=new_pose,
                    color_intrinsics=color_intrinsics,
                    depth_image=None,
                    depth_intrinsics=None,
                    occlusion_backend=compare_backend,
                    ray_caster=ray_caster,
                    instance_mesh_data=instance_mesh_data,
                    modified_scene=scene_context,
                )
                new_status, new_reason_code, new_reason_detail = _resolve_counterfactual_l1_visibility_status(new_metrics)
                if new_status is None:
                    reason_counts["counterfactual_visibility_unresolved"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l2_viewpoint_move",
                        candidate_kind="viewpoint_target",
                        candidate_key=candidate_key,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="counterfactual_visibility_unresolved",
                        reason_detail=f"counterfactual visibility could not be resolved: {new_reason_detail}",
                        evidence={
                            "camera_translation": prompt_direction,
                            "distance_m": float(dist),
                            "original_status": old_status,
                            "original_source": old_source,
                            "new_source": new_source,
                            "new_resolution": new_reason_code,
                            "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                            "new_metrics": _l1_occlusion_metrics_payload(new_metrics),
                        },
                    )
                    continue

                if old_status == new_status:
                    reason_counts["visibility_unchanged"] += 1
                    _emit_generator_candidate(
                        trace_recorder,
                        trace_detail=trace_detail,
                        generator="generate_l2_viewpoint_move",
                        candidate_kind="viewpoint_target",
                        candidate_key=candidate_key,
                        object_ids=object_ids,
                        status="skipped",
                        reason_code="visibility_unchanged",
                        reason_detail="camera motion does not change the L1-style occlusion state of the target",
                        evidence={
                            "camera_translation": prompt_direction,
                            "distance_m": float(dist),
                            "original_status": old_status,
                            "new_status": new_status,
                            "original_source": old_source,
                            "new_source": new_source,
                            "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                            "new_metrics": _l1_occlusion_metrics_payload(new_metrics),
                        },
                    )
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    direction=prompt_direction,
                    direction_with_camera_hint=_direction_with_camera_hint(
                        prompt_direction,
                        moving_subject="camera",
                    ),
                    distance=f"{dist:.0f}m",
                    obj_a=_the(obj["label"]),
                )
                question_text = _with_occlusion_definition(question_text)
                options, answer = generate_options(
                    new_status,
                    L1_OCCLUSION_STATES,
                    n_options=3,
                )
                questions.append({
                    "level": "L2",
                    "type": "viewpoint_move",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": new_status,
                    "obj_a_id": obj["id"],
                    "obj_a_label": obj["label"],
                    "old_visibility": old_status,
                    "new_visibility": new_status,
                    "old_visibility_source": old_source,
                    "new_visibility_source": new_source,
                    "old_visibility_metrics": _l1_occlusion_metrics_payload(old_metrics),
                    "new_visibility_metrics": _l1_occlusion_metrics_payload(new_metrics),
                    "camera_motion_model": "translate_only",
                    "camera_intrinsics_unchanged": True,
                    "camera_orientation_unchanged": True,
                    "mentioned_objects": [
                        _mention("target", obj["label"], obj["id"]),
                    ],
                    "relation_unchanged": False,
                })
                generated_candidate_count += 1
                reason_counts["generated"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l2_viewpoint_move",
                    candidate_kind="viewpoint_target",
                    candidate_key=candidate_key,
                    object_ids=object_ids,
                    status="generated",
                    reason_code="generated",
                    reason_detail="camera motion changes the target's L1-style occlusion state",
                    evidence={
                        "camera_translation": prompt_direction,
                        "distance_m": float(dist),
                        "original_status": old_status,
                        "new_status": new_status,
                        "original_source": old_source,
                        "new_source": new_source,
                        "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                        "new_metrics": _l1_occlusion_metrics_payload(new_metrics),
                    },
                    question_preview=_question_preview_payload(questions[-1]),
                )

    _emit_generator_summary(
        trace_recorder,
        "generate_l2_viewpoint_move",
        generated_count=len(questions),
        candidate_count=candidate_count,
        generated_candidate_count=generated_candidate_count,
        skipped_candidate_count=max(candidate_count - generated_candidate_count, 0),
        reason_counts=dict(reason_counts),
        details={
            "compare_backend": compare_backend,
            "occlusion_mode": "l1_style",
        },
    )
    return questions


def generate_l2_object_remove(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    camera_pose: CameraPose,
    color_intrinsics: CameraIntrinsics | None,
    depth_image,
    depth_intrinsics,
    occlusion_backend: str,
    ray_caster,
    instance_mesh_data: InstanceMeshData | None,
    templates: dict,
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
    trace_detail: str = "light",
) -> list[dict]:
    """Generate L2.3 object-removal questions from visibility changes."""
    questions: list[dict] = []
    if color_intrinsics is None:
        _emit_generator_summary(
            trace_recorder,
            "generate_l2_object_remove",
            generated_count=0,
            candidate_count=0,
            generated_candidate_count=0,
            skipped_candidate_count=0,
            reason_counts={"missing_color_intrinsics": 1},
            details={"occlusion_mode": "l1_style"},
        )
        return questions
    tpl_list = templates.get("L2_object_remove", _default_templates()["L2_object_remove"])
    compare_backend = _counterfactual_occlusion_backend(
        occlusion_backend, ray_caster, instance_mesh_data,
    )
    original_scene_context = _build_modified_scene(ray_caster, instance_mesh_data, set())
    original_visibility: dict[int, tuple[str | None, str, str, _L1OcclusionMetrics]] = {}
    for obj in objects:
        metrics, source_used = _compute_l1_style_visibility_metrics_for_static_target(
            obj=obj,
            camera_pose=camera_pose,
            color_intrinsics=color_intrinsics,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
            occlusion_backend=compare_backend,
            ray_caster=ray_caster,
            instance_mesh_data=instance_mesh_data,
            modified_scene=original_scene_context,
        )
        status, reason_code, _reason_detail = _resolve_counterfactual_l1_visibility_status(metrics)
        original_visibility[int(obj["id"])] = (status, source_used, reason_code, metrics)

    original_ids = {int(obj["id"]) for obj in objects}
    candidate_count = 0
    generated_candidate_count = 0
    reason_counts: Counter[str] = Counter()
    for obj in objects:
        remaining = apply_removal(objects, obj["id"])
        if len(remaining) < 2:
            reason_counts["removal_leaves_too_few_objects"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l2_object_remove",
                candidate_kind="removed_object",
                candidate_key=_candidate_key(obj["id"]),
                object_ids=[int(obj["id"])],
                status="skipped",
                reason_code="removal_leaves_too_few_objects",
                reason_detail="removing this object leaves fewer than two remaining objects to compare",
                evidence={
                    "remaining_object_count": int(len(remaining)),
                },
            )
            continue
        remaining_ids = {int(other["id"]) for other in remaining}
        removed_ids = original_ids - remaining_ids
        removal_scene_context = _build_modified_scene(
            ray_caster, instance_mesh_data, removed_ids,
        )

        for other in remaining:
            candidate_count += 1
            candidate_key = _candidate_key(obj["id"], other["id"])
            object_ids = [int(obj["id"]), int(other["id"])]
            old_status, old_source, old_reason_code, old_metrics = original_visibility.get(
                int(other["id"]),
                (None, "mesh_ray", "missing_original_visibility", _make_l1_occlusion_metrics(0.0, 0.0, 1.0, 0, 0, 0, "mesh_ray")),
            )
            if old_status is None:
                reason_counts["original_visibility_unresolved"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l2_object_remove",
                    candidate_kind="removal_pair",
                    candidate_key=candidate_key,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="original_visibility_unresolved",
                    reason_detail=f"original visibility of the remaining object could not be resolved: {old_reason_code}",
                    evidence={
                        "removed_ids": sorted(int(removed_id) for removed_id in removed_ids),
                        "original_source": old_source,
                        "original_resolution": old_reason_code,
                        "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                    },
                )
                continue

            new_metrics, new_source = _compute_l1_style_visibility_metrics_for_static_target(
                obj=other,
                camera_pose=camera_pose,
                color_intrinsics=color_intrinsics,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend=compare_backend,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
                modified_scene=removal_scene_context,
            )
            new_status, new_reason_code, new_reason_detail = _resolve_counterfactual_l1_visibility_status(new_metrics)
            if new_status is None:
                reason_counts["counterfactual_visibility_unresolved"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l2_object_remove",
                    candidate_kind="removal_pair",
                    candidate_key=candidate_key,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="counterfactual_visibility_unresolved",
                    reason_detail=f"post-removal visibility could not be resolved: {new_reason_detail}",
                    evidence={
                        "removed_ids": sorted(int(removed_id) for removed_id in removed_ids),
                        "original_status": old_status,
                        "original_source": old_source,
                        "new_source": new_source,
                        "new_resolution": new_reason_code,
                        "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                        "new_metrics": _l1_occlusion_metrics_payload(new_metrics),
                    },
                )
                continue

            if old_status == new_status:
                reason_counts["visibility_unchanged"] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l2_object_remove",
                    candidate_kind="removal_pair",
                    candidate_key=candidate_key,
                    object_ids=object_ids,
                    status="skipped",
                    reason_code="visibility_unchanged",
                    reason_detail="removing this object does not change the remaining object's L1-style occlusion state",
                    evidence={
                        "removed_ids": sorted(int(removed_id) for removed_id in removed_ids),
                        "original_status": old_status,
                        "new_status": new_status,
                        "original_source": old_source,
                        "new_source": new_source,
                        "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                        "new_metrics": _l1_occlusion_metrics_payload(new_metrics),
                    },
                )
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                obj_a=_the(obj["label"]),
                obj_b=_the(other["label"]),
            )
            question_text = _with_occlusion_definition(question_text)
            options, answer = generate_options(
                new_status,
                L1_OCCLUSION_STATES,
                n_options=3,
            )
            questions.append({
                "level": "L2",
                "type": "object_remove",
                "question": question_text,
                "options": options,
                "answer": answer,
                "correct_value": new_status,
                "removed_obj_id": obj["id"],
                "removed_obj_label": obj["label"],
                "obj_b_id": other["id"],
                "obj_b_label": other["label"],
                "old_visibility": old_status,
                "new_visibility": new_status,
                "old_visibility_source": old_source,
                "new_visibility_source": new_source,
                "old_visibility_metrics": _l1_occlusion_metrics_payload(old_metrics),
                "new_visibility_metrics": _l1_occlusion_metrics_payload(new_metrics),
                "removed_ids": sorted(int(removed_id) for removed_id in removed_ids),
                "mentioned_objects": [
                    _mention("removed_object", obj["label"], obj["id"]),
                    _mention("remaining_object", other["label"], other["id"]),
                ],
                "relation_unchanged": False,
            })
            generated_candidate_count += 1
            reason_counts["generated"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l2_object_remove",
                candidate_kind="removal_pair",
                candidate_key=candidate_key,
                object_ids=object_ids,
                status="generated",
                reason_code="generated",
                reason_detail="removing the object changes the remaining object's L1-style occlusion state",
                evidence={
                    "removed_ids": sorted(int(removed_id) for removed_id in removed_ids),
                    "original_status": old_status,
                    "new_status": new_status,
                    "original_source": old_source,
                    "new_source": new_source,
                    "original_metrics": _l1_occlusion_metrics_payload(old_metrics),
                    "new_metrics": _l1_occlusion_metrics_payload(new_metrics),
                },
                question_preview=_question_preview_payload(questions[-1]),
            )

    _emit_generator_summary(
        trace_recorder,
        "generate_l2_object_remove",
        generated_count=len(questions),
        candidate_count=candidate_count,
        generated_candidate_count=generated_candidate_count,
        skipped_candidate_count=max(candidate_count - generated_candidate_count, 0),
        reason_counts=dict(reason_counts),
        details={
            "compare_backend": compare_backend,
            "occlusion_mode": "l1_style",
        },
    )
    return questions


# ---------------------------------------------------------------------------
#  L2 generators — new reference frames
# ---------------------------------------------------------------------------

def generate_l2_object_rotate_object_centric(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
) -> list[dict]:
    """L2 object-rotation questions answered in a query-centric object-centric frame."""
    questions_by_object: dict[int, list[dict]] = {}
    referable_object_ids = {int(o["id"]) for o in objects}
    movement_scene_objects = movement_objects if movement_objects is not None else objects
    obj_map = object_map if object_map is not None else {
        int(o["id"]): o for o in movement_scene_objects
    }
    tpl_list = templates.get(
        "L2_object_rotate_object_centric",
        _default_templates()["L2_object_rotate_object_centric"],
    )
    horizontal_answer_pool = list(HORIZONTAL_DIRECTIONS)

    for obj in objects:
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        move_source_id = _resolve_attachment_root_id(obj["id"], attached_by)
        move_source = obj_map.get(move_source_id)
        if move_source is None:
            continue
        moved_ids = set(get_attachment_chain_ids(move_source_id, attachment_graph)) | {move_source_id}
        if obj["id"] not in moved_ids:
            continue
        attachment_remapped = move_source_id != obj["id"]
        if attachment_remapped and move_source_id not in referable_object_ids:
            continue
        has_attachment_chain = len(moved_ids) > 1
        query_center = np.array(obj["center"], dtype=float)

        obj_questions: list[dict] = []
        for face in objects:
            if face["id"] in moved_ids:
                continue
            face_c = np.array(face["center"], dtype=float)
            if not _has_stable_object_centric_facing(query_center, face_c):
                continue

            valid_rotations = find_meaningful_orbit_rotation(
                movement_scene_objects,
                attachment_graph,
                move_source_id,
                face["id"],
                room_bounds=room_bounds,
                collision_objects=collision_objects,
            )
            if not valid_rotations:
                continue

            for rotation in valid_rotations:
                rotated_map = {o["id"]: o for o in rotation["objects"]}
                rotated_query = rotated_map.get(obj["id"])
                if rotated_query is None:
                    continue
                new_query_center = np.array(rotated_query["center"], dtype=float)
                if not _has_stable_object_centric_facing(new_query_center, face_c):
                    continue
                query_delta = new_query_center - query_center

                for ref in objects:
                    if ref["id"] in moved_ids or ref["id"] == face["id"]:
                        continue

                    labels = [
                        obj["label"],
                        ref["label"],
                        face["label"],
                    ]
                    if move_source_id != obj["id"]:
                        labels.append(move_source["label"])
                    if len(set(labels)) < len(labels):
                        continue

                    ref_c = np.array(ref["center"], dtype=float)
                    old_dir, old_amb = primary_direction_object_centric(
                        query_center,
                        face_c,
                        ref_c,
                        anchor_hull_xy=_object_bottom_hull_xy(obj),
                        target_hull_xy=_object_bottom_hull_xy(ref),
                        anchor_bbox_min=np.array(obj["bbox_min"], dtype=float),
                        anchor_bbox_max=np.array(obj["bbox_max"], dtype=float),
                        target_bbox_min=np.array(ref["bbox_min"], dtype=float),
                        target_bbox_max=np.array(ref["bbox_max"], dtype=float),
                    )
                    new_dir, new_amb = primary_direction_object_centric(
                        new_query_center,
                        face_c,
                        ref_c,
                        anchor_hull_xy=_object_bottom_hull_xy(rotated_query),
                        target_hull_xy=_object_bottom_hull_xy(ref),
                        anchor_bbox_min=np.array(rotated_query["bbox_min"], dtype=float),
                        anchor_bbox_max=np.array(rotated_query["bbox_max"], dtype=float),
                        target_bbox_min=np.array(ref["bbox_min"], dtype=float),
                        target_bbox_max=np.array(ref["bbox_max"], dtype=float),
                    )
                    if old_dir not in horizontal_answer_pool or new_dir not in horizontal_answer_pool:
                        continue
                    if max(old_amb, new_amb) > 0.7:
                        continue
                    relation_unchanged = old_dir == new_dir
                    if relation_unchanged and not attachment_remapped:
                        continue

                    tpl = random.choice(tpl_list)
                    question_text = tpl.format(
                        obj_move_source=_the(move_source["label"]),
                        obj_query=_the(obj["label"]),
                        obj_ref=_the(ref["label"]),
                        obj_face=_the(face["label"]),
                        angle=rotation["angle"],
                        rotation_direction=rotation["rotation_direction"],
                    )
                    options, answer = generate_options(new_dir, horizontal_answer_pool)
                    obj_questions.append({
                        "level": "L2",
                        "type": "object_rotate_object_centric",
                        "reference_frame": "object_centric",
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "correct_value": new_dir,
                        "old_correct_value": old_dir,
                        "new_correct_value": new_dir,
                        "moved_obj_id": move_source_id,
                        "moved_obj_label": move_source["label"],
                        "query_obj_id": obj["id"],
                        "query_obj_label": obj["label"],
                        "attachment_remapped": attachment_remapped,
                        "obj_ref_id": ref["id"],
                        "obj_ref_label": ref["label"],
                        "obj_face_id": face["id"],
                        "obj_face_label": face["label"],
                        "facing_anchor_center": new_query_center.tolist(),
                        "facing_target_center": face_c.tolist(),
                        "rotation_angle": rotation["angle"],
                        "rotation_direction": rotation["rotation_direction"],
                        "mentioned_objects": [
                            _mention("moved_object", move_source["label"], move_source_id),
                            _mention("query_object", obj["label"], obj["id"]),
                            _mention("reference_object", ref["label"], ref["id"]),
                            _mention("reference_facing", face["label"], face["id"]),
                        ],
                        "delta": query_delta.tolist(),
                        "relation_unchanged": relation_unchanged,
                        "has_attachment_chain": has_attachment_chain,
                    })

        if obj_questions:
            object_key = int(obj["id"])
            questions_by_object.setdefault(object_key, []).extend(obj_questions)

    return _cap_question_groups(questions_by_object, max_per_object)


def generate_l2_object_move_object_centric(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
) -> list[dict]:
    """Backward-compatible alias for the renamed rotation-based L2 generator."""
    return generate_l2_object_rotate_object_centric(
        objects,
        attachment_graph,
        attached_by,
        camera_pose,
        templates,
        max_per_object=max_per_object,
        room_bounds=room_bounds,
        collision_objects=collision_objects,
        movement_objects=movement_objects,
        object_map=object_map,
    )


def generate_l2_object_move_allocentric(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
    max_per_object: int = 3,
    room_bounds: dict | None = None,
    collision_objects: list[dict] | None = None,
    movement_objects: list[dict] | None = None,
    object_map: dict[int, dict] | None = None,
) -> list[dict]:
    """L2 object-move questions answered in allocentric (cardinal) frame."""
    questions_by_object: dict[int, list[dict]] = {}
    referable_object_ids = {int(o["id"]) for o in objects}
    movement_scene_objects = movement_objects if movement_objects is not None else objects
    obj_map = object_map if object_map is not None else {
        int(o["id"]): o for o in movement_scene_objects
    }
    cam_cardinal = camera_cardinal_direction(camera_pose)
    tpl_list = templates.get(
        "L2_object_move_allocentric",
        _default_templates()["L2_object_move_allocentric"],
    )

    for obj in objects:
        if obj.get("label", "").lower() in EXCLUDED_LABELS:
            continue

        move_source_id = _resolve_attachment_root_id(obj["id"], attached_by)
        move_source = obj_map.get(move_source_id)
        if move_source is None:
            continue
        moved_ids = set(get_attachment_chain_ids(move_source_id, attachment_graph)) | {move_source_id}
        if obj["id"] not in moved_ids:
            continue
        attachment_remapped = move_source_id != obj["id"]
        if attachment_remapped and move_source_id not in referable_object_ids:
            continue

        selected_state = _select_object_move_state(
            movement_scene_objects,
            attachment_graph,
            move_source_id,
            camera_pose,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
            allow_unchanged_attachment=attachment_remapped,
        )
        if selected_state is None:
            continue

        delta = selected_state.delta
        direction_desc = _delta_to_cardinal_description(delta)
        distance_desc = f"{np.linalg.norm(delta):.1f}m"
        new_center = np.array(obj["center"]) + delta

        obj_questions: list[dict] = []
        for ref in objects:
            if ref["id"] in (obj["id"], move_source_id):
                continue
            if ref["label"] == obj["label"]:
                continue

            ref_c = np.array(ref["center"])
            new_bbox_min, new_bbox_max = _translated_bbox(obj, delta)
            new_dir, amb = primary_direction_allocentric(
                new_center,
                ref_c,
                obj_a_hull_xy=_translated_bottom_hull_xy(obj, delta),
                obj_b_hull_xy=_object_bottom_hull_xy(ref),
                obj_a_bbox_min=new_bbox_min,
                obj_a_bbox_max=new_bbox_max,
                obj_b_bbox_min=np.array(ref["bbox_min"], dtype=float),
                obj_b_bbox_max=np.array(ref["bbox_max"], dtype=float),
            )
            if amb > 0.7:
                continue
            if new_dir not in CARDINAL_DIRECTIONS_8:
                continue

            old_dir, _ = primary_direction_allocentric(
                np.array(obj["center"]),
                ref_c,
                obj_a_hull_xy=_object_bottom_hull_xy(obj),
                obj_b_hull_xy=_object_bottom_hull_xy(ref),
                obj_a_bbox_min=np.array(obj["bbox_min"], dtype=float),
                obj_a_bbox_max=np.array(obj["bbox_max"], dtype=float),
                obj_b_bbox_min=np.array(ref["bbox_min"], dtype=float),
                obj_b_bbox_max=np.array(ref["bbox_max"], dtype=float),
            )
            relation_unchanged = old_dir == new_dir
            if relation_unchanged and not attachment_remapped:
                continue

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                obj_move_source=_the(move_source["label"]),
                obj_query=_the(obj["label"]),
                direction=direction_desc,
                distance=distance_desc,
                camera_cardinal=cam_cardinal,
                obj_ref=_the(ref["label"]),
            )
            options, answer = generate_options(new_dir, ALL_DIRECTIONS_ALLOCENTRIC)
            obj_questions.append({
                "level": "L2",
                "type": "object_move_allocentric",
                "reference_frame": "allocentric",
                "question": question_text,
                "options": options,
                "answer": answer,
                "correct_value": new_dir,
                "old_correct_value": old_dir,
                "new_correct_value": new_dir,
                "camera_cardinal": cam_cardinal,
                "moved_obj_id": move_source_id,
                "moved_obj_label": move_source["label"],
                "query_obj_id": obj["id"],
                "query_obj_label": obj["label"],
                "attachment_remapped": attachment_remapped,
                "obj_ref_id": ref["id"],
                "obj_ref_label": ref["label"],
                "mentioned_objects": [
                    _mention("moved_object", move_source["label"], move_source_id),
                    _mention("query_object", obj["label"], obj["id"]),
                    _mention("reference_object", ref["label"], ref["id"]),
                ],
                "delta": delta.tolist(),
                "relation_unchanged": relation_unchanged,
            })

        if obj_questions:
            object_key = int(obj["id"])
            questions_by_object.setdefault(object_key, []).extend(obj_questions)

    return _cap_question_groups(questions_by_object, max_per_object)


# ---------------------------------------------------------------------------
#  Attachment relation generators
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  L3 generators
# ---------------------------------------------------------------------------

def generate_l3_attachment_chain(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    templates: dict,
) -> list[dict]:
    """Generate L3.1 support-chain membership questions (two-hop: A→B→C).

    Tests whether the model can identify ALL objects displaced when A moves.
    Requires 2-hop inference: A supports B (1-hop) AND B supports C (2-hop),
    so both B and C are displaced — not just the direct child B.

    The question does NOT state which objects are on which; the model must
    infer the full chain from the image.

    Options:
      - "{B}"               (1-hop child only  — wrong, misses C)
      - "{C}"               (2-hop grandchild only — wrong, misses B)
      - "Both {B} and {C}" (correct — full chain)
      - "{D}"               (non-chain neighbour — wrong)
    """
    questions: list[dict] = []
    obj_map = {o["id"]: o for o in objects}
    tpl_list = templates.get("L3_attachment_chain", _default_templates()["L3_attachment_chain"])

    for grandparent_id, parent_ids in attachment_graph.items():
        grandparent_id = int(grandparent_id)
        grandparent = obj_map.get(grandparent_id)
        if grandparent is None:
            continue

        for parent_id in parent_ids:
            parent_id = int(parent_id)
            # Second hop: does the intermediate object itself attach to anything?
            grandchild_ids = attachment_graph.get(parent_id) or []
            if not grandchild_ids:
                continue  # no depth-2 chain here

            parent = obj_map.get(parent_id)
            if parent is None:
                continue

            # All objects in this grandparent's chain (not eligible as neighbour D)
            this_chain: set[int] = (
                set(get_attachment_chain_ids(grandparent_id, attachment_graph))
                | {grandparent_id}
            )
            non_chain = [o for o in objects if o["id"] not in this_chain]
            if not non_chain:
                continue

            # Pick closest non-chain object as the distractor (option D)
            gp_center = np.array(grandparent["center"])
            neighbor = min(
                non_chain,
                key=lambda o: np.linalg.norm(np.array(o["center"]) - gp_center),
            )

            for grandchild_id in grandchild_ids:
                grandchild_id = int(grandchild_id)
                grandchild = obj_map.get(grandchild_id)
                if grandchild is None:
                    continue

                # Skip when labels collide — options would be ambiguous
                if len({parent["label"], grandchild["label"], neighbor["label"]}) < 3:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(obj_a=_the(grandparent["label"]))

                opt_parent    = _the(parent["label"])
                opt_grandchild = _the(grandchild["label"])
                opt_both      = f"Both {_the(parent['label'])} and {_the(grandchild['label'])}"
                opt_neighbor  = _the(neighbor["label"])

                options = [opt_parent, opt_grandchild, opt_both, opt_neighbor]
                random.shuffle(options)
                answer_letter = chr(65 + options.index(opt_both))

                questions.append({
                    "level": "L3",
                    "type": "attachment_chain",
                    "question": question_text,
                    "options": options,
                    "answer": answer_letter,
                    "correct_value": opt_both,
                    "chain_depth": 2,
                    "grandparent_id": grandparent_id,
                    "grandparent_label": grandparent["label"],
                    "parent_id": parent_id,
                    "parent_label": parent["label"],
                    "grandchild_id": grandchild_id,
                    "grandchild_label": grandchild["label"],
                    "neighbor_id": neighbor["id"],
                    "neighbor_label": neighbor["label"],
                    "mentioned_objects": [
                        _mention("grandparent", grandparent["label"], grandparent_id),
                        _mention("parent", parent["label"], parent_id),
                        _mention("grandchild", grandchild["label"], grandchild_id),
                        _mention("neighbor", neighbor["label"], neighbor["id"]),
                    ],
                    "relation_unchanged": False,
                })

    return questions


def generate_l3_coordinate_rotation(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_per_angle: int = 5,
) -> list[dict]:
    """Generate L3.2 coordinate-rotation counterfactual questions.

    Objects are rotated around the room centre; the camera pose is held fixed.
    The question asks for the *new direction* of A relative to B as seen from
    the unchanged camera — a 4-option direction question, not Yes/No.

    Using the actual direction as the answer prevents the trivial shortcut of
    always answering "No" (which would be correct for most 90°/180° cases).

    max_per_angle caps questions per rotation angle to avoid flooding when
    the scene has many objects (O(n²) pairs).
    """
    questions: list[dict] = []
    tpl_list = templates.get(
        "L3_coordinate_rotation_agent",
        templates.get(
            "L3_coordinate_rotation",
            _default_templates()["L3_coordinate_rotation_agent"],
        ),
    )

    # Direction changes don't need depth/occlusion; skip it for speed.
    original_relations = compute_all_relations(objects, camera_pose, None, None)

    for angle in (90, 180, 270):
        # rotation_matrix_z uses math convention (positive = counterclockwise).
        # Templates say "clockwise", so negate the angle for the actual rotation.
        rotated = apply_coordinate_rotation(objects, float(-angle))
        # camera_pose intentionally unchanged — objects rotate, camera does not
        new_relations = compute_all_relations(rotated, camera_pose, None, None)
        changed = find_changed_relations(original_relations, new_relations)

        # Collect only direction-changed pairs, then sample to cap
        changed_dir = [ch for ch in changed if "direction_b_rel_a" in ch["changes"]]
        if len(changed_dir) > max_per_angle:
            changed_dir = random.sample(changed_dir, max_per_angle)

        for ch in changed_dir:
            vals = ch["changes"]["direction_b_rel_a"]
            obj_a_label = next((o["label"] for o in objects if o["id"] == ch["obj_a_id"]), "object")
            obj_b_label = next((o["label"] for o in objects if o["id"] == ch["obj_b_id"]), "object")

            tpl = random.choice(tpl_list)
            question_text = tpl.format(
                angle=angle,
                obj_a=_the(obj_a_label),
                obj_b=_the(obj_b_label),
            )
            # The relation engine stores "B relative to A", while this template
            # asks for "A relative to B", so invert the direction before use.
            old_dir = _invert_direction(vals["old"])
            new_dir = _invert_direction(vals["new"])
            options, answer_letter = generate_options(new_dir, ALL_DIRECTIONS)

            questions.append({
                "level": "L3",
                "type": "coordinate_rotation_agent",
                "question": question_text,
                "options": options,
                "answer": answer_letter,
                "correct_value": new_dir,
                "rotation_angle": angle,
                "obj_a_id": ch["obj_a_id"],
                "obj_a_label": obj_a_label,
                "obj_b_id": ch["obj_b_id"],
                "obj_b_label": obj_b_label,
                "mentioned_objects": [
                    _mention("obj_a", obj_a_label, ch["obj_a_id"]),
                    _mention("obj_b", obj_b_label, ch["obj_b_id"]),
                ],
                "old_direction": old_dir,
                "new_direction": new_dir,
                "relation_unchanged": False,
            })

    return questions


def generate_l3_coordinate_rotation_object_centric(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_per_angle: int = 5,
    max_questions: int = 3,
) -> list[dict]:
    """L3 coordinate-rotation questions in object-centric frame.

    After rotating all objects, asks: standing at obj_ref's NEW position and
    facing obj_face's NEW position, where is obj_target?
    """
    questions: list[dict] = []
    tpl_list = templates.get(
        "L3_coordinate_rotation_object_centric",
        _default_templates()["L3_coordinate_rotation_object_centric"],
    )

    for angle in (90, 180, 270):
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {o["id"]: o for o in rotated}

        candidates: list[dict] = []
        n = len(objects)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ref = objects[i]
                    face = objects[j]
                    target = objects[k]
                    if len({ref["label"], face["label"], target["label"]}) < 3:
                        continue

                    # Use the rotated positions for the reference frame
                    ref_rot = rot_map.get(ref["id"])
                    face_rot = rot_map.get(face["id"])
                    target_rot = rot_map.get(target["id"])
                    if ref_rot is None or face_rot is None or target_rot is None:
                        continue
                    ref_rot_c = np.array(ref_rot["center"])
                    face_rot_c = np.array(face_rot["center"])
                    target_rot_c = np.array(target_rot["center"])
                    if not _has_stable_object_centric_facing(ref_rot_c, face_rot_c):
                        continue

                    new_dir, amb = primary_direction_object_centric(
                        ref_rot_c,
                        face_rot_c,
                        target_rot_c,
                        anchor_hull_xy=_object_bottom_hull_xy(ref_rot),
                        target_hull_xy=_object_bottom_hull_xy(target_rot),
                        anchor_bbox_min=np.array(ref_rot["bbox_min"], dtype=float),
                        anchor_bbox_max=np.array(ref_rot["bbox_max"], dtype=float),
                        target_bbox_min=np.array(target_rot["bbox_min"], dtype=float),
                        target_bbox_max=np.array(target_rot["bbox_max"], dtype=float),
                    )
                    if amb > 0.7:
                        continue

                    old_dir, _ = primary_direction_object_centric(
                        np.array(ref["center"]),
                        np.array(face["center"]),
                        np.array(target["center"]),
                        anchor_hull_xy=_object_bottom_hull_xy(ref),
                        target_hull_xy=_object_bottom_hull_xy(target),
                        anchor_bbox_min=np.array(ref["bbox_min"], dtype=float),
                        anchor_bbox_max=np.array(ref["bbox_max"], dtype=float),
                        target_bbox_min=np.array(target["bbox_min"], dtype=float),
                        target_bbox_max=np.array(target["bbox_max"], dtype=float),
                    )

                    tpl = random.choice(tpl_list)
                    question_text = tpl.format(
                        angle=angle,
                        obj_ref=_the(ref["label"]),
                        obj_face=_the(face["label"]),
                        obj_target=_the(target["label"]),
                    )
                    options, answer = generate_options(new_dir, ALL_DIRECTIONS)
                    candidates.append({
                        "level": "L3",
                        "type": "coordinate_rotation_object_centric",
                        "reference_frame": "object_centric",
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "correct_value": new_dir,
                        "rotation_angle": angle,
                        "obj_ref_id": ref["id"],
                        "obj_ref_label": ref["label"],
                        "obj_face_id": face["id"],
                        "obj_face_label": face["label"],
                        "facing_anchor_center": ref_rot_c.tolist(),
                        "facing_target_center": face_rot_c.tolist(),
                        "obj_target_id": target["id"],
                        "obj_target_label": target["label"],
                        "mentioned_objects": [
                            _mention("reference_origin", ref["label"], ref["id"]),
                            _mention("reference_facing", face["label"], face["id"]),
                            _mention("target", target["label"], target["id"]),
                        ],
                        "old_direction": old_dir,
                        "new_direction": new_dir,
                        "relation_unchanged": old_dir == new_dir,
                    })

        if len(candidates) > max_per_angle:
            candidates = random.sample(candidates, max_per_angle)
        questions.extend(candidates)

    if len(questions) > max_questions:
        questions = random.sample(questions, max_questions)

    return questions


def generate_l3_coordinate_rotation_allocentric(
    objects: list[dict],
    camera_pose: CameraPose,
    templates: dict,
    max_per_angle: int = 5,
) -> list[dict]:
    """L3 coordinate-rotation questions in allocentric (cardinal) frame.

    After rotating all objects, asks: on the floor plan, what cardinal
    direction is obj_a from obj_b?  (Camera + room axes stay fixed, so the
    objects' cardinal positions DO change.)
    """
    questions: list[dict] = []
    cam_cardinal = camera_cardinal_direction(camera_pose)
    tpl_list = templates.get(
        "L3_coordinate_rotation_allocentric",
        _default_templates()["L3_coordinate_rotation_allocentric"],
    )

    for angle in (90, 180, 270):
        rotated = apply_coordinate_rotation(objects, float(-angle))
        rot_map = {o["id"]: o for o in rotated}

        candidates: list[dict] = []
        n = len(objects)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = objects[i], objects[j]
                if a["label"] == b["label"]:
                    continue

                a_rot = rot_map.get(a["id"])
                b_rot = rot_map.get(b["id"])
                if a_rot is None or b_rot is None:
                    continue

                new_dir, amb = primary_direction_allocentric(
                    np.array(a_rot["center"]),
                    np.array(b_rot["center"]),
                    obj_a_hull_xy=_object_bottom_hull_xy(a_rot),
                    obj_b_hull_xy=_object_bottom_hull_xy(b_rot),
                    obj_a_bbox_min=np.array(a_rot["bbox_min"], dtype=float),
                    obj_a_bbox_max=np.array(a_rot["bbox_max"], dtype=float),
                    obj_b_bbox_min=np.array(b_rot["bbox_min"], dtype=float),
                    obj_b_bbox_max=np.array(b_rot["bbox_max"], dtype=float),
                )
                if amb > 0.7:
                    continue
                if new_dir not in CARDINAL_DIRECTIONS_8:
                    continue

                old_dir, _ = primary_direction_allocentric(
                    np.array(a["center"]),
                    np.array(b["center"]),
                    obj_a_hull_xy=_object_bottom_hull_xy(a),
                    obj_b_hull_xy=_object_bottom_hull_xy(b),
                    obj_a_bbox_min=np.array(a["bbox_min"], dtype=float),
                    obj_a_bbox_max=np.array(a["bbox_max"], dtype=float),
                    obj_b_bbox_min=np.array(b["bbox_min"], dtype=float),
                    obj_b_bbox_max=np.array(b["bbox_max"], dtype=float),
                )
                if old_dir == new_dir:
                    continue

                tpl = random.choice(tpl_list)
                question_text = tpl.format(
                    angle=angle,
                    camera_cardinal=cam_cardinal,
                    obj_a=_the(a["label"]),
                    obj_b=_the(b["label"]),
                )
                options, answer = generate_options(new_dir, ALL_DIRECTIONS_ALLOCENTRIC)
                candidates.append({
                    "level": "L3",
                    "type": "coordinate_rotation_allocentric",
                    "reference_frame": "allocentric",
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "correct_value": new_dir,
                    "camera_cardinal": cam_cardinal,
                    "rotation_angle": angle,
                    "obj_a_id": a["id"],
                    "obj_a_label": a["label"],
                    "obj_b_id": b["id"],
                    "obj_b_label": b["label"],
                    "mentioned_objects": [
                        _mention("obj_a", a["label"], a["id"]),
                        _mention("obj_b", b["label"], b["id"]),
                    ],
                    "old_direction": old_dir,
                    "new_direction": new_dir,
                    "relation_unchanged": False,
                })

        if len(candidates) > max_per_angle:
            candidates = random.sample(candidates, max_per_angle)
        questions.extend(candidates)

    return questions


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_attachment_chain_ids(obj_id: int, attachment_graph: dict) -> list[int]:
    """Get all dependent IDs (wrapper for import convenience)."""
    from .support_graph import get_attachment_chain
    return get_attachment_chain(obj_id, attachment_graph)


def _resolve_attachment_root_id(obj_id: int, attached_by: dict[int, int]) -> int:
    """Return the lowest attachment ancestor of *obj_id*, or itself if unattached."""
    current = int(obj_id)
    seen: set[int] = set()

    while current not in seen:
        seen.add(current)
        parent = attached_by.get(current)
        if parent is None:
            break
        try:
            current = int(parent)
        except (TypeError, ValueError):
            break

    return current


def _normalize_object_id_set(
    object_ids: list[int] | list[str] | None,
    field_name: str,
) -> set[int]:
    """Best-effort int normalization for object-ID filters."""
    normalized: set[int] = set()
    if object_ids is None:
        return normalized

    for obj_id in object_ids:
        try:
            normalized.add(int(obj_id))
        except (TypeError, ValueError):
            logger.warning("Skipping non-integer %s entry: %r", field_name, obj_id)
    return normalized


def _ensure_question_mentions(
    question: dict[str, Any],
    id_to_object: dict[int, dict],
    label_to_object: dict[str, dict],
) -> dict[str, Any]:
    """Backfill `mentioned_objects` for generators that do not set it explicitly."""
    if question.get("mentioned_objects"):
        return question

    mentions: list[dict[str, Any]] = []
    role_specs = [
        ("obj_a_id", "obj_a_label", "obj_a"),
        ("obj_b_id", "obj_b_label", "obj_b"),
        ("obj_ref_id", "obj_ref_label", "obj_ref"),
        ("obj_face_id", "obj_face_label", "obj_face"),
        ("obj_target_id", "obj_target_label", "obj_target"),
        ("moved_obj_id", "moved_obj_label", "moved_obj"),
        ("query_obj_id", "query_obj_label", "query_obj"),
        ("removed_obj_id", "removed_obj_label", "removed_obj"),
        ("grandparent_id", "grandparent_label", "grandparent"),
        ("parent_id", "parent_label", "parent"),
        ("grandchild_id", "grandchild_label", "grandchild"),
        ("neighbor_id", "neighbor_label", "neighbor"),
    ]
    seen: set[int] = set()
    for id_key, label_key, role in role_specs:
        obj_id = question.get(id_key)
        label = question.get(label_key)
        obj = None
        if obj_id is not None:
            obj = id_to_object.get(int(obj_id))
        elif label:
            obj = label_to_object.get(str(label))
        if obj is None:
            continue
        real_id = int(obj["id"])
        if real_id in seen:
            continue
        mentions.append(_mention(role, obj["label"], real_id))
        seen.add(real_id)

    question["mentioned_objects"] = mentions
    return question


def _emit_generation_trace(
    trace_recorder: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if trace_recorder is not None:
        trace_recorder(payload)


_TRACE_DETAIL_LEVELS = {
    "light": 1,
    "medium": 2,
    "full": 3,
}


def _trace_detail_at_least(trace_detail: str, required: str) -> bool:
    return _TRACE_DETAIL_LEVELS.get(str(trace_detail).strip().lower(), 1) >= _TRACE_DETAIL_LEVELS[required]


def _candidate_key(*parts: Any) -> str:
    return ":".join(str(part) for part in parts)


def _question_preview_payload(question: dict[str, Any]) -> dict[str, Any]:
    preview = {
        "level": question.get("level"),
        "type": question.get("type"),
        "question": question.get("question"),
        "correct_value": question.get("correct_value"),
    }
    object_fields = (
        "obj_a_id",
        "obj_b_id",
        "obj_c_id",
        "obj_ref_id",
        "obj_face_id",
        "obj_target_id",
        "query_obj_id",
        "moved_obj_id",
        "removed_obj_id",
        "parent_id",
        "child_id",
        "grandparent_id",
        "grandchild_id",
        "neighbor_id",
    )
    for field in object_fields:
        if field in question:
            preview[field] = question.get(field)
    return preview


def _emit_generator_context(
    trace_recorder: Callable[[dict[str, Any]], None] | None,
    generator: str,
    details: dict[str, Any],
) -> None:
    _emit_generation_trace(
        trace_recorder,
        {
            "event": "generator_context",
            "stage": "qa_generation",
            "generator": generator,
            "details": details,
        },
    )


def _emit_generator_summary(
    trace_recorder: Callable[[dict[str, Any]], None] | None,
    generator: str,
    *,
    generated_count: int,
    candidate_count: int | None = None,
    generated_candidate_count: int | None = None,
    skipped_candidate_count: int | None = None,
    reason_counts: dict[str, int] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "generator_summary",
        "stage": "qa_generation",
        "generator": generator,
        "generated_count": int(generated_count),
    }
    if candidate_count is not None:
        payload["candidate_count"] = int(candidate_count)
    if generated_candidate_count is not None:
        payload["generated_candidate_count"] = int(generated_candidate_count)
    if skipped_candidate_count is not None:
        payload["skipped_candidate_count"] = int(skipped_candidate_count)
    if reason_counts:
        payload["reason_counts"] = dict(sorted(reason_counts.items()))
    if details:
        payload["details"] = details
    _emit_generation_trace(trace_recorder, payload)


def _emit_generator_candidate(
    trace_recorder: Callable[[dict[str, Any]], None] | None,
    *,
    trace_detail: str,
    generator: str,
    candidate_kind: str,
    candidate_key: str,
    object_ids: list[int] | None,
    status: str,
    reason_code: str,
    reason_detail: str,
    evidence: dict[str, Any] | None = None,
    question_preview: dict[str, Any] | None = None,
) -> None:
    if not _trace_detail_at_least(trace_detail, "full"):
        return
    payload: dict[str, Any] = {
        "event": "generator_candidate",
        "stage": "qa_generation",
        "generator": generator,
        "candidate_kind": candidate_kind,
        "candidate_key": candidate_key,
        "object_ids": list(object_ids or []),
        "status": status,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
    }
    if evidence:
        payload["evidence"] = evidence
    if question_preview:
        payload["question_preview"] = question_preview
    _emit_generation_trace(trace_recorder, payload)


def _enforce_referable_mentions(
    questions: list[dict[str, Any]],
    referable_ids: set[int],
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Drop questions that mention a concrete object outside the referable set."""
    kept: list[dict[str, Any]] = []
    removed = 0
    for question in questions:
        mentions = question.get("mentioned_objects")
        if not isinstance(mentions, list):
            kept.append(question)
            continue

        valid = True
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            obj_id = mention.get("obj_id")
            if obj_id is None:
                continue
            try:
                mention_id = int(obj_id)
            except (TypeError, ValueError):
                valid = False
                break
            if mention_id not in referable_ids:
                valid = False
                break

        if valid:
            kept.append(question)
            continue
        removed += 1
        _emit_generation_trace(
            trace_recorder,
            {
                "event": "question_removed",
                "stage": "qa_generation",
                "filter": "referable_mentions",
                "reason": "mentions_non_referable_object",
                "trace_question_id": question.get("trace_question_id"),
                "question": question,
            },
        )

    if removed:
        logger.info("Referable-mention filter removed %d questions", removed)
    return kept


def _enforce_stable_facing_references(
    questions: list[dict[str, Any]],
    id_to_object: dict[int, dict],
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Drop any question whose "stand at A, face B" reference frame is unstable.

    This is a semantic safety net: any question carrying both `obj_ref_id` and
    `obj_face_id` is treated as using an object-defined facing frame, regardless
    of its formal question type.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for question in questions:
        ref_id = question.get("obj_ref_id")
        face_id = question.get("obj_face_id")
        if ref_id is None or face_id is None:
            kept.append(question)
            continue

        anchor_center = question.get("facing_anchor_center")
        facing_center = question.get("facing_target_center")
        if anchor_center is None or facing_center is None:
            ref_obj = id_to_object.get(int(ref_id))
            face_obj = id_to_object.get(int(face_id))
            if ref_obj is None or face_obj is None:
                kept.append(question)
                continue
            anchor_center = ref_obj["center"]
            facing_center = face_obj["center"]

        if _has_stable_object_centric_facing(
            np.array(anchor_center, dtype=float),
            np.array(facing_center, dtype=float),
        ):
            kept.append(question)
            continue

        removed += 1
        _emit_generation_trace(
            trace_recorder,
            {
                "event": "question_removed",
                "stage": "qa_generation",
                "filter": "stable_facing",
                "reason": "unstable_object_centric_facing",
                "trace_question_id": question.get("trace_question_id"),
                "question": question,
            },
        )

    if removed:
        logger.info("Stable-facing filter removed %d questions", removed)
    return kept


def _delta_to_description(delta: np.ndarray, camera_pose: CameraPose | None = None) -> str:
    """Convert a 3D world-frame delta to a camera-relative direction string.

    Projects the world-frame displacement into the camera coordinate system
    (x=right, y=down, z=forward in OpenCV convention) so that "right" in the
    question text matches what the viewer actually sees in the image.

    Used for ego-centric questions where the reference frame is the camera.
    Falls back to world-frame labels if *camera_pose* is not provided.
    """
    if camera_pose is not None:
        from .utils.coordinate_transform import world_to_camera
        # Transform delta as a direction (translate origin to camera, then difference)
        origin_cam = world_to_camera(np.zeros(3), camera_pose)
        delta_cam = world_to_camera(delta, camera_pose) - origin_cam
        dx, dy, dz = float(delta_cam[0]), float(delta_cam[1]), float(delta_cam[2])
        # OpenCV: x=right, y=down, z=forward
        horiz = abs(dx)
        depth = abs(dz)
        vert = abs(dy)
        dominant = max(horiz, depth, vert)
        if dominant == vert:
            return "down" if dy > 0 else "up"
        elif dominant == depth:
            return "forward" if dz > 0 else "backward"
        else:
            return "right" if dx > 0 else "left"

    # Fallback: world-frame (should not normally be used)
    axis = np.argmax(np.abs(delta))
    sign = delta[axis]
    labels = {0: ("right", "left"), 1: ("forward", "backward"), 2: ("up", "down")}
    pos_label, neg_label = labels[axis]
    return pos_label if sign > 0 else neg_label


def _delta_to_cardinal_description(delta: np.ndarray) -> str:
    """Convert a 3D world-frame delta to an absolute cardinal direction string.

    Convention after ScanNet axis alignment:  +x = east, +y = north, +z = up.

    Used for object-centric and allocentric questions where the movement
    should be described in unambiguous absolute terms.
    """
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    horiz = math.sqrt(dx * dx + dy * dy)
    vert = abs(dz)

    if vert > horiz:
        return "up" if dz > 0 else "down"

    if horiz < 1e-6:
        return "north"

    angle = math.degrees(math.atan2(dx, dy))  # 0=north, 90=east
    if angle < 0:
        angle += 360

    # Bin into 4 cardinal directions (N/E/S/W) using 90-degree bins
    if angle < 45 or angle >= 315:
        return "north"
    elif angle < 135:
        return "east"
    elif angle < 225:
        return "south"
    else:
        return "west"


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def generate_all_questions(
    objects: list[dict],
    attachment_graph: dict[int, list[int]],
    attached_by: dict[int, int],
    camera_pose: CameraPose,
    support_chain_graph: dict[int, list[int]] | None = None,
    support_chain_by: dict[int, int] | None = None,
    color_intrinsics: CameraIntrinsics | None = None,
    depth_image=None,
    depth_intrinsics=None,
    occlusion_backend: str = "mesh_ray",
    ray_caster=None,
    instance_mesh_data: InstanceMeshData | None = None,
    templates: dict | None = None,
    visible_object_ids: list[int] | None = None,
    referable_object_ids: list[int] | None = None,
    label_statuses: dict[str, Any] | None = None,
    label_counts: dict[str, Any] | None = None,
    room_bounds: dict | None = None,
    wall_objects: list[dict] | None = None,
    attachment_edges: list[dict] | None = None,
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
    trace_id_prefix: str = "q",
    trace_detail: str = "light",
) -> list[dict]:
    """Generate all question types for a single scene + frame.

    depth_image: float32 depth map in metres (from ScanNet depth PNG), or None.
    depth_intrinsics: CameraIntrinsics for the depth camera, or None.
    visible_object_ids: if provided, restrict all questions to objects whose
    centre projects into this frame.  Questions about off-screen objects are
    unanswerable from the image and should never be included.
    referable_object_ids: if provided, restrict question generation to the
    object_id subset judged referable by the VLM for this frame.
    label_statuses: if provided, use per-label VLM absent/unique/multiple/unsure
    decisions to guide L1 occlusion generation.
    label_counts: optional compatibility field derived from label_statuses.
    room_bounds: dict with bbox_min/bbox_max from wall/floor mesh, or None.
    wall_objects: visible filtering for ordinary objects does not touch these;
    they are only used to construct allocentric wall-anchor wording.

    Returns a list of question dicts.
    """
    if templates is None:
        templates = _load_templates()
    if attachment_edges is None:
        attachment_edges = []
    if support_chain_graph is None:
        support_chain_graph = attachment_graph
    if support_chain_by is None:
        support_chain_by = attached_by
    attachment_edge_input = len(attachment_edges)
    trace_counter = 0
    original_objects = list(objects)
    enrich_objects_with_distance_geometry(objects, instance_mesh_data)

    def _snapshot_question(question: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(question, ensure_ascii=False))

    def _register_generated_questions(
        generator_name: str,
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        nonlocal trace_counter
        if not questions:
            _emit_generation_trace(
                trace_recorder,
                {
                    "event": "generator_output",
                    "stage": "qa_generation",
                    "generator": generator_name,
                    "count": 0,
                    "question_ids": [],
                    "questions": [],
                },
            )
            return questions

        question_ids: list[str] = []
        snapshots: list[dict[str, Any]] = []
        for question in questions:
            trace_id = question.get("trace_question_id")
            if not trace_id:
                trace_counter += 1
                trace_id = f"{trace_id_prefix}_{trace_counter:04d}"
                question["trace_question_id"] = trace_id
            question["_trace_source"] = generator_name
            question_ids.append(str(trace_id))
            snapshots.append(_snapshot_question(question))

        _emit_generation_trace(
            trace_recorder,
            {
                "event": "generator_output",
                "stage": "qa_generation",
                "generator": generator_name,
                "count": len(questions),
                "question_ids": question_ids,
                "questions": snapshots,
            },
        )
        return questions

    def _apply_question_cap(
        generator_name: str,
        questions: list[dict[str, Any]],
        cap: int,
    ) -> list[dict[str, Any]]:
        if len(questions) <= cap:
            return questions

        kept_questions = random.sample(questions, cap)
        kept_ids = {str(question.get("trace_question_id")) for question in kept_questions}
        removed_ids = [
            str(question.get("trace_question_id"))
            for question in questions
            if str(question.get("trace_question_id")) not in kept_ids
        ]
        _emit_generation_trace(
            trace_recorder,
            {
                "event": "generator_cap_applied",
                "stage": "qa_generation",
                "generator": generator_name,
                "cap": int(cap),
                "input_count": len(questions),
                "output_count": len(kept_questions),
                "removed_question_ids": removed_ids,
            },
        )
        return kept_questions

    def _emit_object_pool_snapshot(
        *,
        l1_occlusion_subject_ids: set[int],
    ) -> None:
        visible_set = (
            _normalize_object_id_set(visible_object_ids, "visible_object_ids")
            if visible_object_ids is not None else {int(obj["id"]) for obj in original_objects}
        )
        excluded_ids = {
            int(obj["id"])
            for obj in original_objects
            if str(obj.get("label", "")).strip().lower() in EXCLUDED_LABELS
        }
        graph_object_ids = {int(obj["id"]) for obj in all_objects_for_graph}
        question_object_ids = {int(obj["id"]) for obj in objects_uniq}
        movement_object_ids = {int(obj["id"]) for obj in movement_objects}
        pool_rows: list[dict[str, Any]] = []
        for obj in original_objects:
            obj_id = int(obj["id"])
            row_reasons: list[str] = []
            if obj_id not in visible_set:
                row_reasons.append("not_visible_in_forced_frame")
            if obj_id in excluded_ids:
                row_reasons.append("excluded_label")
            if obj_id not in referable_set and obj_id not in attachment_context_ids:
                row_reasons.append("not_referable")
            if obj_id in attachment_context_ids and obj_id not in referable_set:
                row_reasons.append("attachment_context_only")
            if obj_id in referable_set and obj_id not in question_object_ids:
                row_reasons.append("filtered_from_question_pool")
            pool_rows.append(
                {
                    "id": obj_id,
                    "label": obj.get("label"),
                    "visible_in_forced_frame": obj_id in visible_set,
                    "excluded_label": obj_id in excluded_ids,
                    "referable": obj_id in referable_set,
                    "attachment_context": obj_id in attachment_context_ids,
                    "graph_pool": obj_id in graph_object_ids,
                    "question_pool": obj_id in question_object_ids,
                    "movement_pool": obj_id in movement_object_ids,
                    "l1_occlusion_pool": obj_id in l1_occlusion_subject_ids,
                    "reasons": row_reasons,
                }
            )
        _emit_generation_trace(
            trace_recorder,
            {
                "event": "object_pool_snapshot",
                "stage": "qa_generation",
                "summary": {
                    "original_object_count": len(original_objects),
                    "visible_object_count": len(visible_set),
                    "excluded_object_count": len(excluded_ids),
                    "referable_object_count": len(referable_set),
                    "attachment_context_count": len(attachment_context_ids),
                    "graph_pool_count": len(graph_object_ids),
                    "question_pool_count": len(question_object_ids),
                    "movement_pool_count": len(movement_object_ids),
                    "l1_occlusion_pool_count": len(l1_occlusion_subject_ids),
                },
                "rows": pool_rows,
            },
        )

    # Restrict to objects visible in this frame so every question can be
    # answered by looking at the image.
    if visible_object_ids is not None:
        vis_set = _normalize_object_id_set(visible_object_ids, "visible_object_ids")
        objects = [o for o in objects if o["id"] in vis_set]
        attachment_graph = {
            k: [c for c in v if c in vis_set]
            for k, v in attachment_graph.items()
            if k in vis_set
        }
        attached_by = {k: v for k, v in attached_by.items() if k in vis_set and v in vis_set}
        support_chain_graph = {
            k: [c for c in v if c in vis_set]
            for k, v in support_chain_graph.items()
            if k in vis_set
        }
        support_chain_by = {
            k: v for k, v in support_chain_by.items() if k in vis_set and v in vis_set
        }
        attachment_edges = [
            e for e in attachment_edges
            if int(e["parent_id"]) in vis_set and int(e["child_id"]) in vis_set
        ]

    attachment_edge_count_visible = len(attachment_edges)

    # Apply the shared hard blacklist once for both graph construction and
    # question generation.
    all_objects_for_graph = [
        o for o in objects
        if o.get("label", "").lower() not in EXCLUDED_LABELS
    ]
    objects_for_questions = list(all_objects_for_graph)
    # L1 occlusion can rely on per-label VLM counts, so keep a pre-referable
    # pool of question-eligible visible objects for that path only.
    l1_occlusion_objects = list(objects_for_questions)
    l2_collision_objects = list(objects_for_questions)
    graph_ids = {int(o["id"]) for o in all_objects_for_graph}
    attachment_edges = [
        e for e in attachment_edges
        if int(e["parent_id"]) in graph_ids and int(e["child_id"]) in graph_ids
    ]
    attachment_edge_count_nonexcluded = len(attachment_edges)
    attachment_context_ids: set[int] = set()

    if referable_object_ids is None:
        raise ValueError(
            "generate_all_questions requires referable_object_ids from VLM referability filtering"
        )

    referable_set = _normalize_object_id_set(
        referable_object_ids,
        "referable_object_ids",
    )
    attachment_parent_ids = {
        int(e["parent_id"])
        for e in attachment_edges
        if int(e["child_id"]) in referable_set
    }
    attachment_context_ids = attachment_parent_ids
    graph_allowed_ids = referable_set | attachment_context_ids
    all_objects_for_graph = [
        o for o in all_objects_for_graph
        if int(o["id"]) in graph_allowed_ids
    ]
    objects_for_questions = [
        o for o in objects_for_questions
        if int(o["id"]) in referable_set
    ]
    attachment_graph = {
        k: [c for c in v if c in graph_allowed_ids]
        for k, v in attachment_graph.items()
        if k in graph_allowed_ids
    }
    attached_by = {
        k: v for k, v in attached_by.items()
        if k in graph_allowed_ids and v in graph_allowed_ids
    }
    attachment_edges = [
        e for e in attachment_edges
        if int(e["parent_id"]) in graph_allowed_ids and int(e["child_id"]) in graph_allowed_ids
    ]
    objects_uniq = list(objects_for_questions)
    referable_question_ids = {int(o["id"]) for o in objects_uniq}
    support_chain_graph = {
        k: filtered_children
        for k, v in support_chain_graph.items()
        if k in referable_question_ids
        for filtered_children in ([c for c in v if c in referable_question_ids],)
        if filtered_children
    }
    support_chain_by = {
        k: v for k, v in support_chain_by.items()
        if k in referable_question_ids and v in referable_question_ids
    }

    graph_eligible_ids = referable_set | attachment_context_ids
    attachment_graph = {
        k: [c for c in v if c in graph_eligible_ids]
        for k, v in attachment_graph.items()
        if k in graph_eligible_ids
    }
    attached_by = {
        k: v for k, v in attached_by.items()
        if k in graph_eligible_ids and v in graph_eligible_ids
    }
    attachment_edges = [
        e for e in attachment_edges
        if int(e["parent_id"]) in graph_eligible_ids and int(e["child_id"]) in graph_eligible_ids
    ]
    movement_objects = [
        o for o in all_objects_for_graph
        if int(o["id"]) in graph_eligible_ids
    ]
    movement_object_map = {int(o["id"]): o for o in movement_objects}
    attachment_edge_lookup = _build_attachment_edge_lookup(attachment_edges)
    l1_occlusion_subject_ids = {
        int(obj["id"])
        for obj in (l1_occlusion_objects if (label_statuses or label_counts) else objects_uniq)
    }
    _emit_object_pool_snapshot(
        l1_occlusion_subject_ids=l1_occlusion_subject_ids,
    )

    logger.info(
        "Attachment filter stats: edges input=%d visible=%d nonexcluded=%d final=%d, graph_objects=%d, question_objects=%d, referable_question_objects=%d",
        attachment_edge_input,
        attachment_edge_count_visible,
        attachment_edge_count_nonexcluded,
        len(attachment_edges),
        len(all_objects_for_graph),
        len(objects_for_questions),
        len(objects_uniq),
    )

    all_questions: list[dict] = []

    # Per-frame caps — keep the benchmark tractable when scenes have many objects
    MAX_L1_DIRECTION = 20
    MAX_L1_DIRECTION_OC = 15   # object-centric
    MAX_L1_DIRECTION_ALLO = 15 # allocentric
    MAX_L1_DISTANCE = 20

    # Ordinary L1 relations ignore depth in normal generation.
    relations = compute_all_relations(objects_uniq, camera_pose, None, None)
    _emit_generator_context(
        trace_recorder,
        "generate_l1_direction",
        {
            "relation_count": len(relations),
            "question_object_count": len(objects_uniq),
        },
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l1_distance",
        {
            "relation_count": len(relations),
            "question_object_count": len(objects_uniq),
        },
    )

    # L1 — collect separately so we can sample before adding
    MAX_L1_OCCLUSION = 15
    l1_dir_qs:  list[dict] = []
    l1_dist_qs: list[dict] = []
    l1_occ_qs:  list[dict] = []
    l1_dir_reason_counts: Counter[str] = Counter()
    l1_dist_reason_counts: Counter[str] = Counter()
    l1_dir_generated_candidate_count = 0
    l1_dist_generated_candidate_count = 0

    for rel in relations:
        relation_key = _candidate_key(rel["obj_a_id"], rel["obj_b_id"])
        relation_object_ids = [int(rel["obj_a_id"]), int(rel["obj_b_id"])]
        if rel["ambiguity_score"] > 0.7:
            l1_dir_reason_counts["ambiguous_direction"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_direction",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="skipped",
                reason_code="ambiguous_direction",
                reason_detail="pairwise ego-centric direction is too close to an ambiguity boundary",
                evidence={
                    "ambiguity_score": float(rel["ambiguity_score"]),
                    "threshold": 0.7,
                },
            )
            q = None
        elif rel["distance_m"] < MIN_DIRECTION_DISTANCE:
            l1_dir_reason_counts["pair_too_close"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_direction",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="skipped",
                reason_code="pair_too_close",
                reason_detail="pairwise direction questions require objects to be farther apart",
                evidence={
                    "distance_m": float(rel["distance_m"]),
                    "min_distance": MIN_DIRECTION_DISTANCE,
                },
            )
            q = None
        elif rel["obj_a_label"] == rel["obj_b_label"]:
            l1_dir_reason_counts["duplicate_labels"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_direction",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="skipped",
                reason_code="duplicate_labels",
                reason_detail="ego-centric direction questions require distinct object labels",
            )
            q = None
        else:
            obj_a = movement_object_map.get(int(rel["obj_a_id"]))
            obj_b = movement_object_map.get(int(rel["obj_b_id"]))
            suppression = None
            if obj_a is not None and obj_b is not None:
                suppression = _direction_suppression_reason(
                    obj_a,
                    obj_b,
                    str(rel["direction_b_rel_a"]),
                    attachment_edge_lookup,
                )
            if suppression is not None:
                reason_code, reason_detail, evidence = suppression
                l1_dir_reason_counts[reason_code] += 1
                _emit_generator_candidate(
                    trace_recorder,
                    trace_detail=trace_detail,
                    generator="generate_l1_direction",
                    candidate_kind="relation_pair",
                    candidate_key=relation_key,
                    object_ids=relation_object_ids,
                    status="skipped",
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    evidence=evidence,
                )
                q = None
            else:
                q = generate_l1_direction(
                    rel,
                    templates,
                    obj_a=obj_a,
                    obj_b=obj_b,
                    attachment_edge_lookup=attachment_edge_lookup,
                )
        if q:
            l1_dir_qs.append(q)
            l1_dir_generated_candidate_count += 1
            l1_dir_reason_counts["generated"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_direction",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="generated",
                reason_code="generated",
                reason_detail="pair yields an unambiguous ego-centric direction question",
                evidence={
                    "direction": rel.get("direction_b_rel_a"),
                    "ambiguity_score": float(rel.get("ambiguity_score", 0.0)),
                    "distance_m": float(rel.get("distance_m", 0.0)),
                },
                question_preview=_question_preview_payload(q),
            )

        if rel["near_boundary"]:
            l1_dist_reason_counts["near_boundary"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_distance",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="skipped",
                reason_code="near_distance_boundary",
                reason_detail="distance falls too close to a bin boundary",
                evidence={
                    "distance_bin": rel.get("distance_bin"),
                    "distance_m": float(rel.get("distance_m", 0.0)),
                },
            )
            q = None
        elif _relation_distance_for_distance_questions(rel) < MIN_DISTANCE_QUESTION_DISTANCE_M:
            l1_dist_reason_counts["distance_too_small"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_distance",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="skipped",
                reason_code="distance_too_small",
                reason_detail="distance questions require object pairs to be at least 0.2m apart",
                evidence={
                    "distance_m": float(rel.get("distance_m", 0.0)),
                    "distance_m_raw": _relation_distance_for_distance_questions(rel),
                    "min_distance": MIN_DISTANCE_QUESTION_DISTANCE_M,
                },
            )
            q = None
        elif rel["obj_a_label"] == rel["obj_b_label"]:
            l1_dist_reason_counts["duplicate_labels"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_distance",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="skipped",
                reason_code="duplicate_labels",
                reason_detail="distance questions require distinct object labels",
            )
            q = None
        else:
            q = generate_l1_distance(rel, templates)
        if q:
            l1_dist_qs.append(q)
            l1_dist_generated_candidate_count += 1
            l1_dist_reason_counts["generated"] += 1
            _emit_generator_candidate(
                trace_recorder,
                trace_detail=trace_detail,
                generator="generate_l1_distance",
                candidate_kind="relation_pair",
                candidate_key=relation_key,
                object_ids=relation_object_ids,
                status="generated",
                reason_code="generated",
                reason_detail="pair yields a stable distance-bin question",
                evidence={
                    "distance_bin": rel.get("distance_bin"),
                    "distance_m": float(rel.get("distance_m", 0.0)),
                },
                question_preview=_question_preview_payload(q),
            )
    l1_dir_qs = _register_generated_questions("generate_l1_direction", l1_dir_qs)
    l1_dist_qs = _register_generated_questions("generate_l1_distance", l1_dist_qs)
    _emit_generator_summary(
        trace_recorder,
        "generate_l1_direction",
        generated_count=len(l1_dir_qs),
        candidate_count=len(relations),
        generated_candidate_count=l1_dir_generated_candidate_count,
        skipped_candidate_count=max(len(relations) - l1_dir_generated_candidate_count, 0),
        reason_counts=dict(l1_dir_reason_counts),
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l1_distance",
        generated_count=len(l1_dist_qs),
        candidate_count=len(relations),
        generated_candidate_count=l1_dist_generated_candidate_count,
        skipped_candidate_count=max(len(relations) - l1_dist_generated_candidate_count, 0),
        reason_counts=dict(l1_dist_reason_counts),
    )

    l1_occlusion_subjects = l1_occlusion_objects if (label_statuses or label_counts) else objects_uniq
    _emit_generator_context(
        trace_recorder,
        "generate_l1_occlusion_questions",
        {
            "subject_object_count": len(l1_occlusion_subjects),
            "question_object_count": len(objects_uniq),
            "label_status_count": len(label_statuses or {}),
            "label_count_count": len(label_counts or {}),
            "referable_object_count": len(referable_set),
        },
    )
    l1_occ_qs = generate_l1_occlusion_questions(
        objects=l1_occlusion_subjects,
        camera_pose=camera_pose,
        color_intrinsics=color_intrinsics,
        depth_image=depth_image,
        depth_intrinsics=depth_intrinsics,
        occlusion_backend=occlusion_backend,
        ray_caster=ray_caster,
        instance_mesh_data=instance_mesh_data,
        templates=templates,
        label_statuses=label_statuses,
        label_counts=label_counts,
        referable_object_ids=referable_object_ids,
    )
    l1_occ_qs = _register_generated_questions("generate_l1_occlusion_questions", l1_occ_qs)

    # L1 new reference frames
    _emit_generator_context(
        trace_recorder,
        "generate_l1_direction_object_centric",
        {
            "question_object_count": len(objects_uniq),
            "potential_triplet_count": len(objects_uniq) * max(len(objects_uniq) - 1, 0) * max(len(objects_uniq) - 2, 0),
            "max_questions": MAX_L1_DIRECTION_OC,
        },
    )
    l1_dir_oc_qs = generate_l1_direction_object_centric(
        objects_uniq, templates, max_questions=MAX_L1_DIRECTION_OC,
        attachment_edge_lookup=attachment_edge_lookup,
        trace_recorder=trace_recorder,
        trace_detail=trace_detail,
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l1_direction_allocentric",
        {
            "question_object_count": len(objects_uniq),
            "potential_pair_count": int((len(objects_uniq) * max(len(objects_uniq) - 1, 0)) / 2),
            "max_questions": MAX_L1_DIRECTION_ALLO,
        },
    )
    l1_dir_allo_qs = generate_l1_direction_allocentric(
        objects_uniq, camera_pose, templates, max_questions=MAX_L1_DIRECTION_ALLO,
        attachment_edge_lookup=attachment_edge_lookup,
        trace_recorder=trace_recorder,
        trace_detail=trace_detail,
    )
    l1_dir_oc_qs = _register_generated_questions("generate_l1_direction_object_centric", l1_dir_oc_qs)
    l1_dir_allo_qs = _register_generated_questions("generate_l1_direction_allocentric", l1_dir_allo_qs)

    l1_dir_qs = _apply_question_cap("generate_l1_direction", l1_dir_qs, MAX_L1_DIRECTION)
    l1_dist_qs = _apply_question_cap("generate_l1_distance", l1_dist_qs, MAX_L1_DISTANCE)
    l1_occ_qs = _apply_question_cap("generate_l1_occlusion_questions", l1_occ_qs, MAX_L1_OCCLUSION)
    all_questions.extend(l1_dir_qs)
    all_questions.extend(l1_dist_qs)
    all_questions.extend(l1_occ_qs)
    all_questions.extend(l1_dir_oc_qs)
    all_questions.extend(l1_dir_allo_qs)
    # Rebuild the movement graph for question-eligible objects plus any
    # attachment-context ancestors kept to preserve attachment roots.
    attachment_graph_uniq = {
        k: [c for c in v if c in graph_eligible_ids]
        for k, v in attachment_graph.items()
        if k in graph_eligible_ids
    }
    attached_by_uniq = {
        k: v for k, v in attached_by.items()
        if k in graph_eligible_ids and v in graph_eligible_ids
    }
    support_chain_graph_uniq = {
        k: filtered_children
        for k, v in support_chain_graph.items()
        if k in referable_question_ids
        for filtered_children in ([c for c in v if c in referable_question_ids],)
        if filtered_children
    }
    support_chain_by_uniq = {
        k: v for k, v in support_chain_by.items()
        if k in referable_question_ids and v in referable_question_ids
    }

    # L2 - ego-centric (existing)
    _emit_generator_context(
        trace_recorder,
        "generate_l2_object_move",
        {
            "question_object_count": len(objects_uniq),
            "movement_object_count": len(movement_objects),
            "attachment_graph_node_count": len(attachment_graph_uniq),
            "occlusion_backend": occlusion_backend,
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l2_object_move",
            generate_l2_object_move(
                objects_uniq,
                attachment_graph_uniq,
                attached_by_uniq,
                camera_pose,
                templates,
                room_bounds=room_bounds,
                collision_objects=l2_collision_objects,
                movement_objects=movement_objects,
                object_map=movement_object_map,
                color_intrinsics=color_intrinsics,
                occlusion_backend=occlusion_backend,
                ray_caster=ray_caster,
                instance_mesh_data=instance_mesh_data,
            ),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l2_object_move",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l2_object_move"
        ),
        details={"audit_mode": "summary_only"},
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l2_viewpoint_move",
        {
            "question_object_count": len(objects_uniq),
            "candidate_motion_count": 12,
            "occlusion_backend": occlusion_backend,
            "occlusion_mode": "l1_style",
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l2_viewpoint_move",
            generate_l2_viewpoint_move(
                objects_uniq,
                camera_pose,
                color_intrinsics,
                depth_image,
                depth_intrinsics,
                occlusion_backend,
                ray_caster,
                instance_mesh_data,
                templates,
                trace_recorder=trace_recorder,
                trace_detail=trace_detail,
            ),
        )
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l2_object_remove",
        {
            "question_object_count": len(objects_uniq),
            "attachment_graph_node_count": len(attachment_graph_uniq),
            "occlusion_backend": occlusion_backend,
            "occlusion_mode": "l1_style",
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l2_object_remove",
            generate_l2_object_remove(
                objects_uniq,
                attachment_graph_uniq,
                camera_pose,
                color_intrinsics,
                depth_image,
                depth_intrinsics,
                occlusion_backend,
                ray_caster,
                instance_mesh_data,
                templates,
                trace_recorder=trace_recorder,
                trace_detail=trace_detail,
            ),
        )
    )
    # L2 - new reference frames
    _emit_generator_context(
        trace_recorder,
        "generate_l2_object_rotate_object_centric",
        {
            "question_object_count": len(objects_uniq),
            "movement_object_count": len(movement_objects),
            "attachment_graph_node_count": len(attachment_graph_uniq),
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l2_object_rotate_object_centric",
            generate_l2_object_rotate_object_centric(
                objects_uniq, attachment_graph_uniq, attached_by_uniq, camera_pose, templates,
                room_bounds=room_bounds,
                collision_objects=l2_collision_objects,
                movement_objects=movement_objects,
                object_map=movement_object_map,
            ),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l2_object_rotate_object_centric",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l2_object_rotate_object_centric"
        ),
        details={"audit_mode": "summary_only"},
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l2_object_move_allocentric",
        {
            "question_object_count": len(objects_uniq),
            "movement_object_count": len(movement_objects),
            "attachment_graph_node_count": len(attachment_graph_uniq),
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l2_object_move_allocentric",
            generate_l2_object_move_allocentric(
                objects_uniq, attachment_graph_uniq, attached_by_uniq, camera_pose, templates,
                room_bounds=room_bounds,
                collision_objects=l2_collision_objects,
                movement_objects=movement_objects,
                object_map=movement_object_map,
            ),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l2_object_move_allocentric",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l2_object_move_allocentric"
        ),
        details={"audit_mode": "summary_only"},
    )
    # L3
    _emit_generator_context(
        trace_recorder,
        "generate_l3_attachment_chain",
        {
            "question_object_count": len(objects_uniq),
            "support_chain_node_count": len(support_chain_graph_uniq),
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l3_attachment_chain",
            generate_l3_attachment_chain(
                objects_uniq,
                support_chain_graph_uniq,
                support_chain_by_uniq,
                camera_pose,
                templates,
            ),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l3_attachment_chain",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l3_attachment_chain"
        ),
        details={"audit_mode": "summary_only"},
    )
    # L3 coordinate rotation - all three reference frames
    _emit_generator_context(
        trace_recorder,
        "generate_l3_coordinate_rotation",
        {
            "question_object_count": len(objects_uniq),
            "rotation_angles": [90, 180, 270],
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l3_coordinate_rotation",
            generate_l3_coordinate_rotation(objects_uniq, camera_pose, templates),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l3_coordinate_rotation",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l3_coordinate_rotation"
        ),
        details={"audit_mode": "summary_only"},
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l3_coordinate_rotation_object_centric",
        {
            "question_object_count": len(objects_uniq),
            "rotation_angles": [90, 180, 270],
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l3_coordinate_rotation_object_centric",
            generate_l3_coordinate_rotation_object_centric(objects_uniq, camera_pose, templates),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l3_coordinate_rotation_object_centric",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l3_coordinate_rotation_object_centric"
        ),
        details={"audit_mode": "summary_only"},
    )
    _emit_generator_context(
        trace_recorder,
        "generate_l3_coordinate_rotation_allocentric",
        {
            "question_object_count": len(objects_uniq),
            "rotation_angles": [90, 180, 270],
        },
    )
    all_questions.extend(
        _register_generated_questions(
            "generate_l3_coordinate_rotation_allocentric",
            generate_l3_coordinate_rotation_allocentric(objects_uniq, camera_pose, templates),
        )
    )
    _emit_generator_summary(
        trace_recorder,
        "generate_l3_coordinate_rotation_allocentric",
        generated_count=sum(
            1 for question in all_questions
            if question.get("_trace_source") == "generate_l3_coordinate_rotation_allocentric"
        ),
        details={"audit_mode": "summary_only"},
    )

    id_to_object = {int(o["id"]): o for o in objects_uniq}
    label_to_object = {str(o["label"]): o for o in objects_uniq}
    for idx, question in enumerate(all_questions):
        all_questions[idx] = _ensure_question_mentions(
            question, id_to_object, label_to_object,
        )

    if trace_recorder is not None:
        all_questions = _enforce_referable_mentions(
            all_questions,
            referable_question_ids,
            trace_recorder=trace_recorder,
        )
        all_questions = _enforce_stable_facing_references(
            all_questions,
            id_to_object,
            trace_recorder=trace_recorder,
        )
    else:
        all_questions = _enforce_referable_mentions(
            all_questions,
            referable_question_ids,
        )
        all_questions = _enforce_stable_facing_references(
            all_questions,
            id_to_object,
        )

    _emit_generation_trace(
        trace_recorder,
        {
            "event": "generation_complete",
            "stage": "qa_generation",
            "count": len(all_questions),
            "question_ids": [
                str(question.get("trace_question_id"))
                for question in all_questions
                if question.get("trace_question_id")
            ],
        },
    )
    logger.info("Generated %d questions total", len(all_questions))
    return all_questions
