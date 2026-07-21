"""Depth-aware, route-aligned auxiliary-frame selection."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .auxiliary_path import MAX_AUXILIARY_FRAMES
from .datasets.base import DepthFrame
from .legacy_auxiliary_path import (
    ROUTE_MIN_OVERLAP_FRAC,
    ROUTE_MIN_PROGRESS_FRAC,
    ROUTE_NEAR_DUPLICATE_ROTATION_DEG,
    ROUTE_NEAR_DUPLICATE_TRANSLATION_M,
    ROUTE_ORIENTATION_THRESHOLD_DEG,
    _coverage_runs,
    _poses_are_near_duplicates,
    _semantic_conflict,
    object_group_center,
    route_visibility_mask,
    sample_route_points,
)
from .utils.colmap_loader import CameraIntrinsics, CameraPose
from .utils.coordinate_transform import (
    get_camera_forward,
    project_camera_points_to_image,
    world_to_camera_batch,
)


DEPTH_MIN_VALID_FRACTION = 0.60
DEPTH_NEIGHBORHOOD_RADIUS = 1
DEPTH_MIN_NEIGHBOR_SAMPLES = 3
DEPTH_ABSOLUTE_TOLERANCE_M = 0.10
DEPTH_RELATIVE_TOLERANCE = 0.02
DEPTH_UNKNOWN_BRIDGE_SAMPLES = 1
ROUTE_HORIZONTAL_DIRECTION_MIN_M = 0.25
ROUTE_LOCAL_PERP_SOFT_M = 0.30
ROUTE_LOCAL_PERP_HARD_M = 0.75
ROUTE_GLOBAL_PERP_SOFT_M = 0.50
ROUTE_GLOBAL_PERP_HARD_M = 1.00
ROUTE_HEIGHT_SOFT_M = 0.20
ROUTE_HEIGHT_HARD_M = 0.50
ROUTE_PARALLEL_SOFT_M = 0.75
ROUTE_DEGENERATE_XY_SOFT_M = 0.75
ROUTE_DEGENERATE_XY_HARD_M = 1.00
ROUTE_ANGLE_SOFT_DEG = 30.0
ROUTE_EDGE_BASE_COST = 0.35
ROUTE_SEARCH_METHOD = "dijkstra_depth_corridor"
VISUAL_PRUNE_ORIENTATION_THRESHOLD_DEG = 80.0
VISUAL_PRUNE_LOCAL_PERP_HARD_M = 0.80
VISUAL_REDUNDANCY_METRIC_VERSION = 1
VISUAL_REDUNDANCY_SAMPLE_STRIDE_PX = 8
VISUAL_REDUNDANCY_MIN_SAMPLES = 32
VISUAL_REDUNDANCY_MIN_BIDIRECTIONAL_OVERLAP = 0.65
VISUAL_REDUNDANCY_MAX_P75_DISPLACEMENT_DIAGONAL = 0.12
VISUAL_REDUNDANCY_MIN_ORB_INLIERS = 24
VISUAL_REDUNDANCY_MIN_ORB_INLIER_RATIO = 0.60
VISUAL_REDUNDANCY_MIN_ORB_GRID_FRACTION = 0.25


@dataclass(frozen=True)
class DepthCorridorAuxiliaryRoute:
    auxiliary_image_names: tuple[str, ...]
    cost: float
    edge_count: int
    route_sample_count: int
    frame_a_coverage_end: float
    frame_b_coverage_start: float
    auxiliary_responsibility_fraction: float
    transition_overlap_fraction: float
    search_method: str
    min_progress_fraction: float
    min_depth_valid_fraction: float
    min_depth_visible_fraction: float
    max_local_perpendicular_m: float
    max_global_perpendicular_m: float
    max_height_change_m: float
    max_parallel_change_m: float
    max_forward_angle_deg: float
    depth_sources: tuple[str, ...]
    pre_prune_auxiliary_count: int
    pruned_auxiliary_frame_count: int
    visual_pruned_auxiliary_frame_count: int
    visual_duplicate_candidate_count: int
    visual_prune_relaxed_angle_edge_count: int
    visual_redundancy_metric_version: int
    semantic_rejected_frame_count: int


@dataclass(frozen=True)
class VisualRedundancyEvidence:
    available: bool
    is_duplicate: bool
    min_bidirectional_overlap: float
    max_p75_displacement_diagonal: float
    orb_inliers: int
    orb_inlier_ratio: float
    orb_min_grid_fraction: float
    reason: str


class DepthVisualRedundancyEvaluator:
    """Detect rendered-view duplicates using depth reprojection plus RGB evidence."""

    def __init__(
        self,
        *,
        poses: Mapping[str, CameraPose],
        depth_frame_for: Callable[[str], DepthFrame | None],
        rgb_evidence_for: Callable[[str, str], Any],
    ) -> None:
        self.poses = poses
        self.depth_frame_for = depth_frame_for
        self.rgb_evidence_for = rgb_evidence_for
        self._cache: dict[tuple[str, str], VisualRedundancyEvidence] = {}

    @staticmethod
    def _directional_reprojection(
        source_pose: CameraPose,
        target_pose: CameraPose,
        source_depth: DepthFrame,
        target_depth: DepthFrame,
    ) -> tuple[float, float, int] | None:
        source_image = np.asarray(source_depth.image_m, dtype=np.float64)
        target_image = np.asarray(target_depth.image_m, dtype=np.float64)
        source_height, source_width = source_image.shape[:2]
        target_height, target_width = target_image.shape[:2]
        offset = VISUAL_REDUNDANCY_SAMPLE_STRIDE_PX // 2
        sample_y = np.arange(
            offset,
            source_height,
            VISUAL_REDUNDANCY_SAMPLE_STRIDE_PX,
            dtype=np.int64,
        )
        sample_x = np.arange(
            offset,
            source_width,
            VISUAL_REDUNDANCY_SAMPLE_STRIDE_PX,
            dtype=np.int64,
        )
        grid_x, grid_y = np.meshgrid(sample_x, sample_y)
        source_z = source_image[grid_y, grid_x]
        valid_source = np.isfinite(source_z) & (source_z > 0.0)
        if int(np.count_nonzero(valid_source)) < VISUAL_REDUNDANCY_MIN_SAMPLES:
            return None

        source_u = grid_x[valid_source].astype(np.float64)
        source_v = grid_y[valid_source].astype(np.float64)
        source_z = source_z[valid_source]
        source_intrinsics = source_depth.intrinsics
        source_camera = np.column_stack(
            (
                (source_u - source_intrinsics.cx) * source_z / source_intrinsics.fx,
                (source_v - source_intrinsics.cy) * source_z / source_intrinsics.fy,
                source_z,
            )
        )
        world_points = (source_camera - source_pose.translation) @ source_pose.rotation
        target_camera = (
            target_pose.rotation @ world_points.T + target_pose.translation[:, None]
        ).T
        target_uv, target_z = project_camera_points_to_image(
            target_camera, target_depth.intrinsics
        )
        finite_projection = (
            np.isfinite(target_uv).all(axis=1)
            & np.isfinite(target_z)
            & (target_z > 0.0)
        )
        target_x = np.zeros(len(target_uv), dtype=np.int64)
        target_y = np.zeros(len(target_uv), dtype=np.int64)
        target_x[finite_projection] = np.rint(
            target_uv[finite_projection, 0]
        ).astype(np.int64)
        target_y[finite_projection] = np.rint(
            target_uv[finite_projection, 1]
        ).astype(np.int64)
        in_frame = (
            finite_projection
            & (target_x >= 0)
            & (target_x < target_width)
            & (target_y >= 0)
            & (target_y < target_height)
        )
        projected_indices = np.flatnonzero(in_frame)
        if projected_indices.size < VISUAL_REDUNDANCY_MIN_SAMPLES:
            return (0.0, 1.0, int(projected_indices.size))
        observed = target_image[
            target_y[projected_indices], target_x[projected_indices]
        ]
        expected = target_z[projected_indices]
        valid_target = np.isfinite(observed) & (observed > 0.0)
        tolerance = np.maximum(
            DEPTH_ABSOLUTE_TOLERANCE_M,
            DEPTH_RELATIVE_TOLERANCE * expected,
        )
        visible = valid_target & (expected <= observed + tolerance)
        visible_indices = projected_indices[visible]
        overlap = float(len(visible_indices) / max(len(projected_indices), 1))
        if len(visible_indices) < VISUAL_REDUNDANCY_MIN_SAMPLES:
            return (overlap, 1.0, len(visible_indices))

        delta_x = (
            target_uv[visible_indices, 0] / max(float(target_width), 1.0)
            - source_u[visible_indices] / max(float(source_width), 1.0)
        )
        delta_y = (
            target_uv[visible_indices, 1] / max(float(target_height), 1.0)
            - source_v[visible_indices] / max(float(source_height), 1.0)
        )
        residual_x = delta_x - float(np.median(delta_x))
        residual_y = delta_y - float(np.median(delta_y))
        normalized_displacement = np.hypot(residual_x, residual_y) / math.sqrt(2.0)
        return (
            overlap,
            float(np.percentile(normalized_displacement, 75.0)),
            len(visible_indices),
        )

    def __call__(self, left_name: str, right_name: str) -> VisualRedundancyEvidence:
        key = tuple(sorted((left_name, right_name)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        left_pose = self.poses.get(left_name)
        right_pose = self.poses.get(right_name)
        left_depth = self.depth_frame_for(left_name)
        right_depth = self.depth_frame_for(right_name)
        if left_pose is None or right_pose is None or left_depth is None or right_depth is None:
            evidence = VisualRedundancyEvidence(
                False, False, 0.0, 1.0, 0, 0.0, 0.0, "missing_pose_or_depth"
            )
            self._cache[key] = evidence
            return evidence

        forward = self._directional_reprojection(
            left_pose, right_pose, left_depth, right_depth
        )
        reverse = self._directional_reprojection(
            right_pose, left_pose, right_depth, left_depth
        )
        if forward is None or reverse is None:
            evidence = VisualRedundancyEvidence(
                False, False, 0.0, 1.0, 0, 0.0, 0.0, "insufficient_depth_samples"
            )
            self._cache[key] = evidence
            return evidence

        try:
            rgb = self.rgb_evidence_for(left_name, right_name)
        except (OSError, TypeError, ValueError):
            rgb = None
        if rgb is None:
            evidence = VisualRedundancyEvidence(
                False,
                False,
                min(forward[0], reverse[0]),
                max(forward[1], reverse[1]),
                0,
                0.0,
                0.0,
                "missing_rgb_evidence",
            )
            self._cache[key] = evidence
            return evidence

        orb_inliers = int(getattr(rgb, "inliers", 0) or 0)
        orb_ratio = float(getattr(rgb, "inlier_ratio", 0.0) or 0.0)
        orb_grid = float(getattr(rgb, "min_grid_fraction", 0.0) or 0.0)
        overlap = min(forward[0], reverse[0])
        displacement = max(forward[1], reverse[1])
        is_duplicate = (
            bool(getattr(rgb, "passed", False))
            and overlap >= VISUAL_REDUNDANCY_MIN_BIDIRECTIONAL_OVERLAP
            and displacement <= VISUAL_REDUNDANCY_MAX_P75_DISPLACEMENT_DIAGONAL
            and orb_inliers >= VISUAL_REDUNDANCY_MIN_ORB_INLIERS
            and orb_ratio >= VISUAL_REDUNDANCY_MIN_ORB_INLIER_RATIO
            and orb_grid >= VISUAL_REDUNDANCY_MIN_ORB_GRID_FRACTION
        )
        evidence = VisualRedundancyEvidence(
            available=True,
            is_duplicate=is_duplicate,
            min_bidirectional_overlap=overlap,
            max_p75_displacement_diagonal=displacement,
            orb_inliers=orb_inliers,
            orb_inlier_ratio=orb_ratio,
            orb_min_grid_fraction=orb_grid,
            reason="duplicate" if is_duplicate else "thresholds_not_met",
        )
        self._cache[key] = evidence
        return evidence


@dataclass(frozen=True)
class _FrameCoverage:
    runs: tuple[tuple[float, float], ...]
    valid_fraction: float
    visible_fraction: float
    source: str


@dataclass(frozen=True)
class _CorridorBasis:
    direction_xy: np.ndarray | None
    lateral_a: float
    lateral_b: float


@dataclass(frozen=True)
class _EdgeMetrics:
    cost: float
    local_perpendicular_m: float
    global_perpendicular_m: float
    height_change_m: float
    parallel_change_m: float
    forward_angle_deg: float


@dataclass(frozen=True)
class _EvaluatedPath:
    cost: float
    overlap: float
    metrics: tuple[_EdgeMetrics, ...]
    coverages: tuple[_FrameCoverage, ...]


def _aggregate_bbox(
    objects: Iterable[Mapping[str, Any]], *, padding_m: float = 0.05
) -> tuple[np.ndarray, np.ndarray] | None:
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    for obj in objects:
        bbox_min = np.asarray(obj.get("bbox_min"), dtype=np.float64)
        bbox_max = np.asarray(obj.get("bbox_max"), dtype=np.float64)
        if bbox_min.shape == (3,) and bbox_max.shape == (3,):
            minima.append(bbox_min)
            maxima.append(bbox_max)
    if not minima:
        return None
    return (
        np.min(np.stack(minima), axis=0) - padding_m,
        np.max(np.stack(maxima), axis=0) + padding_m,
    )


def _points_in_bbox(
    points: np.ndarray, bbox: tuple[np.ndarray, np.ndarray] | None
) -> np.ndarray:
    if bbox is None:
        return np.zeros(len(points), dtype=bool)
    return np.all((points >= bbox[0]) & (points <= bbox[1]), axis=1)


def _depth_visibility(
    points: np.ndarray,
    geometric_mask: np.ndarray,
    pose: CameraPose,
    depth_frame: DepthFrame,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = world_to_camera_batch(points, pose)
    uv, expected_depths = project_camera_points_to_image(
        camera_points, depth_frame.intrinsics
    )
    height, width = depth_frame.image_m.shape[:2]
    valid = np.zeros(len(points), dtype=bool)
    visible = np.zeros(len(points), dtype=bool)
    for index in np.flatnonzero(geometric_mask):
        if expected_depths[index] <= 0.0 or not np.isfinite(uv[index]).all():
            continue
        pixel_x = int(round(float(uv[index, 0])))
        pixel_y = int(round(float(uv[index, 1])))
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            continue
        x0 = max(0, pixel_x - DEPTH_NEIGHBORHOOD_RADIUS)
        x1 = min(width, pixel_x + DEPTH_NEIGHBORHOOD_RADIUS + 1)
        y0 = max(0, pixel_y - DEPTH_NEIGHBORHOOD_RADIUS)
        y1 = min(height, pixel_y + DEPTH_NEIGHBORHOOD_RADIUS + 1)
        neighborhood = np.asarray(depth_frame.image_m[y0:y1, x0:x1], dtype=np.float64)
        samples = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0.0)]
        if samples.size < DEPTH_MIN_NEIGHBOR_SAMPLES:
            continue
        valid[index] = True
        observed_depth = float(np.median(samples))
        tolerance = max(
            DEPTH_ABSOLUTE_TOLERANCE_M,
            DEPTH_RELATIVE_TOLERANCE * float(expected_depths[index]),
        )
        visible[index] = float(expected_depths[index]) <= observed_depth + tolerance
    return valid, visible


def _bridge_single_unknowns(
    visible: np.ndarray, valid: np.ndarray, geometric_mask: np.ndarray
) -> np.ndarray:
    bridged = visible.copy()
    if DEPTH_UNKNOWN_BRIDGE_SAMPLES != 1:
        return bridged
    for index in range(1, len(visible) - 1):
        if (
            geometric_mask[index]
            and not valid[index]
            and visible[index - 1]
            and visible[index + 1]
        ):
            bridged[index] = True
    return bridged


def _forward_angle_deg(left: CameraPose, right: CameraPose) -> float:
    left_forward = np.asarray(get_camera_forward(left), dtype=np.float64)
    right_forward = np.asarray(get_camera_forward(right), dtype=np.float64)
    norm = float(np.linalg.norm(left_forward) * np.linalg.norm(right_forward))
    if norm <= 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(left_forward, right_forward) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _corridor_basis(
    center_a: np.ndarray,
    center_b: np.ndarray,
    pose_a: CameraPose,
    pose_b: CameraPose,
) -> _CorridorBasis:
    direction = np.asarray(center_b[:2] - center_a[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < ROUTE_HORIZONTAL_DIRECTION_MIN_M:
        return _CorridorBasis(None, 0.0, 0.0)
    direction /= norm
    position_a = np.asarray(pose_a.position[:2], dtype=np.float64)
    position_b = np.asarray(pose_b.position[:2], dtype=np.float64)
    lateral_a = float(direction[0] * position_a[1] - direction[1] * position_a[0])
    lateral_b = float(direction[0] * position_b[1] - direction[1] * position_b[0])
    return _CorridorBasis(direction, lateral_a, lateral_b)


def _edge_metrics(
    left: CameraPose,
    right: CameraPose,
    *,
    next_frontier: float,
    depth_visible_fraction: float,
    basis: _CorridorBasis,
    orientation_threshold_deg: float,
    local_perpendicular_hard_m: float = ROUTE_LOCAL_PERP_HARD_M,
) -> _EdgeMetrics | None:
    delta = np.asarray(right.position - left.position, dtype=np.float64)
    height_change = abs(float(delta[2]))
    if height_change > ROUTE_HEIGHT_HARD_M:
        return None
    angle = _forward_angle_deg(left, right)
    if angle >= orientation_threshold_deg:
        return None

    if basis.direction_xy is None:
        xy_distance = float(np.linalg.norm(delta[:2]))
        if xy_distance > ROUTE_DEGENERATE_XY_HARD_M:
            return None
        local_perpendicular = xy_distance
        global_perpendicular = 0.0
        parallel_change = 0.0
        translation_cost = 2.0 * (xy_distance / ROUTE_DEGENERATE_XY_SOFT_M) ** 2
    else:
        direction = basis.direction_xy
        parallel_change = abs(float(np.dot(delta[:2], direction)))
        local_perpendicular = abs(
            float(direction[0] * delta[1] - direction[1] * delta[0])
        )
        right_position = np.asarray(right.position[:2], dtype=np.float64)
        right_lateral = float(
            direction[0] * right_position[1] - direction[1] * right_position[0]
        )
        expected_lateral = (
            (1.0 - next_frontier) * basis.lateral_a
            + next_frontier * basis.lateral_b
        )
        global_perpendicular = abs(right_lateral - expected_lateral)
        if (
            local_perpendicular > local_perpendicular_hard_m
            or global_perpendicular > ROUTE_GLOBAL_PERP_HARD_M
        ):
            return None
        translation_cost = (
            2.0 * (local_perpendicular / ROUTE_LOCAL_PERP_SOFT_M) ** 2
            + (global_perpendicular / ROUTE_GLOBAL_PERP_SOFT_M) ** 2
            + 0.25 * (parallel_change / ROUTE_PARALLEL_SOFT_M) ** 2
        )

    cost = (
        ROUTE_EDGE_BASE_COST
        + translation_cost
        + (height_change / ROUTE_HEIGHT_SOFT_M) ** 2
        + 0.5 * (angle / ROUTE_ANGLE_SOFT_DEG) ** 2
        + 2.0 * (1.0 - float(np.clip(depth_visible_fraction, 0.0, 1.0)))
    )
    return _EdgeMetrics(
        cost=cost,
        local_perpendicular_m=local_perpendicular,
        global_perpendicular_m=global_perpendicular,
        height_change_m=height_change,
        parallel_change_m=parallel_change,
        forward_angle_deg=angle,
    )


def find_depth_corridor_auxiliary_route(
    *,
    center_a: np.ndarray,
    center_b: np.ndarray,
    frame_a_name: str,
    frame_b_name: str,
    poses: Mapping[str, CameraPose],
    intrinsics: CameraIntrinsics,
    depth_frame_for: Callable[[str], DepthFrame | None],
    group_a_objects: Iterable[Mapping[str, Any]],
    group_b_objects: Iterable[Mapping[str, Any]],
    visual_redundancy_for: Callable[[str, str], VisualRedundancyEvidence] | None = None,
    max_auxiliary_frames: int = MAX_AUXILIARY_FRAMES,
    orientation_threshold_deg: float = ROUTE_ORIENTATION_THRESHOLD_DEG,
    min_overlap_frac: float = ROUTE_MIN_OVERLAP_FRAC,
    min_progress_frac: float = ROUTE_MIN_PROGRESS_FRAC,
) -> DepthCorridorAuxiliaryRoute | None:
    if frame_a_name == frame_b_name:
        return None
    if frame_a_name not in poses or frame_b_name not in poses:
        return None
    if max_auxiliary_frames < 0:
        raise ValueError("max_auxiliary_frames must be non-negative")
    group_a = tuple(group_a_objects)
    group_b = tuple(group_b_objects)
    if not group_a or not group_b:
        return None

    center_a = np.asarray(center_a, dtype=np.float64)
    center_b = np.asarray(center_b, dtype=np.float64)
    ts, points = sample_route_points(center_a, center_b)
    sample_interval = 1.0 / max(len(ts) - 1, 1)
    min_progress = max(float(min_progress_frac), sample_interval)
    geometric_masks = {
        name: np.asarray(route_visibility_mask(points, pose, intrinsics), dtype=bool)
        for name, pose in poses.items()
    }
    depth_cache: dict[str, _FrameCoverage | None] = {}
    aggregate_a = _aggregate_bbox(group_a)
    aggregate_b = _aggregate_bbox(group_b)

    def frame_coverage(
        image_name: str, *, endpoint_exemption: tuple[np.ndarray, np.ndarray] | None = None
    ) -> _FrameCoverage | None:
        cache_key = image_name if endpoint_exemption is None else f"{image_name}:endpoint"
        if cache_key in depth_cache:
            return depth_cache[cache_key]
        depth_frame = depth_frame_for(image_name)
        if depth_frame is None:
            depth_cache[cache_key] = None
            return None
        geometric = geometric_masks[image_name]
        valid, visible = _depth_visibility(points, geometric, poses[image_name], depth_frame)
        if endpoint_exemption is not None:
            exempt = geometric & _points_in_bbox(points, endpoint_exemption)
            valid |= exempt
            visible |= exempt
        visible = _bridge_single_unknowns(visible, valid, geometric)
        geometric_count = int(np.count_nonzero(geometric))
        valid_fraction = (
            float(np.count_nonzero(valid & geometric) / geometric_count)
            if geometric_count
            else 0.0
        )
        visible_fraction = (
            float(np.count_nonzero(visible & geometric) / geometric_count)
            if geometric_count
            else 0.0
        )
        coverage = _FrameCoverage(
            runs=tuple(_coverage_runs(visible & geometric, ts)),
            valid_fraction=valid_fraction,
            visible_fraction=visible_fraction,
            source=depth_frame.source,
        )
        depth_cache[cache_key] = coverage
        return coverage

    coverage_a = frame_coverage(frame_a_name, endpoint_exemption=aggregate_a)
    coverage_b = frame_coverage(frame_b_name, endpoint_exemption=aggregate_b)
    if coverage_a is None or coverage_b is None:
        return None
    if (
        coverage_a.valid_fraction < DEPTH_MIN_VALID_FRACTION
        or coverage_b.valid_fraction < DEPTH_MIN_VALID_FRACTION
    ):
        return None
    if not coverage_a.runs or coverage_a.runs[0][0] > 1e-9:
        return None
    if not coverage_b.runs or coverage_b.runs[-1][1] < 1.0 - 1e-9:
        return None
    frame_a_end = coverage_a.runs[0][1]
    frame_b_start = coverage_b.runs[-1][0]
    responsibility = max(0.0, frame_b_start - frame_a_end)

    semantic_rejected = 0
    candidate_coverages: dict[str, _FrameCoverage] = {}
    for image_name, geometric in geometric_masks.items():
        if image_name in {frame_a_name, frame_b_name} or not np.any(geometric):
            continue
        geometric_runs = _coverage_runs(geometric, ts)
        if not any(
            run_end > frame_a_end and run_start <= frame_b_start
            for run_start, run_end in geometric_runs
        ):
            continue
        if _semantic_conflict(poses[image_name], intrinsics, group_a, group_b):
            semantic_rejected += 1
            continue
        coverage = frame_coverage(image_name)
        if coverage is None or coverage.valid_fraction < DEPTH_MIN_VALID_FRACTION:
            continue
        useful_runs = tuple(
            (run_start, run_end)
            for run_start, run_end in coverage.runs
            if run_end > frame_a_end and run_start <= frame_b_start
        )
        if useful_runs:
            candidate_coverages[image_name] = _FrameCoverage(
                runs=useful_runs,
                valid_fraction=coverage.valid_fraction,
                visible_fraction=coverage.visible_fraction,
                source=coverage.source,
            )

    basis = _corridor_basis(
        center_a, center_b, poses[frame_a_name], poses[frame_b_name]
    )

    def transition_overlap(frontier: float, next_start: float) -> float:
        return max(0.0, frontier - next_start)

    def evaluate_fixed_path(
        names: tuple[str, ...],
        *,
        relaxed_edges: frozenset[tuple[str, str]] = frozenset(),
    ) -> _EvaluatedPath | None:
        states: dict[
            float,
            tuple[float, float, tuple[_EdgeMetrics, ...], tuple[_FrameCoverage, ...]],
        ] = {frame_a_end: (0.0, 0.0, (), (coverage_a,))}
        tail_name = frame_a_name
        for image_name in names:
            coverage = candidate_coverages.get(image_name)
            if coverage is None:
                return None
            next_states: dict[
                float,
                tuple[float, float, tuple[_EdgeMetrics, ...], tuple[_FrameCoverage, ...]],
            ] = {}
            for frontier, (cost, overlap_sum, metrics, coverages) in states.items():
                for run_start, run_end in coverage.runs:
                    overlap = transition_overlap(frontier, run_start)
                    if (
                        run_start > frontier + 1e-9
                        or run_end - frontier + 1e-9 < min_progress
                        or overlap + 1e-9 < min_overlap_frac
                    ):
                        continue
                    edge = _edge_metrics(
                        poses[tail_name],
                        poses[image_name],
                        next_frontier=run_end,
                        depth_visible_fraction=coverage.visible_fraction,
                        basis=basis,
                        orientation_threshold_deg=(
                            VISUAL_PRUNE_ORIENTATION_THRESHOLD_DEG + 1e-9
                            if (tail_name, image_name) in relaxed_edges
                            else orientation_threshold_deg
                        ),
                        local_perpendicular_hard_m=(
                            VISUAL_PRUNE_LOCAL_PERP_HARD_M
                            if (tail_name, image_name) in relaxed_edges
                            else ROUTE_LOCAL_PERP_HARD_M
                        ),
                    )
                    if edge is None:
                        continue
                    next_cost = cost + edge.cost
                    previous = next_states.get(run_end)
                    if previous is not None and next_cost >= previous[0] - 1e-12:
                        continue
                    next_states[run_end] = (
                        next_cost,
                        overlap_sum + overlap,
                        metrics + (edge,),
                        coverages + (coverage,),
                    )
            if not next_states:
                return None
            states = next_states
            tail_name = image_name

        best: _EvaluatedPath | None = None
        for frontier, (cost, overlap_sum, metrics, coverages) in states.items():
            final_overlap = transition_overlap(frontier, frame_b_start)
            if (
                frame_b_start > frontier + 1e-9
                or final_overlap + 1e-9 < min_overlap_frac
            ):
                continue
            edge = _edge_metrics(
                poses[tail_name],
                poses[frame_b_name],
                next_frontier=1.0,
                depth_visible_fraction=coverage_b.visible_fraction,
                basis=basis,
                orientation_threshold_deg=(
                    VISUAL_PRUNE_ORIENTATION_THRESHOLD_DEG + 1e-9
                    if (tail_name, frame_b_name) in relaxed_edges
                    else orientation_threshold_deg
                ),
                local_perpendicular_hard_m=(
                    VISUAL_PRUNE_LOCAL_PERP_HARD_M
                    if (tail_name, frame_b_name) in relaxed_edges
                    else ROUTE_LOCAL_PERP_HARD_M
                ),
            )
            if edge is None:
                continue
            result = _EvaluatedPath(
                cost=cost + edge.cost,
                overlap=overlap_sum + final_overlap,
                metrics=metrics + (edge,),
                coverages=coverages + (coverage_b,),
            )
            if best is None or (result.cost, result.overlap) < (best.cost, best.overlap):
                best = result
        return best

    queue: list[tuple[float, int, float, int, str, float, tuple[str, ...]]] = [
        (0.0, 0, 0.0, 0, frame_a_name, frame_a_end, ())
    ]
    best_states: dict[tuple[str, float, int], tuple[float, float]] = {
        (frame_a_name, frame_a_end, 0): (0.0, 0.0)
    }
    serial = 0
    best_complete: tuple[float, int, float, tuple[str, ...]] | None = None
    while queue:
        cost, count, overlap_sum, _serial, tail_name, frontier, path = heapq.heappop(
            queue
        )
        if best_states.get((tail_name, frontier, count)) != (cost, overlap_sum):
            continue
        final_overlap = transition_overlap(frontier, frame_b_start)
        final_edge = None
        if (
            frame_b_start <= frontier + 1e-9
            and final_overlap + 1e-9 >= min_overlap_frac
        ):
            final_edge = _edge_metrics(
                poses[tail_name],
                poses[frame_b_name],
                next_frontier=1.0,
                depth_visible_fraction=coverage_b.visible_fraction,
                basis=basis,
                orientation_threshold_deg=orientation_threshold_deg,
            )
        if final_edge is not None:
            complete = (
                cost + final_edge.cost,
                count,
                overlap_sum + final_overlap,
                path,
            )
            if best_complete is None or complete[:3] < best_complete[:3]:
                best_complete = complete
        if count >= max_auxiliary_frames:
            continue
        for image_name in sorted(candidate_coverages):
            if image_name in path:
                continue
            coverage = candidate_coverages[image_name]
            for run_start, run_end in coverage.runs:
                overlap = transition_overlap(frontier, run_start)
                if (
                    run_start > frontier + 1e-9
                    or run_end - frontier + 1e-9 < min_progress
                    or overlap + 1e-9 < min_overlap_frac
                ):
                    continue
                edge = _edge_metrics(
                    poses[tail_name],
                    poses[image_name],
                    next_frontier=run_end,
                    depth_visible_fraction=coverage.visible_fraction,
                    basis=basis,
                    orientation_threshold_deg=orientation_threshold_deg,
                )
                if edge is None:
                    continue
                next_cost = cost + edge.cost
                next_overlap = overlap_sum + overlap
                next_count = count + 1
                key = (image_name, run_end, next_count)
                previous = best_states.get(key)
                if previous is not None and (next_cost, next_overlap) >= previous:
                    continue
                best_states[key] = (next_cost, next_overlap)
                serial += 1
                heapq.heappush(
                    queue,
                    (
                        next_cost,
                        next_count,
                        next_overlap,
                        serial,
                        image_name,
                        run_end,
                        path + (image_name,),
                    ),
                )

    if best_complete is None:
        return None
    selected_names = best_complete[3]
    evaluated = evaluate_fixed_path(selected_names)
    if evaluated is None:
        return None
    pre_prune_count = len(selected_names)
    names = list(selected_names)
    relaxed_edges: set[tuple[str, str]] = set()
    visual_duplicate_names: set[str] = set()
    visual_pruned_count = 0

    def path_edges(path_names: tuple[str, ...]) -> set[tuple[str, str]]:
        full_path = (frame_a_name,) + path_names + (frame_b_name,)
        return set(zip(full_path, full_path[1:]))

    def visual_prune_pass() -> None:
        nonlocal evaluated, relaxed_edges, visual_pruned_count
        if visual_redundancy_for is None:
            return
        changed = True
        while changed:
            changed = False
            for index, image_name in enumerate(names):
                previous_name = frame_a_name if index == 0 else names[index - 1]
                next_name = frame_b_name if index + 1 == len(names) else names[index + 1]
                try:
                    previous_evidence = visual_redundancy_for(
                        previous_name, image_name
                    )
                    is_duplicate = previous_evidence.is_duplicate
                    if not is_duplicate:
                        is_duplicate = visual_redundancy_for(
                            image_name, next_name
                        ).is_duplicate
                except (OSError, TypeError, ValueError):
                    continue
                if not is_duplicate:
                    continue
                visual_duplicate_names.add(image_name)
                candidate_names = tuple(names[:index] + names[index + 1 :])
                candidate_edges = path_edges(candidate_names)
                new_edge = (previous_name, next_name)
                candidate_relaxed_edges = frozenset(
                    (relaxed_edges & candidate_edges) | {new_edge}
                )
                candidate = evaluate_fixed_path(
                    candidate_names,
                    relaxed_edges=candidate_relaxed_edges,
                )
                if candidate is None:
                    continue
                names.pop(index)
                relaxed_edges = set(candidate_relaxed_edges)
                evaluated = candidate
                visual_pruned_count += 1
                changed = True
                break

    def prune_pass(*, near_pose_only: bool) -> None:
        nonlocal evaluated, relaxed_edges
        changed = True
        while changed:
            changed = False
            for index, image_name in enumerate(names):
                if near_pose_only:
                    previous_name = frame_a_name if index == 0 else names[index - 1]
                    next_name = frame_b_name if index + 1 == len(names) else names[index + 1]
                    if not (
                        _poses_are_near_duplicates(
                            poses[previous_name],
                            poses[image_name],
                            translation_threshold_m=ROUTE_NEAR_DUPLICATE_TRANSLATION_M,
                            rotation_threshold_deg=ROUTE_NEAR_DUPLICATE_ROTATION_DEG,
                        )
                        or _poses_are_near_duplicates(
                            poses[image_name],
                            poses[next_name],
                            translation_threshold_m=ROUTE_NEAR_DUPLICATE_TRANSLATION_M,
                            rotation_threshold_deg=ROUTE_NEAR_DUPLICATE_ROTATION_DEG,
                        )
                    ):
                        continue
                candidate_names = tuple(names[:index] + names[index + 1 :])
                candidate_edges = path_edges(candidate_names)
                candidate_relaxed_edges = frozenset(
                    relaxed_edges & candidate_edges
                )
                candidate = evaluate_fixed_path(
                    candidate_names,
                    relaxed_edges=candidate_relaxed_edges,
                )
                if candidate is None or candidate.cost > evaluated.cost + 1e-12:
                    continue
                names.pop(index)
                relaxed_edges = set(candidate_relaxed_edges)
                evaluated = candidate
                changed = True
                break

    visual_prune_pass()
    prune_pass(near_pose_only=True)
    prune_pass(near_pose_only=False)

    metrics = evaluated.metrics
    coverages = evaluated.coverages
    return DepthCorridorAuxiliaryRoute(
        auxiliary_image_names=tuple(names),
        cost=evaluated.cost + 0.01 * evaluated.overlap,
        edge_count=len(metrics),
        route_sample_count=len(ts),
        frame_a_coverage_end=frame_a_end,
        frame_b_coverage_start=frame_b_start,
        auxiliary_responsibility_fraction=responsibility,
        transition_overlap_fraction=evaluated.overlap,
        search_method=ROUTE_SEARCH_METHOD,
        min_progress_fraction=min_progress,
        min_depth_valid_fraction=min(c.valid_fraction for c in coverages),
        min_depth_visible_fraction=min(c.visible_fraction for c in coverages),
        max_local_perpendicular_m=max(m.local_perpendicular_m for m in metrics),
        max_global_perpendicular_m=max(m.global_perpendicular_m for m in metrics),
        max_height_change_m=max(m.height_change_m for m in metrics),
        max_parallel_change_m=max(m.parallel_change_m for m in metrics),
        max_forward_angle_deg=max(m.forward_angle_deg for m in metrics),
        depth_sources=tuple(sorted({c.source for c in coverages})),
        pre_prune_auxiliary_count=pre_prune_count,
        pruned_auxiliary_frame_count=pre_prune_count - len(names),
        visual_pruned_auxiliary_frame_count=visual_pruned_count,
        visual_duplicate_candidate_count=len(visual_duplicate_names),
        visual_prune_relaxed_angle_edge_count=sum(
            1
            for edge, metric in zip(
                zip(
                    (frame_a_name,) + tuple(names),
                    tuple(names) + (frame_b_name,),
                ),
                metrics,
            )
            if edge in relaxed_edges
            and metric.forward_angle_deg > orientation_threshold_deg + 1e-9
        ),
        visual_redundancy_metric_version=VISUAL_REDUNDANCY_METRIC_VERSION,
        semantic_rejected_frame_count=semantic_rejected,
    )


def find_depth_corridor_auxiliary_route_for_objects(
    *,
    group_a_objects: Iterable[Mapping[str, Any]],
    group_b_objects: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> DepthCorridorAuxiliaryRoute | None:
    group_a = tuple(group_a_objects)
    group_b = tuple(group_b_objects)
    return find_depth_corridor_auxiliary_route(
        center_a=object_group_center(group_a),
        center_b=object_group_center(group_b),
        group_a_objects=group_a,
        group_b_objects=group_b,
        **kwargs,
    )
