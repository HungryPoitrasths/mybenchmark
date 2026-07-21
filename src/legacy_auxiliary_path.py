"""Question-specific geometric auxiliary-frame routing.

This module preserves the route-projection method that predates
``src.auxiliary_path.VisualPoseGraph``.  It is intentionally independent of the
current scene-wide RGB/pose graph so the two methods can be evaluated side by
side.

The original search always started auxiliary coverage at ``t=0`` on the line
from object group A to object group B.  That made auxiliary frames duplicate
route portions already visible in the two reasoning frames.  This version starts
at the end of frame A's endpoint-connected coverage and stops at the beginning
of frame B's endpoint-connected coverage.  A small overlap is still required at
each hand-off; that overlap is continuity evidence, not auxiliary responsibility
for the whole route.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .auxiliary_path import MAX_AUXILIARY_FRAMES
from .utils.colmap_loader import CameraIntrinsics, CameraPose
from .utils.coordinate_transform import (
    get_camera_forward,
    project_camera_points_to_image,
    world_to_camera_batch,
)


ROUTE_SAMPLE_SPACING_M = 0.09
ROUTE_MIN_SAMPLES = 10
ROUTE_MAX_SAMPLES = 40
ROUTE_ORIENTATION_THRESHOLD_DEG = 60.0
ROUTE_MIN_OVERLAP_FRAC = 0.10
ROUTE_MAX_BACKTRACK = 20
ROUTE_MIN_PROGRESS_FRAC = 0.05
ROUTE_NEAR_DUPLICATE_TRANSLATION_M = 0.12
ROUTE_NEAR_DUPLICATE_ROTATION_DEG = 6.0
ROUTE_SEARCH_METHOD = "dijkstra_lexicographic"
# Keep semantic gating aligned with HybridRoutingConfig defaults. The legacy
# route only uses these thresholds to reject frames that show both question
# sides at once; it does not require either side to be visible on its own.
ROUTE_SEMANTIC_MIN_DEPTH_M = 0.3
ROUTE_SEMANTIC_MAX_DEPTH_M = 8.0
ROUTE_SEMANTIC_MIN_BBOX_IN_FRAME_RATIO = 0.20
ROUTE_SEMANTIC_MIN_PROJECTED_AREA_RATIO = 800.0 / (640.0 * 480.0)


@dataclass(frozen=True)
class GeometricAuxiliaryRoute:
    """A route selected only from pose, projection, and camera orientation."""

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
    near_duplicate_translation_m: float
    near_duplicate_rotation_deg: float
    pre_prune_auxiliary_count: int
    pruned_auxiliary_frame_count: int
    semantic_rejected_frame_count: int = 0


def object_group_center(objects: Iterable[Mapping[str, Any]]) -> np.ndarray:
    """Return the mean object center, falling back to each object's bbox center."""

    centers: list[np.ndarray] = []
    for obj in objects:
        raw_center = obj.get("center")
        if raw_center is None:
            bbox_min = np.asarray(obj.get("bbox_min"), dtype=np.float64)
            bbox_max = np.asarray(obj.get("bbox_max"), dtype=np.float64)
            if bbox_min.shape != (3,) or bbox_max.shape != (3,):
                raise ValueError("each object must provide center or valid bbox_min/bbox_max")
            center = 0.5 * (bbox_min + bbox_max)
        else:
            center = np.asarray(raw_center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("each object center must contain three finite values")
        centers.append(center)
    if not centers:
        raise ValueError("objects must not be empty")
    return np.mean(np.stack(centers, axis=0), axis=0)


def _object_projection_metrics(
    obj: Mapping[str, Any],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Return center depth, in-frame bbox-corner ratio, and projected area ratio."""

    bbox_min = np.asarray(obj.get("bbox_min"), dtype=np.float64)
    bbox_max = np.asarray(obj.get("bbox_max"), dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        return 0.0, 0.0, 0.0
    center = np.asarray(obj.get("center", 0.5 * (bbox_min + bbox_max)), dtype=np.float64)
    if center.shape != (3,):
        center = 0.5 * (bbox_min + bbox_max)

    corners = np.asarray(
        [
            [x, y, z]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float64,
    )
    points_camera = (pose.rotation @ corners.T + pose.translation[:, None]).T
    uv, depths = project_camera_points_to_image(points_camera, intrinsics)
    valid = (depths > 0.0) & np.isfinite(uv).all(axis=1)
    center_depth = float(pose.world_to_camera_point(center)[2])
    if not valid.any():
        return center_depth, 0.0, 0.0
    valid_uv = uv[valid]
    in_frame = (
        (valid_uv[:, 0] >= 0.0)
        & (valid_uv[:, 0] < intrinsics.width)
        & (valid_uv[:, 1] >= 0.0)
        & (valid_uv[:, 1] < intrinsics.height)
    )
    bbox_ratio = float(in_frame.sum() / len(valid_uv))
    left = float(np.clip(valid_uv[:, 0].min(), 0.0, intrinsics.width))
    right = float(np.clip(valid_uv[:, 0].max(), 0.0, intrinsics.width))
    top = float(np.clip(valid_uv[:, 1].min(), 0.0, intrinsics.height))
    bottom = float(np.clip(valid_uv[:, 1].max(), 0.0, intrinsics.height))
    area_ratio = max(0.0, right - left) * max(0.0, bottom - top) / max(
        float(intrinsics.width * intrinsics.height), 1.0
    )
    return center_depth, bbox_ratio, area_ratio


def _meaningfully_projected(
    objects: Iterable[Mapping[str, Any]],
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> bool:
    for obj in objects:
        depth, bbox_ratio, area_ratio = _object_projection_metrics(obj, pose, intrinsics)
        if (
            ROUTE_SEMANTIC_MIN_DEPTH_M < depth <= ROUTE_SEMANTIC_MAX_DEPTH_M
            and bbox_ratio >= ROUTE_SEMANTIC_MIN_BBOX_IN_FRAME_RATIO
            and area_ratio >= ROUTE_SEMANTIC_MIN_PROJECTED_AREA_RATIO
        ):
            return True
    return False


def _semantic_conflict(
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    group_a_objects: Iterable[Mapping[str, Any]],
    group_b_objects: Iterable[Mapping[str, Any]],
) -> bool:
    """Whether a frame meaningfully shows both question-side object groups."""

    return _meaningfully_projected(group_a_objects, pose, intrinsics) and _meaningfully_projected(
        group_b_objects, pose, intrinsics
    )


def sample_route_points(
    center_a: np.ndarray,
    center_b: np.ndarray,
    *,
    spacing_m: float = ROUTE_SAMPLE_SPACING_M,
    min_samples: int = ROUTE_MIN_SAMPLES,
    max_samples: int = ROUTE_MAX_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the 3D line from group center A to group center B."""

    start = np.asarray(center_a, dtype=np.float64)
    end = np.asarray(center_b, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("center_a and center_b must each have shape (3,)")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("route centers must contain only finite values")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if min_samples < 2 or max_samples < min_samples:
        raise ValueError("sample limits must satisfy 2 <= min_samples <= max_samples")

    distance = float(np.linalg.norm(end - start))
    sample_count = int(np.clip(round(distance / spacing_m), min_samples, max_samples))
    ts = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    points = start[None, :] + ts[:, None] * (end - start)[None, :]
    return ts, points


def route_visibility_mask(
    route_points: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Return which sampled route points project inside a real camera frame."""

    camera_points = world_to_camera_batch(route_points, pose)
    uv, depths = project_camera_points_to_image(camera_points, intrinsics)
    return (
        (depths > 0.0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < intrinsics.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < intrinsics.height)
    )


def _coverage_runs(mask: np.ndarray, ts: np.ndarray) -> list[tuple[float, float]]:
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for index, visible in enumerate(mask):
        if bool(visible) and start is None:
            start = index
        elif not bool(visible) and start is not None:
            runs.append((float(ts[start]), float(ts[index - 1])))
            start = None
    if start is not None:
        runs.append((float(ts[start]), float(ts[-1])))
    return runs


def _unit_forward(pose: CameraPose) -> np.ndarray:
    forward = np.asarray(get_camera_forward(pose), dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return forward / norm


def _directions_compatible(left: np.ndarray, right: np.ndarray, threshold: float) -> bool:
    return float(np.dot(left, right)) > threshold


def _pose_translation_m(left: CameraPose, right: CameraPose) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.position, dtype=np.float64)
            - np.asarray(right.position, dtype=np.float64)
        )
    )


def _pose_rotation_deg(left: CameraPose, right: CameraPose) -> float:
    left_rotation = np.asarray(left.rotation, dtype=np.float64)
    right_rotation = np.asarray(right.rotation, dtype=np.float64)
    relative = right_rotation @ left_rotation.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _poses_are_near_duplicates(
    left: CameraPose,
    right: CameraPose,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> bool:
    return (
        _pose_translation_m(left, right) < translation_threshold_m
        and _pose_rotation_deg(left, right) < rotation_threshold_deg
    )


def _prune_auxiliary_names(
    auxiliary_names: tuple[str, ...],
    *,
    frame_a_name: str,
    frame_b_name: str,
    poses: Mapping[str, CameraPose],
    route_is_valid: Callable[[tuple[str, ...]], bool],
    near_duplicate_translation_m: float,
    near_duplicate_rotation_deg: float,
) -> tuple[str, ...]:
    """Remove auxiliary frames while preserving the complete route constraints."""

    names = list(auxiliary_names)

    def prune_pass(*, near_pose_only: bool) -> None:
        changed = True
        while changed:
            changed = False
            for index, image_name in enumerate(names):
                if near_pose_only:
                    previous_name = frame_a_name if index == 0 else names[index - 1]
                    next_name = frame_b_name if index + 1 == len(names) else names[index + 1]
                    near_previous = _poses_are_near_duplicates(
                        poses[previous_name],
                        poses[image_name],
                        translation_threshold_m=near_duplicate_translation_m,
                        rotation_threshold_deg=near_duplicate_rotation_deg,
                    )
                    near_next = _poses_are_near_duplicates(
                        poses[image_name],
                        poses[next_name],
                        translation_threshold_m=near_duplicate_translation_m,
                        rotation_threshold_deg=near_duplicate_rotation_deg,
                    )
                    if not (near_previous or near_next):
                        continue

                candidate = tuple(names[:index] + names[index + 1 :])
                if not route_is_valid(candidate):
                    continue
                names.pop(index)
                changed = True
                break

    # Remove obvious near-pose duplicates first, then defensively remove any other
    # auxiliary frame made redundant by equivalent coverage runs.
    prune_pass(near_pose_only=True)
    prune_pass(near_pose_only=False)
    return tuple(names)


def find_geometric_auxiliary_route(
    *,
    center_a: np.ndarray,
    center_b: np.ndarray,
    frame_a_name: str,
    frame_b_name: str,
    poses: dict[str, CameraPose],
    intrinsics: CameraIntrinsics,
    group_a_objects: Iterable[Mapping[str, Any]] | None = None,
    group_b_objects: Iterable[Mapping[str, Any]] | None = None,
    max_auxiliary_frames: int = MAX_AUXILIARY_FRAMES,
    orientation_threshold_deg: float = ROUTE_ORIENTATION_THRESHOLD_DEG,
    min_overlap_frac: float = ROUTE_MIN_OVERLAP_FRAC,
    max_backtrack: int = ROUTE_MAX_BACKTRACK,
    min_progress_frac: float = ROUTE_MIN_PROGRESS_FRAC,
    near_duplicate_translation_m: float = ROUTE_NEAR_DUPLICATE_TRANSLATION_M,
    near_duplicate_rotation_deg: float = ROUTE_NEAR_DUPLICATE_ROTATION_DEG,
) -> GeometricAuxiliaryRoute | None:
    """Find a globally optimal geometric bridge between two reasoning frames.

    Only the interval ``(frame_a_coverage_end, frame_b_coverage_start)`` is the
    responsibility of auxiliary frames.  Candidate frames may extend into the
    main-frame intervals just enough to satisfy ``min_overlap_frac`` at a visual
    hand-off. Dijkstra search minimizes auxiliary-frame count first and total
    redundant transition overlap second. Each auxiliary frame must advance the
    coverage frontier by at least ``min_progress_frac`` or one sample interval.
    ``max_backtrack`` remains in the signature for caller compatibility but no
    longer limits the global search.

    This method deliberately does not read RGB pixels and therefore reproduces
    the old method's main limitation: projected route overlap does not prove that
    consecutive images share visible scene content.
    """

    if frame_a_name == frame_b_name:
        return None
    if frame_a_name not in poses or frame_b_name not in poses:
        return None
    if max_auxiliary_frames < 0:
        raise ValueError("max_auxiliary_frames must be non-negative")
    if not 0.0 <= min_overlap_frac <= 1.0:
        raise ValueError("min_overlap_frac must be in [0, 1]")
    if not 0.0 <= orientation_threshold_deg <= 180.0:
        raise ValueError("orientation_threshold_deg must be in [0, 180]")
    if not 0.0 <= min_progress_frac <= 1.0:
        raise ValueError("min_progress_frac must be in [0, 1]")
    if near_duplicate_translation_m < 0.0:
        raise ValueError("near_duplicate_translation_m must be non-negative")
    if not 0.0 <= near_duplicate_rotation_deg <= 180.0:
        raise ValueError("near_duplicate_rotation_deg must be in [0, 180]")
    _ = max_backtrack

    ts, points = sample_route_points(center_a, center_b)
    masks: dict[str, np.ndarray] = {}
    for name, pose in poses.items():
        mask = np.asarray(route_visibility_mask(points, pose, intrinsics), dtype=bool)
        if mask.shape != ts.shape:
            raise ValueError(
                f"route visibility mask for {name!r} has shape {mask.shape}; "
                f"expected {ts.shape}"
            )
        masks[name] = mask
    runs = {
        name: _coverage_runs(mask, ts)
        for name, mask in masks.items()
    }

    frame_a_runs = runs[frame_a_name]
    frame_b_runs = runs[frame_b_name]
    if not frame_a_runs or frame_a_runs[0][0] > 1e-9:
        return None
    if not frame_b_runs or frame_b_runs[-1][1] < 1.0 - 1e-9:
        return None

    frame_a_end = frame_a_runs[0][1]
    frame_b_start = frame_b_runs[-1][0]
    responsibility = max(0.0, frame_b_start - frame_a_end)
    cosine_threshold = math.cos(math.radians(orientation_threshold_deg))
    forwards = {name: _unit_forward(pose) for name, pose in poses.items()}
    sample_interval = 1.0 / max(len(ts) - 1, 1)
    min_progress = max(float(min_progress_frac), sample_interval)
    semantic_group_a = tuple(group_a_objects or ())
    semantic_group_b = tuple(group_b_objects or ())
    semantic_gate_enabled = bool(semantic_group_a and semantic_group_b)

    def transition_overlap(frontier: float, next_start: float) -> float:
        return max(0.0, frontier - next_start)

    def can_finish(frontier: float, tail_forward: np.ndarray) -> tuple[bool, float]:
        overlap = transition_overlap(frontier, frame_b_start)
        return (
            frame_b_start <= frontier + 1e-9
            and overlap + 1e-9 >= min_overlap_frac
            and _directions_compatible(tail_forward, forwards[frame_b_name], cosine_threshold),
            overlap,
        )

    direct, direct_overlap = can_finish(frame_a_end, forwards[frame_a_name])
    if direct:
        return GeometricAuxiliaryRoute(
            auxiliary_image_names=(),
            cost=0.0,
            edge_count=1,
            route_sample_count=len(ts),
            frame_a_coverage_end=frame_a_end,
            frame_b_coverage_start=frame_b_start,
            auxiliary_responsibility_fraction=responsibility,
            transition_overlap_fraction=direct_overlap,
            search_method=ROUTE_SEARCH_METHOD,
            min_progress_fraction=min_progress,
            near_duplicate_translation_m=near_duplicate_translation_m,
            near_duplicate_rotation_deg=near_duplicate_rotation_deg,
            pre_prune_auxiliary_count=0,
            pruned_auxiliary_frame_count=0,
        )
    if max_auxiliary_frames == 0:
        return None

    semantic_rejected_frame_count = 0
    candidate_runs: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, frame_runs in runs.items():
        if name in {frame_a_name, frame_b_name} or not frame_runs:
            continue
        if semantic_gate_enabled and _semantic_conflict(
            poses[name], intrinsics, semantic_group_a, semantic_group_b
        ):
            semantic_rejected_frame_count += 1
            continue
        candidate_runs[name] = tuple(sorted(frame_runs, key=lambda run: (run[1], run[0])))

    def best_fixed_path(
        auxiliary_names: tuple[str, ...],
    ) -> tuple[float, tuple[tuple[str, float, float], ...]] | None:
        states: dict[float, tuple[float, tuple[tuple[str, float, float], ...]]] = {
            frame_a_end: (0.0, ())
        }
        tail_name = frame_a_name
        for image_name in auxiliary_names:
            if image_name not in candidate_runs:
                return None
            if not _directions_compatible(
                forwards[tail_name], forwards[image_name], cosine_threshold
            ):
                return None
            next_states: dict[
                float, tuple[float, tuple[tuple[str, float, float], ...]]
            ] = {}
            for frontier, (overlap_cost, selected_runs) in states.items():
                for run_start, run_end in candidate_runs[image_name]:
                    overlap = transition_overlap(frontier, run_start)
                    if run_start > frontier + 1e-9:
                        continue
                    if run_end - frontier + 1e-9 < min_progress:
                        continue
                    if overlap + 1e-9 < min_overlap_frac:
                        continue
                    candidate_cost = overlap_cost + overlap
                    previous = next_states.get(run_end)
                    if previous is not None and candidate_cost >= previous[0] - 1e-12:
                        continue
                    next_states[run_end] = (
                        candidate_cost,
                        selected_runs + ((image_name, run_start, run_end),),
                    )
            if not next_states:
                return None
            states = next_states
            tail_name = image_name

        best_result: tuple[float, tuple[tuple[str, float, float], ...]] | None = None
        for frontier, (overlap_cost, selected_runs) in states.items():
            finished, final_overlap = can_finish(frontier, forwards[tail_name])
            if not finished:
                continue
            total_overlap = overlap_cost + final_overlap
            if best_result is None or total_overlap < best_result[0] - 1e-12:
                best_result = total_overlap, selected_runs
        return best_result

    # Heap priority is lexicographic: first auxiliary count, then accumulated
    # redundant overlap. A terminal entry represents the zero-frame-cost edge to B.
    queue: list[
        tuple[int, float, int, int, str, float, tuple[str, ...]]
    ] = [(0, 0.0, 1, 0, frame_a_name, frame_a_end, ())]
    best: dict[tuple[str, float], tuple[int, float]] = {
        (frame_a_name, frame_a_end): (0, 0.0)
    }
    serial = 0
    auxiliary_names: tuple[str, ...] | None = None

    while queue:
        auxiliary_count, overlap_cost, entry_kind, _, tail_name, frontier, path = (
            heapq.heappop(queue)
        )
        if entry_kind == 0:
            auxiliary_names = path
            break
        if best.get((tail_name, frontier)) != (auxiliary_count, overlap_cost):
            continue

        finished, final_overlap = can_finish(frontier, forwards[tail_name])
        if finished:
            serial += 1
            heapq.heappush(
                queue,
                (
                    auxiliary_count,
                    overlap_cost + final_overlap,
                    0,
                    serial,
                    frame_b_name,
                    frontier,
                    path,
                ),
            )

        if auxiliary_count >= max_auxiliary_frames:
            continue
        for image_name in sorted(candidate_runs):
            if image_name in path:
                continue
            if not _directions_compatible(
                forwards[tail_name], forwards[image_name], cosine_threshold
            ):
                continue
            for run_start, run_end in candidate_runs[image_name]:
                overlap = transition_overlap(frontier, run_start)
                if run_start > frontier + 1e-9:
                    continue
                if run_end - frontier + 1e-9 < min_progress:
                    continue
                if overlap + 1e-9 < min_overlap_frac:
                    continue
                next_cost = (auxiliary_count + 1, overlap_cost + overlap)
                state_key = (image_name, run_end)
                previous_cost = best.get(state_key)
                if previous_cost is not None and next_cost >= previous_cost:
                    continue
                best[state_key] = next_cost
                serial += 1
                heapq.heappush(
                    queue,
                    (
                        next_cost[0],
                        next_cost[1],
                        1,
                        serial,
                        image_name,
                        run_end,
                        path + (image_name,),
                    ),
                )

    if auxiliary_names is None:
        return None

    def route_is_valid(names: tuple[str, ...]) -> bool:
        return best_fixed_path(names) is not None

    pre_prune_auxiliary_count = len(auxiliary_names)
    auxiliary_names = _prune_auxiliary_names(
        auxiliary_names,
        frame_a_name=frame_a_name,
        frame_b_name=frame_b_name,
        poses=poses,
        route_is_valid=route_is_valid,
        near_duplicate_translation_m=near_duplicate_translation_m,
        near_duplicate_rotation_deg=near_duplicate_rotation_deg,
    )
    fixed_path = best_fixed_path(auxiliary_names)
    if fixed_path is None:
        return None
    overlap, _selected_runs = fixed_path
    # Frame count dominates; overlap is a small tie-break penalty for duplicated view.
    cost = float(len(auxiliary_names)) + 0.01 * overlap
    return GeometricAuxiliaryRoute(
        auxiliary_image_names=auxiliary_names,
        cost=cost,
        edge_count=len(auxiliary_names) + 1,
        route_sample_count=len(ts),
        frame_a_coverage_end=frame_a_end,
        frame_b_coverage_start=frame_b_start,
        auxiliary_responsibility_fraction=responsibility,
        transition_overlap_fraction=overlap,
        search_method=ROUTE_SEARCH_METHOD,
        min_progress_fraction=min_progress,
        near_duplicate_translation_m=near_duplicate_translation_m,
        near_duplicate_rotation_deg=near_duplicate_rotation_deg,
        pre_prune_auxiliary_count=pre_prune_auxiliary_count,
        pruned_auxiliary_frame_count=pre_prune_auxiliary_count - len(auxiliary_names),
        semantic_rejected_frame_count=semantic_rejected_frame_count,
    )
