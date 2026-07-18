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
import math
from typing import Any, Iterable, Mapping

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
ROUTE_MIN_OVERLAP_FRAC = 0.15
ROUTE_MAX_BACKTRACK = 20


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


def find_geometric_auxiliary_route(
    *,
    center_a: np.ndarray,
    center_b: np.ndarray,
    frame_a_name: str,
    frame_b_name: str,
    poses: dict[str, CameraPose],
    intrinsics: CameraIntrinsics,
    max_auxiliary_frames: int = MAX_AUXILIARY_FRAMES,
    orientation_threshold_deg: float = ROUTE_ORIENTATION_THRESHOLD_DEG,
    min_overlap_frac: float = ROUTE_MIN_OVERLAP_FRAC,
    max_backtrack: int = ROUTE_MAX_BACKTRACK,
) -> GeometricAuxiliaryRoute | None:
    """Find an old-style geometric bridge between two fixed reasoning frames.

    Only the interval ``(frame_a_coverage_end, frame_b_coverage_start)`` is the
    responsibility of auxiliary frames.  Candidate frames may extend into the
    main-frame intervals just enough to satisfy ``min_overlap_frac`` at a visual
    hand-off.  The function returns ``None`` when either main frame does not see
    its route endpoint or no bounded monotonic chain can connect them.

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
        )
    if max_auxiliary_frames == 0:
        return None

    candidate_runs = {
        name: frame_runs
        for name, frame_runs in runs.items()
        if name not in {frame_a_name, frame_b_name} and frame_runs
    }
    budget = [max(0, int(max_backtrack))]

    def search(
        *,
        frontier: float,
        tail_forward: np.ndarray,
        path: tuple[str, ...],
    ) -> tuple[tuple[str, ...], float] | None:
        if len(path) >= max_auxiliary_frames:
            return None

        feasible: list[tuple[float, float, str]] = []
        for name, frame_runs_for_name in candidate_runs.items():
            if name in path:
                continue
            if not _directions_compatible(tail_forward, forwards[name], cosine_threshold):
                continue
            for run_start, run_end in frame_runs_for_name:
                overlap = transition_overlap(frontier, run_start)
                if run_start > frontier + 1e-9 or run_end <= frontier + 1e-9:
                    continue
                if overlap + 1e-9 < min_overlap_frac:
                    continue
                feasible.append((run_end, overlap, name))

        # Prefer maximum progress, then the least redundant transition overlap.
        feasible.sort(key=lambda row: (-row[0], row[1], row[2]))
        for index, (next_frontier, overlap, name) in enumerate(feasible):
            if index > 0:
                if budget[0] <= 0:
                    break
                budget[0] -= 1
            next_path = path + (name,)
            finished, final_overlap = can_finish(next_frontier, forwards[name])
            if finished:
                return next_path, overlap + final_overlap
            result = search(
                frontier=next_frontier,
                tail_forward=forwards[name],
                path=next_path,
            )
            if result is not None:
                route, downstream_overlap = result
                return route, overlap + downstream_overlap
        return None

    found = search(
        frontier=frame_a_end,
        tail_forward=forwards[frame_a_name],
        path=(),
    )
    if found is None:
        return None
    auxiliary_names, overlap = found
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
    )
