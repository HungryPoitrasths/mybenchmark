"""Depth-map based occlusion detection.

Uses ScanNet's pre-rendered depth maps (uint16 PNG, values in mm) to
determine per-object visibility without ray casting.

Algorithm:
    1. Generate 26 sample points on the target object bbox (8 corners +
       12 edge midpoints + 6 face centres).
    2. Transform each point to camera coordinates: p_cam = R @ p_world + t.
    3. Project each point into the image plane.
    4. Measure projected footprint and in-frame coverage to reject objects
       whose visible portion is only a tiny corner near the image boundary.
    5. If a depth map is available, compare each projected depth with the depth
       image and compute visibility_ratio = #visible / #valid_projections.
    6. Classify primarily from visibility_ratio, without the old hard rule
       that required the bbox centre itself to be visible.

Thresholds:
    - projected_area < 400 px or in_frame_ratio < 0.25 -> "not visible"
    - ratio > 0.65 -> "fully visible"
    - 0.2 <= ratio <= 0.65 -> "partially occluded"
    - ratio < 0.2 -> "not visible"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .colmap_loader import CameraIntrinsics, CameraPose
from .coordinate_transform import world_to_camera


MIN_PROJECTED_AREA_PX = 400.0
MIN_IN_FRAME_RATIO = 0.25
FULLY_VISIBLE_RATIO_MIN = 0.65
PARTIALLY_VISIBLE_RATIO_MIN = 0.20


def load_depth_image(depth_path: Path | str) -> np.ndarray:
    """Load a ScanNet depth PNG as a float32 array in metres.

    ScanNet depth maps are stored as 16-bit unsigned integers where the
    value is the depth in **millimetres**.  A pixel value of 0 means
    "no depth measurement" (invalid).
    """
    import cv2

    depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_uint16 is None:
        raise FileNotFoundError(f"Cannot read depth image: {depth_path}")
    return depth_uint16.astype(np.float32) / 1000.0  # mm → metres


def bbox_camera_facing_sample_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    camera_pos: np.ndarray,
) -> np.ndarray:
    """Sample the camera-facing shell of an axis-aligned bounding box.

    A depth map only captures the nearest visible surface of an object. If we
    also sample back faces of the bbox, those points are "occluded" by the
    object's own front surface and large thick objects get penalised
    incorrectly. To avoid that self-occlusion failure mode, we only sample the
    three faces that can be visible from the current viewpoint.
    """
    lo = np.asarray(bbox_min, dtype=np.float64)
    hi = np.asarray(bbox_max, dtype=np.float64)
    mid = (lo + hi) / 2.0

    grid = [
        [lo[0], mid[0], hi[0]],
        [lo[1], mid[1], hi[1]],
        [lo[2], mid[2], hi[2]],
    ]
    face_coords = [
        lo[axis] if camera_pos[axis] <= mid[axis] else hi[axis]
        for axis in range(3)
    ]

    points: set[tuple[float, float, float]] = set()
    for fixed_axis, fixed_value in enumerate(face_coords):
        other_axes = [axis for axis in range(3) if axis != fixed_axis]
        for a in grid[other_axes[0]]:
            for b in grid[other_axes[1]]:
                coords = [0.0, 0.0, 0.0]
                coords[fixed_axis] = float(fixed_value)
                coords[other_axes[0]] = float(a)
                coords[other_axes[1]] = float(b)
                points.add(tuple(coords))

    return np.array(sorted(points), dtype=np.float64)


def _ray_box_entry_depth(
    u: float,
    v: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    camera_pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> float | None:
    """Return the camera-space z depth where pixel ray first enters the bbox."""
    camera_pos = np.asarray(camera_pose.position, dtype=np.float64)
    ray_dir_cam = np.array(
        [
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
            1.0,
        ],
        dtype=np.float64,
    )
    ray_dir_world = camera_pose.rotation.T @ ray_dir_cam

    t_min = 0.0
    t_max = float("inf")
    for axis in range(3):
        origin = float(camera_pos[axis])
        direction = float(ray_dir_world[axis])
        lo = float(bbox_min[axis])
        hi = float(bbox_max[axis])

        if abs(direction) < 1e-8:
            if origin < lo or origin > hi:
                return None
            continue

        t0 = (lo - origin) / direction
        t1 = (hi - origin) / direction
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return None

    if t_max <= 0:
        return None

    hit_world = camera_pos + max(t_min, 0.0) * ray_dir_world
    hit_cam = world_to_camera(hit_world, camera_pose)
    if hit_cam[2] <= 0:
        return None
    return float(hit_cam[2])


def compute_depth_occlusion(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    camera_pose: CameraPose,
    intrinsics: CameraIntrinsics,
    depth_image: np.ndarray,
    depth_tolerance: float = 0.10,
) -> tuple[str, float]:
    """Determine the visibility status of an object from the depth map.

    Args:
        bbox_min: Object bounding box minimum corner (world coords).
        bbox_max: Object bounding box maximum corner (world coords).
        camera_pose: Camera extrinsic parameters (world-to-camera).
        intrinsics: Depth camera intrinsic parameters.
        depth_image: Depth map as float32 array in metres.
        depth_tolerance: Tolerance in metres for depth comparison.

    Returns:
        (status, visibility_ratio) where status is one of
        "fully visible", "partially occluded", "not visible".
    """
    sample_points = bbox_camera_facing_sample_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        camera_pos=camera_pose.position,
    )
    h, w = depth_image.shape[:2]
    projected = []

    visible = 0
    valid = 0

    for pt in sample_points:
        p_cam = world_to_camera(pt, camera_pose)
        if p_cam[2] <= 0:
            continue
        u = intrinsics.fx * p_cam[0] / p_cam[2] + intrinsics.cx
        v = intrinsics.fy * p_cam[1] / p_cam[2] + intrinsics.cy
        u_int = int(round(u))
        v_int = int(round(v))
        projected.append((u, v))
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
            continue
        depth_val = depth_image[v_int, u_int]
        if depth_val <= 0:
            continue
        entry_depth = _ray_box_entry_depth(
            u=u,
            v=v,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            camera_pose=camera_pose,
            intrinsics=intrinsics,
        )
        if entry_depth is None:
            continue
        valid += 1
        if entry_depth - depth_val > depth_tolerance:
            pass  # occluded
        else:
            visible += 1

    if valid == 0:
        return "not visible", 0.0

    projected_area, in_frame_ratio = _projected_bbox_stats(projected, h, w)
    if projected_area < MIN_PROJECTED_AREA_PX or in_frame_ratio < MIN_IN_FRAME_RATIO:
        return "not visible", 0.0

    ratio = visible / valid

    if ratio > FULLY_VISIBLE_RATIO_MIN:
        return "fully visible", ratio
    elif ratio >= PARTIALLY_VISIBLE_RATIO_MIN:
        return "partially occluded", ratio
    else:
        return "not visible", ratio


def _projected_bbox_stats(
    projected_points: list[tuple[float, float]],
    h: int,
    w: int,
) -> tuple[float, float]:
    """Return projected bbox area and fraction of sample points inside frame."""
    if not projected_points:
        return 0.0, 0.0

    us = [float(u) for u, _ in projected_points]
    vs = [float(v) for _, v in projected_points]
    u_min = max(0.0, min(us))
    v_min = max(0.0, min(vs))
    u_max = min(float(w), max(us))
    v_max = min(float(h), max(vs))
    area = max(0.0, u_max - u_min) * max(0.0, v_max - v_min)

    in_frame = sum(
        1 for u, v in projected_points
        if 0 <= u < w and 0 <= v < h
    )
    return float(area), float(in_frame / len(projected_points))
