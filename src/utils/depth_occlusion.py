"""Depth-map based occlusion detection.

Uses ScanNet's pre-rendered depth maps (uint16 PNG, values in mm) to
determine per-object visibility without ray casting.

Algorithm:
    1. Generate 26 sample points on the target object bbox (8 corners +
       12 edge midpoints + 6 face centres).
    2. Transform each point to camera coordinates: p_cam = R @ p_world + t.
    3. Project to the depth-image plane using depth intrinsics:
       u = fx * x/z + cx, v = fy * y/z + cy.
    4. For each point: compare projected depth (z_cam) with the depth map
       value at (u, v).  If z_cam - depth_map[v, u] > tolerance -> occluded.
    5. visibility_ratio = #visible / #valid_projections.

Thresholds:
    - ratio > 0.8 -> "fully visible"
    - 0.2 <= ratio <= 0.8 -> "partially occluded"
    - ratio < 0.2 -> "not visible"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .colmap_loader import CameraIntrinsics, CameraPose
from .coordinate_transform import world_to_camera


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


def _bbox_sample_points(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Return 26 sample points on an axis-aligned bounding box.

    Includes 8 corners, 12 edge midpoints, and 6 face centres.
    Using more points than just corners significantly reduces false
    visibility judgements for large or elongated objects.
    """
    lo, hi = bbox_min, bbox_max
    mid = (lo + hi) / 2.0

    xs = [lo[0], mid[0], hi[0]]
    ys = [lo[1], mid[1], hi[1]]
    zs = [lo[2], mid[2], hi[2]]

    points = set()
    for x in xs:
        for y in ys:
            for z in zs:
                points.add((x, y, z))
    # Remove the interior centre point (mid, mid, mid) — it's not on the surface
    points.discard((mid[0], mid[1], mid[2]))

    return np.array(list(points), dtype=np.float64)


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
    sample_points = _bbox_sample_points(bbox_min, bbox_max)
    h, w = depth_image.shape[:2]

    visible = 0
    valid = 0

    for pt in sample_points:
        # Transform to camera coordinates
        p_cam = world_to_camera(pt, camera_pose)

        # Skip points behind the camera
        if p_cam[2] <= 0:
            continue

        # Project to depth image using depth intrinsics
        u = intrinsics.fx * p_cam[0] / p_cam[2] + intrinsics.cx
        v = intrinsics.fy * p_cam[1] / p_cam[2] + intrinsics.cy

        u_int = int(round(u))
        v_int = int(round(v))

        # Check bounds
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
            continue

        depth_val = depth_image[v_int, u_int]

        # Invalid depth (sensor hole)
        if depth_val <= 0:
            continue

        valid += 1

        # Compare projected depth with depth map
        # If the object corner is further than the depth map reading
        # by more than tolerance → something is in front → occluded
        if p_cam[2] - depth_val > depth_tolerance:
            # Occluded: depth map shows something closer
            pass
        else:
            # Visible: object is at or nearer than depth surface
            visible += 1

    if valid == 0:
        # No sample points projected into the depth image — not in frame
        return "not visible", 0.0

    ratio = visible / valid

    if ratio > 0.8:
        return "fully visible", ratio
    elif ratio >= 0.2:
        return "partially occluded", ratio
    else:
        return "not visible", ratio
