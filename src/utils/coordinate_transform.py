"""Coordinate transformation utilities for CausalSpatial-Bench."""

from __future__ import annotations

import numpy as np

from .colmap_loader import CameraIntrinsics, CameraPose


def world_to_camera(point_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    """Transform a 3D point from world coordinates to camera coordinates.

    Camera convention (OpenCV / COLMAP):
        x → right, y → down, z → forward (into the scene)
    """
    return pose.rotation @ point_world + pose.translation


def camera_to_image(
    point_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Project a camera-space 3D point onto the 2D image plane.

    Returns (u, v) pixel coordinates.  Returns None if the point is behind
    the camera (z <= 0).
    """
    if point_cam[2] <= 0:
        return None
    u = intrinsics.fx * point_cam[0] / point_cam[2] + intrinsics.cx
    v = intrinsics.fy * point_cam[1] / point_cam[2] + intrinsics.cy
    return np.array([u, v])


def project_to_image(
    point_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray | None, float]:
    """Project a world-space point to image pixels.

    Returns:
        (pixel_uv, depth)  where pixel_uv is None if behind the camera.
    """
    p_cam = world_to_camera(point_world, pose)
    depth = p_cam[2]
    uv = camera_to_image(p_cam, intrinsics)
    return uv, depth


def is_in_image(
    uv: np.ndarray | None, intrinsics: CameraIntrinsics, margin: int = 0
) -> bool:
    """Check whether a 2D pixel coordinate falls within image bounds."""
    if uv is None:
        return False
    return (
        margin <= uv[0] < intrinsics.width - margin
        and margin <= uv[1] < intrinsics.height - margin
    )


# ---- Camera axis helpers (world-space directions) ----

def get_camera_right(pose: CameraPose) -> np.ndarray:
    """Unit vector pointing to the camera's right in world coordinates."""
    # Camera x-axis in world = first row of R^T
    return pose.rotation.T[:, 0]


def get_camera_up(pose: CameraPose) -> np.ndarray:
    """Unit vector pointing upward from the camera in world coordinates.

    Note: in OpenCV convention camera y points *down*, so "up" is -y.
    """
    return -pose.rotation.T[:, 1]


def get_camera_forward(pose: CameraPose) -> np.ndarray:
    """Unit vector pointing forward (into the scene) in world coordinates."""
    return pose.rotation.T[:, 2]


# ---- Rotation helpers ----

def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix for rotation about the world z-axis."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
