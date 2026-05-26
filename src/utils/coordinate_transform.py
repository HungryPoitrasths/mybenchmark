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


def world_to_camera_batch(points_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    """Transform ``(N, 3)`` world-space points to camera coordinates."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    rotation = np.asarray(pose.rotation, dtype=np.float64)
    translation = np.asarray(pose.translation, dtype=np.float64)
    return points @ rotation.T + translation


def camera_to_image(
    point_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray | None:
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

    Supports pinhole and distortion models (e.g. ``OPENCV_FISHEYE``)
    declared in *intrinsics*.

    Returns:
        (pixel_uv, depth)  where pixel_uv is None if behind the camera.
    """
    p_cam = world_to_camera(point_world, pose)
    depth = float(p_cam[2])
    if depth <= 0:
        return None, depth
    uv, _depths = project_camera_points_to_image(p_cam.reshape(1, 3), intrinsics)
    uv_flat = uv[0]
    if np.isinf(uv_flat[0]) or np.isinf(uv_flat[1]):
        return None, depth
    return uv_flat, depth


def _project_pinhole_batch(
    points_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Pinhole projection for N camera-space points → (N, 2) uv array.

    Points with z <= 0 get uv = (inf, inf).
    """
    uv = np.full((len(points_cam), 2), np.inf, dtype=np.float64)
    z = points_cam[:, 2]
    valid = z > 0
    if valid.any():
        uv[valid, 0] = (
            intrinsics.fx * points_cam[valid, 0] / z[valid] + intrinsics.cx
        )
        uv[valid, 1] = (
            intrinsics.fy * points_cam[valid, 1] / z[valid] + intrinsics.cy
        )
    return uv


def project_pinhole_batch(
    points_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Public wrapper for pinhole projection of camera-space point batches."""
    return _project_pinhole_batch(points_cam, intrinsics)


def _project_fisheye_batch(
    points_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """OPENCV_FISHEYE projection via cv2.fisheye.projectPoints()."""
    import cv2

    K = intrinsics.to_matrix().astype(np.float64)
    D = np.asarray(intrinsics.distortion_params, dtype=np.float64)
    object_points = points_cam.reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    image_points, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
    return np.asarray(image_points, dtype=np.float64).reshape(-1, 2)


def _project_opencv_batch(
    points_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """OPENCV radial-tangential projection via cv2.projectPoints()."""
    import cv2

    K = intrinsics.to_matrix().astype(np.float64)
    D = np.asarray(intrinsics.distortion_params, dtype=np.float64)
    object_points = points_cam.reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    return np.asarray(image_points, dtype=np.float64).reshape(-1, 2)


def project_camera_points_to_image(
    points_cam: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project N camera-space 3D points to 2D pixel coordinates.

    Supports three distortion models declared in *intrinsics*:

      - ``""`` / ``"PINHOLE"`` — pinhole (no distortion)
      - ``"OPENCV"`` — :func:`cv2.projectPoints` (k1, k2, p1, p2)
      - ``"OPENCV_FISHEYE"`` — :func:`cv2.fisheye.projectPoints` (k1-k4)

    Points with z <= 0 are assigned ``uv = (inf, inf)`` so callers can
    filter them downstream.

    Args:
        points_cam:  ``(N, 3)`` array of camera-frame 3D points.
        intrinsics:  Camera intrinsics (may carry distortion).

    Returns:
        ``(uv, depths)`` where:
          - *uv* is ``(N, 2)`` float64 pixel coordinates
          - *depths* is ``(N,)`` float64 (``points_cam[:, 2]``)

    Raises:
        ValueError:  unknown ``distortion_model``.
    """
    depths = points_cam[:, 2].copy()

    if not intrinsics.is_distorted:
        uv = _project_pinhole_batch(points_cam, intrinsics)
    elif intrinsics.distortion_model == "OPENCV":
        uv = np.full((len(points_cam), 2), np.inf, dtype=np.float64)
        valid = depths > 0
        if valid.any():
            uv[valid] = _project_opencv_batch(points_cam[valid], intrinsics)
    elif intrinsics.distortion_model == "OPENCV_FISHEYE":
        uv = np.full((len(points_cam), 2), np.inf, dtype=np.float64)
        valid = depths > 0
        if valid.any():
            uv[valid] = _project_fisheye_batch(points_cam[valid], intrinsics)
    else:
        raise ValueError(
            f"Unsupported distortion model: {intrinsics.distortion_model!r}. "
            f"Supported: \"\" (pinhole), \"OPENCV\", \"OPENCV_FISHEYE\""
        )

    return uv, depths


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
