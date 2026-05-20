"""Tests for distortion-aware projection (板块 8)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_intrinsics(
    model="",
    params=None,
    w=1920, h=1440,
    fx=1435.0, fy=1436.0, cx=963.0, cy=722.0,
):
    from src.utils.colmap_loader import CameraIntrinsics

    return CameraIntrinsics(
        width=w, height=h,
        fx=fx, fy=fy, cx=cx, cy=cy,
        distortion_model=model,
        distortion_params=(
            np.array(params, dtype=np.float64) if params else None
        ),
    )


# ---------------------------------------------------------------------------
# Pinhole
# ---------------------------------------------------------------------------

class TestPinholeProjection:
    def test_center_point(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics()
        pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        uv, depths = project_camera_points_to_image(pts, intr)
        assert depths[0] == 2.0
        np.testing.assert_allclose(uv[0], [intr.cx, intr.cy], rtol=1e-6)

    def test_behind_camera_returns_inf(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics()
        pts = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
        uv, depths = project_camera_points_to_image(pts, intr)
        assert depths[0] == -1.0
        assert np.isinf(uv[0, 0])
        assert np.isinf(uv[0, 1])

    def test_batch_projection(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics()
        pts = np.array([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, -0.5],
        ], dtype=np.float64)
        uv, depths = project_camera_points_to_image(pts, intr)
        assert uv.shape == (4, 2)
        # Behind-camera point gets inf
        assert np.isinf(uv[3]).all()
        # Front points have finite uv
        assert np.isfinite(uv[:3]).all()
        # First point at optical center
        np.testing.assert_allclose(uv[0], [intr.cx, intr.cy], rtol=1e-6)


# ---------------------------------------------------------------------------
# OPENCV
# ---------------------------------------------------------------------------

class TestOPENCVProjection:
    def test_differs_from_pinhole_off_center(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr_pin = _make_intrinsics()
        intr_cv = _make_intrinsics(
            model="OPENCV", params=[0.067, -0.081, -0.00047, 0.00184],
        )
        # Point near top-left corner
        pts = np.array([[-0.8, -0.6, 2.0]], dtype=np.float64)
        uv_pin, _ = project_camera_points_to_image(pts, intr_pin)
        uv_cv, _ = project_camera_points_to_image(pts, intr_cv)

        # OPENCV model should differ from pinhole off-center
        delta = np.linalg.norm(uv_pin[0] - uv_cv[0])
        assert delta > 0.1, f"Expected OPENCV to differ from pinhole, got delta={delta:.4f}"

    def test_near_center_similar(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr_pin = _make_intrinsics()
        intr_cv = _make_intrinsics(
            model="OPENCV", params=[0.1, -0.1, 0.01, 0.01],
        )
        pts = np.array([[0.01, 0.01, 3.0]], dtype=np.float64)
        uv_pin, _ = project_camera_points_to_image(pts, intr_pin)
        uv_cv, _ = project_camera_points_to_image(pts, intr_cv)

        delta = np.linalg.norm(uv_pin[0] - uv_cv[0])
        assert delta < 5.0, f"Near-center should be similar, got delta={delta:.4f}"

    def test_behind_camera_returns_inf(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics(
            model="OPENCV", params=[0.067, -0.081, -0.00047, 0.00184],
        )
        pts = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
        uv, depths = project_camera_points_to_image(pts, intr)
        assert depths[0] == -1.0
        assert np.isinf(uv[0, 0])


# ---------------------------------------------------------------------------
# OPENCV_FISHEYE
# ---------------------------------------------------------------------------

class TestOPENCVFisheyeProjection:
    def test_center_unchanged(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics(
            w=1752, h=1168,
            fx=790.85, fy=794.94, cx=870.79, cy=583.85,
            model="OPENCV_FISHEYE",
            params=[-0.0313, -0.0037, -0.0024, -7.6e-7],
        )
        pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        uv, _ = project_camera_points_to_image(pts, intr)
        # At optical center, fisheye ≈ pinhole
        np.testing.assert_allclose(uv[0], [intr.cx, intr.cy], rtol=0.01)

    def test_edge_pulls_inward(self):
        from src.utils.coordinate_transform import (
            project_camera_points_to_image,
            _project_pinhole_batch,
        )

        intr = _make_intrinsics(
            w=1752, h=1168,
            fx=790.85, fy=794.94, cx=870.79, cy=583.85,
            model="OPENCV_FISHEYE",
            params=[-0.0313, -0.0037, -0.0024, -7.6e-7],
        )
        # Far edge point
        pts = np.array([[-1.2, 0.0, 0.8]], dtype=np.float64)
        uv_fisheye, _ = project_camera_points_to_image(pts, intr)
        uv_pinhole = _project_pinhole_batch(pts, intr)

        # Fisheye should pull edge points toward center (barrel distortion)
        d_fish = np.linalg.norm(uv_fisheye[0] - np.array([intr.cx, intr.cy]))
        d_pin = np.linalg.norm(uv_pinhole[0] - np.array([intr.cx, intr.cy]))
        assert d_fish < d_pin, f"Fisheye should pull inward: d_fish={d_fish:.1f} d_pin={d_pin:.1f}"

    def test_behind_camera_returns_inf(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics(
            model="OPENCV_FISHEYE",
            params=[-0.0313, -0.0037, -0.0024, -7.6e-7],
        )
        pts = np.array([[0.0, 0.0, -2.0]], dtype=np.float64)
        uv, depths = project_camera_points_to_image(pts, intr)
        assert np.isinf(uv[0, 0])


# ---------------------------------------------------------------------------
# Unknown model rejection
# ---------------------------------------------------------------------------

class TestUnknownModelRejection:
    def test_unknown_model_raises(self):
        from src.utils.coordinate_transform import project_camera_points_to_image

        intr = _make_intrinsics(
            model="RADIAL",
            params=[0.1],
        )
        pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="Unsupported distortion model"):
            project_camera_points_to_image(pts, intr)


# ---------------------------------------------------------------------------
# project_to_image() still works single-point
# ---------------------------------------------------------------------------

class TestProjectToImageLegacy:
    def test_handles_distorted_intrinsics(self):
        from src.utils.colmap_loader import CameraPose
        from src.utils.coordinate_transform import project_to_image

        intr = _make_intrinsics(
            model="OPENCV",
            params=[0.067, -0.081, -0.00047, 0.00184],
        )
        pose = CameraPose(
            image_name="test.jpg",
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([0, 0, 0], dtype=np.float64),
        )
        uv, depth = project_to_image(
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
            pose, intr,
        )
        assert depth > 0
        assert uv is not None

    def test_behind_camera_returns_none(self):
        from src.utils.colmap_loader import CameraPose
        from src.utils.coordinate_transform import project_to_image

        intr = _make_intrinsics()
        pose = CameraPose(
            image_name="test.jpg",
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([0, 0, 0], dtype=np.float64),
        )
        uv, depth = project_to_image(
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            pose, intr,
        )
        assert depth == -1.0
        assert uv is None
