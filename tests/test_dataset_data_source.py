"""Tests for the DataSource abstraction layer (板块 4)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def _write_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    import cv2

    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    assert ok
    path.write_bytes(encoded.tobytes())


# ---------------------------------------------------------------------------
# ScanNetDataSource
# ---------------------------------------------------------------------------

class TestScanNetDataSource:
    def test_scene_id_from_dir_name(self, tmp_path: Path):
        from src.datasets.scannet import ScanNetDataSource

        (tmp_path / "scans" / "scene0000_00").mkdir(parents=True)
        ds = ScanNetDataSource(tmp_path / "scans" / "scene0000_00")
        assert ds.scene_id == "scene0000_00"
        assert ds.scene_dir == tmp_path / "scans" / "scene0000_00"

    def test_image_path_appends_color_dir(self):
        from src.datasets.scannet import ScanNetDataSource

        ds = ScanNetDataSource(Path("/fake/scene0000_00"))
        p = ds.image_path("42.jpg")
        assert p == Path("/fake/scene0000_00/color/42.jpg")


# ---------------------------------------------------------------------------
# ScanNetPPDataSource (iPhone / DSLR)
# ---------------------------------------------------------------------------

class TestScanNetPPDataSourceDefaults:
    def test_default_sensor_is_iphone(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(tmp_path)
        assert ds.sensor == "iphone"

    def test_default_frame_root(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(tmp_path / "0d2ee665be")
        assert ds.frame_root == Path("output") / "scannetpp_iphone_frames"

    def test_explicit_frame_root(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        custom = tmp_path / "my_frames"
        ds = ScanNetPPDataSource(
            tmp_path / "0d2ee665be", frame_root=custom
        )
        assert ds.frame_root == custom

    def test_rejects_unknown_sensor(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        with pytest.raises(ValueError, match="sensor must be"):
            ScanNetPPDataSource(tmp_path, sensor="lidar")


class TestScanNetPPDataSourceiPhoneImagePath:
    def test_image_path_appends_scene_and_frame(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        scene_dir = tmp_path / "0d2ee665be"
        ds = ScanNetPPDataSource(scene_dir, sensor="iphone")
        p = ds.image_path("frame_000010.jpg")
        expected = (
            Path("output")
            / "scannetpp_iphone_frames"
            / "0d2ee665be"
            / "frame_000010.jpg"
        )
        assert p == expected

    def test_image_path_with_custom_frame_root(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        custom = tmp_path / "custom_frames"
        ds = ScanNetPPDataSource(
            tmp_path / "0d2ee665be", sensor="iphone", frame_root=custom
        )
        p = ds.image_path("frame_000010.jpg")
        assert p == custom / "0d2ee665be" / "frame_000010.jpg"


class TestScanNetPPDataSourceDSLR:
    def test_dslr_sensor(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(tmp_path / "0d2ee665be", sensor="dslr")
        assert ds.sensor == "dslr"

    def test_dslr_image_path(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(tmp_path / "0d2ee665be", sensor="dslr")
        p = ds.image_path("DSC00001.JPG")
        assert p == tmp_path / "0d2ee665be" / "dslr" / "resized_images" / "DSC00001.JPG"


class TestScanNetPPDataSourceValidate:
    def _make_minimal_scene(
        self, root: Path, sensor: str = "iphone"
    ) -> Path:
        scene = root / "0d2ee665be"
        (scene / "scans").mkdir(parents=True)

        if sensor == "iphone":
            (scene / "iphone" / "colmap").mkdir(parents=True)

            # cameras.txt
            (scene / "iphone" / "colmap" / "cameras.txt").write_text(
                "# Camera list\n"
                "1 OPENCV 1920 1440 1435.54 1436.06 963.39 722.32 "
                "0.067 -0.081 -0.00047 0.00184\n"
            )

            # images.txt
            (scene / "iphone" / "colmap" / "images.txt").write_text(
                "# Image list\n"
                "0 0.379 0.852 0.329 -0.148 -6.190 -0.430 2.694 1 frame_000000.jpg\n"
                "\n"
                "1 0.380 0.852 0.327 -0.150 -6.192 -0.404 2.704 1 frame_000010.jpg\n"
                "\n"
            )

            # rgb.mkv
            (scene / "iphone" / "rgb.mkv").write_bytes(b"\x00\x00")

        else:
            (scene / "dslr" / "nerfstudio").mkdir(parents=True)
            (scene / "dslr" / "resized_images").mkdir(parents=True)

            transforms = {
                "fl_x": 790.85, "fl_y": 794.94,
                "cx": 870.79, "cy": 583.85,
                "w": 1752, "h": 1168,
                "camera_model": "OPENCV_FISHEYE",
                "k1": -0.0313, "k2": -0.00367,
                "k3": -0.00241, "k4": -7.64e-07,
                "frames": [
                    {
                        "file_path": "DSC00001.JPG",
                        "is_bad": False,
                        "transform_matrix": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ],
            }
            (scene / "dslr" / "nerfstudio" / "transforms.json").write_text(
                json.dumps(transforms)
            )

        return scene

    def test_validate_reports_missing_frames(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        scene = self._make_minimal_scene(tmp_path)
        # Use a frame_root that does NOT exist to guarantee "missing"
        ds = ScanNetPPDataSource(
            scene, sensor="iphone",
            frame_root=tmp_path / "nonexistent_frames",
        )
        result = ds.validate()

        assert result["dataset"] == "scannetpp"
        assert result["sensor"] == "iphone"
        assert result["poses"] == 2
        # Frames were never extracted — expect missing
        assert result["missing_images"] == 2
        assert any("missing extracted iPhone frames" in i for i in result["issues"])

    def test_validate_detects_extracted_frames(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        scene = self._make_minimal_scene(tmp_path)

        # "Extract" frames as empty jpegs
        frame_dir = tmp_path / "frames" / "0d2ee665be"
        frame_dir.mkdir(parents=True)
        dummy = np.zeros((1440, 1920, 3), dtype=np.uint8)
        for name in ["frame_000000.jpg", "frame_000010.jpg"]:
            _write_image_unicode_safe(frame_dir / name, dummy)

        ds = ScanNetPPDataSource(
            scene, sensor="iphone", frame_root=tmp_path / "frames"
        )
        result = ds.validate()

        assert result["missing_images"] == 0
        assert result["unreadable_images"] == 0

    def test_validate_dslr_mode(self, tmp_path: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        scene = self._make_minimal_scene(tmp_path, sensor="dslr")
        # Create a dummy JPEG so the pose passes image-exists check
        dummy = np.zeros((1168, 1752, 3), dtype=np.uint8)
        _write_image_unicode_safe(
            scene / "dslr" / "resized_images" / "DSC00001.JPG",
            dummy,
        )

        ds = ScanNetPPDataSource(scene, sensor="dslr")
        result = ds.validate()

        assert result["dataset"] == "scannetpp"
        assert result["sensor"] == "dslr"
        assert result["poses"] == 1


# ---------------------------------------------------------------------------
# Loaders work with real data (needs ++data/0d2ee665be)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestScanNetPPDataSourceRealData:
    @pytest.fixture(scope="class")
    def scene_dir(self) -> Path:
        p = Path(__file__).resolve().parent.parent / "++data" / "0d2ee665be"
        if not p.is_dir():
            pytest.skip("++data/0d2ee665be not available")
        return p

    def test_load_intrinsics_iphone(self, scene_dir: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(scene_dir, sensor="iphone")
        intr = ds.load_intrinsics()

        assert intr.width == 1920
        assert intr.height == 1440
        assert abs(intr.fx - 1435.54) < 0.1
        assert intr.distortion_model == "OPENCV"
        np.testing.assert_allclose(
            intr.distortion_params,
            [0.067, -0.081, -0.00047, 0.00184],
            rtol=1e-2,
            atol=1e-4,
        )

    def test_load_poses_iphone(self, scene_dir: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(scene_dir, sensor="iphone")
        poses = ds.load_poses()
        assert len(poses) == 418
        assert "frame_000010.jpg" in poses

    def test_validate_iphone_reports_when_no_frames(self, scene_dir: Path):
        from src.datasets.scannetpp import ScanNetPPDataSource

        ds = ScanNetPPDataSource(scene_dir, sensor="iphone")
        result = ds.validate()

        assert result["intrinsics"] == "1920x1440"
        assert result["poses"] == 418
        # Missing frames are expected unless extract script was run
        # We just check the key exists
        assert "missing_images" in result
