"""Tests for ScanNet++ native scene discovery and geometry loading."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Iterator

import numpy as np

from src.datasets.scannetpp import (
    has_scannetpp_dslr,
    has_scannetpp_geometry,
    has_scannetpp_iphone,
    is_scannetpp_scene_dir,
    load_scannetpp_dslr_camera,
    load_scannetpp_dslr_intrinsics,
    load_scannetpp_scene_geometry,
    project_scannetpp_dslr_point,
    resolve_scannetpp_scene_dirs,
)
from src.utils.colmap_loader import CameraPose
from src.scene_parser import parse_scene


def _make_tiny_ply(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    """Write a minimal ASCII PLY file with vertex and face elements."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for vertex in vertices:
        lines.append(f"{vertex[0]} {vertex[1]} {vertex[2]}")
    for face in faces:
        lines.append(f"3 {face[0]} {face[1]} {face[2]}")
    path.write_text("\n".join(lines), encoding="ascii")


class _FakeTriangleMesh:
    def __init__(
        self,
        vertices: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
    ) -> None:
        self.vertices = vertices
        self.triangles = faces


def _read_tiny_ascii_ply(path: str) -> _FakeTriangleMesh:
    """Read the limited ASCII PLY format written by _make_tiny_ply."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    vertex_count = 0
    face_count = 0
    header_end = 0
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.rsplit(" ", 1)[1])
        elif line.startswith("element face "):
            face_count = int(line.rsplit(" ", 1)[1])
        elif line == "end_header":
            header_end = index + 1
            break

    vertex_lines = lines[header_end:header_end + vertex_count]
    face_lines = lines[header_end + vertex_count:header_end + vertex_count + face_count]
    vertices = [tuple(float(part) for part in line.split()) for line in vertex_lines]
    faces = [tuple(int(part) for part in line.split()[1:4]) for line in face_lines]
    return _FakeTriangleMesh(vertices, faces)


@contextmanager
def _fake_open3d() -> Iterator[None]:
    """Install a tiny open3d stub so unit tests do not require the real package."""
    sentinel = object()
    previous = sys.modules.get("open3d", sentinel)
    sys.modules["open3d"] = types.SimpleNamespace(
        io=types.SimpleNamespace(read_triangle_mesh=_read_tiny_ascii_ply)
    )
    try:
        yield
    finally:
        if previous is sentinel:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = previous


def _make_tiny_scene(
    root: Path,
    *,
    include_dslr: bool = False,
    include_iphone: bool = False,
    bad_anno: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a minimal ScanNet++ scene under root."""
    scans = root / "scans"
    scans.mkdir(parents=True, exist_ok=True)

    vertices = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0),
        (2, 0, 0), (3, 0, 0), (2, 1, 0),
        (4, 0, 0), (5, 0, 0), (4, 1, 0),
    ]
    faces = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    _make_tiny_ply(scans / "mesh_aligned_0.05.ply", vertices, faces)

    seg_indices = [100, 100, 100, 200, 200, 200, 300, 300, 300]
    (scans / "segments.json").write_text(
        json.dumps({"segIndices": seg_indices}),
        encoding="utf-8",
    )

    if bad_anno:
        anno = {}
    else:
        anno = {
            "segGroups": [
                {"id": 1, "objectId": 1, "label": "table", "segments": [100]},
                {"id": 2, "objectId": 2, "label": "chair", "segments": [200]},
                {"id": 3, "objectId": 3, "label": "box", "segments": [300]},
            ]
        }
    (scans / "segments_anno.json").write_text(json.dumps(anno), encoding="utf-8")

    if include_dslr:
        dslr = root / "dslr" / "nerfstudio"
        dslr.mkdir(parents=True, exist_ok=True)
        (dslr / "transforms.json").write_text("{}", encoding="utf-8")
        (root / "dslr" / "resized_images").mkdir(parents=True, exist_ok=True)

    if include_iphone:
        iphone_colmap = root / "iphone" / "colmap"
        iphone_colmap.mkdir(parents=True, exist_ok=True)
        (root / "iphone" / "rgb.mkv").write_bytes(b"video")
        (iphone_colmap / "cameras.txt").write_text(
            "1 OPENCV 1920 1440 1435.54 1436.06 963.39 722.32 "
            "0.067 -0.081 -0.00047 0.00184\n",
            encoding="utf-8",
        )
        (iphone_colmap / "images.txt").write_text(
            "0 1 0 0 0 0 0 0 1 frame_000000.jpg\n\n",
            encoding="utf-8",
        )

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


class HasScanNetPPGeometryTests(unittest.TestCase):
    def test_true_for_complete_3d_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            self.assertTrue(has_scannetpp_geometry(root))

    def test_false_when_missing_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            (root / "scans" / "mesh_aligned_0.05.ply").unlink()
            self.assertFalse(has_scannetpp_geometry(root))

    def test_false_when_missing_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            (root / "scans" / "segments.json").unlink()
            self.assertFalse(has_scannetpp_geometry(root))

    def test_false_when_missing_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            (root / "scans" / "segments_anno.json").unlink()
            self.assertFalse(has_scannetpp_geometry(root))

    def test_true_without_dslr_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            self.assertTrue(has_scannetpp_geometry(root))


class IsScanNetPPSceneDirTests(unittest.TestCase):
    def test_false_without_dslr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            self.assertFalse(is_scannetpp_scene_dir(root))

    def test_true_with_dslr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root, include_dslr=True)
            self.assertTrue(is_scannetpp_scene_dir(root))

    def test_sensor_specific_predicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root, include_iphone=True)
            self.assertTrue(has_scannetpp_iphone(root))
            self.assertFalse(has_scannetpp_dslr(root))


class ResolveScanNetPPSceneDirsTests(unittest.TestCase):
    def test_single_scene_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root, include_iphone=True)
            self.assertEqual(resolve_scannetpp_scene_dirs(root), [root.resolve()])

    def test_parent_directory_with_multiple_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            for name in ("aaa", "bbb", "not_a_scene"):
                (parent / name).mkdir()
            _make_tiny_scene(parent / "aaa", include_iphone=True)
            _make_tiny_scene(parent / "bbb", include_iphone=True)

            result = resolve_scannetpp_scene_dirs(parent)
            self.assertEqual(result, sorted([(parent / "aaa").resolve(), (parent / "bbb").resolve()]))

    def test_parent_directory_filters_non_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / "readme.txt").touch()
            (parent / "empty_dir").mkdir()
            _make_tiny_scene(parent / "scene1", include_iphone=True)

            result = resolve_scannetpp_scene_dirs(parent)
            self.assertEqual(result, [(parent / "scene1").resolve()])

    def test_scenes_without_requested_sensor_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / "has_3d_only").mkdir()
            _make_tiny_scene(parent / "has_3d_only")
            self.assertEqual(resolve_scannetpp_scene_dirs(parent), [])

    def test_dslr_sensor_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            _make_tiny_scene(parent / "dslr_scene", include_dslr=True)
            _make_tiny_scene(parent / "iphone_scene", include_iphone=True)
            self.assertEqual(
                resolve_scannetpp_scene_dirs(parent, sensor="dslr"),
                [(parent / "dslr_scene").resolve()],
            )


class LoadScanNetPPGeometryTests(unittest.TestCase):
    def test_loads_vertices_and_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            with _fake_open3d():
                sid, vertices, faces, seg_indices, anno_list = load_scannetpp_scene_geometry(root)

            self.assertEqual(sid, root.name)
            self.assertEqual(len(vertices), 9)
            self.assertEqual(len(faces), 3)
            self.assertEqual(len(seg_indices), 9)
            self.assertEqual(len(anno_list), 3)

    def test_raises_on_seg_indices_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            (root / "scans" / "segments.json").write_text(
                json.dumps({"segIndices": [1, 2]}),
                encoding="utf-8",
            )

            with _fake_open3d():
                with self.assertRaises(ValueError):
                    load_scannetpp_scene_geometry(root)

    def test_raises_on_empty_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root, bad_anno=True)
            with _fake_open3d():
                with self.assertRaises(ValueError):
                    load_scannetpp_scene_geometry(root)

    def test_raises_on_missing_seggroups_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            (root / "scans" / "segments_anno.json").write_text(
                json.dumps({"wrongKey": []}),
                encoding="utf-8",
            )
            with _fake_open3d():
                with self.assertRaises(ValueError):
                    load_scannetpp_scene_geometry(root)


class ParseSceneScanNetPPTests(unittest.TestCase):
    def test_explicit_dataset_scannetpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            with _fake_open3d():
                scene = parse_scene(root, dataset="scannetpp")
            self.assertIsNotNone(scene)
            assert scene is not None
            self.assertEqual(scene["scene_id"], root.name)
            self.assertEqual(len(scene["objects"]), 3)

    def test_auto_detect_scannetpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tiny_scene(root)
            with _fake_open3d():
                scene = parse_scene(root)
            self.assertIsNotNone(scene)
            assert scene is not None
            self.assertEqual(scene["scene_id"], root.name)
            self.assertEqual(len(scene["objects"]), 3)


class ScanNetPPDSLRIntrinsicsTests(unittest.TestCase):
    def test_loads_fisheye_camera_and_legacy_pinhole_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dslr = root / "dslr" / "nerfstudio"
            dslr.mkdir(parents=True)
            (dslr / "transforms.json").write_text(
                json.dumps({
                    "fl_x": 10.0,
                    "fl_y": 11.0,
                    "cx": 5.0,
                    "cy": 6.0,
                    "w": 20,
                    "h": 30,
                    "k1": -0.1,
                    "k2": 0.01,
                    "k3": 0.001,
                    "k4": 0.0001,
                    "camera_model": "OPENCV_FISHEYE",
                    "frames": [],
                }),
                encoding="utf-8",
            )

            camera = load_scannetpp_dslr_camera(root)
            legacy = load_scannetpp_dslr_intrinsics(root)

            self.assertTrue(camera.is_fisheye)
            self.assertEqual(camera.width, 20)
            self.assertEqual(camera.height, 30)
            self.assertAlmostEqual(camera.k1, -0.1)
            self.assertEqual(legacy.width, camera.width)
            self.assertEqual(legacy.fx, camera.fx)

    def test_fisheye_projection_uses_distortion_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dslr = root / "dslr" / "nerfstudio"
            dslr.mkdir(parents=True)
            (dslr / "transforms.json").write_text(
                json.dumps({
                    "fl_x": 100.0,
                    "fl_y": 100.0,
                    "cx": 50.0,
                    "cy": 50.0,
                    "w": 100,
                    "h": 100,
                    "k1": -0.2,
                    "k2": 0.0,
                    "k3": 0.0,
                    "k4": 0.0,
                    "camera_model": "OPENCV_FISHEYE",
                    "frames": [],
                }),
                encoding="utf-8",
            )
            pose = CameraPose(
                image_name="frame.jpg",
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
            )
            camera = load_scannetpp_dslr_camera(root)

            uv, depth = project_scannetpp_dslr_point(
                np.array([1.0, 0.0, 1.0], dtype=np.float64),
                pose,
                camera,
            )

            self.assertAlmostEqual(depth, 1.0)
            assert uv is not None
            pinhole_u = 150.0
            self.assertLess(float(uv[0]), pinhole_u)


if __name__ == "__main__":
    unittest.main()
