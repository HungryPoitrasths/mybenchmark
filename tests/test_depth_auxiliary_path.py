from __future__ import annotations

import json
from pathlib import Path
import zlib

import lz4.block
import numpy as np
import pytest

from src.datasets.base import DepthFrame
from src.datasets.scannetpp_depth import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    ScanNetPPDepthReader,
)
from src.depth_auxiliary_path import (
    _depth_visibility,
    find_depth_corridor_auxiliary_route,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_pose(image_name: str, position: tuple[float, float, float]) -> CameraPose:
    rotation = np.eye(3, dtype=np.float64)
    position_array = np.asarray(position, dtype=np.float64)
    return CameraPose(
        image_name=image_name,
        rotation=rotation,
        translation=-rotation @ position_array,
    )


def make_intrinsics(width: int = 640, height: int = 480) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=320.0,
        fy=320.0,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def write_depth_stream(
    path: Path, arrays: list[np.ndarray], *, encoding: str = "lz4"
) -> None:
    payload = bytearray()
    for array in arrays:
        if encoding == "lz4":
            raw = np.asarray(array, dtype=np.uint16).tobytes()
            compressed = lz4.block.compress(raw, store_size=False)
        else:
            raw = np.asarray(array, dtype=np.float32).tobytes()
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            compressed = compressor.compress(raw) + compressor.flush()
        payload.extend(len(compressed).to_bytes(4, "little"))
        payload.extend(compressed)
    path.write_bytes(bytes(payload))


def write_metadata(path: Path, frame_count: int) -> None:
    metadata = {
        f"frame_{index:06d}": {
            "intrinsic": [
                [1440.0 + index, 0.0, 960.0],
                [0.0, 1440.0 + index, 720.0],
                [0.0, 0.0, 1.0],
            ]
        }
        for index in range(frame_count)
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize("encoding", ["lz4", "zlib"])
def test_scannetpp_depth_reader_random_access_and_units(
    tmp_path: Path, encoding: str
) -> None:
    depth_path = tmp_path / "depth.bin"
    metadata_path = tmp_path / "pose_intrinsic_imu.json"
    if encoding == "lz4":
        arrays = [
            np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 1000 + index * 500, dtype=np.uint16)
            for index in range(2)
        ]
    else:
        arrays = [
            np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 1.0 + index * 0.5, dtype=np.float32)
            for index in range(2)
        ]
    write_depth_stream(depth_path, arrays, encoding=encoding)
    write_metadata(metadata_path, 2)

    reader = ScanNetPPDepthReader(depth_path, metadata_path, cache_size=1)
    frame = reader.load("frame_000001.jpg")

    assert frame is not None
    assert reader.frame_count == 2
    assert frame.image_m.shape == (DEPTH_HEIGHT, DEPTH_WIDTH)
    assert frame.image_m.dtype == np.float32
    assert float(frame.image_m[0, 0]) == pytest.approx(1.5)
    assert frame.valid_ratio == pytest.approx(1.0)
    assert frame.intrinsics.width == DEPTH_WIDTH
    assert frame.intrinsics.height == DEPTH_HEIGHT
    assert frame.intrinsics.fx == pytest.approx((1441.0 * DEPTH_WIDTH) / 1920.0)
    assert frame.intrinsics.cx == pytest.approx(128.0)


def test_depth_visibility_rejects_point_behind_wall() -> None:
    intrinsics = CameraIntrinsics(
        width=DEPTH_WIDTH,
        height=DEPTH_HEIGHT,
        fx=190.0,
        fy=190.0,
        cx=127.5,
        cy=95.5,
    )
    pose = make_pose("frame.jpg", (0.0, 0.0, 0.0))
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    depth = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 1.5, dtype=np.float32)
    frame = DepthFrame(depth, intrinsics, 1.0, "test")

    valid, visible = _depth_visibility(points, np.ones(2, dtype=bool), pose, frame)

    assert valid.tolist() == [True, True]
    assert visible.tolist() == [True, False]


def test_depth_corridor_route_uses_continuous_visible_bridges(monkeypatch) -> None:
    names = ["main_a.jpg", "far.jpg", "bridge_1.jpg", "bridge_2.jpg", "main_b.jpg"]
    poses = {
        "main_a.jpg": make_pose("main_a.jpg", (0.0, 0.0, 0.0)),
        "far.jpg": make_pose("far.jpg", (0.5, 1.2, 0.0)),
        "bridge_1.jpg": make_pose("bridge_1.jpg", (0.3, 0.05, 0.0)),
        "bridge_2.jpg": make_pose("bridge_2.jpg", (0.6, 0.05, 0.0)),
        "main_b.jpg": make_pose("main_b.jpg", (0.9, 0.0, 0.0)),
    }
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
        "far.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
        "bridge_1.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool),
        "bridge_2.jpg": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.depth_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    monkeypatch.setattr(
        "src.depth_auxiliary_path._semantic_conflict",
        lambda *_args, **_kwargs: False,
    )

    def fake_depth_visibility(_points, geometric, pose, _depth_frame):
        valid = geometric.copy()
        visible = geometric.copy()
        if pose.image_name == "far.jpg":
            visible[:] = False
        return valid, visible

    monkeypatch.setattr(
        "src.depth_auxiliary_path._depth_visibility", fake_depth_visibility
    )
    depth_frame = DepthFrame(
        np.ones((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32),
        CameraIntrinsics(DEPTH_WIDTH, DEPTH_HEIGHT, 190.0, 190.0, 127.5, 95.5),
        1.0,
        "test_depth",
    )
    group_a = [{"bbox_min": [-0.05, -0.05, 1.95], "bbox_max": [0.05, 0.05, 2.05]}]
    group_b = [{"bbox_min": [0.85, -0.05, 1.95], "bbox_max": [0.95, 0.05, 2.05]}]

    route = find_depth_corridor_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        depth_frame_for=lambda name: depth_frame if name in names else None,
        group_a_objects=group_a,
        group_b_objects=group_b,
        min_overlap_frac=0.10,
    )

    assert route is not None
    assert route.auxiliary_image_names == ("bridge_1.jpg", "bridge_2.jpg")
    assert "far.jpg" not in route.auxiliary_image_names
    assert route.max_local_perpendicular_m < 0.1
    assert route.depth_sources == ("test_depth",)


def test_depth_corridor_route_rejects_missing_main_depth(monkeypatch) -> None:
    poses = {
        "main_a.jpg": make_pose("main_a.jpg", (0.0, 0.0, 0.0)),
        "main_b.jpg": make_pose("main_b.jpg", (0.5, 0.0, 0.0)),
    }
    monkeypatch.setattr(
        "src.depth_auxiliary_path.route_visibility_mask",
        lambda _points, _pose, _intrinsics: np.ones(10, dtype=bool),
    )

    route = find_depth_corridor_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        depth_frame_for=lambda _name: None,
        group_a_objects=[{"bbox_min": [-0.1, -0.1, 1.9], "bbox_max": [0.1, 0.1, 2.1]}],
        group_b_objects=[{"bbox_min": [0.8, -0.1, 1.9], "bbox_max": [1.0, 0.1, 2.1]}],
    )

    assert route is None
