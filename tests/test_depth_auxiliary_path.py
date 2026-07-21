from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
    DepthVisualRedundancyEvaluator,
    VisualRedundancyEvidence,
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


def make_yaw_pose(
    image_name: str,
    position: tuple[float, float, float],
    yaw_deg: float,
) -> CameraPose:
    angle = np.radians(yaw_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
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
        elif encoding == "deflate":
            raw = np.asarray(array, dtype=np.float32).tobytes()
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            compressed = compressor.compress(raw) + compressor.flush()
        elif encoding == "zlib":
            raw = np.asarray(array, dtype=np.uint16).tobytes()
            compressed = zlib.compress(raw)
        else:
            raise ValueError(f"unsupported test encoding: {encoding}")
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


@pytest.mark.parametrize("encoding", ["lz4", "deflate", "zlib"])
def test_scannetpp_depth_reader_random_access_and_units(
    tmp_path: Path, encoding: str
) -> None:
    depth_path = tmp_path / "depth.bin"
    metadata_path = tmp_path / "pose_intrinsic_imu.json"
    if encoding in {"lz4", "zlib"}:
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


def test_scannetpp_depth_reader_skips_corrupt_frame(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    depth_path = tmp_path / "depth.bin"
    metadata_path = tmp_path / "pose_intrinsic_imu.json"
    valid = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 1250, dtype=np.uint16)
    valid_payload = lz4.block.compress(valid.tobytes(), store_size=False)
    corrupt_payload = b"not a compressed depth frame"
    stream = bytearray()
    for payload in (corrupt_payload, valid_payload):
        stream.extend(len(payload).to_bytes(4, "little"))
        stream.extend(payload)
    depth_path.write_bytes(stream)
    write_metadata(metadata_path, 2)

    reader = ScanNetPPDepthReader(depth_path, metadata_path)

    assert reader.load("frame_000000.jpg") is None
    assert reader.load("frame_000000.jpg") is None
    assert "Skipping unreadable ScanNet++ depth frame 0" in caplog.text
    assert caplog.text.count("Skipping unreadable ScanNet++ depth frame 0") == 1
    valid_frame = reader.load("frame_000001.jpg")
    assert valid_frame is not None
    assert float(valid_frame.image_m[0, 0]) == pytest.approx(1.25)


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


def test_visual_redundancy_combines_translation_and_rotation() -> None:
    intrinsics = CameraIntrinsics(160, 120, 80.0, 80.0, 80.0, 60.0)
    source_name = "source.jpg"
    target_name = "target.jpg"
    poses = {
        source_name: make_yaw_pose(source_name, (0.0, 0.0, 0.0), 0.0),
        target_name: make_yaw_pose(
            target_name,
            (0.5, 0.0, 0.0),
            np.degrees(np.arctan2(0.5, 4.0)),
        ),
    }
    source_depth_frame = DepthFrame(
        np.full((120, 160), 4.0, dtype=np.float32),
        intrinsics,
        1.0,
        "synthetic",
    )
    target_pose = poses[target_name]
    target_u, target_v = np.meshgrid(np.arange(160), np.arange(120))
    target_rays = np.stack(
        (
            (target_u - intrinsics.cx) / intrinsics.fx,
            (target_v - intrinsics.cy) / intrinsics.fy,
            np.ones_like(target_u),
        ),
        axis=-1,
    )
    target_world_rays = target_rays @ target_pose.rotation
    target_depth_frame = DepthFrame(
        (4.0 / target_world_rays[..., 2]).astype(np.float32),
        intrinsics,
        1.0,
        "synthetic",
    )
    evaluator = DepthVisualRedundancyEvaluator(
        poses=poses,
        depth_frame_for=lambda name: (
            source_depth_frame if name == source_name else target_depth_frame
        ),
        rgb_evidence_for=lambda _left, _right: SimpleNamespace(
            passed=True,
            inliers=80,
            inlier_ratio=0.8,
            min_grid_fraction=0.5,
        ),
    )

    evidence = evaluator(source_name, target_name)

    assert evidence.available
    assert evidence.is_duplicate
    assert evidence.min_bidirectional_overlap >= 0.65
    assert evidence.max_p75_displacement_diagonal <= 0.12


def _run_visual_prune_route(
    monkeypatch,
    *,
    next_yaw_deg: float,
    positions: dict[str, tuple[float, float, float]] | None = None,
):
    names = ("main_a.jpg", "duplicate.jpg", "next.jpg", "main_b.jpg")
    positions = positions or {
        "main_a.jpg": (0.0, 0.0, 0.0),
        "duplicate.jpg": (0.25, 0.0, 0.0),
        "next.jpg": (0.5, 0.0, 0.0),
        "main_b.jpg": (0.75, 0.0, 0.0),
    }
    poses = {
        "main_a.jpg": make_yaw_pose("main_a.jpg", positions["main_a.jpg"], 0.0),
        "duplicate.jpg": make_yaw_pose(
            "duplicate.jpg", positions["duplicate.jpg"], 25.0
        ),
        "next.jpg": make_yaw_pose("next.jpg", positions["next.jpg"], next_yaw_deg),
        "main_b.jpg": make_yaw_pose(
            "main_b.jpg", positions["main_b.jpg"], next_yaw_deg
        ),
    }
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
        "duplicate.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool),
        "next.jpg": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.depth_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    monkeypatch.setattr(
        "src.depth_auxiliary_path._semantic_conflict",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "src.depth_auxiliary_path._depth_visibility",
        lambda _points, geometric, _pose, _depth: (geometric.copy(), geometric.copy()),
    )
    depth_frame = DepthFrame(
        np.ones((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32),
        CameraIntrinsics(DEPTH_WIDTH, DEPTH_HEIGHT, 190.0, 190.0, 127.5, 95.5),
        1.0,
        "test_depth",
    )
    duplicate = VisualRedundancyEvidence(
        True, True, 0.8, 0.05, 80, 0.8, 0.5, "duplicate"
    )
    distinct = VisualRedundancyEvidence(
        True, False, 0.4, 0.3, 10, 0.2, 0.1, "thresholds_not_met"
    )

    return find_depth_corridor_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        depth_frame_for=lambda name: depth_frame if name in names else None,
        group_a_objects=[
            {"bbox_min": [-0.05, -0.05, 1.95], "bbox_max": [0.05, 0.05, 2.05]}
        ],
        group_b_objects=[
            {"bbox_min": [0.85, -0.05, 1.95], "bbox_max": [0.95, 0.05, 2.05]}
        ],
        visual_redundancy_for=lambda left, right: (
            duplicate
            if {left, right} == {"main_a.jpg", "duplicate.jpg"}
            else distinct
        ),
        min_overlap_frac=0.10,
    )


@pytest.mark.parametrize("next_yaw_deg", [70.0, 80.0])
def test_visual_prune_relaxes_only_new_edge_through_80_degrees(
    monkeypatch, next_yaw_deg: float
) -> None:
    route = _run_visual_prune_route(monkeypatch, next_yaw_deg=next_yaw_deg)

    assert route is not None
    assert route.auxiliary_image_names == ("next.jpg",)
    assert route.visual_pruned_auxiliary_frame_count == 1
    assert route.visual_duplicate_candidate_count == 1
    assert route.visual_prune_relaxed_angle_edge_count == 1
    assert route.max_forward_angle_deg == pytest.approx(next_yaw_deg)


def test_visual_prune_rejects_new_edge_above_80_degrees(monkeypatch) -> None:
    route = _run_visual_prune_route(monkeypatch, next_yaw_deg=81.0)

    assert route is not None
    assert route.auxiliary_image_names == ("duplicate.jpg", "next.jpg")
    assert route.visual_pruned_auxiliary_frame_count == 0
    assert route.visual_duplicate_candidate_count == 1
    assert route.visual_prune_relaxed_angle_edge_count == 0


def test_visual_prune_allows_new_local_perpendicular_edge_through_080m(
    monkeypatch,
) -> None:
    route = _run_visual_prune_route(
        monkeypatch,
        next_yaw_deg=70.0,
        positions={
            "main_a.jpg": (0.0, 0.0, 0.0),
            "duplicate.jpg": (0.25, 0.3, 0.0),
            "next.jpg": (0.5, 0.78, 0.0),
            "main_b.jpg": (0.75, 0.78, 0.0),
        },
    )

    assert route is not None
    assert route.auxiliary_image_names == ("next.jpg",)
    assert route.visual_pruned_auxiliary_frame_count == 1
    assert route.max_local_perpendicular_m == pytest.approx(0.78)


def test_visual_prune_preserves_duplicate_when_geometric_hard_limit_fails(
    monkeypatch,
) -> None:
    route = _run_visual_prune_route(
        monkeypatch,
        next_yaw_deg=70.0,
        positions={
            "main_a.jpg": (0.0, 0.0, 0.0),
            "duplicate.jpg": (0.25, 0.4, 0.0),
            "next.jpg": (0.5, 1.0, 0.0),
            "main_b.jpg": (0.75, 1.0, 0.0),
        },
    )

    assert route is not None
    assert route.auxiliary_image_names == ("duplicate.jpg", "next.jpg")
    assert route.visual_pruned_auxiliary_frame_count == 0
