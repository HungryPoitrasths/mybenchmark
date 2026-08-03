from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.reroute_benchmark_auxiliary_frames as reroute_script
from src.datasets.base import DepthFrame
from src.depth_auxiliary_path import (
    _CorridorBasis,
    _edge_metrics,
    find_depth_corridor_auxiliary_route,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_pose(
    image_name: str,
    position: tuple[float, float, float],
    *,
    yaw_deg: float = 0.0,
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


@pytest.mark.parametrize(
    ("right", "basis", "next_frontier"),
    [
        (
            make_pose("right.jpg", (0.1, 0.0, 0.0), yaw_deg=90.0),
            _CorridorBasis(np.array([1.0, 0.0]), 0.0, 0.0),
            1.0,
        ),
        (
            make_pose("right.jpg", (0.1, 0.0, 0.6)),
            _CorridorBasis(np.array([1.0, 0.0]), 0.0, 0.0),
            1.0,
        ),
        (
            make_pose("right.jpg", (0.1, 0.8, 0.0)),
            _CorridorBasis(np.array([1.0, 0.0]), 0.0, 0.0),
            1.0,
        ),
        (
            make_pose("right.jpg", (0.1, 0.2, 0.0)),
            _CorridorBasis(np.array([1.0, 0.0]), 0.0, 3.0),
            1.0,
        ),
        (make_pose("right.jpg", (1.1, 0.0, 0.0)), _CorridorBasis(None, 0.0, 0.0), 1.0),
    ],
    ids=(
        "orientation",
        "height",
        "local_perpendicular",
        "global_perpendicular",
        "degenerate_xy",
    ),
)
def test_edge_metrics_can_disable_all_camera_motion_hard_limits(
    right: CameraPose,
    basis: _CorridorBasis,
    next_frontier: float,
) -> None:
    left = make_pose("left.jpg", (0.0, 0.0, 0.0))
    kwargs = {
        "next_frontier": next_frontier,
        "depth_visible_fraction": 1.0,
        "basis": basis,
        "orientation_threshold_deg": 60.0,
    }

    assert _edge_metrics(left, right, **kwargs) is None
    relaxed = _edge_metrics(
        left,
        right,
        **kwargs,
        enforce_camera_motion_hard_limits=False,
    )

    assert relaxed is not None
    assert np.isfinite(relaxed.cost)


def test_relaxed_edges_keep_camera_motion_soft_costs() -> None:
    left = make_pose("left.jpg", (0.0, 0.0, 0.0))
    basis = _CorridorBasis(np.array([1.0, 0.0]), 0.0, 0.0)
    smooth = _edge_metrics(
        left,
        make_pose("smooth.jpg", (0.1, 0.05, 0.0), yaw_deg=5.0),
        next_frontier=1.0,
        depth_visible_fraction=1.0,
        basis=basis,
        orientation_threshold_deg=60.0,
        enforce_camera_motion_hard_limits=False,
    )
    jump = _edge_metrics(
        left,
        make_pose("jump.jpg", (0.1, 1.2, 0.8), yaw_deg=100.0),
        next_frontier=1.0,
        depth_visible_fraction=1.0,
        basis=basis,
        orientation_threshold_deg=60.0,
        enforce_camera_motion_hard_limits=False,
    )

    assert smooth is not None
    assert jump is not None
    assert smooth.cost < jump.cost


def test_depth_route_threads_relaxed_policy_to_direct_edge(monkeypatch) -> None:
    poses = {
        "main_a.jpg": make_pose("main_a.jpg", (0.0, 0.0, 0.0)),
        "main_b.jpg": make_pose("main_b.jpg", (0.2, 0.0, 0.0), yaw_deg=90.0),
    }
    monkeypatch.setattr(
        "src.depth_auxiliary_path.route_visibility_mask",
        lambda _points, _pose, _intrinsics: np.ones(10, dtype=bool),
    )
    monkeypatch.setattr(
        "src.depth_auxiliary_path._depth_visibility",
        lambda _points, geometric, _pose, _depth: (
            geometric.copy(),
            geometric.copy(),
        ),
    )
    intrinsics = CameraIntrinsics(640, 480, 320.0, 320.0, 320.0, 240.0)
    depth_frame = DepthFrame(
        np.ones((480, 640), dtype=np.float32), intrinsics, 1.0, "test"
    )
    kwargs = {
        "center_a": np.array([0.0, 0.0, 2.0]),
        "center_b": np.array([0.9, 0.0, 2.0]),
        "frame_a_name": "main_a.jpg",
        "frame_b_name": "main_b.jpg",
        "poses": poses,
        "intrinsics": intrinsics,
        "depth_frame_for": lambda _name: depth_frame,
        "group_a_objects": [
            {"bbox_min": [-0.1, -0.1, 1.9], "bbox_max": [0.1, 0.1, 2.1]}
        ],
        "group_b_objects": [
            {"bbox_min": [0.8, -0.1, 1.9], "bbox_max": [1.0, 0.1, 2.1]}
        ],
        "max_auxiliary_frames": 0,
    }

    assert find_depth_corridor_auxiliary_route(**kwargs) is None
    route = find_depth_corridor_auxiliary_route(
        **kwargs, enforce_camera_motion_hard_limits=False
    )

    assert route is not None
    assert route.auxiliary_image_names == ()
    assert route.max_forward_angle_deg == pytest.approx(90.0)


def test_preferred_candidates_survive_pose_cap_and_restore_depth_route(
    monkeypatch,
) -> None:
    poses = {
        "main_a.jpg": make_pose("main_a.jpg", (0.0, 0.0, 0.0)),
        "main_b.jpg": make_pose("main_b.jpg", (0.2, 0.0, 0.0)),
        "ranked.jpg": make_pose("ranked.jpg", (0.1, 0.0, 0.0)),
        "preferred.jpg": make_pose("preferred.jpg", (0.1, 0.0, 0.0)),
    }

    def interval_mask(length: int, start: float, end: float) -> np.ndarray:
        samples = np.linspace(0.0, 1.0, length)
        return (samples >= start) & (samples <= end)

    def geometric_mask(points, pose, _intrinsics):
        intervals = {
            "main_a.jpg": (0.0, 0.45),
            "main_b.jpg": (0.65, 1.0),
            "ranked.jpg": (0.25, 1.0),
            "preferred.jpg": (0.25, 0.9),
        }
        return interval_mask(len(points), *intervals[pose.image_name])

    def depth_visibility(points, geometric, pose, _depth):
        visible = geometric.copy()
        if pose.image_name == "ranked.jpg":
            samples = np.linspace(0.0, 1.0, len(points))
            visible &= (samples <= 0.5) | (samples >= 0.65)
        return geometric.copy(), visible

    monkeypatch.setattr(
        "src.depth_auxiliary_path.route_visibility_mask", geometric_mask
    )
    monkeypatch.setattr("src.depth_auxiliary_path._depth_visibility", depth_visibility)
    monkeypatch.setattr(
        "src.depth_auxiliary_path._semantic_conflict",
        lambda *_args, **_kwargs: False,
    )
    intrinsics = CameraIntrinsics(640, 480, 320.0, 320.0, 320.0, 240.0)
    depth_frame = DepthFrame(
        np.ones((480, 640), dtype=np.float32), intrinsics, 1.0, "test"
    )
    kwargs = {
        "center_a": np.array([0.0, 0.0, 2.0]),
        "center_b": np.array([1.0, 0.0, 2.0]),
        "frame_a_name": "main_a.jpg",
        "frame_b_name": "main_b.jpg",
        "poses": poses,
        "intrinsics": intrinsics,
        "depth_frame_for": lambda _name: depth_frame,
        "group_a_objects": [
            {"bbox_min": [-0.1, -0.1, 1.9], "bbox_max": [0.1, 0.1, 2.1]}
        ],
        "group_b_objects": [
            {"bbox_min": [0.9, -0.1, 1.9], "bbox_max": [1.1, 0.1, 2.1]}
        ],
        "max_auxiliary_frames": 1,
        "max_candidate_poses": 1,
        "enforce_camera_motion_hard_limits": False,
    }

    assert find_depth_corridor_auxiliary_route(**kwargs) is None
    route = find_depth_corridor_auxiliary_route(
        **kwargs,
        preferred_candidate_names=("preferred.jpg",),
    )

    assert route is not None
    assert route.auxiliary_image_names == ("preferred.jpg",)


def fake_route(*names: str) -> SimpleNamespace:
    return SimpleNamespace(
        auxiliary_image_names=names,
        cost=3.5,
        edge_count=len(names) + 1,
        route_sample_count=10,
        frame_a_coverage_end=0.3,
        frame_b_coverage_start=0.7,
        auxiliary_responsibility_fraction=0.4,
        transition_overlap_fraction=0.2,
        search_method="dijkstra_depth_corridor",
        min_progress_fraction=0.1,
        min_depth_valid_fraction=0.8,
        min_depth_visible_fraction=0.7,
        max_local_perpendicular_m=2.0,
        max_global_perpendicular_m=3.0,
        max_height_change_m=1.0,
        max_parallel_change_m=0.5,
        max_forward_angle_deg=120.0,
        depth_sources=("test",),
        pre_prune_auxiliary_count=len(names),
        pruned_auxiliary_frame_count=0,
        visual_pruned_auxiliary_frame_count=0,
        visual_duplicate_candidate_count=0,
        visual_prune_relaxed_angle_edge_count=0,
        visual_redundancy_metric_version=1,
        semantic_rejected_frame_count=0,
    )


def test_reroute_payload_updates_only_auxiliary_fields_and_reuses_route(
    monkeypatch,
) -> None:
    original = {
        "name": "benchmark",
        "statistics": {"total": 3},
        "questions": [
            {
                "type": "occlusion",
                "scene_id": "scene0001_00",
                "image_name": "single.jpg",
            },
            {
                "question_uid": "q1",
                "type": "distance",
                "scene_id": "scene0001_00",
                "image_name": "first.jpg",
                "reasoning_frame_2": "last.jpg",
                "object_frame_groups": {"frame_1": [1], "frame_2": [2]},
                "auxiliary_image_names": ["old.jpg"],
                "auxiliary_route": {"cost": 99.0},
                "answer": "A",
            },
            {
                "question_uid": "q2",
                "type": "distance",
                "scene_id": "scene0001_00",
                "image_name": "first.jpg",
                "reasoning_frame_2": "last.jpg",
                "object_frame_groups": {"frame_1": [1], "frame_2": [2]},
                "auxiliary_image_names": ["old.jpg"],
                "answer": "B",
            },
        ],
    }
    data_source = SimpleNamespace(load_depth_frame=lambda _name: None)
    resources = reroute_script.SceneRoutingResources(
        data_source=data_source,
        objects_by_id={
            1: {"id": 1, "center": [0.0, 0.0, 0.0]},
            2: {"id": 2, "center": [1.0, 0.0, 0.0]},
        },
        poses={"first.jpg": object(), "last.jpg": object()},
        intrinsics=object(),
        geometry_cache=object(),
        visual_redundancy=object(),
    )
    calls = []

    def route_stub(**kwargs):
        calls.append(kwargs)
        return fake_route("new.jpg")

    monkeypatch.setattr(
        reroute_script, "find_depth_corridor_auxiliary_route", route_stub
    )
    config = reroute_script.RerouteConfig(None, None, None)

    output, report = reroute_script.reroute_payload(
        original,
        config,
        resource_loader=lambda _scene_id, _config: resources,
    )

    assert original["questions"][1]["auxiliary_image_names"] == ["old.jpg"]
    assert output["statistics"] == original["statistics"]
    assert output["questions"][0] == original["questions"][0]
    assert output["questions"][1]["image_name"] == "first.jpg"
    assert output["questions"][1]["reasoning_frame_2"] == "last.jpg"
    assert output["questions"][1]["question_uid"] == "q1"
    assert output["questions"][1]["answer"] == "A"
    assert output["questions"][1]["auxiliary_image_names"] == ["new.jpg"]
    assert (
        output["questions"][1]["auxiliary_route"]["camera_motion_hard_limits_enabled"]
        is False
    )
    assert len(calls) == 1
    assert calls[0]["enforce_camera_motion_hard_limits"] is False
    assert calls[0]["preferred_candidate_names"] == ("old.jpg",)
    assert report["policy"]["original_auxiliary_candidates_pinned"] is True
    assert report["summary"] == {
        "total_question_count": 3,
        "cross_frame_question_count": 2,
        "single_frame_question_count": 1,
        "reroute_succeeded_count": 2,
        "auxiliary_selection_changed_count": 2,
        "auxiliary_selection_unchanged_count": 0,
        "reroute_failed_count": 0,
        "route_cache_hit_count": 1,
    }


def test_reroute_payload_keeps_original_selection_when_scene_loading_fails() -> None:
    original = {
        "questions": [
            {
                "question_uid": "q1",
                "type": "distance",
                "scene_id": "scene0001_00",
                "image_name": "first.jpg",
                "reasoning_frame_2": "last.jpg",
                "object_frame_groups": {"frame_1": [1], "frame_2": [2]},
                "auxiliary_image_names": ["old.jpg"],
                "auxiliary_route": {"cost": 9.0},
            }
        ]
    }

    def fail_loader(_scene_id, _config):
        raise FileNotFoundError("missing scene")

    output, report = reroute_script.reroute_payload(
        original,
        reroute_script.RerouteConfig(None, None, None),
        resource_loader=fail_loader,
    )

    assert output == original
    assert report["summary"]["reroute_failed_count"] == 1
    assert report["failures"][0]["reason"].startswith("scene_resource_error")


def test_main_reads_utf8_bom_and_writes_default_outputs(tmp_path) -> None:
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        json.dumps(
            {
                "name": "test",
                "questions": [
                    {
                        "type": "occlusion",
                        "scene_id": "scene0001_00",
                        "image_name": "single.jpg",
                    }
                ],
            }
        ),
        encoding="utf-8-sig",
    )

    assert reroute_script.main(["--input", str(input_path)]) == 0

    output_path = tmp_path / "benchmark.relaxed_camera_edges.json"
    report_path = tmp_path / "benchmark.relaxed_camera_edges.report.json"
    assert json.loads(output_path.read_text(encoding="utf-8"))["name"] == "test"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["single_frame_question_count"] == 1
