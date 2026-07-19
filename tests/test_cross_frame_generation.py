from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.auxiliary_path import AuxiliaryRoute, VisualPoseEdge, VisualPoseGraph
from src.legacy_auxiliary_path import (
    _prune_auxiliary_names,
    find_geometric_auxiliary_route,
    object_group_center,
)
from src.qa_generator import (
    ReasoningFrameContext,
    _annotate_cross_frame_questions,
)
from src.referability_checks import _question_referability_role_ids
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=320, height=240, fx=200.0, fy=200.0, cx=160.0, cy=120.0)


def make_pose(name: str, x: float = 0.0) -> CameraPose:
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array([-x, 0.0, 0.0], dtype=np.float64)
    return CameraPose(image_name=name, rotation=rotation, translation=translation)


def make_object(obj_id: int, label: str) -> dict:
    return {
        "id": obj_id,
        "label": label,
        "center": [float(obj_id), 0.0, 0.5],
        "bbox_min": [float(obj_id) - 0.1, -0.1, 0.0],
        "bbox_max": [float(obj_id) + 0.1, 0.1, 1.0],
    }


def make_context(
    name: str,
    *,
    regular: set[int],
    attachment: set[int] | None = None,
) -> ReasoningFrameContext:
    return ReasoningFrameContext(
        image_name=name,
        camera_pose=make_pose(name),
        regular_referable_ids=frozenset(regular),
        attachment_referable_ids=frozenset(attachment or set()),
        cache_entry={"frame_usable": True},
    )


def test_occlusion_binds_movement_to_frame_1_and_visibility_to_frame_2() -> None:
    objects = {1: make_object(1, "table"), 2: make_object(2, "lamp")}
    question = {
        "type": "object_move_occlusion",
        "question": "question",
        "moved_obj_id": 1,
        "target_obj_id": 2,
    }
    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context("first.jpg", regular=set(), attachment={1}),
        frame_2=make_context("last.jpg", regular={2}),
        objects_by_id=objects,
    )
    assert len(result) == 1
    assert result[0]["camera_bindings"] == {
        "movement": "frame_1",
        "visibility": "frame_2",
    }
    assert result[0]["object_frame_groups"] == {"frame_1": [1], "frame_2": [2]}


def test_v2_occlusion_binds_query_and_reference_to_distinct_roles() -> None:
    objects = {
        1: make_object(1, "table"),
        2: make_object(2, "lamp"),
        3: make_object(3, "sofa"),
    }
    question = {
        "type": "object_move_occlusion",
        "occlusion_semantics_version": 2,
        "question": "question",
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "obj_ref_id": 3,
    }
    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context("first.jpg", regular=set(), attachment={1, 2}),
        frame_2=make_context("last.jpg", regular={3}),
        objects_by_id=objects,
    )
    assert len(result) == 1
    assert result[0]["camera_bindings"] == {
        "movement": "frame_1",
        "visibility": "frame_1",
    }
    assert result[0]["object_frame_groups"] == {"frame_1": [1, 2], "frame_2": [3]}


def test_v2_occlusion_referability_roles_keep_reference_ordinary() -> None:
    attachment_ids, ordinary_ids = _question_referability_role_ids(
        {
            "type": "object_move_occlusion",
            "attachment_remapped": True,
            "moved_obj_id": 1,
            "query_obj_id": 2,
            "target_obj_id": 2,
            "obj_ref_id": 3,
        }
    )

    assert attachment_ids == {1, 2}
    assert ordinary_ids == {3}


def test_v2_cross_frame_generation_uses_frame_1_camera_for_occlusion() -> None:
    objects = {
        1: make_object(1, "table"),
        2: make_object(2, "lamp"),
        3: make_object(3, "sofa"),
    }
    frame_1 = make_context("first.jpg", regular=set(), attachment={1, 2})
    frame_2_a = make_context("last-a.jpg", regular={3})
    frame_2_b = make_context("last-b.jpg", regular={3})
    generated = {
        "type": "object_move_occlusion",
        "occlusion_semantics_version": 2,
        "question": "question",
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "obj_ref_id": 3,
    }
    with patch("src.qa_generator.generate_l2_object_move", return_value=[dict(generated)]) as generate_mock:
        from src.qa_generator import generate_cross_frame_questions

        result_a = generate_cross_frame_questions(
            objects=list(objects.values()),
            attachment_graph={1: [2]},
            attached_by={2: 1},
            frame_1=frame_1,
            frame_2=frame_2_a,
            color_intrinsics=make_intrinsics(),
            only_question_types=["L2_object_move_occlusion"],
        )
        first_camera = generate_mock.call_args.args[3]
        result_b = generate_cross_frame_questions(
            objects=list(objects.values()),
            attachment_graph={1: [2]},
            attached_by={2: 1},
            frame_1=frame_1,
            frame_2=frame_2_b,
            color_intrinsics=make_intrinsics(),
            only_question_types=["L2_object_move_occlusion"],
        )
        second_camera = generate_mock.call_args.args[3]

    assert first_camera is frame_1.camera_pose
    assert second_camera is frame_1.camera_pose
    assert result_a[0]["camera_bindings"] == result_b[0]["camera_bindings"]


def test_generic_frame_2_role_requires_regular_referability() -> None:
    objects = {1: make_object(1, "table"), 2: make_object(2, "lamp")}
    question = {
        "type": "object_move_occlusion",
        "question": "question",
        "moved_obj_id": 1,
        "target_obj_id": 2,
    }
    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context("first.jpg", regular=set(), attachment={1}),
        frame_2=make_context("last.jpg", regular=set(), attachment={2}),
        objects_by_id=objects,
    )
    assert result == []


def test_opposite_frame_union_signal_rejects_semantic_overlap() -> None:
    objects = {1: make_object(1, "table"), 2: make_object(2, "lamp")}
    question = {
        "type": "object_move_occlusion",
        "question": "question",
        "moved_obj_id": 1,
        "target_obj_id": 2,
    }
    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context("first.jpg", regular=set(), attachment={1}),
        frame_2=make_context("last.jpg", regular={2}, attachment={1}),
        objects_by_id=objects,
    )
    assert result == []


def test_object_rotation_supports_both_ref_face_layouts() -> None:
    objects = {index: make_object(index, f"object-{index}") for index in range(1, 5)}
    base = {
        "type": "object_rotate_object_centric",
        "question": "question",
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "obj_ref_id": 3,
        "obj_face_id": 4,
    }
    ref_first = _annotate_cross_frame_questions(
        [dict(base)],
        frame_1=make_context("first.jpg", regular={2, 3}, attachment={1}),
        frame_2=make_context("last.jpg", regular={4}),
        objects_by_id=objects,
        layout_id="ref_in_frame_1",
    )
    face_first = _annotate_cross_frame_questions(
        [dict(base)],
        frame_1=make_context("first.jpg", regular={2, 4}, attachment={1}),
        frame_2=make_context("last.jpg", regular={3}),
        objects_by_id=objects,
        layout_id="face_in_frame_1",
    )
    assert ref_first[0]["object_frame_groups"] == {"frame_1": [1, 2, 3], "frame_2": [4]}
    assert face_first[0]["object_frame_groups"] == {"frame_1": [1, 2, 4], "frame_2": [3]}


def test_visual_pose_route_respects_auxiliary_limit_and_direction() -> None:
    names = [f"frame_{index:06d}.jpg" for index in range(4)]
    graph = VisualPoseGraph(
        poses={name: make_pose(name, float(index)) for index, name in enumerate(names)},
        image_path_for=lambda name: Path(name),
    )
    graph.edges = {
        names[0]: [VisualPoseEdge(names[1], 1.0, 0.0, 40, 0.5, 1.0)],
        names[1]: [
            VisualPoseEdge(names[0], 1.0, 0.0, 40, 0.5, 1.0),
            VisualPoseEdge(names[2], 1.0, 0.0, 40, 0.5, 1.0),
        ],
        names[2]: [
            VisualPoseEdge(names[1], 1.0, 0.0, 40, 0.5, 1.0),
            VisualPoseEdge(names[3], 1.0, 0.0, 40, 0.5, 1.0),
        ],
        names[3]: [VisualPoseEdge(names[2], 1.0, 0.0, 40, 0.5, 1.0)],
    }
    route = graph.find_route(names[3], names[0], max_auxiliary_frames=2)
    assert isinstance(route, AuxiliaryRoute)
    assert route.auxiliary_image_names == (names[2], names[1])
    assert graph.find_route(names[3], names[0], max_auxiliary_frames=1) is None


def test_visual_pose_graph_build_applies_pose_and_overlap_thresholds(monkeypatch) -> None:
    names = [f"frame_{index:06d}.jpg" for index in range(3)]
    graph = VisualPoseGraph(
        poses={
            names[0]: make_pose(names[0], 0.0),
            names[1]: make_pose(names[1], 0.9),
            names[2]: make_pose(names[2], 2.0),
        },
        image_path_for=lambda name: Path(name),
    )
    monkeypatch.setattr(
        graph,
        "_load_gray_and_quality",
        lambda _name: (np.zeros((16, 16), dtype=np.uint8), 100.0, 20.0),
    )

    def overlap(left: str, right: str) -> tuple[int, float, int]:
        if {left, right} == {names[0], names[1]}:
            return 40, 0.5, 4
        return 23, 0.5, 4

    monkeypatch.setattr(graph, "_visual_overlap", overlap)
    graph.build()

    assert [edge.target for edge in graph.edges[names[0]]] == [names[1]]
    assert [edge.target for edge in graph.edges[names[1]]] == [names[0]]
    assert names[2] not in graph.edges
    assert graph.diagnostics()["graph_edge_count"] == 1
    assert graph.rejected_edge_counts["translation"] >= 1


def test_visual_pose_graph_cache_round_trip_and_image_invalidation(tmp_path, monkeypatch) -> None:
    names = ["frame_000000.jpg", "frame_000001.jpg"]
    for name in names:
        (tmp_path / name).write_bytes(b"initial-image-state")
    poses = {name: make_pose(name, index * 0.2) for index, name in enumerate(names)}
    graph = VisualPoseGraph(
        poses=poses,
        image_path_for=lambda name: tmp_path / name,
        flash_frame_names=set(names),
    )
    monkeypatch.setattr(
        graph,
        "_load_gray_and_quality",
        lambda _name: (np.zeros((16, 16), dtype=np.uint8), 100.0, 20.0),
    )
    monkeypatch.setattr(graph, "_visual_overlap", lambda _left, _right: (40, 0.5, 4))
    graph.build()
    cache_path = tmp_path / "graph.json"
    graph.save_cache(cache_path)

    restored = VisualPoseGraph(
        poses=poses,
        image_path_for=lambda name: tmp_path / name,
        flash_frame_names=set(names),
    )
    assert restored.load_cache(cache_path)
    assert restored.diagnostics() == graph.diagnostics()
    assert restored.find_route(names[0], names[1]) is not None

    (tmp_path / names[1]).write_bytes(b"changed-image-state-with-different-size")
    invalidated = VisualPoseGraph(
        poses=poses,
        image_path_for=lambda name: tmp_path / name,
        flash_frame_names=set(names),
    )
    assert not invalidated.load_cache(cache_path)


def test_geometric_route_starts_after_frame_a_coverage(monkeypatch) -> None:
    names = ["main_a.jpg", "bridge.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "bridge.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.legacy_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )

    route = find_geometric_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        min_overlap_frac=0.1,
    )

    assert route is not None
    assert route.auxiliary_image_names == ("bridge.jpg",)
    assert route.frame_a_coverage_end == pytest.approx(1.0 / 3.0)
    assert route.frame_b_coverage_start == pytest.approx(2.0 / 3.0)
    assert route.auxiliary_responsibility_fraction == pytest.approx(1.0 / 3.0)
    assert route.search_method == "dijkstra_lexicographic"
    assert route.min_progress_fraction == pytest.approx(1.0 / 9.0)


def test_geometric_route_group_center_falls_back_to_bbox() -> None:
    center = object_group_center([
        {"center": [0.0, 0.0, 2.0]},
        {"bbox_min": [1.0, -1.0, 1.0], "bbox_max": [3.0, 1.0, 3.0]},
    ])

    assert center == pytest.approx(np.array([1.0, 0.0, 2.0]))


def test_geometric_route_uses_no_auxiliary_when_main_frames_connect(monkeypatch) -> None:
    names = ["main_a.jpg", "unused.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool),
        "unused.jpg": np.ones(10, dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.legacy_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )

    route = find_geometric_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        min_overlap_frac=0.1,
    )

    assert route is not None
    assert route.auxiliary_image_names == ()
    assert route.auxiliary_responsibility_fraction == 0.0


def test_geometric_route_dijkstra_escapes_greedy_dead_end(monkeypatch) -> None:
    names = ["main_a.jpg", "decoy.jpg", "bridge_1.jpg", "bridge_2.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "decoy.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "bridge_1.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool),
        "bridge_2.jpg": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool),
    }
    angles = {
        "main_a.jpg": 0.0,
        "decoy.jpg": 50.0,
        "bridge_1.jpg": -30.0,
        "bridge_2.jpg": -50.0,
        "main_b.jpg": -50.0,
    }
    monkeypatch.setattr(
        "src.legacy_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    monkeypatch.setattr(
        "src.legacy_auxiliary_path._unit_forward",
        lambda pose: np.array([
            math.sin(math.radians(angles[pose.image_name])),
            0.0,
            math.cos(math.radians(angles[pose.image_name])),
        ]),
    )

    route = find_geometric_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        min_overlap_frac=0.1,
        max_backtrack=0,
    )

    assert route is not None
    assert route.auxiliary_image_names == ("bridge_1.jpg", "bridge_2.jpg")


def test_geometric_route_minimizes_total_redundant_overlap(monkeypatch) -> None:
    names = ["main_a.jpg", "high_overlap.jpg", "low_overlap.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "high_overlap.jpg": np.ones(10, dtype=bool),
        "low_overlap.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.legacy_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )

    route = find_geometric_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        min_overlap_frac=0.1,
    )

    assert route is not None
    assert route.auxiliary_image_names == ("low_overlap.jpg",)


def test_geometric_route_enforces_minimum_progress(monkeypatch) -> None:
    names = ["main_a.jpg", "tiny_step.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1] * 11 + [0] * 29, dtype=bool),
        "tiny_step.jpg": np.array([0] * 6 + [1] * 6 + [0] * 28, dtype=bool),
        "main_b.jpg": np.array([0] * 7 + [1] * 33, dtype=bool),
    }
    monkeypatch.setattr(
        "src.legacy_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    kwargs = {
        "center_a": np.array([0.0, 0.0, 2.0]),
        "center_b": np.array([4.0, 0.0, 2.0]),
        "frame_a_name": "main_a.jpg",
        "frame_b_name": "main_b.jpg",
        "poses": poses,
        "intrinsics": make_intrinsics(),
        "min_overlap_frac": 0.1,
    }

    assert find_geometric_auxiliary_route(**kwargs, min_progress_frac=0.05) is None
    route = find_geometric_auxiliary_route(**kwargs, min_progress_frac=0.02)
    assert route is not None
    assert route.auxiliary_image_names == ("tiny_step.jpg",)


def test_auxiliary_pruning_checks_near_pose_and_route_validity() -> None:
    poses = {
        "main_a.jpg": make_pose("main_a.jpg", 0.0),
        "near.jpg": make_pose("near.jpg", 0.05),
        "far_redundant.jpg": make_pose("far_redundant.jpg", 0.8),
        "needed.jpg": make_pose("needed.jpg", 1.4),
        "main_b.jpg": make_pose("main_b.jpg", 2.0),
    }

    pruned = _prune_auxiliary_names(
        ("near.jpg", "far_redundant.jpg", "needed.jpg"),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        route_is_valid=lambda names: "needed.jpg" in names,
        near_duplicate_translation_m=0.12,
        near_duplicate_rotation_deg=6.0,
    )

    assert pruned == ("needed.jpg",)
