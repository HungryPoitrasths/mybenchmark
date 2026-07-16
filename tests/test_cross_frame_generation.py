from __future__ import annotations

from pathlib import Path

import numpy as np

from src.auxiliary_path import AuxiliaryRoute, VisualPoseEdge, VisualPoseGraph
from src.qa_generator import (
    ReasoningFrameContext,
    _annotate_cross_frame_questions,
)
from src.utils.colmap_loader import CameraPose


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
