from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.auxiliary_path import AuxiliaryRoute, VisualPoseEdge, VisualPoseGraph
from src.legacy_auxiliary_path import (
    _prune_auxiliary_names,
    _semantic_conflict,
    find_geometric_auxiliary_route,
    object_group_center,
)
from src.qa_generator import (
    ReasoningFrameContext,
    _CROSS_FRAME_ANSWER_PAIR_DISTANCE_DEFINITION_KEY,
    _CROSS_FRAME_ANSWER_PAIR_DISTANCE_M_KEY,
    _CROSS_FRAME_ANSWER_PAIR_IDS_KEY,
    _annotate_cross_frame_answer_pair_distance,
    _annotate_cross_frame_questions,
    _clear_cross_frame_distance_metadata,
    _cross_frame_answer_pair_ids,
    _prioritize_cross_frame_questions_by_distance,
    _prioritize_cross_frame_reference_objects,
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


def make_point_object(obj_id: int, label: str, x: float) -> dict:
    return {
        "id": obj_id,
        "label": label,
        "center": [x, 0.0, 0.5],
        "bbox_min": [x, 0.0, 0.0],
        "bbox_max": [x, 0.0, 1.0],
    }


def make_context(
    name: str,
    *,
    regular: set[int],
    attachment: set[int] | None = None,
    cross_frame_visible: set[int] | None = None,
) -> ReasoningFrameContext:
    return ReasoningFrameContext(
        image_name=name,
        camera_pose=make_pose(name),
        regular_referable_ids=frozenset(regular),
        attachment_referable_ids=frozenset(attachment or set()),
        cross_frame_visible_ids=frozenset(cross_frame_visible or set()),
        cache_entry={"frame_usable": True},
    )


@pytest.mark.parametrize(
    ("question", "expected_pair"),
    [
        (
            {"type": "direction_agent", "obj_a_id": 1, "obj_b_id": 2},
            (1, 2),
        ),
        (
            {"type": "distance", "obj_a_id": 1, "obj_b_id": 2},
            (1, 2),
        ),
        (
            {"type": "direction_allocentric", "obj_a_id": 1, "obj_b_id": 2},
            (1, 2),
        ),
        (
            {
                "type": "direction_object_centric",
                "obj_ref_id": 1,
                "obj_face_id": 3,
                "obj_target_id": 2,
            },
            (1, 2),
        ),
        (
            {"type": "object_move_agent", "moved_obj_id": 90, "query_obj_id": 1, "obj_c_id": 2},
            (1, 2),
        ),
        (
            {"type": "object_move_distance", "moved_obj_id": 90, "query_obj_id": 1, "obj_c_id": 2},
            (1, 2),
        ),
        (
            {"type": "object_move_occlusion", "moved_obj_id": 90, "query_obj_id": 1, "obj_ref_id": 2},
            (1, 2),
        ),
        (
            {"type": "object_move_object_centric", "moved_obj_id": 90, "query_obj_id": 1, "obj_ref_id": 2},
            (1, 2),
        ),
        (
            {
                "type": "object_rotate_object_centric",
                "moved_obj_id": 90,
                "query_obj_id": 1,
                "obj_ref_id": 2,
                "obj_face_id": 91,
            },
            (1, 2),
        ),
        (
            {"type": "object_move_allocentric", "moved_obj_id": 90, "query_obj_id": 1, "obj_ref_id": 2},
            (1, 2),
        ),
        (
            {"type": "attachment_move", "root_id": 90, "query_obj_id": 1, "obj_ref_id": 2},
            (1, 2),
        ),
        (
            {"type": "coordinate_rotation_agent", "obj_a_id": 1, "obj_b_id": 2},
            (1, 2),
        ),
        (
            {
                "type": "coordinate_rotation_object_centric",
                "obj_ref_id": 1,
                "obj_face_id": 91,
                "obj_target_id": 2,
            },
            (1, 2),
        ),
        (
            {"type": "coordinate_rotation_allocentric", "obj_a_id": 1, "obj_b_id": 2},
            (1, 2),
        ),
        (
            {"type": "object_move_occlusion", "moved_obj_id": 1, "target_obj_id": 2},
            (1, 2),
        ),
    ],
)
def test_cross_frame_answer_pair_uses_the_asked_relation(
    question: dict,
    expected_pair: tuple[int, int],
) -> None:
    assert _cross_frame_answer_pair_ids(question) == expected_pair


def test_cross_frame_answer_pair_distance_uses_inclusive_surface_distance_limit() -> None:
    objects = {
        1: make_point_object(1, "left", 0.0),
        2: make_point_object(2, "right", 4.0),
    }
    question = {"type": "coordinate_rotation_agent", "obj_a_id": 1, "obj_b_id": 2}

    assert _annotate_cross_frame_answer_pair_distance(question, objects)
    assert question[_CROSS_FRAME_ANSWER_PAIR_IDS_KEY] == [1, 2]
    assert question[_CROSS_FRAME_ANSWER_PAIR_DISTANCE_M_KEY] == pytest.approx(4.0)
    assert (
        question[_CROSS_FRAME_ANSWER_PAIR_DISTANCE_DEFINITION_KEY]
        == "aabb_closest_point_approx"
    )

    selected, diagnostics = _prioritize_cross_frame_questions_by_distance([question])
    assert selected == [question]
    assert diagnostics["within_limit_question_count"] == 1
    assert diagnostics["fallback_type_count"] == 0


def test_cross_frame_distance_priority_keeps_all_routed_pairs_farthest_first() -> None:
    objects = {
        obj_id: make_point_object(obj_id, f"object-{obj_id}", x)
        for obj_id, x in ((1, 0.0), (2, 1.0), (3, 3.0), (4, 4.0), (5, 5.0))
    }
    questions = [
        {
            "type": "coordinate_rotation_agent",
            "obj_a_id": 1,
            "obj_b_id": obj_id,
            "correct_value": f"answer-{obj_id}",
        }
        for obj_id in (2, 3, 4, 5)
    ]
    for question in questions:
        assert _annotate_cross_frame_answer_pair_distance(question, objects)

    selected, diagnostics = _prioritize_cross_frame_questions_by_distance(questions)

    assert [question["obj_b_id"] for question in selected] == [5, 4, 3, 2]
    assert diagnostics["over_limit_question_count"] == 1
    assert diagnostics["over_limit_retained_question_count"] == 1
    assert diagnostics["over_limit_dropped_question_count"] == 0
    assert diagnostics["fallback_type_count"] == 0


def test_cross_frame_distance_priority_keeps_types_with_only_over_limit_pairs() -> None:
    objects = {
        obj_id: make_point_object(obj_id, f"object-{obj_id}", x)
        for obj_id, x in ((1, 0.0), (2, 5.0), (3, 6.0), (4, 4.0))
    }
    fallback_questions = [
        {
            "type": "object_move_allocentric",
            "query_obj_id": 1,
            "obj_ref_id": obj_id,
            "correct_value": answer,
        }
        for obj_id, answer in ((2, "north"), (2, "south"), (3, "east"))
    ]
    in_limit_question = {
        "type": "coordinate_rotation_agent",
        "obj_a_id": 1,
        "obj_b_id": 4,
    }
    questions = [*fallback_questions, in_limit_question]
    for question in questions:
        assert _annotate_cross_frame_answer_pair_distance(question, objects)

    selected, diagnostics = _prioritize_cross_frame_questions_by_distance(questions)

    assert [
        question["obj_ref_id"]
        for question in selected
        if question["type"] == "object_move_allocentric"
    ] == [3, 2, 2]
    assert in_limit_question in selected
    assert diagnostics["fallback_type_count"] == 0
    assert diagnostics["by_type"]["object_move_allocentric"]["over_limit_question_count"] == 3
    assert diagnostics["by_type"]["object_move_allocentric"]["kept_question_count"] == 3


def test_cross_frame_reference_budget_uses_farthest_in_limit_objects() -> None:
    anchor = make_point_object(1, "anchor", 0.0)
    references = [
        make_point_object(obj_id, f"object-{obj_id}", x)
        for obj_id, x in ((2, 1.0), (3, 2.0), (4, 3.0), (5, 4.0), (6, 5.0))
    ]

    selected = _prioritize_cross_frame_reference_objects(
        anchor,
        references,
        max_candidates=3,
    )

    assert [obj["id"] for obj in selected] == [5, 4, 3]


def test_cross_frame_reference_budget_falls_back_to_nearest_over_limit_object() -> None:
    anchor = make_point_object(1, "anchor", 0.0)
    references = [
        make_point_object(obj_id, f"object-{obj_id}", x)
        for obj_id, x in ((2, 6.0), (3, 5.0), (4, 7.0))
    ]

    selected = _prioritize_cross_frame_reference_objects(
        anchor,
        references,
        max_candidates=3,
    )

    assert [obj["id"] for obj in selected] == [3]


def test_cross_frame_distance_priority_runs_before_view_and_object_caps() -> None:
    import scripts.run_pipeline as run_pipeline

    objects = {
        obj_id: make_point_object(obj_id, f"object-{obj_id}", x)
        for obj_id, x in ((1, 0.0), (2, 1.0), (3, 3.0), (4, 4.0))
    }
    questions = []
    for ref_id, route_cost in ((2, 1.0), (3, 2.0), (4, 3.0)):
        question = {
            "scene_id": "scene0000_00",
            "level": "L2",
            "type": "object_move_agent",
            "image_name": "first.jpg",
            "reasoning_frame_2": "last.jpg",
            "cross_frame_layout": "source_query_to_ref",
            "object_frame_groups": {"frame_1": [90, 1], "frame_2": [ref_id]},
            "moved_obj_id": 90,
            "query_obj_id": 1,
            "obj_c_id": ref_id,
            "correct_value": f"answer-{ref_id}",
            "_cross_frame_pair_score": route_cost,
        }
        assert _annotate_cross_frame_answer_pair_distance(question, objects)
        questions.append(question)

    prioritized, _diagnostics = _prioritize_cross_frame_questions_by_distance(questions)
    retained = run_pipeline._retain_best_cross_frame_views(prioritized)
    capped = run_pipeline._apply_scene_type_cap(
        retained,
        scene_type_cap=0,
        frame_type_cap=0,
        frame_type_object_cap=0,
    )
    for question in capped:
        _clear_cross_frame_distance_metadata(question)

    assert [question["obj_c_id"] for question in capped] == [4, 3]
    assert all(
        not any(key.startswith("_cross_frame_answer_pair_") for key in question)
        for question in capped
    )


def test_cross_frame_view_selection_keeps_lowest_route_cost_per_semantic_question() -> None:
    import scripts.run_pipeline as run_pipeline

    questions = []
    for image_name, route_cost in (("slow.jpg", 5.0), ("best.jpg", 1.0), ("ok.jpg", 3.0)):
        questions.append({
            "type": "coordinate_rotation_agent",
            "image_name": image_name,
            "reasoning_frame_2": "last.jpg",
            "cross_frame_layout": "a_to_b",
            "object_frame_groups": {"frame_1": [1], "frame_2": [2]},
            "obj_a_id": 1,
            "obj_b_id": 2,
            "rotation_angle": 90,
            "correct_value": "left",
            _CROSS_FRAME_ANSWER_PAIR_IDS_KEY: [1, 2],
            _CROSS_FRAME_ANSWER_PAIR_DISTANCE_M_KEY: 3.5,
            "_cross_frame_pair_score": route_cost,
        })

    retained = run_pipeline._retain_best_cross_frame_views(questions)

    assert [question["image_name"] for question in retained] == ["best.jpg"]
    assert all("_cross_frame_pair_score" not in question for question in retained)


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
        "visibility": "frame_2",
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


def test_v2_cross_frame_generation_uses_frame_2_camera_for_occlusion() -> None:
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
        first_movement_camera = generate_mock.call_args.args[3]
        first_occlusion_camera = generate_mock.call_args.kwargs["occlusion_camera_pose"]
        result_b = generate_cross_frame_questions(
            objects=list(objects.values()),
            attachment_graph={1: [2]},
            attached_by={2: 1},
            frame_1=frame_1,
            frame_2=frame_2_b,
            color_intrinsics=make_intrinsics(),
            only_question_types=["L2_object_move_occlusion"],
        )
        second_movement_camera = generate_mock.call_args.args[3]
        second_occlusion_camera = generate_mock.call_args.kwargs["occlusion_camera_pose"]

    assert first_movement_camera is frame_1.camera_pose
    assert second_movement_camera is frame_1.camera_pose
    assert first_occlusion_camera is frame_2_a.camera_pose
    assert second_occlusion_camera is frame_2_b.camera_pose
    assert result_a[0]["camera_bindings"] == result_b[0]["camera_bindings"]


def test_direct_cross_frame_generation_keeps_over_limit_pairs_and_cleans_metadata() -> None:
    from src.qa_generator import generate_cross_frame_questions

    objects = {
        1: make_point_object(1, "anchor", 0.0),
        2: make_point_object(2, "at-limit", 4.0),
        3: make_point_object(3, "over-limit", 5.0),
    }
    generated = [
        {
            "type": "coordinate_rotation_agent",
            "question": f"question-{obj_b_id}",
            "obj_a_id": 1,
            "obj_b_id": obj_b_id,
            "rotation_angle": 90,
            "correct_value": "left",
        }
        for obj_b_id in (2, 3)
    ]

    with patch(
        "src.qa_generator.generate_l3_coordinate_rotation",
        return_value=generated,
    ):
        result = generate_cross_frame_questions(
            objects=list(objects.values()),
            attachment_graph={},
            attached_by={},
            frame_1=make_context("first.jpg", regular={1}),
            frame_2=make_context("last.jpg", regular={2, 3}),
            color_intrinsics=make_intrinsics(),
            only_question_types=["L3_coordinate_rotation_agent"],
        )

    assert [question["obj_b_id"] for question in result] == [3, 2]
    assert all(
        not any(key.startswith("_cross_frame_answer_pair_") for key in question)
        for question in result
    )


def test_direct_cross_frame_generation_can_preserve_distance_metadata() -> None:
    from src.qa_generator import generate_cross_frame_questions

    objects = {
        1: make_point_object(1, "anchor", 0.0),
        2: make_point_object(2, "at-limit", 4.0),
        3: make_point_object(3, "over-limit", 5.0),
    }
    generated = [
        {
            "type": "coordinate_rotation_agent",
            "question": f"question-{obj_b_id}",
            "obj_a_id": 1,
            "obj_b_id": obj_b_id,
            "rotation_angle": 90,
            "correct_value": "left",
        }
        for obj_b_id in (2, 3)
    ]

    with patch(
        "src.qa_generator.generate_l3_coordinate_rotation",
        return_value=generated,
    ):
        result = generate_cross_frame_questions(
            objects=list(objects.values()),
            attachment_graph={},
            attached_by={},
            frame_1=make_context("first.jpg", regular={1}),
            frame_2=make_context("last.jpg", regular={2, 3}),
            color_intrinsics=make_intrinsics(),
            only_question_types=["L3_coordinate_rotation_agent"],
            preserve_distance_metadata=True,
        )

    assert [question["obj_b_id"] for question in result] == [2, 3]
    assert [
        question[_CROSS_FRAME_ANSWER_PAIR_DISTANCE_M_KEY]
        for question in result
    ] == [4.0, 5.0]
    assert all(
        question[_CROSS_FRAME_ANSWER_PAIR_IDS_KEY] == [1, question["obj_b_id"]]
        for question in result
    )


def test_l1_cross_frame_generation_uses_expected_layouts_and_camera_bindings() -> None:
    from src.qa_generator import generate_cross_frame_questions

    objects = [
        make_point_object(1, "table", 0.0),
        make_point_object(2, "lamp", 1.5),
        make_point_object(3, "sofa", 2.5),
        make_point_object(4, "cabinet", 3.5),
    ]
    frame_1 = make_context("first.jpg", regular={1})
    frame_2 = make_context("last.jpg", regular={2, 3, 4})

    result = generate_cross_frame_questions(
        objects=objects,
        attachment_graph={},
        attached_by={},
        frame_1=frame_1,
        frame_2=frame_2,
        color_intrinsics=make_intrinsics(),
        only_question_types=[
            "L1_direction_agent",
            "L1_distance",
            "L1_direction_object_centric",
            "L1_direction_allocentric",
        ],
    )

    by_type: dict[str, list[dict]] = {}
    for question in result:
        by_type.setdefault(question["type"], []).append(question)
        assert question["image_name"] == "first.jpg"
        assert question["reasoning_frame_2"] == "last.jpg"
        assert set(question["object_frame_groups"]["frame_1"]) == {1}
        assert set(question["object_frame_groups"]["frame_1"]).isdisjoint(
            question["object_frame_groups"]["frame_2"]
        )
        assert question["question"].startswith(
            "A sequence of views follows a visually continuous camera path"
        )

    assert set(by_type) == {
        "direction_agent",
        "distance",
        "direction_object_centric",
        "direction_allocentric",
    }
    assert {q["cross_frame_layout"] for q in by_type["direction_agent"]} == {"a_to_b"}
    assert {q["cross_frame_layout"] for q in by_type["distance"]} == {"a_to_b"}
    assert {q["cross_frame_layout"] for q in by_type["direction_allocentric"]} == {"a_to_b"}
    assert {q["cross_frame_layout"] for q in by_type["direction_object_centric"]} == {
        "ref_in_frame_1",
        "face_in_frame_1",
    }
    assert all(
        question["question"].count("first main view") >= 2
        for question_type in ("direction_agent", "direction_allocentric")
        for question in by_type[question_type]
    )
    for question in by_type["direction_object_centric"]:
        if question["cross_frame_layout"] == "ref_in_frame_1":
            assert question["obj_ref_id"] == 1
            assert {
                question["obj_face_id"],
                question["obj_target_id"],
            } == set(question["object_frame_groups"]["frame_2"])
        else:
            assert question["obj_face_id"] == 1
            assert {
                question["obj_ref_id"],
                question["obj_target_id"],
            } == set(question["object_frame_groups"]["frame_2"])
    assert by_type["direction_agent"][0]["camera_bindings"] == {"answer": "frame_1"}
    assert by_type["distance"][0]["camera_bindings"] == {
        "answer": "camera_independent"
    }
    assert by_type["direction_allocentric"][0]["camera_bindings"] == {
        "cardinal_hint": "frame_1",
        "answer": "world",
    }
    assert by_type["direction_object_centric"][0]["camera_bindings"] == {
        "answer": "object_defined"
    }


def test_l1_cross_frame_generation_uses_first_main_view_camera() -> None:
    from src.qa_generator import generate_cross_frame_questions

    objects = [make_object(1, "table"), make_object(2, "lamp")]
    frame_1 = make_context("first.jpg", regular={1})
    frame_2 = make_context("last.jpg", regular={2})

    with (
        patch("src.qa_generator.compute_all_relations", return_value=[]) as relations_mock,
        patch(
            "src.qa_generator.generate_l1_direction_allocentric",
            return_value=[],
        ) as allocentric_mock,
    ):
        generate_cross_frame_questions(
            objects=objects,
            attachment_graph={},
            attached_by={},
            frame_1=frame_1,
            frame_2=frame_2,
            color_intrinsics=make_intrinsics(),
            only_question_types=[
                "L1_direction_agent",
                "L1_direction_allocentric",
            ],
        )

    assert relations_mock.call_args.args[1] is frame_1.camera_pose
    assert allocentric_mock.call_args.args[1] is frame_1.camera_pose


def test_l1_cross_frame_annotation_rejects_overlap_without_single_frame_fallback() -> None:
    objects = {1: make_object(1, "table"), 2: make_object(2, "lamp")}
    question = {
        "level": "L1",
        "type": "direction_agent",
        "question": "question",
        "obj_a_id": 1,
        "obj_b_id": 2,
    }

    referability_overlap = _annotate_cross_frame_questions(
        [dict(question)],
        frame_1=make_context("first.jpg", regular={1, 2}),
        frame_2=make_context("last.jpg", regular={2}),
        objects_by_id=objects,
    )
    visibility_overlap = _annotate_cross_frame_questions(
        [dict(question)],
        frame_1=make_context(
            "first.jpg",
            regular={1},
            cross_frame_visible={2},
        ),
        frame_2=make_context("last.jpg", regular={2}),
        objects_by_id=objects,
    )

    assert referability_overlap == []
    assert visibility_overlap == []


def test_l1_cross_frame_direction_preserves_attachment_suppression() -> None:
    from src.qa_generator import generate_cross_frame_questions

    objects = [
        make_point_object(1, "table", 0.0),
        make_point_object(2, "lamp", 1.5),
    ]
    result = generate_cross_frame_questions(
        objects=objects,
        attachment_graph={1: [2]},
        attached_by={2: 1},
        frame_1=make_context("first.jpg", regular={1}),
        frame_2=make_context("last.jpg", regular={2}),
        color_intrinsics=make_intrinsics(),
        only_question_types=["L1_direction_agent", "L1_distance"],
        attachment_edges=[{"parent_id": 1, "child_id": 2, "type": "support"}],
    )

    assert {question["type"] for question in result} == {"distance"}


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


def test_visible_but_nonreferable_opposite_role_reports_cross_frame_rejection() -> None:
    objects = {1: make_object(1, "table"), 2: make_object(2, "toilet paper")}
    question = {
        "type": "object_move_occlusion",
        "question": "question",
        "moved_obj_id": 1,
        "target_obj_id": 2,
    }
    rejection_counts: Counter[str] = Counter()
    rejection_details: list[dict] = []

    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context(
            "first.jpg",
            regular=set(),
            attachment={1},
            cross_frame_visible={1, 2},
        ),
        frame_2=make_context("last.jpg", regular={2}, cross_frame_visible={2}),
        objects_by_id=objects,
        rejection_counts=rejection_counts,
        rejection_details=rejection_details,
    )

    assert result == []
    assert rejection_counts["question_main_frame_visibility_overlap_rejected"] == 1
    assert rejection_details[0]["frame_1_conflicting_object_ids"] == [2]
    assert rejection_details[0]["frame_2_conflicting_object_ids"] == []


def test_frame_2_visibility_of_frame_1_role_rejects_cross_frame_question() -> None:
    objects = {1: make_object(1, "table"), 2: make_object(2, "lamp")}
    question = {
        "type": "object_move_occlusion",
        "question": "question",
        "moved_obj_id": 1,
        "target_obj_id": 2,
    }

    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context(
            "first.jpg",
            regular=set(),
            attachment={1},
            cross_frame_visible={1},
        ),
        frame_2=make_context("last.jpg", regular={2}, cross_frame_visible={1, 2}),
        objects_by_id=objects,
    )

    assert result == []


def test_scene_0d2ee665be_frame_001200_rejects_visible_toilet_paper() -> None:
    objects = {
        34: make_object(34, "table"),
        42: make_object(42, "keyboard"),
        64: make_object(64, "toilet paper"),
    }
    question = {
        "type": "object_move_agent",
        "question": "question",
        "moved_obj_id": 34,
        "query_obj_id": 42,
        "obj_c_id": 64,
    }

    result = _annotate_cross_frame_questions(
        [question],
        frame_1=make_context(
            "frame_001200.jpg",
            regular=set(),
            attachment={34, 42},
            cross_frame_visible={34, 42, 64},
        ),
        frame_2=make_context(
            "frame_001480.jpg",
            regular={64},
            cross_frame_visible={64},
        ),
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


def test_geometric_route_rejects_auxiliary_frame_showing_both_question_groups(
    monkeypatch,
) -> None:
    names = ["main_a.jpg", "both_groups.jpg", "one_group.jpg", "main_b.jpg"]
    poses = {name: make_pose(name) for name in names}
    masks = {
        "main_a.jpg": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "both_groups.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "one_group.jpg": np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "main_b.jpg": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    }
    monkeypatch.setattr(
        "src.legacy_auxiliary_path.route_visibility_mask",
        lambda _points, pose, _intrinsics: masks[pose.image_name],
    )
    monkeypatch.setattr(
        "src.legacy_auxiliary_path._semantic_conflict",
        lambda pose, _intrinsics, _group_a, _group_b: pose.image_name == "both_groups.jpg",
    )

    route = find_geometric_auxiliary_route(
        center_a=np.array([0.0, 0.0, 2.0]),
        center_b=np.array([0.9, 0.0, 2.0]),
        frame_a_name="main_a.jpg",
        frame_b_name="main_b.jpg",
        poses=poses,
        intrinsics=make_intrinsics(),
        group_a_objects=[make_object(1, "frame-1-object")],
        group_b_objects=[make_object(2, "frame-2-object")],
        min_overlap_frac=0.1,
    )

    assert route is not None
    assert route.auxiliary_image_names == ("one_group.jpg",)
    assert route.semantic_rejected_frame_count == 1


def test_geometric_semantic_conflict_requires_visible_objects_from_both_groups() -> None:
    pose = make_pose("frame.jpg")
    intrinsics = make_intrinsics()

    def projected_object(obj_id: int, label: str, x: float) -> dict:
        return {
            "id": obj_id,
            "label": label,
            "center": [x, 0.0, 2.0],
            "bbox_min": [x - 0.2, -0.2, 1.8],
            "bbox_max": [x + 0.2, 0.2, 2.2],
        }

    visible_a = projected_object(1, "frame-1-object", -0.3)
    visible_b = projected_object(2, "frame-2-object", 0.3)
    outside_view = projected_object(3, "outside-view", 100.0)

    assert _semantic_conflict(pose, intrinsics, [visible_a], [visible_b])
    assert not _semantic_conflict(pose, intrinsics, [visible_a], [outside_view])
    assert not _semantic_conflict(pose, intrinsics, [outside_view], [visible_b])


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
