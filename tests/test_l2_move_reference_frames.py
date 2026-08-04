from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.qa_generator import (
    L2_OBJECT_MOVE_SEMANTICS_VERSION,
    SceneMotionCache,
    _allocentric_ground_move_directions,
    _camera_ground_move_directions,
    _candidate_deltas,
    _generate_l2_distance_questions_for_object,
    _iter_valid_object_move_states,
    _movement_direction_for_delta,
    _object_pair_ground_move_directions,
    _scaled_move_candidates,
    generate_l2_object_move,
    generate_l2_object_move_allocentric,
    generate_l2_object_move_object_centric,
)
from src.utils.colmap_loader import CameraPose
from src.virtual_ops import apply_movement_selective


def _object(obj_id: int, label: str, center: tuple[float, float, float]) -> dict:
    center_array = np.asarray(center, dtype=np.float64)
    half_extent = np.asarray([0.1, 0.1, 0.1], dtype=np.float64)
    return {
        "id": obj_id,
        "label": label,
        "center": center_array.tolist(),
        "bbox_min": (center_array - half_extent).tolist(),
        "bbox_max": (center_array + half_extent).tolist(),
    }


def _identity_pose() -> CameraPose:
    return CameraPose(
        image_name="frame.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def _rolled_pitched_pose() -> CameraPose:
    yaw, pitch, roll = np.deg2rad([31.0, -24.0, 17.0])
    rz = np.asarray([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    ry = np.asarray([
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
    ])
    rx = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll), np.cos(roll)],
    ])
    camera_to_world = rz @ ry @ rx
    return CameraPose(
        image_name="rolled.jpg",
        rotation=camera_to_world.T,
        translation=np.zeros(3, dtype=np.float64),
    )


def _attachment_scene() -> tuple[list[dict], dict[int, list[int]]]:
    return [
        _object(1, "table", (1.0, 0.0, 0.5)),
        _object(2, "book", (2.0, 0.0, 0.5)),
        _object(3, "chair", (3.0, 0.0, 0.5)),
    ], {1: [2]}


def test_camera_ground_actions_follow_projected_camera_axes() -> None:
    directions = _camera_ground_move_directions(_rolled_pitched_pose())

    assert [label for label, _vector in directions] == [
        "forward",
        "forward-right",
        "right",
        "backward-right",
        "backward",
        "backward-left",
        "left",
        "forward-left",
    ]
    vectors = [vector for _label, vector in directions]
    rotation_t = _rolled_pitched_pose().rotation.T
    expected_forward = np.asarray(rotation_t[:, 2], dtype=float)
    expected_forward[2] = 0.0
    expected_forward /= np.linalg.norm(expected_forward)
    expected_right = np.asarray(rotation_t[:, 0], dtype=float)
    expected_right[2] = 0.0
    expected_right /= np.linalg.norm(expected_right)
    np.testing.assert_allclose(vectors[0], expected_forward)
    np.testing.assert_allclose(vectors[2], expected_right)
    for index, vector in enumerate(vectors):
        assert vector[2] == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        vectors[1],
        (expected_forward + expected_right)
        / np.linalg.norm(expected_forward + expected_right),
    )


def test_scaled_camera_actions_preserve_requested_metric_distance() -> None:
    candidates = _scaled_move_candidates(
        _camera_ground_move_directions(_rolled_pitched_pose()),
        magnitudes=(2.5,),
    )

    assert len(candidates) == 8
    assert all(np.linalg.norm(delta) == pytest.approx(2.5) for _label, delta in candidates)


def test_object_pair_frame_uses_query_to_reference_as_frozen_forward() -> None:
    query = np.asarray([1.0, 2.0, 0.5])
    reference = np.asarray([4.0, 6.0, 1.5])
    directions = dict(_object_pair_ground_move_directions(query, reference))
    expected_forward = np.asarray([3.0, 4.0, 0.0]) / 5.0

    np.testing.assert_allclose(directions["forward"], expected_forward)
    np.testing.assert_allclose(
        directions["right"],
        [expected_forward[1], -expected_forward[0], 0.0],
    )
    assert _object_pair_ground_move_directions(query, query) == ()


def test_direction_lookup_rejects_a_mislabeled_diagonal() -> None:
    candidates = (("right", np.asarray([2.5, 0.0, 0.0])),)

    with pytest.raises(ValueError, match="not part of the declared action frame"):
        _movement_direction_for_delta(np.asarray([2.0, 1.0, 0.0]), candidates)


def test_scene_motion_cache_separates_action_frame_signatures() -> None:
    objects = [_object(1, "box", (0.0, 0.0, 0.0))]
    bounds = {"bbox_min": [-5.0, -5.0, -1.0], "bbox_max": [5.0, 5.0, 1.0]}
    cache = SceneMotionCache()

    with patch("src.qa_generator.has_terminal_bbox_collision", return_value=False):
        x_states = list(_iter_valid_object_move_states(
            objects,
            {},
            1,
            room_bounds=bounds,
            motion_cache=cache,
            candidate_deltas=(np.asarray([1.0, 0.0, 0.0]),),
        ))
        y_states = list(_iter_valid_object_move_states(
            objects,
            {},
            1,
            room_bounds=bounds,
            motion_cache=cache,
            candidate_deltas=(np.asarray([0.0, 1.0, 0.0]),),
        ))

    np.testing.assert_allclose(x_states[0][0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(y_states[0][0], [0.0, 1.0, 0.0])
    assert cache.diagnostics()["move_misses"] == 2
    assert cache.diagnostics()["move_source_count"] == 2


def test_agent_generator_uses_strict_camera_action_metadata() -> None:
    objects, graph = _attachment_scene()
    moved_objects = apply_movement_selective(
        objects,
        graph,
        1,
        np.asarray([0.5, 0.0, 0.0]),
    )
    selected_state = SimpleNamespace(
        delta=np.asarray([0.5, 0.0, 0.0]),
        moved_objects=moved_objects,
        moved_ids={1, 2},
    )

    def select_state(*_args, **kwargs):
        assert _movement_direction_for_delta(
            selected_state.delta,
            tuple(zip(
                [label for label, _vector in _camera_ground_move_directions(_identity_pose())]
                * 6,
                kwargs["candidate_deltas"],
            )),
        ) == "right"
        return selected_state

    with (
        patch("src.qa_generator._select_object_move_state", side_effect=select_state),
        patch("src.qa_generator.compute_all_relations", return_value=[{
            "obj_a_id": 2,
            "obj_b_id": 3,
            "direction_b_rel_a": "left",
        }]),
        patch("src.qa_generator.compute_direction_relations", return_value=[{
            "obj_a_id": 2,
            "obj_b_id": 3,
            "direction_b_rel_a": "front",
        }]),
    ):
        questions = generate_l2_object_move(
            objects=objects,
            attachment_graph=graph,
            attached_by={2: 1},
            camera_pose=_identity_pose(),
            templates={},
            enabled_l2_object_move_types={"object_move_agent"},
        )

    question = next(question for question in questions if question["obj_c_id"] == 3)
    assert question["movement_semantics_version"] == L2_OBJECT_MOVE_SEMANTICS_VERSION
    assert question["movement_reference_frame"] == "agent"
    assert question["movement_camera_binding"] == "frame_1"
    assert question["movement_direction"] == "right"
    assert question["movement_distance_m"] == pytest.approx(0.5)


def test_agent_generator_rejects_direction_answers_near_a_bin_boundary() -> None:
    objects, graph = _attachment_scene()
    delta = np.asarray([0.5, 0.0, 0.0])
    selected_state = SimpleNamespace(
        delta=delta,
        moved_objects=apply_movement_selective(objects, graph, 1, delta),
        moved_ids={1, 2},
    )

    with (
        patch("src.qa_generator._select_object_move_state", return_value=selected_state),
        patch("src.qa_generator.compute_all_relations", return_value=[{
            "obj_a_id": 2,
            "obj_b_id": 3,
            "direction_b_rel_a": "left",
            "ambiguity_score": 0.99,
        }]),
        patch("src.qa_generator.compute_direction_relations", return_value=[{
            "obj_a_id": 2,
            "obj_b_id": 3,
            "direction_b_rel_a": "front",
            "ambiguity_score": 0.0,
        }]),
        patch("src.qa_generator._iter_additional_object_move_states", return_value=[]),
    ):
        questions = generate_l2_object_move(
            objects=objects,
            attachment_graph=graph,
            attached_by={2: 1},
            camera_pose=_identity_pose(),
            templates={},
            enabled_l2_object_move_types={"object_move_agent"},
        )

    assert not any(question["type"] == "object_move_agent" for question in questions)


def test_distance_generator_uses_same_camera_action_table() -> None:
    parent = _object(1, "table", (-0.5, 0.0, 0.5))
    query = _object(2, "book", (0.0, 0.0, 0.5))
    reference = _object(3, "chair", (1.4, 0.0, 0.5))
    candidates = _scaled_move_candidates(_camera_ground_move_directions(_identity_pose()))

    questions = _generate_l2_distance_questions_for_object(
        query_obj=query,
        move_source=parent,
        move_source_id=1,
        attachment_remapped=True,
        relations=[{
            "obj_a_id": 2,
            "obj_b_id": 3,
            "distance_bin": "close (1.0-2.0m)",
            "distance_bin_id": "close",
            "distance_m": 1.2,
        }],
        movement_scene_objects=[parent, query, reference],
        attachment_graph={1: [2]},
        camera_pose=_identity_pose(),
        templates={},
        obj_map={1: parent, 2: query, 3: reference},
        fixed_delta=np.asarray([-1.0, 0.0, 0.0]),
        movement_candidates=candidates,
    )

    assert questions
    assert questions[0]["movement_direction"] == "left"
    assert questions[0]["movement_reference_frame"] == "agent"
    assert questions[0]["movement_camera_binding"] == "frame_1"


def test_object_centric_generator_freezes_initial_query_reference_frame() -> None:
    objects, graph = _attachment_scene()
    delta = np.asarray([0.5, 0.0, 0.0])
    moved_objects = apply_movement_selective(objects, graph, 1, delta)
    facing_calls: list[tuple[np.ndarray, np.ndarray]] = []

    from src.relation_engine import primary_direction_object_centric as real_direction

    def record_direction(anchor_center, facing_center, target_center, **kwargs):
        facing_calls.append((np.asarray(anchor_center), np.asarray(facing_center)))
        return real_direction(anchor_center, facing_center, target_center, **kwargs)

    def valid_states(*_args, **kwargs):
        assert any(np.allclose(candidate, delta) for candidate in kwargs["candidate_deltas"])
        return [(delta, moved_objects, {1, 2})]

    with (
        patch("src.qa_generator._iter_valid_object_move_states", side_effect=valid_states),
        patch("src.qa_generator.primary_direction_object_centric", side_effect=record_direction),
    ):
        questions = generate_l2_object_move_object_centric(
            objects=objects,
            attachment_graph=graph,
            attached_by={2: 1},
            camera_pose=_identity_pose(),
            templates={},
        )

    question = next(question for question in questions if question["obj_ref_id"] == 3)
    assert question["movement_direction"] == "forward"
    assert question["movement_reference_frame"] == "object_centric"
    assert question["movement_frame_query_obj_id"] == 2
    assert question["movement_frame_reference_obj_id"] == 3
    assert question["movement_frame_frozen"] is True
    initial_anchor, initial_facing = facing_calls[0]
    moved_anchor, moved_facing = facing_calls[1]
    np.testing.assert_allclose(initial_facing - initial_anchor, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(moved_facing - moved_anchor, [1.0, 0.0, 0.0])


def test_allocentric_generator_uses_world_cardinal_action_metadata() -> None:
    objects, graph = _attachment_scene()
    delta = np.asarray([0.5, 0.0, 0.0])
    moved_objects = apply_movement_selective(objects, graph, 1, delta)
    selected_state = SimpleNamespace(
        delta=delta,
        moved_objects=moved_objects,
        moved_ids={1, 2},
    )

    def select_state(*_args, **kwargs):
        expected = _candidate_deltas(
            _scaled_move_candidates(_allocentric_ground_move_directions())
        )
        assert len(kwargs["candidate_deltas"]) == len(expected)
        assert all(
            np.allclose(actual, wanted)
            for actual, wanted in zip(kwargs["candidate_deltas"], expected)
        )
        return selected_state

    with patch("src.qa_generator._select_object_move_state", side_effect=select_state):
        questions = generate_l2_object_move_allocentric(
            objects=objects,
            attachment_graph=graph,
            attached_by={2: 1},
            camera_pose=_identity_pose(),
            templates={},
        )

    question = next(question for question in questions if question["obj_ref_id"] == 3)
    assert question["movement_direction"] == "east"
    assert question["movement_reference_frame"] == "allocentric"
    assert question["movement_world_axes"] == "scannet_aligned_xy"
