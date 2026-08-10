from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.recompute_l2_move_benchmark import (
    CandidateEvaluation,
    CandidateSpec,
    MOVE_MAGNITUDES_M,
    MotionState,
    SceneResources,
    _deterministic_options,
    _motion_state,
    _update_options,
    build_candidate_schedule,
    choose_candidate,
    cross_check_legacy_text,
    detect_dataset,
    normalize_v2_object_centric_template,
    recompute_baseline,
    recover_legacy_object_centric_movement_from_text,
    recover_legacy_movement,
    repair_benchmark,
    repaired_dedup_key,
    strict_directions_for_question,
)
from src.qa_generator import (
    L2_OBJECT_MOVE_SEMANTICS_VERSION,
    _allocentric_ground_move_directions,
    _camera_ground_move_directions,
    _object_pair_ground_move_directions,
)
from src.relation_engine import HORIZONTAL_DIRECTIONS
from src.utils.colmap_loader import CameraPose


def _object(obj_id: int, label: str, center: tuple[float, float, float]) -> dict:
    center_array = np.asarray(center, dtype=float)
    half = np.asarray([0.1, 0.1, 0.1], dtype=float)
    return {
        "id": obj_id,
        "label": label,
        "center": center_array.tolist(),
        "bbox_min": (center_array - half).tolist(),
        "bbox_max": (center_array + half).tolist(),
    }


def _pose() -> CameraPose:
    return CameraPose(
        image_name="frame.jpg",
        rotation=np.eye(3, dtype=float),
        translation=np.asarray([0.0, -4.0, 0.0], dtype=float),
    )


def _resources(
    objects: list[dict] | None = None,
    graph: dict[int, list[int]] | None = None,
) -> SceneResources:
    return SceneResources(
        scene_id="hashscene",
        dataset="scannetpp",
        scene_dir=Path("unused"),
        metadata_path=None,
        objects=objects or [
            _object(1, "table", (0.0, 0.0, 0.0)),
            _object(2, "book", (1.0, 0.0, 0.0)),
            _object(3, "chair", (3.0, 0.0, 0.0)),
        ],
        attachment_graph=graph or {1: [2]},
        room_bounds={"bbox_min": [-10, -10, -2], "bbox_max": [10, 10, 2]},
        poses={"frame.jpg": _pose()},
        distance_geometry="not_needed",
    )


def _base_question(qtype: str = "object_move_agent") -> dict:
    question = {
        "level": "L2",
        "type": qtype,
        "scene_id": "hashscene",
        "image_name": "frame.jpg",
        "reasoning_frame_2": "other.jpg",
        "moved_obj_id": 1,
        "moved_obj_label": "table",
        "query_obj_id": 2,
        "query_obj_label": "book",
        "attachment_remapped": True,
        "attachment_pair_id": "1->2",
        "delta": [1.0, 0.0, 0.0],
        "old_correct_value": "left",
        "correct_value": "right",
        "options": ["front", "right", "back", "left"],
        "answer": "B",
        "object_frame_groups": {"frame_1": [1, 2], "frame_2": [3]},
    }
    if qtype in {"object_move_agent", "object_move_distance"}:
        question.update({
            "obj_b_id": 2,
            "obj_b_label": "book",
            "obj_c_id": 3,
            "obj_c_label": "chair",
            "question": (
                "From the first main view's camera perspective, imagine moving "
                "the table right by 1.0m. What happens?"
            ),
        })
    else:
        question.update({
            "obj_ref_id": 3,
            "obj_ref_label": "chair",
        })
    return question


def test_dataset_detection_uses_scene_id_convention() -> None:
    assert detect_dataset("scene0025_01") == "scannet"
    assert detect_dataset("fb5a96b1a2") == "scannetpp"


def test_legacy_direction_recovery_and_text_cross_check() -> None:
    question = _base_question()
    direction, distance = recover_legacy_movement(
        question,
        _pose(),
        _resources().objects_by_id,
    )

    assert direction == "right"
    assert distance == pytest.approx(1.0)
    assert cross_check_legacy_text(question, direction, distance) == (True, [])
    bad = copy.deepcopy(question)
    bad["question"] = bad["question"].replace("right", "left")
    ok, reasons = cross_check_legacy_text(bad, direction, distance)
    assert not ok
    assert "legacy_direction_text_mismatch" in reasons


def test_legacy_object_centric_text_recovery_needs_no_pose() -> None:
    question = _base_question("object_move_object_centric")
    question["question"] = (
        "Imagine you are the table initially facing the camera. "
        "If you were shifted backward-left by 2.5m, what happens?"
    )

    assert recover_legacy_object_centric_movement_from_text(question) == (
        "backward-left",
        2.5,
    )


def test_object_centric_text_cross_check_allows_an_inserted_object_label() -> None:
    question = _base_question("object_move_object_centric")
    question["question"] = (
        "After moving the table forward-right by 2.5m in its frozen frame, "
        "where is the chair?"
    )

    assert cross_check_legacy_text(question, "forward-right", 2.5) == (True, [])


def test_known_legacy_diagonal_is_not_retained_as_strict_right() -> None:
    question = _base_question()
    question["delta"] = [1.767767, 1.767767, 0.0]
    question["question"] = question["question"].replace("1.0m", "2.5m")
    direction, distance = recover_legacy_movement(
        question,
        _pose(),
        _resources().objects_by_id,
    )
    schedule = build_candidate_schedule(
        _camera_ground_move_directions(_pose()),
        direction,
        distance,
    )

    assert direction == "right"
    np.testing.assert_allclose(schedule[0].delta, [2.5, 0.0, 0.0])
    assert not np.allclose(schedule[0].delta, question["delta"])


def test_strict_coordinate_systems_produce_expected_deltas() -> None:
    resources = _resources()
    agent = _base_question("object_move_agent")
    allocentric = _base_question("object_move_allocentric")
    allocentric["question"] = "If the table is moved 1.0m to the east, what happens?"
    object_centric = _base_question("object_move_object_centric")
    object_centric["question"] = "If the table were shifted right by 1.0m, what happens?"

    agent_dirs = dict(strict_directions_for_question(agent, _pose(), resources.objects_by_id))
    alloc_dirs = dict(strict_directions_for_question(allocentric, _pose(), resources.objects_by_id))
    object_dirs = dict(
        strict_directions_for_question(object_centric, _pose(), resources.objects_by_id)
    )

    np.testing.assert_allclose(agent_dirs["right"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(alloc_dirs["east"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(object_dirs["forward"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(object_dirs["right"], [1.0, 0.0, 0.0])


def test_object_centric_baseline_uses_frozen_query_to_camera_frame() -> None:
    question = _base_question("object_move_object_centric")
    result = recompute_baseline(question, _pose(), _resources().objects_by_id)

    assert result.valid
    assert result.new_value == "right"
    assert result.ambiguity is not None and result.ambiguity < 0.7


def test_candidate_schedule_has_locked_fallback_order() -> None:
    schedule = build_candidate_schedule(
        _allocentric_ground_move_directions(),
        "east",
        1.0,
    )

    assert [(item.direction, item.distance_m, item.phase) for item in schedule[:4]] == [
        ("east", 1.0, "original_direction_original_distance"),
        ("east", 1.5, "same_direction_farther"),
        ("east", 2.0, "same_direction_farther"),
        ("east", 2.5, "same_direction_farther"),
    ]
    first_other = next(
        index for index, item in enumerate(schedule) if item.phase == "other_direction_original_distance"
    )
    assert first_other == len(MOVE_MAGNITUDES_M)
    assert schedule[first_other].distance_m == 1.0
    assert len(schedule) == 48


def test_choose_candidate_skips_unchanged_until_farther_changed_candidate() -> None:
    schedule = build_candidate_schedule(
        _allocentric_ground_move_directions(),
        "east",
        1.0,
    )
    baseline = CandidateEvaluation(True, new_value="east")

    def evaluate(_question, candidate, *_args):
        value = "north" if candidate.direction == "east" and candidate.distance_m == 1.5 else "east"
        return CandidateEvaluation(True, new_value=value)

    with patch(
        "scripts.recompute_l2_move_benchmark.evaluate_candidate",
        side_effect=evaluate,
    ):
        selected, result, _rejections = choose_candidate(
            _base_question("object_move_allocentric"),
            schedule,
            baseline,
            _pose(),
            _resources(),
        )

    assert selected is not None and result is not None
    assert selected.phase == "same_direction_farther"
    assert selected.distance_m == 1.5
    assert result.new_value == "north"


def test_choose_candidate_can_fall_back_to_other_direction_or_none() -> None:
    schedule = build_candidate_schedule(
        _allocentric_ground_move_directions(),
        "east",
        3.0,
    )
    baseline = CandidateEvaluation(True, new_value="east")

    def other_direction_change(_question, candidate, *_args):
        if candidate.phase == "other_direction_original_distance":
            return CandidateEvaluation(True, new_value="north")
        return CandidateEvaluation(False, "collision")

    with patch(
        "scripts.recompute_l2_move_benchmark.evaluate_candidate",
        side_effect=other_direction_change,
    ):
        selected, result, _ = choose_candidate(
            _base_question("object_move_allocentric"), schedule, baseline, _pose(), _resources()
        )
    assert selected is not None and result is not None
    assert selected.phase == "other_direction_original_distance"

    with patch(
        "scripts.recompute_l2_move_benchmark.evaluate_candidate",
        return_value=CandidateEvaluation(False, "collision"),
    ):
        selected, result, rejections = choose_candidate(
            _base_question("object_move_allocentric"), schedule, baseline, _pose(), _resources()
        )
    assert selected is None and result is None
    assert rejections == {"collision": 48}


def test_motion_state_rejects_room_exit_and_collision() -> None:
    outside_resources = _resources()
    outside_resources.room_bounds = {
        "bbox_min": [-0.2, -0.2, -1],
        "bbox_max": [3.2, 0.2, 1],
    }
    outside = _motion_state(outside_resources, 1, np.asarray([0.0, 1.0, 0.0]))
    assert not outside.valid
    assert outside.reason == "outside_room"

    collision_objects = [
        _object(1, "table", (0.0, 0.0, 0.0)),
        _object(2, "book", (0.0, 0.0, 0.4)),
        _object(3, "chair", (1.0, 0.0, 0.0)),
    ]
    collision_resources = _resources(collision_objects, {1: [2]})
    collision = _motion_state(collision_resources, 1, np.asarray([1.0, 0.0, 0.0]))
    assert not collision.valid
    assert collision.reason == "terminal_collision"


def test_option_update_preserves_order_or_regenerates_deterministically() -> None:
    question = _base_question()
    original_options = list(question["options"])
    assert _update_options(question, "left", 12)
    assert question["options"] == original_options
    assert question["answer"] == "D"

    first = _base_question()
    second = _base_question()
    assert not _update_options(first, "above", 99)
    assert not _update_options(second, "above", 99)
    assert first["options"] == second["options"]
    assert first["answer"] == second["answer"]
    assert _deterministic_options("front", list(HORIZONTAL_DIRECTIONS), 7) == _deterministic_options(
        "front", list(HORIZONTAL_DIRECTIONS), 7
    )


def test_repaired_dedup_key_includes_delta_frames_and_roles() -> None:
    question = _base_question()
    same = copy.deepcopy(question)
    changed_delta = copy.deepcopy(question)
    changed_delta["delta"] = [1.5, 0.0, 0.0]
    changed_frame = copy.deepcopy(question)
    changed_frame["reasoning_frame_2"] = "different.jpg"

    assert repaired_dedup_key(question) == repaired_dedup_key(same)
    assert repaired_dedup_key(question) != repaired_dedup_key(changed_delta)
    assert repaired_dedup_key(question) != repaired_dedup_key(changed_frame)


def test_v2_template_normalization_preserves_existing_multiview_prefix() -> None:
    question = _base_question("object_move_object_centric")
    prefix = "A sequence of views follows a visually continuous camera path. "
    question.update({
        "movement_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "movement_direction": "right",
        "movement_distance_m": 1.0,
        "movement_reference_frame": "moved_object_facing_first_camera",
        "movement_frame_anchor_obj_id": 1,
        "movement_frame_frozen": True,
        "question": (
            prefix
            + "Use a fixed object-centric coordinate frame defined by the initial scene: "
            "old wording"
        ),
    })
    from scripts.recompute_l2_move_benchmark import _load_templates

    normalized = normalize_v2_object_centric_template(
        question,
        0,
        None,
        _load_templates(),
    )

    assert normalized["question"].startswith(prefix)
    assert "Freeze both objects' initial horizontal forward/right axes" in normalized["question"]
    assert "Use a fixed object-centric" not in normalized["question"]
    assert normalized["answer"] == question["answer"]


def test_repair_merge_keeps_non_targets_deep_equal_and_in_order() -> None:
    non_target_a = {"level": "L1", "type": "distance", "marker": "a", "answer": "A"}
    target = _base_question()
    non_target_b = {
        "level": "L3",
        "type": "coordinate_rotation_agent",
        "marker": "b",
        "answer": "B",
    }
    benchmark = {
        "name": "test",
        "version": "1",
        "statistics": {"total": 3},
        "questions": [non_target_a, target, non_target_b],
    }
    repaired_target = copy.deepcopy(target)
    repaired_target.update({
        "movement_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
        "relation_unchanged": False,
    })

    with patch(
        "scripts.recompute_l2_move_benchmark._repair_one",
        return_value=(
            repaired_target,
            {"source_index": 1, "status": "candidate_repaired", "reason": None},
        ),
    ):
        output, audit = repair_benchmark(
            benchmark,
            resources_by_scene={"hashscene": _resources()},
            scene_errors={},
            templates={},
        )

    assert output["questions"][0] == non_target_a
    assert output["questions"][0] is non_target_a
    assert output["questions"][2] == non_target_b
    assert output["questions"][2] is non_target_b
    assert output["questions"][1]["movement_semantics_version"] == 2
    assert audit["aggregate"]["kept_repaired_count"] == 1


def test_repair_aborts_instead_of_mass_dropping_missing_scene_resources() -> None:
    benchmark = {
        "questions": [_base_question("object_move_object_centric")],
        "statistics": {},
    }

    with pytest.raises(RuntimeError, match="systemic scene-resource failures"):
        repair_benchmark(
            benchmark,
            resources_by_scene={},
            scene_errors={"hashscene": "missing raw scene"},
            templates={},
            target_types={"object_move_object_centric"},
        )


def test_targeted_legacy_repair_deduplicates_against_preserved_v2() -> None:
    legacy = _base_question("object_move_object_centric")
    legacy["question"] = "If the table were shifted right by 1.0m, what happens?"
    preserved = copy.deepcopy(legacy)
    preserved["movement_semantics_version"] = L2_OBJECT_MOVE_SEMANTICS_VERSION
    preserved["question"] = "current v2 wording"
    other_type = _base_question("object_move_agent")
    benchmark = {
        "name": "test",
        "version": "1",
        "statistics": {},
        "questions": [preserved, legacy, other_type],
    }
    repaired = copy.deepcopy(legacy)
    repaired["movement_semantics_version"] = L2_OBJECT_MOVE_SEMANTICS_VERSION

    def fake_repair(index, _question, *_args, **_kwargs):
        return repaired, {"source_index": index, "status": "candidate_repaired"}

    with patch(
        "scripts.recompute_l2_move_benchmark._repair_one",
        side_effect=fake_repair,
    ):
        output, audit = repair_benchmark(
            benchmark,
            resources_by_scene={"hashscene": _resources()},
            scene_errors={},
            templates={},
            target_types={"object_move_object_centric"},
            legacy_only=True,
            rebalance=False,
            deduplicate_against_preserved=True,
        )

    assert output["questions"] == [preserved, other_type]
    assert audit["questions"][0]["reason"] == "duplicate_preserved_question"


def test_attachment_balance_applies_exact_scene_and_batch_caps() -> None:
    questions = []
    for index in range(8):
        question = _base_question()
        question["scene_id"] = f"scene{index % 2}"
        question["image_name"] = f"changed{index}.jpg"
        question["attachment_pair_id"] = f"1->{index + 10}"
        question["relation_unchanged"] = False
        questions.append(question)
    for index in range(6):
        question = _base_question()
        question["scene_id"] = "same_scene"
        question["image_name"] = f"unchanged{index}.jpg"
        question["attachment_pair_id"] = f"2->{index + 20}"
        question["relation_unchanged"] = True
        questions.append(question)

    from src.quality_control import balance_l2_attachment_per_scene

    kept = balance_l2_attachment_per_scene(questions)
    unchanged = [question for question in kept if question["relation_unchanged"]]
    assert len([question for question in kept if not question["relation_unchanged"]]) == 8
    assert len(unchanged) == 2  # floor(8 / 4), stricter than the per-scene cap of 3


def test_utf8_json_write_has_no_bom_and_does_not_touch_input(tmp_path: Path) -> None:
    from scripts.recompute_l2_move_benchmark import _write_json

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_bytes = b'{"immutable": true}\n'
    input_path.write_bytes(input_bytes)
    _write_json(output_path, {"text": "\u684c\u5b50"})

    assert input_path.read_bytes() == input_bytes
    output_bytes = output_path.read_bytes()
    assert not output_bytes.startswith(b"\xef\xbb\xbf")
    assert json.loads(output_bytes.decode("utf-8"))["text"] == "\u684c\u5b50"
