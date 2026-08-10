from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import merge_strict_object_centric_train_addon as merge
from src.qa_generator import _object_facing_ground_axes


class FakeResource:
    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self.poses = {"frame.jpg": SimpleNamespace(position=np.array([0.0, 2.0, 1.0]))}
        self.objects_by_id = {
            1: {"id": 1, "center": [0.0, 0.0, 0.0]},
            2: {"id": 2, "center": [1.0, 0.0, 0.0]},
            3: {"id": 3, "center": [-1.0, 0.0, 0.0]},
        }


def strict_question(scene_id: str = "train-scene") -> dict:
    resource = FakeResource(scene_id)
    camera_position = resource.poses["frame.jpg"].position
    movement_forward, movement_right = _object_facing_ground_axes(
        np.array(resource.objects_by_id[1]["center"]), camera_position
    )
    answer_forward, answer_right = _object_facing_ground_axes(
        np.array(resource.objects_by_id[2]["center"]), camera_position
    )
    assert movement_forward is not None and answer_forward is not None
    return {
        "level": "L2",
        "type": "object_move_object_centric",
        "scene_id": scene_id,
        "image_name": "frame.jpg",
        "auxiliary_image_names": ["route-a.jpg", "route-b.jpg"],
        "question": "After moving the chair right by 0.5m, where is it?",
        "options": ["left", "right", "front", "behind"],
        "answer": "B",
        "correct_value": "right",
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "obj_ref_id": 3,
        "movement_semantics_version": 2,
        "movement_direction": "right",
        "movement_distance_m": 0.5,
        "delta": (movement_right * 0.5).tolist(),
        "movement_reference_frame": "moved_object_facing_first_camera",
        "movement_frame_anchor_obj_id": 1,
        "movement_camera_binding": "frame_1",
        "movement_frame_forward_world": movement_forward.tolist(),
        "movement_frame_right_world": movement_right.tolist(),
        "movement_frame_frozen": True,
        "answer_reference_frame": "query_object_facing_first_camera",
        "answer_frame_anchor_obj_id": 2,
        "answer_camera_binding": "frame_1",
        "answer_frame_forward_world": answer_forward.tolist(),
        "answer_frame_right_world": answer_right.tolist(),
        "answer_frame_frozen": True,
    }


def source(question: dict, index: int = 0) -> merge.SourceQuestion:
    return merge.SourceQuestion(Path("shard/benchmark.json"), index, question)


def document(*questions: dict) -> dict:
    return {"version": "1.0", "questions": list(questions), "statistics": {}}


def prepare(canonical: dict, validation: dict, rows: list[merge.SourceQuestion]):
    resources = {
        row.question["scene_id"]: FakeResource(row.question["scene_id"])
        for row in rows
    }
    return merge.prepare_merge(
        canonical,
        validation,
        rows,
        resources=resources,
        image_validator=lambda _record: None,
        expected_source_count=None,
        expected_existing_count=None,
        expected_append_count=None,
    )


def test_rejects_non_strict_source_question() -> None:
    question = strict_question()
    question["movement_semantics_version"] = 1

    with pytest.raises(AssertionError, match="non-v2"):
        prepare(document(), document(), [source(question)])


def test_deduplicates_only_exact_text_and_ordered_image_route() -> None:
    existing = strict_question()
    merged, audit = prepare(document(copy.deepcopy(existing)), document(), [source(existing)])

    assert merged["questions"] == [existing]
    assert audit["already_present_count"] == 1
    assert audit["appended_count"] == 0
    assert audit["already_present"][0]["reason"] == (
        "same_question_text_and_complete_ordered_image_route"
    )


def test_rejects_val_scene_or_visible_identity_overlap() -> None:
    question = strict_question()
    val_question = copy.deepcopy(question)
    val_question["scene_id"] = "different-scene"

    with pytest.raises(AssertionError, match="train/val isolation"):
        prepare(document(), document(val_question), [source(question)])


def test_preserves_existing_rows_and_appends_new_question() -> None:
    existing = {
        "level": "L1",
        "type": "direction_agent",
        "scene_id": "old-train-scene",
        "image_name": "old.jpg",
        "question": "Where is the table?",
        "options": ["left", "right", "front", "behind"],
        "answer": "A",
    }
    canonical = document(copy.deepcopy(existing))
    addition = strict_question()

    merged, audit = prepare(canonical, document(), [source(addition)])

    assert merged["questions"][0] == existing
    assert merged["questions"][1] == addition
    assert canonical["questions"] == [existing]
    assert audit["appended_count"] == 1


def test_failure_does_not_write_canonical_file(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    original = b'{"questions": [{"sentinel": true}]}\n'
    benchmark.write_bytes(original)
    question = strict_question()

    with pytest.raises(KeyError, match="missing scene resources"):
        merge.prepare_merge(
            document(),
            document(),
            [source(question)],
            resources={},
            image_validator=lambda _record: None,
            expected_source_count=None,
            expected_existing_count=None,
            expected_append_count=None,
        )

    assert benchmark.read_bytes() == original
