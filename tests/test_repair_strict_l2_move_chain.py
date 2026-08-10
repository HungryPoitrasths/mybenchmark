from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.repair_strict_l2_move_chain import (
    _apply_trusted_attachment_evidence,
    _load_trusted_attachment_evidence,
    _sample_exact_per_type,
    _validate_strict_questions,
    _visible_identity,
)
from scripts.recompute_l2_move_benchmark import SceneResources


def _strict_question() -> dict:
    distance = 2.5
    diagonal = distance / np.sqrt(2.0)
    return {
        "level": "L2",
        "type": "object_move_object_centric",
        "scene_id": "scene-a",
        "image_name": "first.jpg",
        "reasoning_frame_2": "last.jpg",
        "question": "Strict camera-facing question",
        "options": ["front", "right", "back", "left"],
        "answer": "B",
        "correct_value": "right",
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "obj_ref_id": 3,
        "movement_semantics_version": 2,
        "movement_direction": "forward-right",
        "movement_distance_m": distance,
        "delta": [diagonal, diagonal, 0.0],
        "movement_reference_frame": "moved_object_facing_first_camera",
        "movement_frame_anchor_obj_id": 1,
        "movement_camera_binding": "frame_1",
        "movement_frame_forward_world": [0.0, 1.0, 0.0],
        "movement_frame_right_world": [1.0, 0.0, 0.0],
        "movement_frame_frozen": True,
        "answer_reference_frame": "query_object_facing_first_camera",
        "answer_frame_anchor_obj_id": 2,
        "answer_camera_binding": "frame_1",
        "answer_frame_forward_world": [-1.0, 0.0, 0.0],
        "answer_frame_right_world": [0.0, 1.0, 0.0],
        "answer_frame_frozen": True,
    }


def test_strict_validator_checks_delta_against_frozen_movement_axes() -> None:
    question = _strict_question()
    _validate_strict_questions([question])

    invalid = copy.deepcopy(question)
    invalid["delta"] = [2.5, 0.0, 0.0]
    with pytest.raises(AssertionError, match="delta does not match"):
        _validate_strict_questions([invalid])


def test_per_type_sample_contains_only_exact_canonical_text_and_routes() -> None:
    canonical_questions = []
    for question_type in ("object_move_agent", "object_move_object_centric"):
        for index in range(3):
            question = {
                "type": question_type,
                "level": "L2",
                "scene_id": "scene-a",
                "image_name": f"{question_type}-{index}-first.jpg",
                "reasoning_frame_2": f"{question_type}-{index}-last.jpg",
                "question": f"canonical {question_type} {index}",
                "options": ["front"],
                "answer": "A",
                "correct_value": "front",
            }
            canonical_questions.append(question)
    canonical = {"questions": canonical_questions}
    source = {
        "questions": [
            copy.deepcopy(canonical_questions[0]),
            {**canonical_questions[3], "question": "obsolete simplified wording"},
        ]
    }

    sampled, report = _sample_exact_per_type(source, canonical=canonical, per_type=2)

    assert len(sampled["questions"]) == 4
    canonical_identities = {_visible_identity(q) for q in canonical_questions}
    assert all(_visible_identity(q) in canonical_identities for q in sampled["questions"])
    assert report["object_move_agent"]["preserved_exact"] == 1
    assert report["object_move_object_centric"]["preserved_exact"] == 0


def test_trusted_attachment_pass_adds_missing_graph_edge(tmp_path) -> None:
    result = {
        "schema_version": "l2-attachment-audit-result-v1",
        "benchmark_sha256": "source-hash",
        "check_id": "check-1",
        "scene_id": "scene-a",
        "frame_key": "scene-a|first.jpg",
        "parent": {"obj_id": 45},
        "child": {"obj_id": 44},
        "relation": "supported_by",
        "verdict": "pass",
        "confidence": 0.99,
    }
    path = tmp_path / "result.json"
    path.write_text(__import__("json").dumps(result), encoding="utf-8")
    evidence = _load_trusted_attachment_evidence(
        tmp_path, expected_benchmark_sha256="source-hash"
    )
    resource = SceneResources(
        scene_id="scene-a",
        dataset="scannetpp",
        scene_dir=None,
        metadata_path=None,
        objects=[],
        attachment_graph={45: []},
        room_bounds=None,
        poses={},
        distance_geometry="not_needed",
    )

    report = _apply_trusted_attachment_evidence({"scene-a": resource}, evidence)

    assert resource.attachment_graph == {45: [44]}
    assert resource.trusted_attachment_evidence[(45, 44)]["check_id"] == "check-1"
    assert report["applied_missing_edges"] == 1


def test_trusted_attachment_rejects_stale_benchmark_hash(tmp_path) -> None:
    result = {
        "schema_version": "l2-attachment-audit-result-v1",
        "benchmark_sha256": "stale-hash",
        "check_id": "check-1",
        "scene_id": "scene-a",
        "parent": {"obj_id": 1},
        "child": {"obj_id": 2},
        "verdict": "pass",
    }
    (tmp_path / "result.json").write_text(
        __import__("json").dumps(result), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="benchmark hash mismatch"):
        _load_trusted_attachment_evidence(
            tmp_path, expected_benchmark_sha256="current-hash"
        )
