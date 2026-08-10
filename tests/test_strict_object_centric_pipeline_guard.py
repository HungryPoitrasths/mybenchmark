from __future__ import annotations

import json

import pytest

import scripts.run_pipeline as run_pipeline_module


def _strict_question() -> dict:
    return {
        "type": "object_move_object_centric",
        "question": "strict camera-facing move",
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "movement_semantics_version": (
            run_pipeline_module.L2_OBJECT_MOVE_SEMANTICS_VERSION
        ),
        "movement_reference_frame": "moved_object_facing_first_camera",
        "movement_camera_binding": "frame_1",
        "movement_frame_anchor_obj_id": 1,
        "movement_frame_forward_world": [1.0, 0.0, 0.0],
        "movement_frame_right_world": [0.0, -1.0, 0.0],
        "movement_frame_frozen": True,
        "answer_reference_frame": "query_object_facing_first_camera",
        "answer_camera_binding": "frame_1",
        "answer_frame_anchor_obj_id": 2,
        "answer_frame_forward_world": [0.0, 1.0, 0.0],
        "answer_frame_right_world": [1.0, 0.0, 0.0],
        "answer_frame_frozen": True,
        "movement_direction": "right",
        "movement_distance_m": 2.0,
        "delta": [0.0, -2.0, 0.0],
    }


def test_pipeline_outputs_record_strict_object_centric_profile() -> None:
    status = run_pipeline_module._build_empty_pipeline_scene_status_doc()
    benchmark = run_pipeline_module._build_benchmark_payload([])

    assert (
        status["object_move_object_centric_semantics"]
        == run_pipeline_module.OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE
    )
    assert (
        benchmark["object_move_object_centric_semantics"]
        == run_pipeline_module.OBJECT_MOVE_OBJECT_CENTRIC_SEMANTICS_PROFILE
    )


def test_scene_status_rejects_missing_strict_object_centric_profile(tmp_path) -> None:
    status_path = tmp_path / "scene_status.json"
    status_path.write_text(
        json.dumps({
            "version": run_pipeline_module.PIPELINE_SCENE_STATUS_VERSION,
            "object_move_semantics_version": (
                run_pipeline_module.L2_OBJECT_MOVE_SEMANTICS_VERSION
            ),
            "completed_scenes": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="object_move_object_centric semantics profile '<missing>'",
    ):
        run_pipeline_module._load_pipeline_scene_status_doc(status_path)


def test_pipeline_guard_accepts_camera_facing_record() -> None:
    question = _strict_question()

    run_pipeline_module._validate_strict_object_centric_questions(
        [question],
        source="test",
    )
    benchmark = run_pipeline_module._build_benchmark_payload([question])

    assert benchmark["questions"] == [question]


def test_pipeline_guard_rejects_query_reference_record() -> None:
    question = _strict_question()
    question.update({
        "movement_reference_frame": "object_centric",
        "movement_frame_query_obj_id": 2,
        "movement_frame_reference_obj_id": 3,
    })

    with pytest.raises(RuntimeError, match="legacy query-to-reference"):
        run_pipeline_module._validate_strict_object_centric_questions(
            [question],
            source="test",
        )


def test_raw_scene_cache_rejects_query_reference_record(tmp_path) -> None:
    raw_questions_dir = tmp_path / "_raw_questions_scene_cache"
    raw_questions_dir.mkdir()
    scene_id = "scene0000_00"
    question = _strict_question()
    question.update({
        "scene_id": scene_id,
        "movement_reference_frame": "object_centric",
        "movement_frame_query_obj_id": 2,
        "movement_frame_reference_obj_id": 3,
    })
    (raw_questions_dir / f"{scene_id}.json").write_text(
        json.dumps([question]),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="legacy query-to-reference"):
        run_pipeline_module._load_cached_scene_questions(
            raw_questions_dir,
            scene_ids=[scene_id],
            scene_type_cap=0,
            frame_type_cap=0,
            frame_type_object_cap=0,
        )
