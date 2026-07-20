from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_pipeline_frame_caps",
    PROJECT_ROOT / "scripts" / "run_pipeline.py",
)
assert SPEC is not None and SPEC.loader is not None
run_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_pipeline)


def _question(
    *,
    level: str,
    question_type: str,
    image_name: str,
    object_id: int,
    index: int,
    scene_id: str = "scene0000_00",
) -> dict:
    return {
        "scene_id": scene_id,
        "level": level,
        "type": question_type,
        "image_name": image_name,
        "query_obj_id": object_id,
        "question": f"question {index}",
    }


def test_l2_caps_each_object_at_two_without_capping_the_frame_type_total() -> None:
    questions = []
    index = 0
    for object_id in (7, 8):
        for _ in range(4):
            questions.append(
                _question(
                    level="L2",
                    question_type="object_move_agent",
                    image_name="000100.jpg",
                    object_id=object_id,
                    index=index,
                )
            )
            index += 1

    kept = run_pipeline._apply_scene_type_cap(
        questions,
        scene_type_cap=0,
        frame_type_cap=0,
        frame_type_object_cap=0,
    )

    assert [question["question"] for question in kept] == [
        "question 0",
        "question 1",
        "question 4",
        "question 5",
    ]


def test_l3_caps_each_object_at_one_without_capping_the_frame_type_total() -> None:
    questions = [
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name="000100.jpg",
            object_id=7,
            index=0,
        ),
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name="000100.jpg",
            object_id=7,
            index=1,
        ),
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name="000100.jpg",
            object_id=8,
            index=2,
        ),
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name="000100.jpg",
            object_id=9,
            index=3,
        ),
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name="000100.jpg",
            object_id=10,
            index=4,
        ),
    ]

    kept = run_pipeline._apply_scene_type_cap(
        questions,
        scene_type_cap=0,
        frame_type_cap=0,
        frame_type_object_cap=0,
    )

    assert [question["question"] for question in kept] == [
        "question 0",
        "question 2",
        "question 3",
        "question 4",
    ]


def test_l2_l3_object_caps_are_independent_per_scene_frame_and_type() -> None:
    questions = []
    index = 0
    for image_name in ("000100.jpg", "000200.jpg"):
        for question_type in ("object_move_distance", "object_move_occlusion"):
            for object_id in (7, 8, 9):
                questions.append(
                    _question(
                        level="L2",
                        question_type=question_type,
                        image_name=image_name,
                        object_id=object_id,
                        index=index,
                    )
                )
                index += 1

    kept = run_pipeline._apply_scene_type_cap(
        questions,
        scene_type_cap=0,
        frame_type_cap=0,
        frame_type_object_cap=0,
    )

    counts: dict[tuple[str, str], int] = {}
    for question in kept:
        key = (question["image_name"], question["type"])
        counts[key] = counts.get(key, 0) + 1

    assert counts == {
        ("000100.jpg", "object_move_distance"): 3,
        ("000100.jpg", "object_move_occlusion"): 3,
        ("000200.jpg", "object_move_distance"): 3,
        ("000200.jpg", "object_move_occlusion"): 3,
    }

    other_scene = [
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name="000100.jpg",
            object_id=7,
            index=index,
            scene_id=scene_id,
        )
        for index, scene_id in enumerate(("scene0000_00", "scene0001_00"))
    ]
    assert len(
        run_pipeline._apply_scene_type_cap(
            other_scene,
            scene_type_cap=0,
            frame_type_cap=0,
            frame_type_object_cap=0,
        )
    ) == 2


def test_attachment_chain_uses_grandparent_as_its_primary_object() -> None:
    questions = [
        {
            "scene_id": "scene0000_00",
            "level": "L3",
            "type": "attachment_chain",
            "image_name": "000100.jpg",
            "grandparent_id": 7,
            "parent_id": parent_id,
            "grandchild_id": parent_id + 100,
            "question": f"chain {parent_id}",
        }
        for parent_id in (8, 9, 10)
    ]

    kept = run_pipeline._apply_scene_type_cap(
        questions,
        scene_type_cap=0,
        frame_type_cap=0,
        frame_type_object_cap=0,
    )

    assert [question["question"] for question in kept] == ["chain 8"]


def test_split_scene_question_hard_caps() -> None:
    assert run_pipeline._split_scene_question_hard_cap("val") == 50
    assert run_pipeline._split_scene_question_hard_cap("train") == 100
    assert run_pipeline._split_scene_question_hard_cap("all") == 0
    assert run_pipeline._split_scene_question_hard_cap(None) == 0


def test_hard_cap_applies_to_every_scene_question_type() -> None:
    questions = [
        _question(
            level="L2",
            question_type="object_move_agent",
            image_name=f"{index:06d}.jpg",
            object_id=index,
            index=index,
        )
        for index in range(55)
    ] + [
        _question(
            level="L3",
            question_type="coordinate_rotation_agent",
            image_name=f"{index:06d}.jpg",
            object_id=index,
            index=100 + index,
        )
        for index in range(55)
    ]

    kept = run_pipeline._apply_scene_type_cap(
        questions,
        scene_type_cap=0,
        frame_type_cap=0,
        frame_type_object_cap=0,
        scene_question_hard_cap=50,
    )

    kept_counts = Counter(question["type"] for question in kept)
    assert kept_counts == {
        "object_move_agent": 50,
        "coordinate_rotation_agent": 50,
    }


def test_hard_cap_budget_reaches_zero_for_early_stop() -> None:
    budgets = run_pipeline._remaining_scene_type_budgets(
        Counter({"object_move_agent": 50, "attachment_move": 49}),
        scene_type_cap=8,
        allowed_types={"object_move_agent", "attachment_move"},
        scene_question_hard_cap=50,
    )

    assert budgets == {"attachment_move": 1, "object_move_agent": 0}
