from collections import Counter

from src.quality_control import (
    balance_answer_distribution,
    full_quality_pipeline,
    quality_filter,
)


def _make_question(
    qtype: str,
    idx: int,
    correct_value: str,
    *,
    answer: str = "A",
) -> dict:
    options = [correct_value, f"{correct_value} distractor 1", f"{correct_value} distractor 2", f"{correct_value} distractor 3"]
    return {
        "level": "L2",
        "type": qtype,
        "scene_id": f"scene{idx:04d}_00",
        "image_name": f"{idx:06d}.jpg",
        "question": f"{qtype} question {idx}",
        "options": options,
        "answer": answer,
        "correct_value": correct_value,
        "obj_a_label": f"anchor {idx}",
        "obj_b_label": f"target {idx}",
    }


def test_full_quality_pipeline_does_not_downsample_imbalanced_correct_values() -> None:
    questions = []
    idx = 0
    value_sets = {
        "distance": ["very close", "very close", "very close", "very close", "far"],
        "direction_object_centric": ["left", "left", "left", "left", "right"],
        "direction_allocentric": ["north", "north", "north", "north", "south"],
    }
    for qtype, values in value_sets.items():
        for correct_value in values:
            questions.append(_make_question(qtype, idx, correct_value))
            idx += 1

    filtered = full_quality_pipeline(questions)

    assert len(filtered) == len(questions)
    for qtype, values in value_sets.items():
        kept_values = Counter(
            q["correct_value"] for q in filtered
            if q["type"] == qtype
        )
        assert kept_values == Counter(values)


def test_balance_answer_distribution_reorders_options_without_deleting_questions() -> None:
    questions = [
        _make_question("distance", idx, f"value {idx}")
        for idx in range(8)
    ]

    balanced = balance_answer_distribution(questions)

    assert len(balanced) == len(questions)
    assert Counter(q["answer"] for q in balanced)["A"] < len(questions)
    for q in balanced:
        assert q["options"][ord(q["answer"]) - ord("A")] == q["correct_value"]


def test_quality_filter_keeps_identical_text_when_object_ids_differ_across_frames() -> None:
    questions = [
        {
            **_make_question("coordinate_rotation_agent", idx, "left"),
            "level": "L3",
            "scene_id": "scene0001_00",
            "image_name": f"{idx:06d}.jpg",
            "question": "After rotating the room, where is the object?",
            "obj_a_id": idx * 2 + 1,
            "obj_b_id": idx * 2 + 2,
        }
        for idx in range(4)
    ]

    filtered = quality_filter(questions)

    assert len(filtered) == len(questions)


def test_quality_filter_dedups_identical_text_with_same_object_ids_across_frames() -> None:
    questions = [
        {
            **_make_question("coordinate_rotation_agent", idx, "left"),
            "level": "L3",
            "scene_id": "scene0001_00",
            "image_name": f"{idx:06d}.jpg",
            "question": "After rotating the room, where is the object?",
            "obj_a_id": 1,
            "obj_b_id": 2,
        }
        for idx in range(4)
    ]

    filtered = quality_filter(questions)

    assert len(filtered) == 1


def test_quality_filter_dedups_non_coordinate_l3_questions_across_frames() -> None:
    questions = [
        {
            **_make_question("attachment_chain", idx, "both objects"),
            "level": "L3",
            "scene_id": "scene0001_00",
            "image_name": f"{idx:06d}.jpg",
            "question": "If the desk moves, what else moves with it?",
            "grandparent_id": 1,
            "parent_id": 2,
            "grandchild_id": 3,
            "neighbor_id": 4,
        }
        for idx in range(3)
    ]

    filtered = quality_filter(questions)

    assert len(filtered) == 1


def test_full_quality_pipeline_caps_l1_occlusion_not_visible_ratio() -> None:
    questions = [
        {
            **_make_question("occlusion", idx, "not visible" if idx < 4 else "not occluded"),
            "level": "L1",
            "type": "occlusion",
            "correct_value": "not visible" if idx < 4 else "not occluded",
            "obj_a_id": idx + 1,
        }
        for idx in range(6)
    ]

    filtered = full_quality_pipeline(questions)
    occlusion_questions = [
        q for q in filtered
        if q["level"] == "L1" and q["type"] == "occlusion"
    ]
    not_visible_count = sum(
        1 for q in occlusion_questions
        if q["correct_value"] == "not visible"
    )

    assert not_visible_count / len(occlusion_questions) <= 1.0 / 3.0
