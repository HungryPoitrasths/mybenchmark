from __future__ import annotations

from scripts.finalize_diverse_strict_val import (
    DOMINANT_COUNT,
    DOMINANT_PAIR,
    DOMINANT_SCENE,
    _select_dominant,
)
from scripts.repair_strict_l2_move_chain import _visible_identity


def _question(index: int) -> dict:
    return {
        "type": "object_move_object_centric",
        "scene_id": DOMINANT_SCENE,
        "attachment_pair_id": DOMINANT_PAIR,
        "question": f"question-{index}",
        "image_name": f"first-{index % 5}.jpg",
        "reasoning_frame_2": f"last-{index % 7}.jpg",
        "obj_ref_id": index % 11,
        "movement_direction": ("forward", "right", "forward-left")[index % 3],
        "movement_distance_m": (1.0, 1.5, 2.0, 2.5)[index % 4],
        "correct_value": ("front", "right", "back", "left")[index % 4],
    }


def test_dominant_pair_selection_is_deterministic_and_keeps_mandatory() -> None:
    candidates = [_question(index) for index in range(86)]
    mandatory = {_visible_identity(q) for q in candidates[:14]}

    first = _select_dominant(candidates, mandatory)
    second = _select_dominant(list(reversed(candidates)), mandatory)

    assert len(first) == DOMINANT_COUNT
    assert mandatory <= first
    assert first == second
    assert len({q["obj_ref_id"] for q in candidates if _visible_identity(q) in first}) == 11
