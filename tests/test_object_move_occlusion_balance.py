import unittest

from scripts.run_pipeline import _balance_scene_object_move_occlusion_negatives


def make_question(index: int, relation: str, *, valid_roles: bool = True) -> dict:
    frame_1_ids = [1, 2] if valid_roles else [1]
    return {
        "type": "object_move_occlusion",
        "image_name": "first.jpg",
        "reasoning_frame_2": "last.jpg",
        "object_frame_groups": {
            "frame_1": frame_1_ids,
            "frame_2": [100 + index],
        },
        "moved_obj_id": 1,
        "query_obj_id": 2,
        "obj_ref_id": 100 + index,
        "new_pairwise_occlusion_relation": relation,
        "new_query_blocking_ratio": index / 100.0,
        "new_ref_blocking_ratio": 0.0,
        "directed_search_score": {"predicted_coverage": index / 100.0},
    }


class ObjectMoveOcclusionBalanceTests(unittest.TestCase):
    def test_neither_count_does_not_exceed_positive_count(self) -> None:
        questions = [
            make_question(1, "query_occluded_by_reference"),
            make_question(2, "reference_occluded_by_query"),
            *[make_question(index, "neither") for index in range(3, 9)],
        ]

        kept, diagnostics = _balance_scene_object_move_occlusion_negatives(questions)

        negatives = [
            question for question in kept
            if question["new_pairwise_occlusion_relation"] == "neither"
        ]
        self.assertEqual(len(negatives), 2)
        self.assertEqual(diagnostics["positive_count"], 2)
        self.assertEqual(diagnostics["negative_dropped_count"], 4)

    def test_zero_positive_scene_keeps_at_most_three_neither(self) -> None:
        questions = [make_question(index, "neither") for index in range(1, 8)]

        kept, diagnostics = _balance_scene_object_move_occlusion_negatives(questions)

        self.assertEqual(len(kept), 3)
        self.assertEqual(diagnostics["negative_kept_count"], 3)

    def test_invalid_cross_frame_roles_are_rejected(self) -> None:
        questions = [
            make_question(1, "query_occluded_by_reference"),
            make_question(2, "neither", valid_roles=False),
        ]

        kept, diagnostics = _balance_scene_object_move_occlusion_negatives(questions)

        self.assertEqual(len(kept), 1)
        self.assertEqual(diagnostics["invalid_frame_role_dropped_count"], 1)

    def test_unknown_relation_is_not_treated_as_positive(self) -> None:
        questions = [
            make_question(1, "unsupported_relation"),
            make_question(2, "neither"),
        ]

        kept, diagnostics = _balance_scene_object_move_occlusion_negatives(questions)

        self.assertEqual([question["new_pairwise_occlusion_relation"] for question in kept], ["neither"])
        self.assertEqual(diagnostics["positive_count"], 0)
        self.assertEqual(diagnostics["invalid_frame_role_dropped_count"], 1)


if __name__ == "__main__":
    unittest.main()
