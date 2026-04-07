import unittest

from scripts.make_scene_debug_viewer import (
    build_frames_from_debug_doc,
    normalize_question_items,
    render_questions,
)


class MakeSceneDebugViewerTests(unittest.TestCase):
    def test_normalize_question_items_preserves_referability_audit(self) -> None:
        items = normalize_question_items(
            [
                {
                    "level": "L1",
                    "type": "direction_agent",
                    "question": "Where is the chair relative to the curtain?",
                    "options": ["left", "right"],
                    "answer": "A",
                    "question_referability_audit": {
                        "decision": "drop",
                        "reason_codes": ["mentioned_label_not_unique"],
                        "frame_referable_object_ids": [1],
                        "mentioned_objects": [
                            {
                                "role": "reference",
                                "label": "curtain",
                                "candidate_object_ids": [2, 3],
                                "referable_object_ids": [],
                                "passes_referability_check": False,
                                "reason_codes": ["mentioned_label_not_unique"],
                            }
                        ],
                    },
                }
            ],
            question_limit_per_frame=10,
        )

        self.assertEqual(items[0]["question_referability_audit"]["decision"], "drop")
        self.assertEqual(
            items[0]["question_referability_audit"]["mentioned_objects"][0]["candidate_object_ids"],
            [2, 3],
        )

    def test_build_frames_from_debug_doc_prefers_generated_questions_for_audit(self) -> None:
        frames = build_frames_from_debug_doc(
            {
                "frames": [
                    {
                        "image_name": "000123.jpg",
                        "generated_questions": [
                            {
                                "level": "L1",
                                "type": "direction_agent",
                                "question": "generated question",
                                "options": ["left", "right"],
                                "answer": "A",
                                "question_referability_audit": {
                                    "decision": "drop",
                                    "reason_codes": ["mentioned_label_not_unique"],
                                    "frame_referable_object_ids": [1],
                                    "mentioned_objects": [],
                                },
                            }
                        ],
                        "final_questions": [
                            {
                                "level": "L1",
                                "type": "direction_agent",
                                "question": "final question",
                                "options": ["left", "right"],
                                "answer": "A",
                            }
                        ],
                    }
                ]
            },
            scene_dir=None,
            scene_questions={},
            question_limit_per_frame=10,
            image_mode="inline",
        )

        self.assertEqual(frames[0]["questions"][0]["question"], "generated question")
        self.assertEqual(
            frames[0]["questions"][0]["question_referability_audit"]["decision"],
            "drop",
        )

    def test_render_questions_includes_referability_audit(self) -> None:
        html = render_questions(
            {
                "questions": [
                    {
                        "level": "L1",
                        "type": "direction_agent",
                        "question": "Where is the chair relative to the curtain?",
                        "options": ["left", "right"],
                        "answer": "A",
                        "prediction": None,
                        "prediction_correct": None,
                        "question_referability_audit": {
                            "decision": "drop",
                            "reason_codes": ["mentioned_label_not_unique"],
                            "frame_referable_object_ids": [1],
                            "mentioned_objects": [
                                {
                                    "role": "reference",
                                    "label": "curtain",
                                    "obj_id": None,
                                    "label_status": "multiple",
                                    "candidate_object_ids": [2, 3],
                                    "referable_object_ids": [],
                                    "passes_referability_check": False,
                                    "reason_codes": ["mentioned_label_not_unique"],
                                }
                            ],
                        },
                    }
                ]
            }
        )

        self.assertIn("referability drop", html)
        self.assertIn("reason codes: mentioned_label_not_unique", html)
        self.assertIn("reference: label=curtain", html)


if __name__ == "__main__":
    unittest.main()
