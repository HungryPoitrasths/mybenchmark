import unittest

from scripts.build_fixed_type_eval_set import sample_questions


class BuildFixedTypeEvalSetTests(unittest.TestCase):
    def test_sample_questions_filters_invalid_self_attachment_query(self) -> None:
        questions = [
            {
                "_dataset": "pilot",
                "_source_benchmark": "unused.json",
                "question_uid": "l2-invalid",
                "_rank": 1,
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "level": "L2",
                "type": "object_move_agent",
                "question": "Invalid self-query attachment row.",
                "options": ["A", "B"],
                "answer": "A",
                "attachment_remapped": True,
                "has_attachment_chain": True,
                "moved_obj_id": 10,
                "query_obj_id": 10,
                "obj_c_id": 20,
            },
            {
                "_dataset": "pilot",
                "_source_benchmark": "unused.json",
                "question_uid": "l2-valid",
                "_rank": 2,
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "level": "L2",
                "type": "object_move_agent",
                "question": "Valid child-query attachment row.",
                "options": ["A", "B"],
                "answer": "B",
                "attachment_remapped": True,
                "has_attachment_chain": True,
                "moved_obj_id": 10,
                "query_obj_id": 11,
                "obj_c_id": 20,
            },
            {
                "_dataset": "pilot",
                "_source_benchmark": "unused.json",
                "question_uid": "l1-ok",
                "_rank": 3,
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "level": "L1",
                "type": "distance",
                "question": "Normal L1 row.",
                "options": ["A", "B"],
                "answer": "A",
                "obj_a_id": 1,
                "obj_b_id": 2,
            },
        ]

        sampled, stats = sample_questions(
            questions,
            target_per_level=10,
            frame_cap=2,
            scene_cap=8,
        )

        self.assertEqual([question["question_uid"] for question in sampled], ["l1-ok", "l2-valid"])
        self.assertEqual(stats["validation_filters"]["invalid_self_attachment_query"]["total"], 1)
        self.assertEqual(
            stats["validation_filters"]["invalid_self_attachment_query"]["by_level_type"],
            {"L2": {"object_move_agent": 1}},
        )
        l2_stats = stats["levels"]["L2"]["types"]["object_move_agent"]
        self.assertEqual(l2_stats["available_raw"], 2)
        self.assertEqual(l2_stats["filtered_invalid_self_attachment_query"], 1)
        self.assertEqual(l2_stats["available_after_validation_filter"], 1)
        self.assertEqual(l2_stats["available_after_attachment_filter"], 1)
        self.assertEqual(l2_stats["sampled"], 1)


if __name__ == "__main__":
    unittest.main()
