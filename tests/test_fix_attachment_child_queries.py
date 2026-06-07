import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.fix_attachment_child_queries import (
    _stable_rank,
    fix_attachment_child_queries,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class FixAttachmentChildQueriesTests(unittest.TestCase):
    def test_fix_attachment_child_queries_replaces_with_exact_child_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source_benchmark.json"
            source_question = {
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "level": "L2",
                "type": "object_move_agent",
                "question": "What is the mug relative to the lamp after moving the table?",
                "options": ["left", "right", "front", "back"],
                "answer": "B",
                "correct_value": "right",
                "old_correct_value": "front",
                "new_correct_value": "right",
                "moved_obj_id": 10,
                "moved_obj_label": "table",
                "query_obj_id": 11,
                "query_obj_label": "mug",
                "obj_b_id": 11,
                "obj_b_label": "mug",
                "obj_c_id": 20,
                "obj_c_label": "lamp",
                "delta": [3.0, 0.0, 0.0],
                "mentioned_objects": [
                    {"role": "moved_object", "label": "table", "obj_id": 10},
                    {"role": "query_object", "label": "mug", "obj_id": 11},
                    {"role": "relation_obj_b", "label": "mug", "obj_id": 11},
                    {"role": "relation_obj_c", "label": "lamp", "obj_id": 20},
                ],
                "question_referability_audit": {"decision": "pass"},
                "trace_question_id": "q_child",
                "trace_reason": "attachment_agent_relation_change",
                "attachment_remapped": True,
                "has_attachment_chain": True,
                "attachment_parent_id": 10,
                "attachment_child_id": 11,
                "attachment_pair_id": "10->11",
            }
            _write_json(source_path, {"questions": [source_question]})

            payload = {
                "metadata": {"seed": 7},
                "questions": [
                    {
                        "_dataset": "pilot",
                        "_source_benchmark": str(source_path),
                        "_rank": 123,
                        "question_uid": "old_uid",
                        "scene_id": "scene0000_00",
                        "image_name": "000.jpg",
                        "level": "L2",
                        "type": "object_move_agent",
                        "question": "What is the table relative to the lamp after moving the table?",
                        "options": ["left", "right", "front", "back"],
                        "answer": "A",
                        "correct_value": "left",
                        "old_correct_value": "back",
                        "new_correct_value": "left",
                        "moved_obj_id": 10,
                        "moved_obj_label": "table",
                        "query_obj_id": 10,
                        "query_obj_label": "table",
                        "obj_b_id": 10,
                        "obj_b_label": "table",
                        "obj_c_id": 20,
                        "obj_c_label": "lamp",
                        "delta": [3.0, 0.0, 0.0],
                        "mentioned_objects": [{"role": "query_object", "label": "table", "obj_id": 10}],
                        "question_referability_audit": {"decision": "manual_review"},
                        "trace_question_id": "q_bad",
                        "trace_reason": "attachment_agent_relation_preserved_fallback",
                        "attachment_remapped": True,
                        "has_attachment_chain": True,
                        "attachment_parent_id": 10,
                        "attachment_child_id": 11,
                        "attachment_pair_id": "10->11",
                    }
                ],
            }

            fixed_payload, report = fix_attachment_child_queries(payload)

        self.assertEqual(report["target_count"], 1)
        self.assertEqual(report["fixed_count"], 1)
        self.assertEqual(report["skipped_count"], 0)

        fixed_question = fixed_payload["questions"][0]
        self.assertEqual(fixed_question["question"], source_question["question"])
        self.assertEqual(fixed_question["query_obj_id"], 11)
        self.assertEqual(fixed_question["query_obj_label"], "mug")
        self.assertEqual(fixed_question["obj_b_id"], 11)
        self.assertEqual(fixed_question["answer"], "B")
        self.assertEqual(fixed_question["trace_question_id"], "q_child")
        self.assertEqual(fixed_question["_dataset"], "pilot")
        self.assertEqual(fixed_question["_source_benchmark"], str(source_path))
        self.assertEqual(fixed_question["attachment_parent_id"], 10)
        self.assertEqual(fixed_question["attachment_child_id"], 11)
        self.assertEqual(fixed_question["attachment_pair_id"], "10->11")
        self.assertNotEqual(fixed_question["question_uid"], "old_uid")
        self.assertEqual(fixed_question["_rank"], _stable_rank(7, fixed_question["question_uid"]))

    def test_fix_attachment_child_queries_reports_no_exact_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source_benchmark.json"
            _write_json(
                source_path,
                {
                    "questions": [
                        {
                            "scene_id": "scene0000_00",
                            "image_name": "000.jpg",
                            "level": "L2",
                            "type": "object_move_agent",
                            "question": "Different child and reference object.",
                            "moved_obj_id": 10,
                            "query_obj_id": 11,
                            "obj_c_id": 99,
                            "delta": [3.0, 0.0, 0.0],
                        }
                    ]
                },
            )
            original_question = {
                "_dataset": "pilot",
                "_source_benchmark": str(source_path),
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "level": "L2",
                "type": "object_move_agent",
                "question": "Broken self-query row.",
                "moved_obj_id": 10,
                "query_obj_id": 10,
                "obj_c_id": 20,
                "delta": [3.0, 0.0, 0.0],
                "attachment_remapped": True,
                "has_attachment_chain": True,
                "attachment_child_id": 11,
                "trace_question_id": "q_bad",
            }

            fixed_payload, report = fix_attachment_child_queries({"questions": [original_question]})

        self.assertEqual(report["target_count"], 1)
        self.assertEqual(report["fixed_count"], 0)
        self.assertEqual(report["skipped_count"], 1)
        self.assertEqual(report["skipped"][0]["skip_reason"], "no_exact_child_match")
        self.assertEqual(fixed_payload["questions"][0], original_question)

    def test_main_writes_output_and_metadata_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "benchmark_subset.json"
            output_path = root / "benchmark_subset.fixed.json"
            report_path = root / "benchmark_subset.fix_report.json"
            source_path = root / "source_benchmark.json"
            _write_json(
                source_path,
                {
                    "questions": [
                        {
                            "scene_id": "scene0000_00",
                            "image_name": "000.jpg",
                            "level": "L2",
                            "type": "object_move_agent",
                            "question": "What is the mug relative to the lamp after moving the table?",
                            "options": ["left", "right"],
                            "answer": "B",
                            "moved_obj_id": 10,
                            "query_obj_id": 11,
                            "obj_b_id": 11,
                            "obj_c_id": 20,
                            "delta": [3.0, 0.0, 0.0],
                        }
                    ]
                },
            )
            _write_json(
                input_path,
                {
                    "metadata": {"seed": 5},
                    "questions": [
                        {
                            "_dataset": "pilot",
                            "_source_benchmark": str(source_path),
                            "scene_id": "scene0000_00",
                            "image_name": "000.jpg",
                            "level": "L2",
                            "type": "object_move_agent",
                            "question": "Broken self-query row.",
                            "options": ["left", "right"],
                            "answer": "A",
                            "moved_obj_id": 10,
                            "query_obj_id": 10,
                            "obj_b_id": 10,
                            "obj_c_id": 20,
                            "delta": [3.0, 0.0, 0.0],
                            "attachment_remapped": True,
                            "has_attachment_chain": True,
                            "attachment_child_id": 11,
                        }
                    ],
                },
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "fix_attachment_child_queries.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ],
            ):
                main()

            written_payload = json.loads(output_path.read_text(encoding="utf-8"))
            written_report = json.loads(report_path.read_text(encoding="utf-8"))

        summary = written_payload["metadata"]["postprocess"]["self_attachment_query_fix"]
        self.assertEqual(summary["input_path"], str(input_path))
        self.assertEqual(summary["output_path"], str(output_path))
        self.assertEqual(summary["report_path"], str(report_path))
        self.assertEqual(summary["fixed_count"], 1)
        self.assertEqual(written_report["fixed_count"], 1)


if __name__ == "__main__":
    unittest.main()
