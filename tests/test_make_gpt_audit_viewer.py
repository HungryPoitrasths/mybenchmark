import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.audit_benchmark_gpt import CHECK_NAMES, question_fingerprint
from scripts.make_gpt_audit_viewer import build_viewer_html, generate_viewer, join_flagged_questions


def flagged_result(question: dict, source_index: int = 0) -> dict:
    checks = {}
    for name in CHECK_NAMES:
        verdict = "fail" if name == "continuity" else ("pass" if name in {"referability", "fairness"} else "not_applicable")
        checks[name] = {
            "verdict": verdict,
            "summary_zh": "两张图之间跳变明显" if verdict == "fail" else "通过",
            "issues": (
                [{"code": "large_viewpoint_jump", "message_zh": "视角跳变过大", "image_indices": [1, 2], "object_labels": []}]
                if verdict == "fail" else []
            ),
        }
    stage = {"model": "gpt-5.2", "status": "ok", "result": {"checks": checks}, "validation_errors": [], "error": None}
    return {
        "source_index": source_index,
        "question_fingerprint": question_fingerprint(question, source_index),
        "final_status": "flagged",
        "final_source": "review",
        "applicable_checks": ["referability", "continuity", "fairness"],
        "problem_checks": ["continuity"],
        "primary_result": {**stage, "model": "gpt-4.1-mini"},
        "review_result": stage,
        "final_result": stage,
    }


class MakeGptAuditViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "scene_id": "abc12345",
            "level": "L2",
            "type": "object_move_agent",
            "image_name": "first.jpg",
            "auxiliary_image_names": ["bridge.jpg"],
            "reasoning_frame_2": "last.jpg",
            "question": "Where is the lamp?",
            "options": ["left", "right"],
            "answer": "B",
            "correct_value": "right",
        }

    def test_join_rejects_changed_benchmark(self) -> None:
        result = flagged_result(self.question)
        changed = {**self.question, "question": "Changed"}
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            join_flagged_questions([changed], [result])

    def test_join_excludes_non_flagged_results(self) -> None:
        result = flagged_result(self.question)
        result["final_status"] = "passed"
        self.assertEqual(join_flagged_questions([self.question], [result]), [])

    def test_html_contains_all_images_issues_traces_and_edit_controls(self) -> None:
        result = flagged_result(self.question)
        with mock.patch("scripts.make_gpt_audit_viewer._image_data_url", return_value="data:image/jpeg;base64,encoded"):
            rendered = build_viewer_html(
                [(self.question, result)],
                title="Audit",
                output_filename="review.html",
                scannet_roots=[],
                scannetpp_roots=[Path("images")],
            )
        self.assertEqual(rendered.count('data:image/jpeg;base64,encoded'), 3)
        self.assertIn("首张主图", rendered)
        self.assertIn("过渡图", rendered)
        self.assertIn("末张主图", rendered)
        self.assertIn("Where is the lamp?", rendered)
        self.assertIn("视角跳变过大", rendered)
        self.assertIn("GPT-4.1-mini 初审", rendered)
        self.assertIn("GPT-5.2 复核", rendered)
        self.assertIn('data-deleted="false"', rendered)
        self.assertIn("Export Edited HTML", rendered)
        self.assertIn("review_edited.html", rendered)
        self.assertIn('id="category"', rendered)
        self.assertIn('id="qtype"', rendered)

    def test_generate_viewer_writes_self_contained_html(self) -> None:
        result = flagged_result(self.question)
        with tempfile.TemporaryDirectory(dir="tests") as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmark.json"
            audit_path = root / "flagged.json"
            output_path = root / "review.html"
            benchmark_path.write_text(
                json.dumps({"questions": [self.question]}), encoding="utf-8"
            )
            audit_path.write_text(
                json.dumps({"results": [result]}), encoding="utf-8"
            )
            with mock.patch(
                "scripts.make_gpt_audit_viewer._image_data_url",
                return_value="data:image/jpeg;base64,encoded",
            ):
                stats = generate_viewer(
                    benchmark_path=benchmark_path,
                    audit_path=audit_path,
                    output_path=output_path,
                    scannet_roots=[],
                    scannetpp_roots=[root / "images"],
                )
            rendered = output_path.read_text(encoding="utf-8")
        self.assertEqual(stats, {"flagged": 1, "source_questions": 1})
        self.assertIn("GPT 审核问题复核", rendered)
        self.assertIn('data-source-index="0"', rendered)


if __name__ == "__main__":
    unittest.main()
