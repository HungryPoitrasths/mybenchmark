import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from scripts.recompute_attachment_child_queries import (
    main,
    recompute_attachment_child_queries,
)
from src.utils.colmap_loader import CameraPose


class RecomputeAttachmentChildQueriesTests(unittest.TestCase):
    def _base_target_question(self, *, image_name: str, trace_question_id: str) -> dict:
        return {
            "_dataset": "0-9",
            "_source_benchmark": "output/scannetpp_polit/0-9/benchmark.json",
            "_rank": 10,
            "question_uid": f"old-{trace_question_id}",
            "trace_question_id": trace_question_id,
            "scene_id": "0d2ee665be",
            "image_name": image_name,
            "level": "L2",
            "type": "object_move_distance",
            "question": "From the camera's perspective, if the table is moved right by 3.0m, what is the approximate shortest distance between the table and the shelf, measured from their closest points?",
            "options": [
                "very close (<1.0m)",
                "close (1.0-2.0m)",
                "moderate (2.0-3.3m)",
                "far (>3.3m)",
            ],
            "answer": "D",
            "correct_value": "far (>3.3m)",
            "moved_obj_id": 10,
            "moved_obj_label": "table",
            "query_obj_id": 10,
            "query_obj_label": "table",
            "obj_b_id": 10,
            "obj_b_label": "table",
            "obj_c_id": 30,
            "obj_c_label": "shelf",
            "delta": [3.0, 0.0, 0.0],
            "attachment_remapped": True,
            "has_attachment_chain": True,
            "attachment_parent_id": 10,
            "attachment_child_id": 11,
            "attachment_pair_id": "10->11",
        }

    def _scene_context(self) -> dict:
        objects = [
            {
                "id": 10,
                "label": "table",
                "center": [0.0, 0.0, 0.5],
                "bbox_min": [-0.5, -0.5, 0.0],
                "bbox_max": [0.5, 0.5, 1.0],
            },
            {
                "id": 11,
                "label": "cup",
                "center": [0.0, 0.0, 1.1],
                "bbox_min": [-0.1, -0.1, 1.0],
                "bbox_max": [0.1, 0.1, 1.2],
            },
            {
                "id": 30,
                "label": "shelf",
                "center": [5.0, 0.0, 0.5],
                "bbox_min": [4.5, -0.5, 0.0],
                "bbox_max": [5.5, 0.5, 1.0],
            },
        ]
        obj_map = {int(obj["id"]): obj for obj in objects}
        return {
            "dataset": "scannetpp",
            "scene_dir": Path("/fake/0d2ee665be"),
            "scene": {"objects": objects},
            "objects": objects,
            "obj_map": obj_map,
            "attachment_graph": {10: [11]},
            "poses": {
                "frame_000000.jpg": CameraPose(
                    image_name="frame_000000.jpg",
                    rotation=np.eye(3, dtype=np.float64),
                    translation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                )
                ,
                "frame_000001.jpg": CameraPose(
                    image_name="frame_000001.jpg",
                    rotation=np.eye(3, dtype=np.float64),
                    translation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                )
            },
        }

    def test_recompute_attachment_child_queries_updates_distance_question(self) -> None:
        payload = {
            "metadata": {"seed": 7},
            "questions": [
                self._base_target_question(image_name="frame_000000.jpg", trace_question_id="t0")
            ],
        }

        with mock.patch(
            "scripts.recompute_attachment_child_queries._load_scene_context",
            return_value=self._scene_context(),
        ):
            fixed_payload, report = recompute_attachment_child_queries(
                payload,
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
            )

        self.assertEqual(report["target_count"], 1)
        self.assertEqual(report["fixed_count"], 1)
        self.assertEqual(report["skipped_count"], 0)
        self.assertEqual(report["processed_count"], 1)
        self.assertEqual(report["restored_count"], 0)
        fixed = fixed_payload["questions"][0]
        self.assertEqual(fixed["query_obj_id"], 11)
        self.assertEqual(fixed["query_obj_label"], "cup")
        self.assertEqual(fixed["obj_b_id"], 11)
        self.assertTrue(fixed["child_query_recomputed"])
        self.assertIn("the cup", fixed["question"])
        self.assertIn("the table", fixed["question"])
        self.assertEqual(fixed["options"][ord(fixed["answer"]) - 65], fixed["correct_value"])
        self.assertNotEqual(fixed["question_uid"], "old")

    def test_recompute_attachment_child_queries_reports_missing_pose(self) -> None:
        payload = {
            "questions": [
                {
                    "_source_benchmark": "output/scannetpp_polit/0-9/benchmark.json",
                    "scene_id": "0d2ee665be",
                    "image_name": "missing.jpg",
                    "level": "L2",
                    "type": "object_move_distance",
                    "moved_obj_id": 10,
                    "query_obj_id": 10,
                    "attachment_child_id": 11,
                    "attachment_remapped": True,
                }
            ]
        }

        with mock.patch(
            "scripts.recompute_attachment_child_queries._load_scene_context",
            return_value=self._scene_context(),
        ):
            fixed_payload, report = recompute_attachment_child_queries(
                payload,
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
            )

        self.assertEqual(report["fixed_count"], 0)
        self.assertEqual(report["skipped_count"], 1)
        self.assertEqual(report["processed_count"], 1)
        self.assertEqual(report["skipped"][0]["skip_reason"], "pose_missing_for_frame")
        self.assertEqual(fixed_payload["questions"][0]["query_obj_id"], 10)

    def test_recompute_attachment_child_queries_resume_skips_completed_question(self) -> None:
        payload = {
            "metadata": {"seed": 7},
            "questions": [
                self._base_target_question(image_name="frame_000000.jpg", trace_question_id="t0"),
                self._base_target_question(image_name="frame_000001.jpg", trace_question_id="t1"),
            ],
        }

        with mock.patch(
            "scripts.recompute_attachment_child_queries._load_scene_context",
            return_value=self._scene_context(),
        ):
            first_output, first_report = recompute_attachment_child_queries(
                {"metadata": {"seed": 7}, "questions": [payload["questions"][0]]},
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
            )

        resume_output = {"metadata": {"seed": 7}, "questions": [first_output["questions"][0]]}
        resume_report = {"processed": [first_report["processed"][0]]}

        logs: list[str] = []
        with mock.patch(
            "scripts.recompute_attachment_child_queries._load_scene_context",
            return_value=self._scene_context(),
        ) as mocked_loader:
            fixed_payload, report = recompute_attachment_child_queries(
                payload,
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
                resume_output_payload=resume_output,
                resume_report=resume_report,
                progress_every=1,
                log=logs.append,
            )

        self.assertEqual(mocked_loader.call_count, 1)
        self.assertEqual(report["target_count"], 2)
        self.assertEqual(report["fixed_count"], 2)
        self.assertEqual(report["restored_count"], 1)
        self.assertEqual(report["processed_count"], 2)
        self.assertIn("resume", logs[1])
        self.assertTrue(fixed_payload["questions"][0]["child_query_recomputed"])
        self.assertTrue(fixed_payload["questions"][1]["child_query_recomputed"])

    def test_recompute_attachment_child_queries_checkpoint_and_progress_logging(self) -> None:
        payload = {
            "metadata": {"seed": 7},
            "questions": [self._base_target_question(image_name="frame_000000.jpg", trace_question_id="t0")],
        }
        checkpoints: list[tuple[dict, dict]] = []
        logs: list[str] = []

        with mock.patch(
            "scripts.recompute_attachment_child_queries._load_scene_context",
            return_value=self._scene_context(),
        ):
            _, report = recompute_attachment_child_queries(
                payload,
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
                progress_every=1,
                checkpoint_every=1,
                log=logs.append,
                checkpoint_callback=lambda output_payload, checkpoint_report: checkpoints.append((output_payload, checkpoint_report)),
            )

        self.assertEqual(report["fixed_count"], 1)
        self.assertEqual(len(checkpoints), 1)
        checkpoint_payload, checkpoint_report = checkpoints[0]
        self.assertEqual(len(checkpoint_payload["questions"]), 1)
        self.assertEqual(checkpoint_report["processed_count"], 1)
        self.assertTrue(any(message.startswith("[scene ") for message in logs))
        self.assertTrue(any(message.startswith("[progress] 1/1") for message in logs))

    def test_main_writes_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "subset.json"
            output_path = root / "subset.fixed.json"
            report_path = root / "subset.report.json"
            input_path.write_text(
                json.dumps(
                    {
                        "metadata": {"seed": 7},
                        "questions": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "recompute_attachment_child_queries.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ],
            ):
                main()

            written = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        summary = written["metadata"]["postprocess"]["self_attachment_child_recompute"]
        self.assertEqual(summary["fixed_count"], 0)
        self.assertEqual(report["target_count"], 0)
        self.assertEqual(summary["restored_count"], 0)


if __name__ == "__main__":
    unittest.main()
