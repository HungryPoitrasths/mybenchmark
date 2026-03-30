import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import scripts.run_pipeline as run_pipeline_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose

TEST_TMP_ROOT = Path(__file__).resolve().parent / "_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


def make_camera_pose(image_name: str) -> CameraPose:
    return CameraPose(
        image_name=image_name,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )


def make_object(obj_id: int, label: str) -> dict:
    return {
        "id": obj_id,
        "label": label,
        "center": [0.0, 0.0, 1.0],
        "bbox_min": [-0.1, -0.1, 0.9],
        "bbox_max": [0.1, 0.1, 1.1],
    }


def make_case_dir(prefix: str) -> Path:
    path = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class RunPipelineReferabilityTests(unittest.TestCase):
    def test_collect_question_presence_labels_prefers_mentioned_objects(self) -> None:
        labels = run_pipeline_module._collect_question_presence_labels(
            {
                "mentioned_objects": [
                    {"label": "Cup"},
                    {"label": "cup"},
                    {"label": "Table"},
                ],
                "obj_a_label": "Chair",
                "obj_b_label": "Lamp",
            }
        )

        self.assertEqual(labels, ["Cup", "Table"])

    def test_load_referability_cache_rejects_old_version(self) -> None:
        case_dir = make_case_dir("cache")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "3.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "expected 5.0"):
            run_pipeline_module._load_referability_cache(cache_path)

    def test_run_pipeline_requires_referability_cache(self) -> None:
        root = make_case_dir("pipeline_requires_cache")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        data_root.mkdir(parents=True)

        with self.assertRaisesRegex(ValueError, "requires a referability_cache"):
            run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                use_occlusion=False,
                referability_cache=None,
                write_frame_debug=False,
            )

    def test_run_pipeline_uses_cached_candidate_pool_directly(self) -> None:
        root = make_case_dir("pipeline")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "5.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [2, 1],
                        "referable_object_ids": [1],
                        "label_counts": {"cup": 1, "table": 1},
                        "candidate_labels": ["cup", "table"],
                        "label_to_object_ids": {"cup": [1], "table": [2]},
                    }
                }
            },
        }

        scene = {
            "scene_id": scene_id,
            "objects": [
                make_object(1, "cup"),
                make_object(2, "table"),
            ],
            "attachment_edges": [
                {"parent_id": 2, "child_id": 1, "type": "attachment"},
            ],
            "room_bounds": None,
            "wall_objects": [],
        }

        captured: dict[str, object] = {}

        def fake_generate_all_questions(**kwargs):
            captured["visible_object_ids"] = list(kwargs["visible_object_ids"])
            captured["referable_object_ids"] = list(kwargs["referable_object_ids"] or [])
            captured["label_counts"] = dict(kwargs["label_counts"] or {})
            return [
                {
                    "question": "Is the cup on the table?",
                    "answer": "A",
                    "options": ["yes", "no"],
                    "type": "attachment",
                    "level": "L1",
                }
            ]

        with (
            patch.object(run_pipeline_module, "parse_scene", return_value=scene),
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "has_nontrivial_attachment", return_value=True),
            patch.object(run_pipeline_module, "_load_scene_geometry", return_value=None),
            patch.object(run_pipeline_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(run_pipeline_module, "load_scannet_poses", return_value={image_name: make_camera_pose(image_name)}),
            patch.object(run_pipeline_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(run_pipeline_module, "load_instance_mesh_data", return_value=object()),
            patch.object(run_pipeline_module, "generate_all_questions", side_effect=fake_generate_all_questions),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                write_frame_debug=False,
            )

        self.assertEqual(captured["visible_object_ids"], [1, 2])
        self.assertEqual(captured["referable_object_ids"], [1])
        self.assertEqual(captured["label_counts"], {"cup": 1, "table": 1})
        self.assertEqual(len(questions), 1)

    def test_run_question_presence_review_writes_flagged_outputs(self) -> None:
        root = make_case_dir("presence_review")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        color_dir = data_root / scene_id / "color"
        color_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        (color_dir / image_name).write_bytes(b"not-a-real-jpeg")

        questions = [
            {
                "scene_id": scene_id,
                "image_name": image_name,
                "question": "Is the cup on the table?",
                "answer": "A",
                "options": ["yes", "no"],
                "type": "direction_agent",
                "level": "L1",
                "mentioned_objects": [
                    {"label": "cup"},
                    {"label": "table"},
                ],
            }
        ]

        def fake_builder(vlm_url, vlm_model):
            def fake_review(image_path, question, labels):
                self.assertEqual(image_path.name, image_name)
                self.assertEqual(labels, ["cup", "table"])
                return {
                    "decision": "manual_review",
                    "flagged_labels": ["cup"],
                    "object_reviews": [
                        {"label": "cup", "status": "absent", "reason": "not visible"},
                        {"label": "table", "status": "present", "reason": "visible"},
                    ],
                }

            return "fake-vlm", fake_review

        with patch.object(run_pipeline_module, "_build_question_presence_reviewer", side_effect=fake_builder):
            summary = run_pipeline_module._run_question_presence_review(
                questions=questions,
                data_root=data_root,
                output_dir=output_dir,
                vlm_url="http://example.com/v1",
                vlm_model="fake-vlm",
                workers=1,
            )

        self.assertEqual(summary["model"], "fake-vlm")
        self.assertEqual(summary["reviewed_question_count"], 1)
        self.assertEqual(summary["manual_review_count"], 1)

        review_json = json.loads((output_dir / "question_presence_review.json").read_text(encoding="utf-8"))
        flagged_json = json.loads((output_dir / "question_presence_review_flagged.json").read_text(encoding="utf-8"))
        flagged_html = (output_dir / "question_presence_review_flagged.html").read_text(encoding="utf-8")

        self.assertEqual(review_json["manual_review_count"], 1)
        self.assertEqual(flagged_json["manual_review_count"], 1)
        self.assertEqual(len(flagged_json["questions"]), 1)
        self.assertEqual(
            flagged_json["questions"][0]["manual_review_reason"],
            "VLM flagged mentioned objects: cup=absent",
        )
        self.assertIn("VLM flagged mentioned objects: cup=absent", flagged_html)


if __name__ == "__main__":
    unittest.main()
