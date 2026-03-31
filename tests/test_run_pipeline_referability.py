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


def make_fake_review_response(text: str):
    return type(
        "FakeResponse",
        (),
        {
            "choices": [
                type(
                    "FakeChoice",
                    (),
                    {"message": type("FakeMessage", (), {"content": text})()},
                )()
            ]
        },
    )()


def make_fake_review_client(text: str):
    completions = Mock()
    completions.create.return_value = make_fake_review_response(text)
    chat = Mock()
    chat.completions = completions
    client = Mock()
    client.chat = chat
    return client


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

    def test_question_answer_prompt_uses_actual_option_letters(self) -> None:
        prompt = run_pipeline_module._question_answer_prompt(
            {
                "question": "Is the cup visible?",
                "options": ["yes", "no"],
            }
        )

        self.assertIsNotNone(prompt)
        self.assertIn("Answer with a single letter only (A or B).", prompt)
        self.assertNotIn("C, or D", prompt)

    def test_parse_mcq_answer_rejects_ambiguous_multi_letter_responses(self) -> None:
        self.assertEqual(run_pipeline_module._parse_mcq_answer("Answer: B"), "B")
        self.assertIsNone(run_pipeline_module._parse_mcq_answer("A or B"))

    def test_combine_manual_review_reasons_deduplicates(self) -> None:
        combined = run_pipeline_module._combine_manual_review_reasons(
            ["same reason", "same reason", ""]
        )

        self.assertEqual(combined, "same reason")

    def test_load_referability_cache_rejects_old_version(self) -> None:
        case_dir = make_case_dir("cache")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "3.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "expected 6.0"):
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
            "version": "6.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [2, 1],
                        "referable_object_ids": [1],
                        "label_statuses": {"cup": "unique", "table": "unique"},
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
            captured["label_statuses"] = dict(kwargs["label_statuses"] or {})
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
        self.assertEqual(captured["label_statuses"], {"cup": "unique", "table": "unique"})
        self.assertEqual(captured["label_counts"], {"cup": 1, "table": 1})
        self.assertEqual(len(questions), 1)

    def test_run_pipeline_uses_dedicated_question_review_vlm_config(self) -> None:
        root = make_case_dir("pipeline_question_review_vlm")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "6.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [2, 1],
                        "referable_object_ids": [1],
                        "label_statuses": {"cup": "unique", "table": "unique"},
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
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                return_value=[
                    {
                        "scene_id": scene_id,
                        "image_name": image_name,
                        "question": "Where is the cup relative to the table?",
                        "answer": "A",
                        "options": ["left", "right", "front", "behind"],
                        "type": "direction_agent",
                        "level": "L1",
                    }
                ],
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
            patch.object(run_pipeline_module, "_run_question_presence_review") as review_mock,
        ):
            run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                question_review_vlm_url="http://question-review.example/v1",
                question_review_vlm_model="question-review-model",
                run_question_presence_review=True,
                write_frame_debug=False,
            )

        review_mock.assert_called_once()
        self.assertEqual(review_mock.call_args.kwargs["vlm_url"], "http://question-review.example/v1")
        self.assertEqual(review_mock.call_args.kwargs["vlm_model"], "question-review-model")

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

        def fake_presence_review(image_path, question, labels):
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

        def fake_answer_review(image_path, question):
            self.assertEqual(image_path.name, image_name)
            return {
                "decision": "pass",
                "predicted_answer": "A",
                "gold_answer": "A",
                "predicted_option": "yes",
                "gold_option": "yes",
                "reason": "",
                "raw_response": "A",
            }

        with (
            patch.object(run_pipeline_module, "_resolve_question_review_vlm", return_value=(object(), "fake-vlm")),
            patch.object(run_pipeline_module, "_make_question_presence_reviewer", return_value=("fake-vlm", fake_presence_review)),
            patch.object(run_pipeline_module, "_make_question_answer_reviewer", return_value=("fake-vlm", fake_answer_review)),
        ):
            summary = run_pipeline_module._run_question_presence_review(
                questions=questions,
                data_root=data_root,
                output_dir=output_dir,
                vlm_url="http://example.com/v1",
                vlm_model="fake-vlm",
                workers=1,
            )

        self.assertEqual(summary["model"], "fake-vlm")
        self.assertEqual(summary["answer_review_model"], "fake-vlm")
        self.assertEqual(summary["reviewed_question_count"], 1)
        self.assertEqual(summary["manual_review_count"], 1)
        self.assertEqual(summary["answer_review_question_count"], 1)
        self.assertEqual(summary["answer_mismatch_count"], 0)

        review_json = json.loads((output_dir / "question_presence_review.json").read_text(encoding="utf-8"))
        flagged_json = json.loads((output_dir / "question_presence_review_flagged.json").read_text(encoding="utf-8"))
        flagged_html = (output_dir / "question_presence_review_flagged.html").read_text(encoding="utf-8")

        self.assertEqual(review_json["manual_review_count"], 1)
        self.assertEqual(flagged_json["manual_review_count"], 1)
        self.assertEqual(review_json["answer_mismatch_count"], 0)
        self.assertEqual(len(flagged_json["questions"]), 1)
        self.assertEqual(
            flagged_json["questions"][0]["manual_review_reason"],
            "VLM flagged mentioned objects: cup=absent",
        )
        self.assertEqual(
            flagged_json["questions"][0]["question_answer_review"]["decision"],
            "pass",
        )
        self.assertIn("VLM flagged mentioned objects: cup=absent", flagged_html)

    def test_run_question_presence_review_flags_answer_mismatch_for_target_types(self) -> None:
        root = make_case_dir("answer_review")
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
                "question": "Where is the lamp relative to the chair?",
                "answer": "A",
                "options": ["left", "right", "front", "behind"],
                "type": "direction_agent",
                "level": "L1",
                "mentioned_objects": [
                    {"label": "lamp"},
                    {"label": "chair"},
                ],
            }
        ]

        def fake_presence_review(image_path, question, labels):
            self.assertEqual(image_path.name, image_name)
            self.assertEqual(labels, ["lamp", "chair"])
            return {
                "decision": "pass",
                "flagged_labels": [],
                "object_reviews": [
                    {"label": "lamp", "status": "present", "reason": "visible"},
                    {"label": "chair", "status": "present", "reason": "visible"},
                ],
            }

        def fake_answer_review(image_path, question):
            self.assertEqual(image_path.name, image_name)
            return {
                "decision": "manual_review",
                "predicted_answer": "B",
                "gold_answer": "A",
                "predicted_option": "right",
                "gold_option": "left",
                "reason": "model answered B but gold answer is A",
                "raw_response": "B",
            }

        with (
            patch.object(run_pipeline_module, "_resolve_question_review_vlm", return_value=(object(), "fake-vlm")),
            patch.object(run_pipeline_module, "_make_question_presence_reviewer", return_value=("fake-vlm", fake_presence_review)),
            patch.object(run_pipeline_module, "_make_question_answer_reviewer", return_value=("fake-vlm", fake_answer_review)),
        ):
            summary = run_pipeline_module._run_question_presence_review(
                questions=questions,
                data_root=data_root,
                output_dir=output_dir,
                vlm_url="http://example.com/v1",
                vlm_model="fake-vlm",
                workers=1,
            )

        self.assertEqual(summary["manual_review_count"], 1)
        self.assertEqual(summary["answer_review_question_count"], 1)
        self.assertEqual(summary["answer_mismatch_count"], 1)

        flagged_json = json.loads((output_dir / "question_presence_review_flagged.json").read_text(encoding="utf-8"))
        self.assertEqual(flagged_json["manual_review_count"], 1)
        self.assertEqual(flagged_json["answer_mismatch_count"], 1)
        self.assertEqual(len(flagged_json["questions"]), 1)
        self.assertEqual(
            flagged_json["questions"][0]["manual_review_reason"],
            "VLM answered B (right) but gold answer is A (left)",
        )
        self.assertEqual(
            flagged_json["questions"][0]["question_presence_review"]["decision"],
            "pass",
        )
        self.assertEqual(
            flagged_json["questions"][0]["question_answer_review"]["decision"],
            "manual_review",
        )

    def test_review_question_object_presence_skips_answer_review_for_non_target_type(self) -> None:
        root = make_case_dir("skip_answer_review")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        color_dir = data_root / scene_id / "color"
        color_dir.mkdir(parents=True)
        (color_dir / image_name).write_bytes(b"not-a-real-jpeg")

        question = {
            "scene_id": scene_id,
            "image_name": image_name,
            "question": "Is the cup on the table?",
            "answer": "A",
            "options": ["yes", "no"],
            "type": "attachment_chain",
            "mentioned_objects": [{"label": "cup"}, {"label": "table"}],
        }

        reviewed = run_pipeline_module._review_question_object_presence(
            lambda *_args, **_kwargs: {
                "decision": "pass",
                "flagged_labels": [],
                "object_reviews": [],
            },
            lambda *_args, **_kwargs: self.fail("answer review should be skipped"),
            question_index=0,
            question=question,
            data_root=data_root,
        )

        self.assertEqual(reviewed["question_presence_review"]["decision"], "pass")
        self.assertEqual(reviewed["question_answer_review"]["decision"], "skipped")
        self.assertNotIn("manual_review_reason", reviewed)

    def test_review_question_object_presence_combines_presence_and_answer_reasons(self) -> None:
        root = make_case_dir("combine_review")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        color_dir = data_root / scene_id / "color"
        color_dir.mkdir(parents=True)
        (color_dir / image_name).write_bytes(b"not-a-real-jpeg")

        question = {
            "scene_id": scene_id,
            "image_name": image_name,
            "question": "Where is the lamp relative to the chair?",
            "answer": "A",
            "options": ["left", "right", "front", "behind"],
            "type": "direction_agent",
            "mentioned_objects": [{"label": "lamp"}, {"label": "chair"}],
        }

        reviewed = run_pipeline_module._review_question_object_presence(
            lambda *_args, **_kwargs: {
                "decision": "manual_review",
                "flagged_labels": ["lamp"],
                "object_reviews": [
                    {"label": "lamp", "status": "absent", "reason": "not visible"},
                    {"label": "chair", "status": "present", "reason": "visible"},
                ],
            },
            lambda *_args, **_kwargs: {
                "decision": "manual_review",
                "predicted_answer": "B",
                "gold_answer": "A",
                "predicted_option": "right",
                "gold_option": "left",
                "reason": "model answered B but gold answer is A",
                "raw_response": "B",
            },
            question_index=0,
            question=question,
            data_root=data_root,
        )

        self.assertEqual(
            reviewed["manual_review_reason"],
            "VLM flagged mentioned objects: lamp=absent | VLM answered B (right) but gold answer is A (left)",
        )

    def test_make_question_answer_reviewer_flags_invalid_gold_answer(self) -> None:
        client = make_fake_review_client("A")
        _model, review_fn = run_pipeline_module._make_question_answer_reviewer(client, "fake-vlm")

        review = review_fn(
            Path("image.jpg"),
            {
                "question": "Is the cup visible?",
                "answer": "yes",
                "options": ["yes", "no"],
            },
        )

        self.assertEqual(review["decision"], "manual_review")
        self.assertEqual(review["reason"], "invalid gold answer: YES")

    def test_make_question_answer_reviewer_flags_missing_options(self) -> None:
        client = make_fake_review_client("A")
        _model, review_fn = run_pipeline_module._make_question_answer_reviewer(client, "fake-vlm")

        review = review_fn(
            Path("image.jpg"),
            {
                "question": "Is the cup visible?",
                "answer": "A",
                "options": [],
            },
        )

        self.assertEqual(review["decision"], "manual_review")
        self.assertEqual(review["reason"], "missing question text or options")
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
