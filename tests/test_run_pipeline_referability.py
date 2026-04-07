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


def make_frame_context(
    *,
    image_path: Path,
    objects: list[dict] | None = None,
    invalid_obj_ids: set[int] | None = None,
    has_projection_context: bool = True,
) -> dict:
    objects = list(objects or [])
    invalid_obj_ids = set(invalid_obj_ids or set())
    objects_by_id = {int(obj["id"]): obj for obj in objects}
    crop_by_obj_id = {}
    for obj_id in objects_by_id:
        crop_by_obj_id[obj_id] = {
            "valid": obj_id not in invalid_obj_ids,
            "reason": "invalid_crop" if obj_id in invalid_obj_ids else "",
            "roi_bounds_px": [10, 30, 12, 36],
            "image_b64": "Y3JvcA==",
            "mime": "image/jpeg",
        }
    return {
        "scene_id": image_path.parent.parent.name,
        "image_name": image_path.name,
        "image_path": image_path,
        "image_exists": image_path.exists(),
        "image_b64": "ZnVsbA==",
        "mime": "image/jpeg",
        "objects_by_id": objects_by_id,
        "crop_by_obj_id": crop_by_obj_id,
        "has_projection_context": has_projection_context,
        "context_errors": [],
    }


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
    def test_resolve_vlm_api_key_warns_when_env_is_missing(self) -> None:
        with (
            patch.dict(run_pipeline_module.os.environ, {}, clear=True),
            patch.object(run_pipeline_module.logger, "warning") as warning_mock,
        ):
            api_key = run_pipeline_module._resolve_vlm_api_key(
                purpose="question post-review",
                missing_key_hint="Set one of the supported API key environment variables.",
            )

        self.assertEqual(api_key, run_pipeline_module.PLACEHOLDER_VLM_API_KEY)
        warning_mock.assert_called_once()
        self.assertIn("placeholder API key", warning_mock.call_args.args[0])

    def test_collect_question_presence_targets_prefers_mentioned_objects(self) -> None:
        targets = run_pipeline_module._collect_question_presence_targets(
            {
                "mentioned_objects": [
                    {"label": "Cup", "obj_id": 1, "role": "target"},
                    {"label": "cup", "obj_id": 1, "role": "query_obj"},
                    {"label": "Table", "obj_id": 2, "role": "reference"},
                ],
                "obj_a_label": "Chair",
                "obj_b_label": "Lamp",
            },
            objects_by_id={
                1: make_object(1, "cup"),
                2: make_object(2, "table"),
            },
        )

        self.assertEqual(
            targets,
            [
                {"label": "Cup", "obj_id": 1, "roles": ["query_obj", "target"]},
                {"label": "Table", "obj_id": 2, "roles": ["reference"]},
            ],
        )

    def test_collect_question_presence_targets_uses_fallback_obj_ids(self) -> None:
        targets = run_pipeline_module._collect_question_presence_targets(
            {
                "obj_a_id": 2,
                "obj_a_label": "table",
                "obj_b_id": 1,
                "obj_b_label": "cup",
            },
            objects_by_id={
                1: make_object(1, "cup"),
                2: make_object(2, "table"),
            },
        )

        self.assertEqual(
            targets,
            [
                {"label": "table", "obj_id": 2, "roles": ["obj_a"]},
                {"label": "cup", "obj_id": 1, "roles": ["obj_b"]},
            ],
        )

    def test_build_question_referability_audit_drops_ambiguous_nonreferable_label(self) -> None:
        audit = run_pipeline_module._build_question_referability_audit(
            {
                "question": "Where is the chair relative to the curtain?",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                    {"role": "reference", "label": "curtain"},
                ],
            },
            objects_by_id={
                1: make_object(1, "chair"),
                2: make_object(2, "curtain"),
                3: make_object(3, "curtain"),
            },
            referability_entry={
                "label_statuses": {"chair": "unique", "curtain": "multiple"},
                "label_to_object_ids": {"chair": [1], "curtain": [2, 3]},
            },
            frame_referable_ids=[1],
        )

        self.assertEqual(audit["decision"], "drop")
        self.assertEqual(
            audit["reason_codes"],
            ["mentioned_label_not_unique", "mentioned_label_not_resolved"],
        )
        self.assertEqual(audit["frame_referable_object_ids"], [1])
        self.assertEqual(len(audit["mentioned_objects"]), 2)
        self.assertTrue(audit["mentioned_objects"][0]["passes_referability_check"])
        self.assertFalse(audit["mentioned_objects"][1]["passes_referability_check"])
        self.assertEqual(audit["mentioned_objects"][1]["label"], "curtain")
        self.assertEqual(audit["mentioned_objects"][1]["candidate_object_ids"], [2, 3])
        self.assertEqual(audit["mentioned_objects"][1]["referable_object_ids"], [])

    def test_apply_question_referability_filter_keeps_uniquely_referable_mentions(self) -> None:
        kept, audited = run_pipeline_module._apply_question_referability_filter(
            [
                {
                    "scene_id": "scene0000_00",
                    "image_name": "000123.jpg",
                    "question": "Where is the chair relative to the table?",
                    "mentioned_objects": [
                        {"role": "target", "label": "chair", "obj_id": 1},
                        {"role": "reference", "label": "table", "obj_id": 2},
                    ],
                }
            ],
            objects_by_id={
                1: make_object(1, "chair"),
                2: make_object(2, "table"),
            },
            referability_entry={
                "label_statuses": {"chair": "unique", "table": "unique"},
                "label_to_object_ids": {"chair": [1], "table": [2]},
            },
            frame_referable_ids=[1, 2],
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(audited), 1)
        self.assertEqual(
            kept[0]["question_referability_audit"]["decision"],
            "pass",
        )
        self.assertEqual(
            kept[0]["question_referability_audit"]["reason_codes"],
            [],
        )

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

    def test_make_question_presence_reviewer_uses_instance_targets_and_dynamic_tokens(self) -> None:
        client = make_fake_review_client(
            '{"objects":['
            '{"obj_id":1,"status":"present","reason":"visible"},'
            '{"obj_id":2,"status":"absent","reason":"not visible"},'
            '{"obj_id":3,"status":"unsure","reason":"component of larger desk"}'
            "]}"
        )
        _model, review_fn = run_pipeline_module._make_question_presence_reviewer(client, "fake-vlm")

        review = review_fn(
            {
                "image_name": "000123.jpg",
                "image_b64": "ZnVsbA==",
                "mime": "image/jpeg",
            },
            {"question": "Where is the cup relative to the table?"},
            [
                {
                    "label": "cup",
                    "obj_id": 1,
                    "roles": ["target"],
                    "crop_image_b64": "Y3JvcDE=",
                    "crop_mime": "image/jpeg",
                    "roi_bounds_px": [1, 2, 3, 4],
                },
                {
                    "label": "table",
                    "obj_id": 2,
                    "roles": ["reference"],
                    "crop_image_b64": "Y3JvcDI=",
                    "crop_mime": "image/jpeg",
                    "roi_bounds_px": [5, 6, 7, 8],
                },
                {
                    "label": "cabinet",
                    "obj_id": 3,
                    "roles": ["reference"],
                    "crop_image_b64": "Y3JvcDM=",
                    "crop_mime": "image/jpeg",
                    "roi_bounds_px": [9, 10, 11, 12],
                },
            ],
        )

        create_kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(create_kwargs["max_tokens"], 384)
        content = create_kwargs["messages"][0]["content"]
        self.assertEqual(sum(1 for item in content if item["type"] == "image_url"), 4)
        self.assertIn('"obj_id": 1', content[-1]["text"])
        self.assertIn("component/substructure", content[-1]["text"])
        self.assertEqual(review["object_reviews"][0]["obj_id"], 1)
        self.assertEqual(review["object_reviews"][1]["status"], "absent")
        self.assertEqual(review["object_reviews"][2]["status"], "unsure")

    def test_combine_manual_review_reasons_deduplicates(self) -> None:
        combined = run_pipeline_module._combine_manual_review_reasons(
            ["same reason", "same reason", ""]
        )

        self.assertEqual(combined, "same reason")

    def test_call_question_review_vlm_retries_concurrency_limit_errors(self) -> None:
        calls: list[int] = []

        def flaky_call():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError(
                    "Error code: 429 - {'error': {'code': 'concurrent_request_limit_exceeded', 'message': 'Too many concurrent requests.'}}"
                )
            return "ok"

        with patch.object(run_pipeline_module.time, "sleep") as sleep_mock:
            result = run_pipeline_module._call_question_review_vlm(
                flaky_call,
                context="presence review",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)
        sleep_mock.assert_called_once_with(run_pipeline_module.QUESTION_REVIEW_RETRY_DELAY_SECONDS)

    def test_call_question_review_vlm_surfaces_authentication_hint(self) -> None:
        def auth_failure():
            raise RuntimeError("Error code: 401 - {'error': {'message': 'Unauthorized'}}")

        with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY or OPENAI_API_KEY"):
            run_pipeline_module._call_question_review_vlm(
                auth_failure,
                context="presence review",
            )

    def test_load_referability_cache_rejects_old_version(self) -> None:
        case_dir = make_case_dir("cache")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "3.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "expected 7.0"):
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
            "version": "7.0",
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

    def test_run_pipeline_drops_questions_with_ambiguous_nonreferable_mentions(self) -> None:
        root = make_case_dir("pipeline_referability_backstop")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "7.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [3, 2, 1],
                        "referable_object_ids": [1],
                        "label_statuses": {
                            "chair": "unique",
                            "curtain": "multiple",
                        },
                        "label_counts": {
                            "chair": 1,
                            "curtain": 2,
                        },
                        "candidate_labels": ["chair", "curtain"],
                        "label_to_object_ids": {
                            "chair": [1],
                            "curtain": [2, 3],
                        },
                    }
                }
            },
        }

        scene = {
            "scene_id": scene_id,
            "objects": [
                make_object(1, "chair"),
                make_object(2, "curtain"),
                make_object(3, "curtain"),
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
                        "question": "From the camera's viewpoint, the chair is in which direction relative to the curtain?",
                        "answer": "A",
                        "options": ["back-right", "front-left", "front-right", "back-left"],
                        "type": "direction_agent",
                        "level": "L1",
                        "mentioned_objects": [
                            {"role": "target", "label": "chair", "obj_id": 1},
                            {"role": "reference", "label": "curtain"},
                        ],
                    }
                ],
            ),
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
                write_frame_debug=True,
            )

        self.assertEqual(questions, [])

        frame_debug = json.loads(
            (output_dir / "frame_debug" / f"{scene_id}.json").read_text(encoding="utf-8")
        )
        generated_questions = frame_debug["frames"][0]["generated_questions"]
        self.assertEqual(len(generated_questions), 1)
        self.assertEqual(
            generated_questions[0]["question_referability_audit"]["decision"],
            "drop",
        )
        self.assertEqual(
            generated_questions[0]["question_referability_audit"]["reason_codes"],
            ["mentioned_label_not_unique", "mentioned_label_not_resolved"],
        )
        self.assertEqual(frame_debug["frames"][0]["final_question_count"], 0)

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
            "version": "7.0",
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
                    {"label": "cup", "obj_id": 1, "role": "target"},
                    {"label": "table", "obj_id": 2, "role": "reference"},
                ],
            }
        ]

        frame_context = {
            (scene_id, image_name): make_frame_context(
                image_path=color_dir / image_name,
                objects=[make_object(1, "cup"), make_object(2, "table")],
            )
        }

        def fake_presence_review(frame_context_arg, question, targets):
            self.assertEqual(frame_context_arg["image_name"], image_name)
            self.assertEqual([target["obj_id"] for target in targets], [1, 2])
            return {
                "object_reviews": [
                    {
                        "label": "cup",
                        "obj_id": 1,
                        "roles": ["target"],
                        "status": "absent",
                        "reason": "not visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                    {
                        "label": "table",
                        "obj_id": 2,
                        "roles": ["reference"],
                        "status": "present",
                        "reason": "visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                ],
                "raw_response": '{"objects":[]}',
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
            patch.object(run_pipeline_module, "_prebuild_question_review_frame_contexts", return_value=frame_context),
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
            "VLM flagged mentioned objects: cup#1=absent",
        )
        self.assertEqual(
            flagged_json["questions"][0]["question_answer_review"]["decision"],
            "pass",
        )
        self.assertEqual(
            flagged_json["questions"][0]["question_presence_review"]["review_mode"],
            "instance",
        )
        self.assertEqual(
            flagged_json["questions"][0]["question_presence_review"]["flagged_object_ids"],
            [1],
        )
        self.assertIn("VLM flagged mentioned objects: cup#1=absent", flagged_html)

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
                "question": "Where is the cup relative to the chair?",
                "answer": "A",
                "options": ["left", "right", "front", "behind"],
                "type": "direction_agent",
                "level": "L1",
                "mentioned_objects": [
                    {"label": "cup", "obj_id": 1, "role": "target"},
                    {"label": "chair", "obj_id": 2, "role": "reference"},
                ],
            }
        ]

        frame_context = {
            (scene_id, image_name): make_frame_context(
                image_path=color_dir / image_name,
                objects=[make_object(1, "cup"), make_object(2, "chair")],
            )
        }

        def fake_presence_review(frame_context_arg, question, targets):
            self.assertEqual(frame_context_arg["image_name"], image_name)
            self.assertEqual([target["obj_id"] for target in targets], [1, 2])
            return {
                "object_reviews": [
                    {
                        "label": "cup",
                        "obj_id": 1,
                        "roles": ["target"],
                        "status": "present",
                        "reason": "visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                    {
                        "label": "chair",
                        "obj_id": 2,
                        "roles": ["reference"],
                        "status": "present",
                        "reason": "visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                ],
                "raw_response": '{"objects":[]}',
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
            patch.object(run_pipeline_module, "_prebuild_question_review_frame_contexts", return_value=frame_context),
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
            "mentioned_objects": [
                {"label": "cup", "obj_id": 1, "role": "target"},
                {"label": "table", "obj_id": 2, "role": "reference"},
            ],
        }
        frame_context_by_key = {
            (scene_id, image_name): make_frame_context(
                image_path=color_dir / image_name,
                objects=[make_object(1, "cup"), make_object(2, "table")],
            )
        }

        reviewed = run_pipeline_module._review_question_object_presence(
            lambda *_args, **_kwargs: {
                "object_reviews": [
                    {
                        "label": "cup",
                        "obj_id": 1,
                        "roles": ["target"],
                        "status": "present",
                        "reason": "visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                    {
                        "label": "table",
                        "obj_id": 2,
                        "roles": ["reference"],
                        "status": "present",
                        "reason": "visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                ],
                "raw_response": "",
            },
            lambda *_args, **_kwargs: self.fail("answer review should be skipped"),
            question_index=0,
            question=question,
            data_root=data_root,
            frame_context_by_key=frame_context_by_key,
        )

        self.assertEqual(reviewed["question_presence_review"]["decision"], "pass")
        self.assertEqual(reviewed["question_answer_review"]["decision"], "skipped")
        self.assertNotIn("manual_review_reason", reviewed)

    def test_review_question_object_presence_marks_missing_obj_id_for_manual_review(self) -> None:
        root = make_case_dir("missing_obj_id_review")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        color_dir = data_root / scene_id / "color"
        color_dir.mkdir(parents=True)
        (color_dir / image_name).write_bytes(b"not-a-real-jpeg")

        reviewed = run_pipeline_module._review_question_object_presence(
            lambda *_args, **_kwargs: self.fail("presence review should not run"),
            lambda *_args, **_kwargs: self.fail("answer review should be skipped"),
            question_index=0,
            question={
                "scene_id": scene_id,
                "image_name": image_name,
                "question": "Where is the cup relative to the table?",
                "type": "attachment_chain",
                "mentioned_objects": [{"label": "cup", "role": "target"}],
            },
            data_root=data_root,
            frame_context_by_key={},
        )

        self.assertEqual(reviewed["question_presence_review"]["decision"], "manual_review")
        self.assertEqual(
            reviewed["question_presence_review"]["object_reviews"][0]["reason"],
            "missing_obj_id",
        )
        self.assertEqual(
            reviewed["manual_review_reason"],
            "VLM flagged mentioned objects: cup=unsure",
        )

    def test_review_question_object_presence_marks_invalid_crop_for_manual_review(self) -> None:
        root = make_case_dir("invalid_crop_review")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        color_dir = data_root / scene_id / "color"
        color_dir.mkdir(parents=True)
        (color_dir / image_name).write_bytes(b"not-a-real-jpeg")
        frame_context_by_key = {
            (scene_id, image_name): make_frame_context(
                image_path=color_dir / image_name,
                objects=[make_object(1, "cup"), make_object(2, "table")],
                invalid_obj_ids={1},
            )
        }

        reviewed = run_pipeline_module._review_question_object_presence(
            lambda *_args, **_kwargs: self.fail("presence review should not run for invalid crops"),
            lambda *_args, **_kwargs: self.fail("answer review should be skipped"),
            question_index=0,
            question={
                "scene_id": scene_id,
                "image_name": image_name,
                "question": "Where is the cup relative to the table?",
                "type": "attachment_chain",
                "mentioned_objects": [
                    {"label": "cup", "obj_id": 1, "role": "target"},
                ],
            },
            data_root=data_root,
            frame_context_by_key=frame_context_by_key,
        )

        self.assertEqual(reviewed["question_presence_review"]["decision"], "manual_review")
        self.assertEqual(
            reviewed["question_presence_review"]["object_reviews"][0]["reason"],
            "invalid_crop",
        )

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
            "question": "Where is the cup relative to the chair?",
            "answer": "A",
            "options": ["left", "right", "front", "behind"],
            "type": "direction_agent",
            "mentioned_objects": [
                {"label": "cup", "obj_id": 1, "role": "target"},
                {"label": "chair", "obj_id": 2, "role": "reference"},
            ],
        }
        frame_context_by_key = {
            (scene_id, image_name): make_frame_context(
                image_path=color_dir / image_name,
                objects=[make_object(1, "cup"), make_object(2, "chair")],
            )
        }

        reviewed = run_pipeline_module._review_question_object_presence(
            lambda *_args, **_kwargs: {
                "object_reviews": [
                    {
                        "label": "cup",
                        "obj_id": 1,
                        "roles": ["target"],
                        "status": "absent",
                        "reason": "not visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                    {
                        "label": "chair",
                        "obj_id": 2,
                        "roles": ["reference"],
                        "status": "present",
                        "reason": "visible",
                        "roi_bounds_px": [10, 30, 12, 36],
                    },
                ],
                "raw_response": "",
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
            frame_context_by_key=frame_context_by_key,
        )

        self.assertEqual(
            reviewed["manual_review_reason"],
            "VLM flagged mentioned objects: cup#1=absent | VLM answered B (right) but gold answer is A (left)",
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
