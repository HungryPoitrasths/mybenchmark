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
    def test_build_frame_debug_entry_records_occlusion_eligible_object_ids(self) -> None:
        objects = [make_object(1, "lamp"), make_object(2, "table")]
        entry = run_pipeline_module._build_frame_debug_entry(
            image_name="000123.jpg",
            scene_objects=objects,
            objects_by_id={int(obj["id"]): obj for obj in objects},
            selector_visible_ids=[1, 2],
            pipeline_visible_ids=[1, 2],
            occlusion_eligible_object_ids=[2, 1],
            referability_entry=None,
            frame_attachment_rows=[],
        )

        self.assertEqual(entry["occlusion_eligible_object_ids"], [1, 2])

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

    def test_question_presence_prompt_uses_crop_index_only(self) -> None:
        prompt = run_pipeline_module._question_presence_prompt(
            "Where is the chair relative to the table?",
            [
                {"label": "chair", "obj_id": 1, "roles": ["target"]},
                {"label": "table", "obj_id": 2, "roles": ["reference"]},
            ],
        )
        prompt_lower = prompt.lower()

        self.assertNotIn("obj_id", prompt_lower)
        self.assertIn("judge each crop_index independently", prompt_lower)
        self.assertIn('"crop_index": 1', prompt)
        self.assertIn('"crop_index": 2', prompt)
        self.assertLess(prompt.index('"crop_index": 1'), prompt.index('"crop_index": 2'))

    def test_question_presence_reviewer_maps_crop_index_back_to_internal_obj_id(self) -> None:
        raw_text = json.dumps(
            {
                "objects": [
                    {"crop_index": 2, "status": "absent", "reason": "blocked by another object"}
                ]
            },
            ensure_ascii=False,
        )
        fake_response = Mock(
            choices=[Mock(message=Mock(content=raw_text))]
        )
        _, review_fn = run_pipeline_module._make_question_presence_reviewer(Mock(), "fake-model")

        with patch.object(
            run_pipeline_module,
            "_call_question_review_vlm",
            return_value=fake_response,
        ):
            review = review_fn(
                {"image_b64": "full-frame", "mime": "image/jpeg", "image_name": "000123.jpg"},
                {"question": "Is the chair left of the table?"},
                [
                    {
                        "label": "chair",
                        "obj_id": 11,
                        "roles": ["target"],
                        "roi_bounds_px": [1, 2, 30, 40],
                        "crop_image_b64": "crop-1",
                        "crop_mime": "image/jpeg",
                    },
                    {
                        "label": "table",
                        "obj_id": 22,
                        "roles": ["reference"],
                        "roi_bounds_px": [5, 6, 50, 60],
                        "crop_image_b64": "crop-2",
                        "crop_mime": "image/jpeg",
                    },
                ],
            )

        self.assertEqual(review["raw_response"], raw_text)
        self.assertEqual(review["object_reviews"][0]["obj_id"], 11)
        self.assertEqual(review["object_reviews"][0]["reason"], "missing_obj_id_in_vlm_response")
        self.assertEqual(review["object_reviews"][1]["obj_id"], 22)
        self.assertEqual(review["object_reviews"][1]["status"], "absent")
        self.assertEqual(review["object_reviews"][1]["reason"], "blocked by another object")

    def test_question_presence_reviewer_accepts_legacy_obj_id_response(self) -> None:
        raw_text = json.dumps(
            {
                "objects": [
                    {"obj_id": 22, "status": "present", "reason": "clearly visible"}
                ]
            },
            ensure_ascii=False,
        )
        fake_response = Mock(
            choices=[Mock(message=Mock(content=raw_text))]
        )
        _, review_fn = run_pipeline_module._make_question_presence_reviewer(Mock(), "fake-model")

        with patch.object(
            run_pipeline_module,
            "_call_question_review_vlm",
            return_value=fake_response,
        ):
            review = review_fn(
                {"image_b64": "full-frame", "mime": "image/jpeg", "image_name": "000123.jpg"},
                {"question": "Is the chair left of the table?"},
                [
                    {
                        "label": "chair",
                        "obj_id": 11,
                        "roles": ["target"],
                        "roi_bounds_px": [1, 2, 30, 40],
                        "crop_image_b64": "crop-1",
                        "crop_mime": "image/jpeg",
                    },
                    {
                        "label": "table",
                        "obj_id": 22,
                        "roles": ["reference"],
                        "roi_bounds_px": [5, 6, 50, 60],
                        "crop_image_b64": "crop-2",
                        "crop_mime": "image/jpeg",
                    },
                ],
            )

        self.assertEqual(review["object_reviews"][0]["reason"], "missing_obj_id_in_vlm_response")
        self.assertEqual(review["object_reviews"][1]["obj_id"], 22)
        self.assertEqual(review["object_reviews"][1]["status"], "present")
        self.assertEqual(review["object_reviews"][1]["reason"], "clearly visible")

    def test_review_question_object_presence_keeps_missing_obj_id_target_out_of_vlm(self) -> None:
        answer_review_fn = Mock()
        review_fn = Mock(
            return_value={
                "object_reviews": [
                    run_pipeline_module._build_presence_review_entry(
                        {
                            "label": "table",
                            "obj_id": 2,
                            "roles": ["reference"],
                            "roi_bounds_px": [10, 20, 40, 60],
                        },
                        status="present",
                        reason="clear crop",
                    )
                ],
                "raw_response": '{"objects":[{"crop_index":1,"status":"present","reason":"clear crop"}]}',
            }
        )

        reviewed = run_pipeline_module._review_question_object_presence(
            review_fn,
            answer_review_fn,
            question_index=0,
            question={
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "question": "Is the chair left of the table?",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": None},
                    {"role": "reference", "label": "table", "obj_id": 2},
                ],
            },
            data_root=Path("."),
            frame_context_by_key={
                ("scene0000_00", "000123.jpg"): {
                    "image_path": Path("unused.jpg"),
                    "image_exists": True,
                    "has_projection_context": True,
                    "image_b64": "full-frame",
                    "mime": "image/jpeg",
                    "objects_by_id": {2: make_object(2, "table")},
                    "crop_by_obj_id": {
                        2: {
                            "valid": True,
                            "image_b64": "crop-2",
                            "mime": "image/jpeg",
                            "roi_bounds_px": [10, 20, 40, 60],
                        }
                    },
                }
            },
        )

        review_fn.assert_called_once()
        answer_review_fn.assert_not_called()
        sent_targets = review_fn.call_args.args[2]
        self.assertEqual(len(sent_targets), 1)
        self.assertEqual(sent_targets[0]["obj_id"], 2)
        self.assertEqual(sent_targets[0]["label"], "table")

        object_reviews = reviewed["question_presence_review"]["object_reviews"]
        self.assertEqual(len(object_reviews), 2)
        missing_obj_review = next(
            item for item in object_reviews if item.get("obj_id") is None
        )
        resolved_review = next(
            item for item in object_reviews if item.get("obj_id") == 2
        )
        self.assertEqual(missing_obj_review["status"], "unsure")
        self.assertEqual(missing_obj_review["reason"], "missing_obj_id")
        self.assertEqual(resolved_review["status"], "present")

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

    def test_apply_question_referability_filter_raises_on_nonreferable_mention(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Referability backstop detected"):
            run_pipeline_module._apply_question_referability_filter(
                [
                    {
                        "scene_id": "scene0000_00",
                        "image_name": "000123.jpg",
                        "question": "Where is the chair relative to the curtain?",
                        "type": "direction_agent",
                        "mentioned_objects": [
                            {"role": "target", "label": "chair", "obj_id": 1},
                            {"role": "reference", "label": "curtain"},
                        ],
                    }
                ],
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

    def test_l1_not_visible_occlusion_passes_audit(self) -> None:
        audit = run_pipeline_module._build_question_referability_audit(
            {
                "type": "occlusion",
                "correct_value": "not visible",
                "question": "Is the lamp visible?",
                "mentioned_objects": [
                    {"role": "target", "label": "lamp", "obj_id": None},
                ],
                "obj_a_label": "lamp",
                "obj_a_id": None,
            },
            objects_by_id={5: make_object(5, "lamp")},
            referability_entry={
                "label_statuses": {"lamp": "absent"},
                "label_to_object_ids": {"lamp": [5]},
            },
            frame_referable_ids=[],
        )

        self.assertEqual(audit["decision"], "pass")
        self.assertEqual(audit["reason_codes"], [])
        self.assertEqual(len(audit["mentioned_objects"]), 1)
        self.assertTrue(audit["mentioned_objects"][0]["exempt"])

    def test_object_move_occlusion_target_is_not_exempt(self) -> None:
        audit = run_pipeline_module._build_question_referability_audit(
            {
                "type": "object_move_occlusion",
                "correct_value": "not visible",
                "question": "If the table moves, is the lamp visible?",
                "mentioned_objects": [
                    {"role": "moved_object", "label": "table", "obj_id": 1},
                    {"role": "target_object", "label": "lamp", "obj_id": 2},
                ],
            },
            objects_by_id={
                1: make_object(1, "table"),
                2: make_object(2, "lamp"),
            },
            referability_entry={
                "label_statuses": {"table": "unique", "lamp": "absent"},
                "label_to_object_ids": {"table": [1], "lamp": [2]},
            },
            frame_referable_ids=[1],
        )

        self.assertEqual(audit["decision"], "drop")
        self.assertIn("mentioned_nonreferable_object", audit["reason_codes"])

    def test_build_question_referability_audit_drops_same_object_used_by_multiple_roles(self) -> None:
        audit = run_pipeline_module._build_question_referability_audit(
            {
                "type": "direction_agent",
                "question": "Where is the chair relative to itself?",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                    {"role": "reference", "label": "chair", "obj_id": 1},
                ],
            },
            objects_by_id={1: make_object(1, "chair")},
            referability_entry={
                "label_statuses": {"chair": "unique"},
                "label_to_object_ids": {"chair": [1]},
            },
            frame_referable_ids=[1],
        )

        self.assertEqual(audit["decision"], "drop")
        self.assertEqual(audit["reason_codes"], ["mentioned_object_multi_role"])
        self.assertEqual(
            audit["mentioned_objects"][0]["same_object_roles"],
            ["reference", "target"],
        )
        self.assertEqual(
            audit["mentioned_objects"][1]["same_object_roles"],
            ["reference", "target"],
        )

    def test_build_question_referability_audit_ignores_legacy_alias_when_explicit_role_matches(self) -> None:
        audit = run_pipeline_module._build_question_referability_audit(
            {
                "type": "object_move_agent",
                "question": "If the table moves, where is the cup?",
                "query_obj_id": 1,
                "query_obj_label": "cup",
                "mentioned_objects": [
                    {"role": "query_object", "label": "cup", "obj_id": 1},
                ],
            },
            objects_by_id={1: make_object(1, "cup")},
            referability_entry={
                "label_statuses": {"cup": "unique"},
                "label_to_object_ids": {"cup": [1]},
            },
            frame_referable_ids=[1],
        )

        self.assertEqual(audit["decision"], "pass")
        self.assertEqual(audit["reason_codes"], [])
        self.assertEqual(
            audit["mentioned_objects"][0]["explicit_roles"],
            ["query_object"],
        )
        self.assertEqual(
            audit["mentioned_objects"][0]["fallback_roles"],
            ["query_obj"],
        )

    def test_load_referability_cache_rejects_old_version(self) -> None:
        case_dir = make_case_dir("cache")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "3.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "expected 13.0"):
            run_pipeline_module._load_referability_cache(cache_path)

    def test_has_l1_visibility_candidates_only_keeps_absent_labels(self) -> None:
        self.assertTrue(
            run_pipeline_module._has_l1_visibility_candidates({"lamp": "absent"})
        )
        self.assertFalse(
            run_pipeline_module._has_l1_visibility_candidates(
                {"chair": "unique", "table": "multiple", "sofa": "unsure"}
            )
        )

    def test_run_pipeline_uses_crop_unique_label_even_when_full_frame_marks_absent(self) -> None:
        root = make_case_dir("pipeline_l1_absent_candidate")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [1],
                        "crop_label_statuses": {"lamp": "unique"},
                        "crop_label_counts": {"lamp": 1},
                        "crop_referable_object_ids": [1],
                        "full_frame_label_reviews": [
                            {
                                "label": "lamp",
                                "status": "absent",
                                "crop_status": "unique",
                                "crop_clear_count": 1,
                                "crop_referable_object_id": 1,
                                "raw_response": '{"status":"absent"}',
                            }
                        ],
                        "full_frame_label_statuses": {"lamp": "absent"},
                        "full_frame_label_counts": {"lamp": 0},
                        "referable_object_ids": [1],
                        "label_statuses": {"lamp": "unique"},
                        "label_counts": {"lamp": 1},
                        "candidate_labels": ["lamp"],
                        "label_to_object_ids": {"lamp": [1]},
                    }
                }
            },
        }

        scene = {
            "scene_id": scene_id,
            "objects": [
                make_object(1, "lamp"),
                make_object(2, "table"),
            ],
            "attachment_edges": [
                {"parent_id": 2, "child_id": 1, "type": "attachment"},
            ],
            "room_bounds": None,
            "wall_objects": [],
        }

        captured: dict[str, object] = {"called": False}

        def fake_generate_all_questions(**kwargs):
            captured["called"] = True
            captured["visible_object_ids"] = list(kwargs["visible_object_ids"])
            captured["referable_object_ids"] = list(kwargs["referable_object_ids"] or [])
            captured["occlusion_eligible_object_ids"] = list(kwargs["occlusion_eligible_object_ids"] or [])
            captured["mention_in_frame_ratio_by_obj_id"] = dict(
                kwargs.get("mention_in_frame_ratio_by_obj_id") or {}
            )
            captured["label_statuses"] = dict(kwargs["label_statuses"] or {})
            captured["label_counts"] = dict(kwargs["label_counts"] or {})
            return []

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
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                },
            ),
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
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertTrue(captured["called"])
        self.assertEqual(captured["visible_object_ids"], [1])
        self.assertEqual(captured["referable_object_ids"], [1])
        self.assertEqual(captured["occlusion_eligible_object_ids"], [1])
        self.assertEqual(captured["mention_in_frame_ratio_by_obj_id"], {1: 0.95})
        self.assertEqual(captured["label_statuses"], {"lamp": "unique"})
        self.assertEqual(captured["label_counts"], {"lamp": 1})
        self.assertEqual(questions, [])

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
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [2, 1],
                        "crop_label_statuses": {"cup": "unique", "table": "unique"},
                        "crop_label_counts": {"cup": 1, "table": 1},
                        "crop_referable_object_ids": [1, 2],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
                        "referable_object_ids": [1, 2],
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
            captured["occlusion_eligible_object_ids"] = list(kwargs["occlusion_eligible_object_ids"] or [])
            captured["mention_in_frame_ratio_by_obj_id"] = dict(
                kwargs.get("mention_in_frame_ratio_by_obj_id") or {}
            )
            captured["label_statuses"] = dict(kwargs["label_statuses"] or {})
            captured["label_counts"] = dict(kwargs["label_counts"] or {})
            captured["label_to_object_ids"] = dict(kwargs["label_to_object_ids"] or {})
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
            patch.object(
                run_pipeline_module,
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                    2: {"bbox_in_frame_ratio": 0.85},
                },
            ),
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
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertEqual(captured["visible_object_ids"], [1, 2])
        self.assertEqual(captured["referable_object_ids"], [1, 2])
        self.assertEqual(captured["occlusion_eligible_object_ids"], [1, 2])
        self.assertEqual(captured["mention_in_frame_ratio_by_obj_id"], {1: 0.95, 2: 0.85})
        self.assertEqual(captured["label_statuses"], {"cup": "unique", "table": "unique"})
        self.assertEqual(captured["label_counts"], {"cup": 1, "table": 1})
        self.assertEqual(captured["label_to_object_ids"], {"cup": [1], "table": [2]})
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
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [3, 2, 1],
                        "crop_label_statuses": {"chair": "unique", "curtain": "multiple"},
                        "crop_label_counts": {"chair": 1, "curtain": 2},
                        "crop_referable_object_ids": [1],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
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
            with self.assertRaisesRegex(AssertionError, "Referability backstop detected"):
                run_pipeline_module.run_pipeline(
                    data_root=data_root,
                    output_dir=output_dir,
                    max_scenes=10,
                    max_frames=10,
                    use_occlusion=False,
                    referability_cache=referability_cache,
                    write_frame_debug=True,
                )

if __name__ == "__main__":
    unittest.main()
