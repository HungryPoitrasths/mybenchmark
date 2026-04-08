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

    def test_load_referability_cache_rejects_old_version(self) -> None:
        case_dir = make_case_dir("cache")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "3.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "expected 9.0"):
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
            "version": "9.0",
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
            "version": "9.0",
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

if __name__ == "__main__":
    unittest.main()
