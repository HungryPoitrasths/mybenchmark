import json
import shutil
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import scripts.run_single_frame_trace as trace_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose

TEST_TMP_ROOT = Path(__file__).resolve().parent / "_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


def make_case_dir(prefix: str) -> Path:
    path = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def make_scene(scene_id: str) -> dict:
    return {
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


def make_referability_entry() -> dict:
    return {
        "frame_usable": True,
        "frame_reject_reason": None,
        "frame_quality_score": 78,
        "frame_quality_severely_out_of_focus": False,
        "frame_quality_usable_for_spatial_reasoning": True,
        "frame_quality_reason": "clear enough with minor softness",
        "frame_selection_score": 78002,
        "selector_visible_object_ids": [1, 2],
        "selector_visible_label_counts": {"cup": 1, "table": 1},
        "candidate_visible_object_ids": [1, 2],
        "candidate_visibility_source": "depth_refined",
        "candidate_visible_label_counts": {"cup": 1, "table": 1},
        "candidate_labels": ["cup", "table"],
        "label_to_object_ids": {"cup": [1], "table": [2]},
        "object_reviews": {
            "1": {
                "obj_id": 1,
                "label": "cup",
                "bbox_in_frame_ratio": 1.0,
                "local_outcome": "reviewed",
                "local_reason": "",
                "vlm_status": "clear",
                "raw_response": '{"status":"clear"}',
                "ray_visibility_review": {
                    "applied": False,
                    "decision": "not_applicable",
                    "reason": "not_crop_unique",
                    "stage1": None,
                    "stage2": None,
                },
            },
            "2": {
                "obj_id": 2,
                "label": "table",
                "bbox_in_frame_ratio": 0.85,
                "local_outcome": "reviewed",
                "local_reason": "",
                "vlm_status": "clear",
                "raw_response": '{"status":"clear"}',
                "ray_visibility_review": {
                    "applied": False,
                    "decision": "not_applicable",
                    "reason": "not_crop_unique",
                    "stage1": None,
                    "stage2": None,
                },
            },
        },
        "crop_label_statuses": {"cup": "unique", "table": "unique"},
        "crop_label_counts": {"cup": 1, "table": 1},
        "crop_referable_object_ids": [1, 2],
        "full_frame_label_reviews": [],
        "full_frame_label_statuses": {},
        "full_frame_label_counts": {},
        "label_statuses": {"cup": "unique", "table": "unique"},
        "label_counts": {"cup": 1, "table": 1},
        "referable_object_ids": [1, 2],
    }


def make_fake_questions() -> list[dict]:
    return [
        {
            "question": "Is the cup on the table?",
            "answer": "A",
            "options": ["yes", "no"],
            "correct_value": "yes",
            "type": "attachment_chain",
            "level": "L3",
        }
    ]


def make_near_duplicate_questions() -> list[dict]:
    return [
        {
            "question": "If the cup moves, will it still be attached to the table?",
            "answer": "A",
            "options": ["yes", "no"],
            "correct_value": "yes",
            "type": "attachment_chain",
            "level": "L3",
            "obj_a_id": 1,
            "obj_b_id": 2,
            "parent_id": 2,
            "child_id": 1,
        },
        {
            "question": "After moving the cup, does the cup remain attached to the table?",
            "answer": "B",
            "options": ["yes", "no"],
            "correct_value": "no",
            "type": "attachment_chain",
            "level": "L3",
            "obj_a_id": 1,
            "obj_b_id": 2,
            "parent_id": 2,
            "child_id": 1,
        },
    ]


class RunSingleFrameTraceTests(unittest.TestCase):
    def _make_paths(self) -> tuple[Path, Path, str, str]:
        root = make_case_dir("single_frame_trace")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / "color").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")
        (scene_dir / "color" / image_name).write_bytes(b"fake-jpg")
        return data_root, output_dir, scene_id, image_name

    def _patch_common(self, scene_id: str, image_name: str):
        scene = make_scene(scene_id)
        return (
            patch.object(trace_module, "parse_scene", return_value=scene),
            patch.object(trace_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(trace_module, "get_scene_attachment_graph", return_value={2: [1]}),
            patch.object(trace_module, "get_scene_attached_by", return_value={1: [2]}),
            patch.object(trace_module, "get_scene_support_chain_graph", return_value={2: [1]}),
            patch.object(trace_module, "get_scene_support_chain_by", return_value={1: [2]}),
            patch.object(trace_module, "has_nontrivial_attachment", return_value=True),
            patch.object(trace_module, "_load_scene_geometry", return_value=None),
            patch.object(trace_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(trace_module, "load_scannet_poses", return_value={image_name: make_camera_pose(image_name)}),
            patch.object(trace_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(trace_module, "load_scannet_depth_intrinsics", return_value=None),
            patch.object(trace_module, "load_instance_mesh_data", return_value=object()),
            patch.object(trace_module.RayCaster, "from_ply", return_value=Mock()),
        )

    def test_run_single_frame_trace_uses_cached_referability_entry(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        referability_cache = {
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: make_referability_entry(),
                }
            },
        }
        captured: dict[str, object] = {}

        def fake_generate_all_questions(**_kwargs):
            captured["occlusion_eligible_object_ids"] = list(
                _kwargs.get("occlusion_eligible_object_ids") or []
            )
            captured["mention_in_frame_ratio_by_obj_id"] = dict(
                _kwargs.get("mention_in_frame_ratio_by_obj_id") or {}
            )
            trace_recorder = _kwargs.get("trace_recorder")
            if trace_recorder is not None:
                trace_recorder(
                    {
                        "event": "object_pool_snapshot",
                        "stage": "qa_generation",
                        "details": {
                            "summary": {"original_object_count": 2, "question_object_count": 1},
                            "rows": [
                                {"id": 1, "label": "cup", "reasons": ["referable"], "tags": ["question_pool"]},
                                {"id": 2, "label": "table", "reasons": ["attachment_context"], "tags": ["graph_only"]},
                            ],
                        },
                    }
                )
                trace_recorder(
                    {
                        "event": "generator_summary",
                        "stage": "qa_generation",
                        "generator": "fake_generator",
                        "generated_count": 1,
                        "reason_counts": {"generated": 1},
                    }
                )
            return make_fake_questions()

        with ExitStack() as stack:
            for mocked in self._patch_common(scene_id, image_name):
                stack.enter_context(mocked)
            stack.enter_context(
                patch.object(trace_module, "generate_all_questions", side_effect=fake_generate_all_questions)
            )
            stack.enter_context(
                patch.object(
                    trace_module,
                    "_compute_single_frame_referability_entry",
                    side_effect=AssertionError("should not fallback online"),
                )
            )
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
            )

        self.assertEqual(trace_doc["status"], "completed")
        self.assertEqual(trace_doc["input"]["referability_source"], "cache")
        trace_json = json.loads((output_dir / "single_frame" / scene_id / "000123" / "trace.json").read_text(encoding="utf-8"))
        final_json = json.loads((output_dir / "single_frame" / scene_id / "000123" / "final_questions.json").read_text(encoding="utf-8"))
        html_text = (output_dir / "single_frame" / scene_id / "000123" / "trace.html").read_text(encoding="utf-8")
        referability_audit = json.loads(Path(trace_json["artifacts"]["audits"]["referability"]).read_text(encoding="utf-8"))
        reason_index = json.loads(Path(trace_json["artifacts"]["audits"]["reason_index"]).read_text(encoding="utf-8"))

        self.assertEqual(final_json["question_count"], 1)
        self.assertIn("trace_question_id", final_json["questions"][0])
        self.assertEqual(trace_json["question_lifecycle"][0]["status"], "kept")
        self.assertEqual(trace_json["input"]["trace_detail"], "full")
        self.assertEqual(trace_json["input"]["trace_vlm_payload"], "summary")
        self.assertEqual(captured["occlusion_eligible_object_ids"], [1, 2])
        self.assertEqual(captured["mention_in_frame_ratio_by_obj_id"], {1: 1.0, 2: 0.85})
        self.assertEqual(trace_json["frame_context"]["occlusion_eligible_object_ids"], [1, 2])
        self.assertIn("object_pool", trace_json["artifacts"]["audits"])
        self.assertIn("reason_index", trace_json["artifacts"]["audits"])
        self.assertIn("generator:fake_generator", trace_json["artifacts"]["audits"])
        self.assertNotIn("raw_response", json.dumps(referability_audit, ensure_ascii=False))
        self.assertIn("question_count", reason_index)
        self.assertIn("Is the cup on the table?", html_text)
        self.assertIn("data:image/jpeg;base64,", html_text)
        self.assertIn("Root Cause Summary", html_text)
        self.assertIn("Object Pool Audit", html_text)

    def test_run_single_frame_trace_falls_back_to_online_referability(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        referability_cache = {"version": "13.0", "frames": {scene_id: {}}}

        def fake_generate_all_questions(**_kwargs):
            return make_fake_questions()

        with ExitStack() as stack:
            for mocked in self._patch_common(scene_id, image_name):
                stack.enter_context(mocked)
            stack.enter_context(
                patch.object(trace_module, "generate_all_questions", side_effect=fake_generate_all_questions)
            )
            fallback_mock = stack.enter_context(
                patch.object(
                    trace_module,
                    "_compute_single_frame_referability_entry",
                    return_value=(make_referability_entry(), "online"),
                )
            )
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
            )

        self.assertEqual(trace_doc["status"], "completed")
        self.assertEqual(trace_doc["input"]["referability_source"], "online")
        fallback_mock.assert_called_once()

    def test_run_single_frame_trace_keeps_full_vlm_payload_when_requested(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        referability_cache = {
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: make_referability_entry(),
                }
            },
        }

        def fake_generate_all_questions(**_kwargs):
            return make_fake_questions()

        with ExitStack() as stack:
            for mocked in self._patch_common(scene_id, image_name):
                stack.enter_context(mocked)
            stack.enter_context(
                patch.object(trace_module, "generate_all_questions", side_effect=fake_generate_all_questions)
            )
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
                trace_vlm_payload="full",
            )

        trace_json = json.loads((output_dir / "single_frame" / scene_id / "000123" / "trace.json").read_text(encoding="utf-8"))
        referability_audit = json.loads(Path(trace_json["artifacts"]["audits"]["referability"]).read_text(encoding="utf-8"))
        self.assertEqual(trace_doc["status"], "completed")
        self.assertIn("raw_response", json.dumps(referability_audit, ensure_ascii=False))

    def test_run_single_frame_trace_stops_when_pose_missing(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        referability_cache = {"version": "13.0", "frames": {}}

        with ExitStack() as stack:
            stack.enter_context(patch.object(trace_module, "parse_scene", return_value=make_scene(scene_id)))
            stack.enter_context(patch.object(trace_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None))
            stack.enter_context(patch.object(trace_module, "get_scene_attachment_graph", return_value={2: [1]}))
            stack.enter_context(patch.object(trace_module, "get_scene_attached_by", return_value={1: [2]}))
            stack.enter_context(patch.object(trace_module, "get_scene_support_chain_graph", return_value={2: [1]}))
            stack.enter_context(patch.object(trace_module, "get_scene_support_chain_by", return_value={1: [2]}))
            stack.enter_context(patch.object(trace_module, "has_nontrivial_attachment", return_value=True))
            stack.enter_context(patch.object(trace_module, "_load_scene_geometry", return_value=None))
            stack.enter_context(patch.object(trace_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)))
            stack.enter_context(patch.object(trace_module, "load_scannet_poses", return_value={}))
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
            )

        self.assertEqual(trace_doc["status"], "stopped")
        self.assertEqual(trace_doc["stop_reason"], "missing_pose")
        self.assertEqual(trace_doc["stop_details"]["requested_image_name"], image_name)
        self.assertEqual(trace_doc["stop_details"]["available_pose_count"], 0)
        final_json = json.loads((output_dir / "single_frame" / scene_id / "000123" / "final_questions.json").read_text(encoding="utf-8"))
        html_text = (output_dir / "single_frame" / scene_id / "000123" / "trace.html").read_text(encoding="utf-8")
        self.assertEqual(final_json["question_count"], 0)
        self.assertIn("missing_pose", html_text)
        self.assertIn(image_name, html_text)
        self.assertIn("Root Cause Summary", html_text)

    def test_run_single_frame_trace_stops_when_vlm_frame_review_rejects_frame(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        rejected_entry = make_referability_entry()
        rejected_entry["frame_usable"] = False
        rejected_entry["frame_reject_reason"] = "out_of_focus"
        rejected_entry["referable_object_ids"] = []
        referability_cache = {
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: rejected_entry,
                }
            },
        }

        with ExitStack() as stack:
            for mocked in self._patch_common(scene_id, image_name):
                stack.enter_context(mocked)
            stack.enter_context(
                patch.object(
                    trace_module,
                    "generate_all_questions",
                    side_effect=AssertionError("should stop before question generation"),
                )
            )
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
            )

        self.assertEqual(trace_doc["status"], "stopped")
        self.assertEqual(trace_doc["stop_reason"], "frame_rejected_by_vlm_frame_review")
        self.assertEqual(trace_doc["stop_details"]["frame_reject_reason"], "out_of_focus")
        html_text = (output_dir / "single_frame" / scene_id / "000123" / "trace.html").read_text(encoding="utf-8")
        self.assertIn("frame_rejected_by_vlm_frame_review", html_text)
        self.assertIn("out_of_focus", html_text)

    def test_run_single_frame_trace_uses_crop_unique_label_when_full_frame_marks_absent(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        absent_entry = make_referability_entry()
        absent_entry["crop_label_statuses"] = {"cup": "unique"}
        absent_entry["crop_label_counts"] = {"cup": 1}
        absent_entry["crop_referable_object_ids"] = [1]
        absent_entry["full_frame_label_reviews"] = [
            {
                "label": "cup",
                "status": "absent",
                "crop_status": "unique",
                "crop_clear_count": 1,
                "crop_referable_object_id": 1,
                "raw_response": '{"status":"absent"}',
            }
        ]
        absent_entry["full_frame_label_statuses"] = {"cup": "absent"}
        absent_entry["full_frame_label_counts"] = {"cup": 0}
        absent_entry["label_statuses"] = {"cup": "unique"}
        absent_entry["label_counts"] = {"cup": 1}
        absent_entry["referable_object_ids"] = [1]
        referability_cache = {
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: absent_entry,
                }
            },
        }

        def fake_generate_all_questions(**_kwargs):
            return make_fake_questions()

        with ExitStack() as stack:
            for mocked in self._patch_common(scene_id, image_name):
                stack.enter_context(mocked)
            generator_mock = stack.enter_context(
                patch.object(trace_module, "generate_all_questions", side_effect=fake_generate_all_questions)
            )
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
            )

        self.assertEqual(trace_doc["status"], "completed")
        generator_mock.assert_called_once()

    def test_run_single_frame_trace_records_detailed_near_duplicate_reason(self) -> None:
        data_root, output_dir, scene_id, image_name = self._make_paths()
        referability_cache = {
            "version": "13.0",
            "frames": {
                scene_id: {
                    image_name: make_referability_entry(),
                }
            },
        }

        def fake_generate_all_questions(**_kwargs):
            return make_near_duplicate_questions()

        with ExitStack() as stack:
            for mocked in self._patch_common(scene_id, image_name):
                stack.enter_context(mocked)
            stack.enter_context(
                patch.object(trace_module, "generate_all_questions", side_effect=fake_generate_all_questions)
            )
            stack.enter_context(
                patch.object(
                    trace_module,
                    "_compute_single_frame_referability_entry",
                    side_effect=AssertionError("should not fallback online"),
                )
            )
            trace_doc = trace_module.run_single_frame_trace(
                data_root=data_root,
                scene_id=scene_id,
                image_name=image_name,
                output_dir=output_dir,
                referability_cache=referability_cache,
                use_occlusion=False,
            )

        self.assertEqual(trace_doc["status"], "completed")
        lifecycle = trace_doc["question_lifecycle"]
        removed = [row for row in lifecycle if row["status"] == "removed"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["removal_reason"], "near_duplicate")
        self.assertIn("same scene/frame/type", removed[0]["removal_detail"])
        self.assertTrue(removed[0]["duplicate_of_trace_question_id"])
        self.assertIn("attachment-id signature", removed[0]["removal_detail"])

        html_text = (output_dir / "single_frame" / scene_id / "000123" / "trace.html").read_text(encoding="utf-8")
        trace_json = json.loads((output_dir / "single_frame" / scene_id / "000123" / "trace.json").read_text(encoding="utf-8"))
        quality_filter_audit = json.loads(Path(trace_json["artifacts"]["audits"]["quality_filter"]).read_text(encoding="utf-8"))
        self.assertIn("Removal Detail", html_text)
        self.assertIn("attachment-id signature", html_text)
        self.assertIn("Question-Centric Audit", html_text)
        self.assertEqual(len(quality_filter_audit["removed_questions"]), 1)


if __name__ == "__main__":
    unittest.main()
