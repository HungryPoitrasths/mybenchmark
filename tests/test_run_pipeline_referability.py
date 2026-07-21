import json
import shutil
import tempfile
import unittest
import uuid
from collections import Counter
from pathlib import Path
import contextlib
from unittest.mock import Mock, patch

import numpy as np

import scripts.run_pipeline as run_pipeline_module
import scripts.run_vlm_referability as referability_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose

TEST_TMP_ROOT = Path(__file__).resolve().parent / "_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


def make_camera_pose(image_name: str) -> CameraPose:
    return CameraPose(
        image_name=image_name,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_pose_at(image_name: str, position_x: float) -> CameraPose:
    """Like make_camera_pose, but translated along world x (still forward=+z)."""
    return CameraPose(
        image_name=image_name,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.array([-position_x, 0.0, 0.0], dtype=np.float64),
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


def make_simple_scene(
    scene_id: str,
    *,
    child_label: str = "chair",
    parent_label: str = "table",
) -> dict:
    return {
        "scene_id": scene_id,
        "objects": [
            make_object(1, child_label),
            make_object(2, parent_label),
        ],
        "attachment_edges": [
            {"parent_id": 2, "child_id": 1, "type": "attachment"},
        ],
        "room_bounds": None,
        "wall_objects": [],
    }


def make_simple_referability_cache(scene_frames: dict[str, list[str]]) -> dict:
    cache_frames: dict[str, dict[str, dict]] = {}
    for scene_id, image_names in scene_frames.items():
        cache_frames[scene_id] = {}
        for image_name in image_names:
            cache_frames[scene_id][image_name] = {
                "frame_usable": True,
                "candidate_visible_object_ids": [1, 2],
                "crop_label_statuses": {"chair": "unique", "table": "unique"},
                "crop_label_counts": {"chair": 1, "table": 1},
                "crop_referable_object_ids": [1, 2],
                "full_frame_label_reviews": [],
                "full_frame_label_statuses": {},
                "full_frame_label_counts": {},
                "attachment_referable_object_ids": [1, 2],
                "referable_object_ids": [1, 2],
                "label_statuses": {"chair": "unique", "table": "unique"},
                "label_counts": {"chair": 1, "table": 1},
                "out_of_frame_label_reviews": [],
                "out_of_frame_not_visible_labels": [],
                "out_of_frame_label_to_object_ids": {},
                "out_of_frame_vlm_early_stop": False,
                "candidate_labels": ["chair", "table"],
                "label_to_object_ids": {"chair": [1], "table": [2]},
            }
    return {
        "version": "20.0",
        "frames": cache_frames,
    }


def write_neighbor_edited_html(
    cache_path: Path,
    html_text: str | None = None,
    filename: str = "edited.html",
) -> Path:
    edited_html_path = cache_path.parent / filename
    edited_html_path.write_text(
        html_text or "<!doctype html><html lang=\"en\"><body></body></html>",
        encoding="utf-8",
    )
    return edited_html_path


def write_scene_edited_html(
    cache_path: Path,
    scene_id: str,
    html_text: str | None = None,
) -> Path:
    edited_html_path = run_pipeline_module._referability_cache_scene_edited_html_path(
        cache_path,
        scene_id,
    )
    edited_html_path.write_text(
        html_text or "<!doctype html><html lang=\"en\"><body></body></html>",
        encoding="utf-8",
    )
    return edited_html_path


def make_referability_batch_doc(
    *,
    scene_id: str,
    model: str = "fake-vlm",
    alias_config_version: str = "test-alias",
) -> dict:
    return {
        "version": run_pipeline_module.EXPECTED_REFERABILITY_CACHE_VERSION,
        "model": model,
        "alias_config_version": alias_config_version,
        "referability_backend": "crop_vlm_with_mesh_ray",
        "label_batch_size": 1,
        "frames": {},
        "scene_grouping": {
            scene_id: {
                "scene_id": scene_id,
                "split": "train",
                "pipeline_outcome": "processed",
                "scene_skip_reason": None,
                "final_cacheable_frame_count": 0,
            }
        },
        "scene_status": {
            scene_id: {
                "scene_id": scene_id,
                "processed": True,
                "pipeline_outcome": "processed",
                "split": "train",
                "has_cache_frames": False,
                "final_cacheable_frame_count": 0,
                "scene_skip_reason": None,
            }
        },
    }


def make_attachment_pair_review_html(
    *,
    scene_id: str,
    image_name: str,
    parent_id: int,
    parent_label: str,
    parent_surface_text: str,
    child_id: int,
    child_label: str,
    child_surface_text: str,
    deleted: bool = False,
) -> str:
    deleted_value = "true" if deleted else "false"
    return f"""<!doctype html>
<html lang="en">
<body>
  <article class="pair-card" data-scene-id="{scene_id}" data-image-name="{image_name}" data-group-id="{scene_id}:group_0" data-pair-id="{parent_id}-&gt;{child_id}" data-parent-id="{parent_id}" data-parent-label="{parent_label}" data-child-id="{child_id}" data-child-label="{child_label}" data-deleted="{deleted_value}">
    <input type="text" name="parent_surface_text" value="{parent_surface_text}">
    <input type="text" name="child_surface_text" value="{child_surface_text}">
  </article>
</body>
</html>"""


class RunPipelineReferabilityTests(unittest.TestCase):
    def test_build_reasoning_context_filters_mesh_visible_ids_by_bbox_ratio(self) -> None:
        image_name = "000123.jpg"
        objects = [make_object(1, "table"), make_object(2, "toilet paper")]
        entry = {
            "frame_usable": True,
            "candidate_visibility_source": "mesh_ray_refined",
            "candidate_visible_object_ids": [1, 2],
            "referable_object_ids": [1],
            "attachment_referable_object_ids": [],
            "visibility_audit_by_object_id": {
                "1": {"obj_id": 1, "bbox_in_frame_ratio": 0.20},
                "2": {"obj_id": 2, "bbox_in_frame_ratio": 0.199},
            },
        }

        contexts = run_pipeline_module._build_reasoning_frame_contexts(
            frames=[{"image_name": image_name}],
            scene_frames={image_name: entry},
            poses={image_name: make_camera_pose(image_name)},
            scene_objects=objects,
            color_intrinsics=make_camera_intrinsics(),
        )

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].cross_frame_visible_ids, frozenset({1}))

    def test_build_reasoning_context_uses_independent_cross_frame_bbox_threshold(self) -> None:
        image_name = "000123.jpg"
        entry = {
            "frame_usable": True,
            "candidate_visibility_source": "mesh_ray_refined",
            "candidate_visible_object_ids": [1],
            "referable_object_ids": [1],
            "attachment_referable_object_ids": [],
            "visibility_audit_by_object_id": {
                "1": {"obj_id": 1, "bbox_in_frame_ratio": 0.70},
            },
        }

        with patch.object(
            run_pipeline_module,
            "CROSS_FRAME_EXCLUSION_BBOX_IN_FRAME_RATIO_MIN",
            0.75,
        ):
            contexts = run_pipeline_module._build_reasoning_frame_contexts(
                frames=[{"image_name": image_name}],
                scene_frames={image_name: entry},
                poses={image_name: make_camera_pose(image_name)},
                scene_objects=[make_object(1, "table")],
                color_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].cross_frame_visible_ids, frozenset())

    def test_build_reasoning_context_rejects_untrusted_visibility_source(self) -> None:
        image_name = "000123.jpg"
        entry = {
            "frame_usable": True,
            "candidate_visibility_source": "projection_fallback",
            "candidate_visible_object_ids": [1],
            "referable_object_ids": [1],
            "attachment_referable_object_ids": [],
            "visibility_audit_by_object_id": {
                "1": {"obj_id": 1, "bbox_in_frame_ratio": 1.0},
            },
        }

        contexts = run_pipeline_module._build_reasoning_frame_contexts(
            frames=[{"image_name": image_name}],
            scene_frames={image_name: entry},
            poses={image_name: make_camera_pose(image_name)},
            scene_objects=[make_object(1, "table")],
            color_intrinsics=make_camera_intrinsics(),
        )

        self.assertEqual(contexts, [])

    def test_cross_only_resource_requirements_follow_requested_types(self) -> None:
        self.assertEqual(
            run_pipeline_module._scene_resource_requirements(
                single_frame_requested_types=[],
                cross_frame_requested_types=["L3_coordinate_rotation_agent"],
                occlusion_backend="mesh_ray",
            ),
            (False, False),
        )
        self.assertEqual(
            run_pipeline_module._scene_resource_requirements(
                single_frame_requested_types=[],
                cross_frame_requested_types=["L2_object_move_distance"],
                occlusion_backend="mesh_ray",
            ),
            (False, False),
        )
        self.assertEqual(
            run_pipeline_module._scene_resource_requirements(
                single_frame_requested_types=[],
                cross_frame_requested_types=["L2_object_move_occlusion"],
                occlusion_backend="mesh_ray",
            ),
            (True, True),
        )
        self.assertEqual(
            run_pipeline_module._scene_resource_requirements(
                single_frame_requested_types=[],
                cross_frame_requested_types=["L1_distance"],
                occlusion_backend="mesh_ray",
            ),
            (False, True),
        )

    def test_l1_non_occlusion_types_are_cross_frame_only(self) -> None:
        cross_frame_l1_types = {
            "L1_direction_agent",
            "L1_distance",
            "L1_direction_object_centric",
            "L1_direction_allocentric",
        }

        self.assertTrue(
            cross_frame_l1_types <= run_pipeline_module.CROSS_FRAME_PUBLIC_QUESTION_TYPES
        )
        self.assertTrue(
            cross_frame_l1_types.isdisjoint(
                run_pipeline_module.SINGLE_FRAME_PUBLIC_QUESTION_TYPES
            )
        )
        self.assertIn(
            "L1_occlusion",
            run_pipeline_module.SINGLE_FRAME_PUBLIC_QUESTION_TYPES,
        )
        self.assertNotIn(
            "L1_occlusion",
            run_pipeline_module.CROSS_FRAME_PUBLIC_QUESTION_TYPES,
        )

    def test_scene_status_v5_is_rejected_after_l1_cap_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "scene_status.json"
            status_path.write_text(
                json.dumps({"version": 5, "completed_scenes": {}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "expected 6"):
                run_pipeline_module._load_pipeline_scene_status_doc(status_path)

    def test_scene_type_cap_limits_all_l1_types(self) -> None:
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": f"d-{idx}.jpg",
                "type": "direction_agent",
                "obj_a_id": idx,
                "question": f"d {idx}",
            }
            for idx in range(7)
        ] + [
            {
                "scene_id": "scene0000_00",
                "image_name": f"m-{idx}.jpg",
                "type": "distance",
                "obj_a_id": idx,
                "question": f"m {idx}",
            }
            for idx in range(6)
        ] + [
            {
                "scene_id": "scene0000_00",
                "image_name": f"o-{idx}.jpg",
                "type": "occlusion",
                "obj_a_id": idx,
                "question": f"o {idx}",
            }
            for idx in range(7)
        ]

        kept = run_pipeline_module._apply_scene_type_cap(
            questions,
            scene_type_cap=5,
        )

        self.assertEqual([q["question"] for q in kept if q["type"] == "direction_agent"], [f"d {idx}" for idx in range(5)])
        self.assertEqual([q["question"] for q in kept if q["type"] == "distance"], [f"m {idx}" for idx in range(5)])
        self.assertEqual([q["question"] for q in kept if q["type"] == "occlusion"], [f"o {idx}" for idx in range(5)])

    def test_scene_type_cap_does_not_limit_object_move_or_rotate_types(self) -> None:
        questions = [
            {"scene_id": "scene0000_00", "type": "object_move_object_centric", "question": f"move {idx}"}
            for idx in range(6)
        ] + [
            {"scene_id": "scene0000_00", "type": "object_rotate_object_centric", "question": f"rotate {idx}"}
            for idx in range(6)
        ]

        kept = run_pipeline_module._apply_scene_type_cap(
            questions,
            scene_type_cap=5,
        )

        self.assertEqual(len(kept), 12)
        self.assertEqual(
            [q["question"] for q in kept if q["type"] == "object_move_object_centric"],
            [f"move {idx}" for idx in range(6)],
        )
        self.assertEqual(
            [q["question"] for q in kept if q["type"] == "object_rotate_object_centric"],
            [f"rotate {idx}" for idx in range(6)],
        )

    def test_load_cached_scene_questions_applies_scene_type_cap(self) -> None:
        root = make_case_dir("cached_scene_cap")
        self.addCleanup(shutil.rmtree, root, True)
        raw_questions_dir = root / "_raw_questions_scene_cache"
        raw_questions_dir.mkdir()
        scene_id = "scene0000_00"
        cached_questions = [
            {
                "scene_id": scene_id,
                "image_name": f"{idx:06d}.jpg",
                "type": "occlusion",
                "question": f"occlusion {idx}",
                "obj_a_id": idx,
            }
            for idx in range(8)
        ]
        (raw_questions_dir / f"{scene_id}.json").write_text(
            json.dumps(cached_questions),
            encoding="utf-8",
        )

        loaded, raw_count = run_pipeline_module._load_cached_scene_questions(
            raw_questions_dir,
            scene_ids=[scene_id],
            scene_type_cap=5,
            frame_type_cap=0,
            frame_type_object_cap=0,
        )

        self.assertEqual(raw_count, 8)
        self.assertEqual(len(loaded), 5)
        self.assertEqual([q["question"] for q in loaded], [f"occlusion {idx}" for idx in range(5)])

    def test_remaining_scene_type_budgets_tracks_all_requested_l1_types(self) -> None:
        budgets = run_pipeline_module._remaining_scene_type_budgets(
            Counter(
                {
                    "occlusion": 3,
                    "object_move_agent": 99,
                    "distance": 4,
                }
            ),
            scene_type_cap=5,
            allowed_types={"occlusion", "distance", "object_move_agent"},
        )

        self.assertEqual(budgets, {"distance": 1, "occlusion": 2})

    def test_scene_cap_only_bounds_l1_while_l2_l3_use_object_frame_caps(self) -> None:
        # The configurable scene cap remains L1-only. L2/L3 instead use the
        # fixed per-object/frame limits applied by the incremental cap helper.
        questions = [
            {"scene_id": "scene0000_00", "type": "occlusion", "question": f"o {idx}"}
            for idx in range(7)
        ] + [
            {
                "scene_id": "scene0000_00",
                "type": "object_move_object_centric",
                "query_obj_id": 1,
                "question": f"move {idx}",
            }
            for idx in range(7)
        ] + [
            {
                "scene_id": "scene0000_00",
                "type": "attachment_chain",
                "grandparent_id": 1,
                "parent_id": idx + 10,
                "question": f"chain {idx}",
            }
            for idx in range(7)
        ]

        kept = run_pipeline_module._apply_scene_type_cap(
            questions,
            scene_type_cap=5,
        )

        self.assertEqual(
            [q["question"] for q in kept if q["type"] == "occlusion"],
            [f"o {idx}" for idx in range(5)],
        )
        self.assertEqual(
            [q["question"] for q in kept if q["type"] == "object_move_object_centric"],
            ["move 0", "move 1"],
        )
        self.assertEqual(
            [q["question"] for q in kept if q["type"] == "attachment_chain"],
            ["chain 0"],
        )

    def test_remaining_scene_type_budgets_omits_l2_l3_types(self) -> None:
        budgets = run_pipeline_module._remaining_scene_type_budgets(
            Counter({"occlusion": 3, "object_move_agent": 99, "attachment_chain": 99}),
            scene_type_cap=5,
            allowed_types={"occlusion", "object_move_agent", "attachment_chain"},
        )

        self.assertEqual(budgets, {"occlusion": 2})

    def test_remaining_scene_type_budgets_returns_none_when_no_l1_types_targeted(self) -> None:
        budgets = run_pipeline_module._remaining_scene_type_budgets(
            Counter({"object_move_agent": 99}),
            scene_type_cap=5,
            allowed_types={"object_move_agent"},
        )

        self.assertIsNone(budgets)

    def test_apply_question_dinox_audit_records_all_unique_mentioned_labels(self) -> None:
        chair_mask = np.zeros((20, 30), dtype=bool)
        chair_mask[2:10, 3:11] = True
        table_mask = np.zeros((20, 30), dtype=bool)
        table_mask[5:18, 12:25] = True
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                    {"role": "reference", "label": "table", "obj_id": 2},
                    {"role": "distractor", "label": "chair", "obj_id": 3},
                ],
            }
        ]

        with (
            patch.object(
                run_pipeline_module.cv2,
                "imread",
                return_value=np.zeros((20, 30, 3), dtype=np.uint8),
            ),
            patch.object(
                run_pipeline_module,
                "_call_question_dinox_detection",
                side_effect=[
                    [
                        {
                            "bbox": [3.0, 2.0, 11.0, 10.0],
                            "mask": chair_mask,
                            "score": 0.95,
                            "area_px": int(chair_mask.sum()),
                            "category": "chair",
                        }
                    ],
                    [
                        {
                            "bbox": [12.0, 5.0, 25.0, 18.0],
                            "mask": table_mask,
                            "score": 0.85,
                            "area_px": int(table_mask.sum()),
                            "category": "table",
                        }
                    ],
                ],
            ) as detection_mock,
        ):
            audited = run_pipeline_module._apply_question_dinox_audit(
                questions=questions,
                data_root=Path("data"),
            )

        self.assertEqual(detection_mock.call_count, 2)
        audit = audited[0]["question_dinox_audit"]
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(len(audit["labels"]), 2)
        chair_entry = next(item for item in audit["labels"] if item["label"] == "chair")
        self.assertEqual(chair_entry["roles"], ["distractor", "target"])
        self.assertEqual(chair_entry["mentioned_object_ids"], [1, 3])
        self.assertEqual(chair_entry["raw_detection_count"], 1)
        self.assertEqual(chair_entry["loose_detection_count"], 1)
        self.assertEqual(chair_entry["raw_detections"][0]["mask_bounds_px"], [3, 11, 2, 10])

    def test_apply_question_dinox_audit_reuses_cached_frame_label_results(self) -> None:
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                ],
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "mentioned_objects": [
                    {"role": "reference", "label": "chair", "obj_id": 2},
                ],
            },
        ]
        mask = np.zeros((20, 30), dtype=bool)
        mask[1:6, 4:10] = True

        with (
            patch.object(
                run_pipeline_module.cv2,
                "imread",
                return_value=np.zeros((20, 30, 3), dtype=np.uint8),
            ),
            patch.object(
                run_pipeline_module,
                "_call_question_dinox_detection",
                return_value=[
                    {
                        "bbox": [4.0, 1.0, 10.0, 6.0],
                        "mask": mask,
                        "score": 0.90,
                        "area_px": int(mask.sum()),
                        "category": "chair",
                    }
                ],
            ) as detection_mock,
        ):
            audited = run_pipeline_module._apply_question_dinox_audit(
                questions=questions,
                data_root=Path("data"),
            )

        self.assertEqual(detection_mock.call_count, 1)
        self.assertEqual(audited[0]["question_dinox_audit"]["labels"][0]["raw_detection_count"], 1)
        self.assertEqual(audited[1]["question_dinox_audit"]["labels"][0]["raw_detection_count"], 1)

    def test_apply_question_post_generation_audit_flags_mesh_mismatch_for_review(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:10, 3:11] = True
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                ],
            }
        ]
        frame_context = {
            ("scene0000_00", "000123.jpg"): {
                "scene_dir": Path("data/scene0000_00"),
                "image_path": Path("data/scene0000_00/color/000123.jpg"),
                "objects_by_id": {1: make_object(1, "chair")},
                "pose": make_camera_pose("000123.jpg"),
                "color_intrinsics": make_camera_intrinsics(),
                "visibility_by_obj_id": {
                    1: {
                        "roi_bounds_px": [3, 11, 2, 10],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    }
                },
                "crop_by_obj_id": {
                    1: {
                        "valid": True,
                        "reason": "",
                        "roi_bounds_px": [3, 11, 2, 10],
                        "crop_bounds_px": [1, 13, 0, 12],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    }
                },
                "has_projection_context": True,
            }
        }

        with (
            patch.object(
                run_pipeline_module,
                "_prebuild_question_review_frame_contexts",
                return_value=frame_context,
            ),
            patch.object(
                run_pipeline_module.cv2,
                "imread",
                return_value=np.zeros((20, 30, 3), dtype=np.uint8),
            ),
            patch.object(
                run_pipeline_module,
                "_call_question_dinox_detection",
                return_value=[
                    {
                        "bbox": [3.0, 2.0, 11.0, 10.0],
                        "mask": mask,
                        "score": 0.95,
                        "area_px": int(mask.sum()),
                        "category": "chair",
                    }
                ],
            ),
            patch.object(
                run_pipeline_module,
                "load_instance_mesh_data",
                return_value=object(),
            ),
            patch.object(
                run_pipeline_module,
                "load_scannet_depth_intrinsics",
                return_value=None,
            ),
            patch.object(
                run_pipeline_module,
                "_compute_topology_quality_for_object",
                return_value={"status": "pass", "reason_codes": []},
            ),
            patch.object(
                run_pipeline_module,
                "_strong_detection_min_area",
                return_value=1,
            ),
            patch.object(
                run_pipeline_module,
                "_compute_mesh_mask_quality_for_object",
                return_value={
                    "status": "fail",
                    "reason_codes": ["low_iou", "high_under_coverage"],
                    "iou": 0.2,
                    "under_coverage": 0.6,
                    "over_coverage": 0.1,
                    "area_ratio": 1.2,
                    "depth_bad_ratio": None,
                },
            ),
        ):
            audited = run_pipeline_module._apply_question_post_generation_audit(
                questions=questions,
                data_root=Path("data"),
                output_dir=Path("output"),
            )

        post_review = audited[0]["question_post_generation_review"]
        self.assertEqual(post_review["decision"], "manual_review")
        self.assertIn("mesh_low_iou:chair#1", post_review["reason_codes"])
        self.assertIn("mesh_high_under_coverage:chair#1", post_review["reason_codes"])
        mesh_review = audited[0]["question_mesh_audit"]["objects"][0]
        self.assertEqual(mesh_review["decision"], "manual_review")
        self.assertEqual(mesh_review["reason"], "mesh_mask_mismatch")
        self.assertEqual(mesh_review["mesh_mask_reason_codes"], ["low_iou", "high_under_coverage"])

    def test_apply_question_post_generation_audit_flags_multiple_dinox_detections_for_review(self) -> None:
        mask_a = np.zeros((20, 30), dtype=bool)
        mask_a[2:10, 3:11] = True
        mask_b = np.zeros((20, 30), dtype=bool)
        mask_b[2:10, 15:23] = True
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                ],
            }
        ]
        frame_context = {
            ("scene0000_00", "000123.jpg"): {
                "scene_dir": Path("data/scene0000_00"),
                "image_path": Path("data/scene0000_00/color/000123.jpg"),
                "objects_by_id": {1: make_object(1, "chair")},
                "pose": make_camera_pose("000123.jpg"),
                "color_intrinsics": make_camera_intrinsics(),
                "visibility_by_obj_id": {
                    1: {
                        "roi_bounds_px": [3, 11, 2, 10],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    }
                },
                "crop_by_obj_id": {
                    1: {
                        "valid": True,
                        "reason": "",
                        "roi_bounds_px": [3, 11, 2, 10],
                        "crop_bounds_px": [1, 13, 0, 12],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    }
                },
                "has_projection_context": True,
            }
        }

        with (
            patch.object(
                run_pipeline_module,
                "_prebuild_question_review_frame_contexts",
                return_value=frame_context,
            ),
            patch.object(
                run_pipeline_module.cv2,
                "imread",
                return_value=np.zeros((20, 30, 3), dtype=np.uint8),
            ),
            patch.object(
                run_pipeline_module,
                "_call_question_dinox_detection",
                return_value=[
                    {
                        "bbox": [3.0, 2.0, 11.0, 10.0],
                        "mask": mask_a,
                        "score": 0.95,
                        "area_px": int(mask_a.sum()),
                        "category": "chair",
                    },
                    {
                        "bbox": [15.0, 2.0, 23.0, 10.0],
                        "mask": mask_b,
                        "score": 0.96,
                        "area_px": int(mask_b.sum()),
                        "category": "chair",
                    },
                ],
            ),
            patch.object(
                run_pipeline_module,
                "load_instance_mesh_data",
                return_value=object(),
            ),
            patch.object(
                run_pipeline_module,
                "load_scannet_depth_intrinsics",
                return_value=None,
            ),
            patch.object(
                run_pipeline_module,
                "_compute_topology_quality_for_object",
                return_value={"status": "pass", "reason_codes": []},
            ),
            patch.object(
                run_pipeline_module,
                "_strong_detection_min_area",
                return_value=1,
            ),
            patch.object(
                run_pipeline_module,
                "_compute_mesh_mask_quality_for_object",
                return_value={
                    "status": "pass",
                    "reason_codes": [],
                    "iou": 0.9,
                    "under_coverage": 0.1,
                    "over_coverage": 0.1,
                    "area_ratio": 1.0,
                    "depth_bad_ratio": None,
                },
            ),
        ):
            audited = run_pipeline_module._apply_question_post_generation_audit(
                questions=questions,
                data_root=Path("data"),
                output_dir=Path("output"),
            )

        post_review = audited[0]["question_post_generation_review"]
        self.assertEqual(post_review["decision"], "manual_review")
        self.assertIn("dinox_multiple_strong_detections:chair", post_review["reason_codes"])
        label_review = audited[0]["question_dinox_audit"]["labels"][0]
        self.assertEqual(label_review["decision"], "manual_review")
        self.assertEqual(label_review["strong_detection_count"], 2)
        self.assertEqual(label_review["matched_object_ids"], [1])

    def test_apply_question_post_generation_audit_flags_partial_same_label_match_for_review(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:10, 3:11] = True
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "mentioned_objects": [
                    {"role": "target", "label": "chair", "obj_id": 1},
                    {"role": "distractor", "label": "chair", "obj_id": 3},
                ],
            }
        ]
        frame_context = {
            ("scene0000_00", "000123.jpg"): {
                "scene_dir": Path("data/scene0000_00"),
                "image_path": Path("data/scene0000_00/color/000123.jpg"),
                "objects_by_id": {
                    1: make_object(1, "chair"),
                    3: make_object(3, "chair"),
                },
                "pose": make_camera_pose("000123.jpg"),
                "color_intrinsics": make_camera_intrinsics(),
                "visibility_by_obj_id": {
                    1: {
                        "roi_bounds_px": [3, 11, 2, 10],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    },
                    3: {
                        "roi_bounds_px": [20, 28, 2, 10],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    },
                },
                "crop_by_obj_id": {
                    1: {
                        "valid": True,
                        "reason": "",
                        "roi_bounds_px": [3, 11, 2, 10],
                        "crop_bounds_px": [1, 13, 0, 12],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    },
                    3: {
                        "valid": True,
                        "reason": "",
                        "roi_bounds_px": [20, 28, 2, 10],
                        "crop_bounds_px": [18, 30, 0, 12],
                        "projected_area_px": 900.0,
                        "bbox_in_frame_ratio": 0.9,
                    },
                },
                "has_projection_context": True,
            }
        }

        with (
            patch.object(
                run_pipeline_module,
                "_prebuild_question_review_frame_contexts",
                return_value=frame_context,
            ),
            patch.object(
                run_pipeline_module.cv2,
                "imread",
                return_value=np.zeros((20, 30, 3), dtype=np.uint8),
            ),
            patch.object(
                run_pipeline_module,
                "_call_question_dinox_detection",
                return_value=[
                    {
                        "bbox": [3.0, 2.0, 11.0, 10.0],
                        "mask": mask,
                        "score": 0.95,
                        "area_px": int(mask.sum()),
                        "category": "chair",
                    }
                ],
            ),
            patch.object(
                run_pipeline_module,
                "_strong_detection_min_area",
                return_value=1,
            ),
            patch.object(
                run_pipeline_module,
                "_run_question_post_mesh_stage",
                return_value={
                    "audit": {"status": "skipped", "objects": []},
                    "reason_codes": [],
                    "flagged_labels": [],
                    "flagged_object_ids": [],
                },
            ),
        ):
            audited = run_pipeline_module._apply_question_post_generation_audit(
                questions=questions,
                data_root=Path("data"),
                output_dir=Path("output"),
            )

        post_review = audited[0]["question_post_generation_review"]
        self.assertEqual(post_review["decision"], "manual_review")
        self.assertIn("dinox_detection_misses_target:chair", post_review["reason_codes"])
        label_review = audited[0]["question_dinox_audit"]["labels"][0]
        self.assertEqual(label_review["decision"], "manual_review")
        self.assertEqual(label_review["matched_object_ids"], [1])
        self.assertEqual(label_review["unmatched_object_ids"], [3])

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

    def test_build_frame_debug_entry_records_attachment_referability_fields(self) -> None:
        objects = [make_object(9, "desk"), make_object(31, "monitor")]
        entry = run_pipeline_module._build_frame_debug_entry(
            image_name="1942.jpg",
            scene_objects=objects,
            objects_by_id={int(obj["id"]): obj for obj in objects},
            selector_visible_ids=[9, 31],
            pipeline_visible_ids=[9, 31],
            occlusion_eligible_object_ids=[9, 31],
            pipeline_referable_object_ids=[9],
            pipeline_attachment_referable_object_ids=[9, 31],
            pipeline_attachment_referable_pairs=[(9, 31)],
            referability_entry={
                "referable_object_ids": [9],
                "attachment_referable_object_ids": [9, 31],
                "attachment_referable_pairs": [[9, 31]],
            },
            frame_attachment_rows=[],
        )

        self.assertEqual(entry["pipeline_attachment_referable_object_ids_used_for_generation"], [9, 31])
        self.assertEqual(entry["pipeline_attachment_referable_pairs_used_for_generation"], [[9, 31]])
        self.assertEqual(entry["attachment_referable_object_ids"], [9, 31])
        self.assertEqual(entry["attachment_referable_pairs"], [[9, 31]])

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
        self.assertIn(
            "recognizable as the given label from the crop itself",
            prompt_lower,
        )
        self.assertIn(
            "not provide enough evidence to tell that the object is the given label",
            prompt_lower,
        )
        self.assertIn(
            "even if the blocking object is very small",
            prompt_lower,
        )
        self.assertIn(
            "different object or label",
            prompt_lower,
        )

    def test_should_run_question_presence_review_only_for_l1_occlusion(self) -> None:
        self.assertTrue(
            run_pipeline_module._should_run_question_presence_review(
                {"level": "L1", "type": "occlusion"}
            )
        )
        self.assertFalse(
            run_pipeline_module._should_run_question_presence_review(
                {"level": "L2", "type": "occlusion"}
            )
        )
        self.assertFalse(
            run_pipeline_module._should_run_question_presence_review(
                {"level": "L1", "type": "distance"}
            )
        )

    def test_should_run_attachment_pair_review_skips_self_pairs_and_non_l2_cases(self) -> None:
        self.assertTrue(
            run_pipeline_module._should_run_attachment_pair_review(
                {
                    "level": "L2",
                    "type": "object_move_agent",
                    "moved_obj_id": 1,
                    "query_obj_id": 2,
                }
            )
        )
        self.assertFalse(
            run_pipeline_module._should_run_attachment_pair_review(
                {
                    "level": "L2",
                    "type": "object_move_agent",
                    "moved_obj_id": 1,
                    "query_obj_id": 1,
                }
            )
        )
        self.assertFalse(
            run_pipeline_module._should_run_attachment_pair_review(
                {
                    "level": "L1",
                    "type": "object_move_agent",
                    "moved_obj_id": 1,
                    "query_obj_id": 2,
                }
            )
        )
        self.assertFalse(
            run_pipeline_module._should_run_attachment_pair_review(
                {
                    "level": "L2",
                    "type": "viewpoint_move",
                    "moved_obj_id": 1,
                    "query_obj_id": 2,
                }
            )
        )

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
        self.assertNotIn("question_answer_review", reviewed)

    def test_run_question_presence_review_writes_combined_payloads(self) -> None:
        root = make_case_dir("question_presence_review_combined")
        self.addCleanup(shutil.rmtree, root, True)
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        reviewed_presence_questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "question": "Is the chair left of the table?",
                "level": "L1",
                "type": "occlusion",
                "manual_review_reason": "existing manual note",
                "question_presence_review": {
                    "review_mode": "instance",
                    "decision": "pass",
                    "flagged_labels": [],
                    "flagged_object_ids": [],
                    "object_reviews": [],
                    "raw_response": "",
                },
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000124.jpg",
                "question": "Is the lamp near the sofa?",
                "level": "L1",
                "type": "occlusion",
                "question_post_generation_review": {
                    "decision": "manual_review",
                    "reason_codes": ["mesh_low_iou:lamp#3"],
                },
                "question_presence_review": {
                    "review_mode": "instance",
                    "decision": "manual_review",
                    "flagged_labels": ["lamp"],
                    "flagged_object_ids": [3],
                    "object_reviews": [],
                    "raw_response": "",
                },
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000125.jpg",
                "question": "Is the desk behind the chair?",
                "level": "L1",
                "type": "occlusion",
                "question_presence_review": {
                    "review_mode": "instance",
                    "decision": "pass",
                    "flagged_labels": [],
                    "flagged_object_ids": [],
                    "object_reviews": [],
                    "raw_response": "",
                },
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000128.jpg",
                "question": "If the bed moves, where is the pillow relative to the chair?",
                "level": "L2",
                "type": "object_move_agent",
                "manual_review_reason": "Attachment-pair review flagged: not a distinct pair",
                "question_attachment_pair_review": {
                    "decision": "manual_review",
                    "reason": "not a distinct pair",
                    "moved_obj_id": 1,
                    "query_obj_id": 2,
                    "moved_obj_label": "bed",
                    "query_obj_label": "pillow",
                    "raw_response": '{"decision":"manual_review","reason":"not a distinct pair"}',
                },
            },
        ]
        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "question": "Is the chair left of the table?",
                "level": "L1",
                "type": "occlusion",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000124.jpg",
                "question": "Is the lamp near the sofa?",
                "level": "L1",
                "type": "occlusion",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000125.jpg",
                "question": "Is the desk behind the chair?",
                "level": "L1",
                "type": "occlusion",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000126.jpg",
                "question": "How far is the desk from the chair?",
                "level": "L1",
                "type": "distance",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000127.jpg",
                "question": "Is the chair occluded after moving the table?",
                "level": "L2",
                "type": "occlusion",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000128.jpg",
                "question": "If the bed moves, where is the pillow relative to the chair?",
                "level": "L2",
                "type": "object_move_agent",
                "moved_obj_id": 1,
                "query_obj_id": 2,
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000129.jpg",
                "question": "If the bed moves, where is the bed relative to itself?",
                "level": "L2",
                "type": "object_move_agent",
                "moved_obj_id": 1,
                "query_obj_id": 1,
            },
        ]

        def fake_presence_review(_review_fn, *, question_index: int, question: dict[str, object], **kwargs):
            result = dict(reviewed_presence_questions[question_index])
            result["benchmark_index"] = question_index
            self.assertEqual(question["question"], reviewed_presence_questions[question_index]["question"])
            return result

        def fake_attachment_pair_review(_review_fn, *, question_index: int, question: dict[str, object], **kwargs):
            self.assertEqual(question_index, 5)
            self.assertEqual(question["question"], reviewed_presence_questions[3]["question"])
            result = dict(reviewed_presence_questions[3])
            result["benchmark_index"] = question_index
            return result

        with (
            patch.object(run_pipeline_module, "_resolve_question_review_vlm", return_value=(Mock(), "fake-vlm")),
            patch.object(run_pipeline_module, "_make_question_presence_reviewer", return_value=("fake-vlm", Mock())),
            patch.object(run_pipeline_module, "_make_attachment_pair_reviewer", return_value=("fake-vlm", Mock())),
            patch.object(run_pipeline_module, "_prebuild_question_review_frame_contexts", return_value={}),
            patch.object(run_pipeline_module, "_review_question_object_presence", side_effect=fake_presence_review),
            patch.object(run_pipeline_module, "_review_question_attachment_pair", side_effect=fake_attachment_pair_review),
            patch("scripts.make_viewer.build_viewer_html", return_value="<html>flagged</html>"),
        ):
            result = run_pipeline_module._run_question_presence_review(
                questions=questions,
                data_root=Path("."),
                output_dir=output_dir,
                vlm_url="http://fake-vlm.local",
                vlm_model="fake-vlm",
                workers=1,
            )

        self.assertEqual(result["model"], "fake-vlm")
        self.assertEqual(result["reviewed_question_count"], 4)
        self.assertEqual(result["manual_review_count"], 3)
        self.assertEqual(result["referability_issue_count"], 1)
        self.assertEqual(result["attachment_pair_issue_count"], 1)
        self.assertEqual(result["post_generation_issue_count"], 1)
        self.assertNotIn("answer_review_model", result)
        self.assertNotIn("answer_review_question_count", result)
        self.assertNotIn("answer_mismatch_count", result)
        self.assertEqual(len(result["questions"]), 4)
        self.assertTrue(result["review_json_path"].exists())
        self.assertTrue(result["flagged_json_path"].exists())
        self.assertTrue(result["flagged_html_path"].exists())
        self.assertFalse((output_dir / "attachment_pair_review.json").exists())

        review_payload = json.loads(result["review_json_path"].read_text(encoding="utf-8"))
        flagged_payload = json.loads(result["flagged_json_path"].read_text(encoding="utf-8"))

        self.assertEqual(review_payload["model"], "fake-vlm")
        self.assertEqual(review_payload["reviewed_question_count"], 4)
        self.assertEqual(review_payload["manual_review_count"], 3)
        self.assertEqual(review_payload["referability_issue_count"], 1)
        self.assertEqual(review_payload["attachment_pair_issue_count"], 1)
        self.assertEqual(review_payload["post_generation_issue_count"], 1)
        self.assertNotIn("answer_review_model", review_payload)
        self.assertNotIn("answer_review_question_count", review_payload)
        self.assertNotIn("answer_mismatch_count", review_payload)
        self.assertTrue(all("question_answer_review" not in q for q in review_payload["questions"]))
        self.assertEqual(
            [question["question"] for question in review_payload["questions"]],
            [
                "Is the chair left of the table?",
                "Is the lamp near the sofa?",
                "Is the desk behind the chair?",
                "If the bed moves, where is the pillow relative to the chair?",
            ],
        )

        self.assertEqual(flagged_payload["reviewed_question_count"], 4)
        self.assertEqual(flagged_payload["manual_review_count"], 3)
        self.assertEqual(flagged_payload["attachment_pair_issue_count"], 1)
        self.assertEqual(len(flagged_payload["questions"]), 3)
        self.assertEqual(
            [question["question"] for question in flagged_payload["questions"]],
            [
                "Is the chair left of the table?",
                "Is the lamp near the sofa?",
                "If the bed moves, where is the pillow relative to the chair?",
            ],
        )
        self.assertTrue(all("question_answer_review" not in q for q in flagged_payload["questions"]))

    def test_run_question_presence_review_only_reviews_non_self_l2_attachment_pairs(self) -> None:
        root = make_case_dir("question_presence_review_non_self_attachment_only")
        self.addCleanup(shutil.rmtree, root, True)
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        questions = [
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "level": "L2",
                "type": "object_move_agent",
                "question": "If the bed moves, where is the pillow relative to the chair?",
                "moved_obj_id": 1,
                "moved_obj_label": "bed",
                "query_obj_id": 2,
                "query_obj_label": "pillow",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "level": "L2",
                "type": "object_move_agent",
                "question": "If the bed moves, where is the chair relative to itself?",
                "moved_obj_id": 1,
                "moved_obj_label": "bed",
                "query_obj_id": 1,
                "query_obj_label": "bed",
            },
            {
                "scene_id": "scene0000_00",
                "image_name": "000123.jpg",
                "level": "L2",
                "type": "occlusion",
                "question": "Is the chair occluded after moving the bed?",
                "moved_obj_id": 1,
                "query_obj_id": 2,
            },
        ]

        reviewed_non_self = {
            **questions[0],
            "benchmark_index": 0,
            "question_attachment_pair_review": {
                "decision": "pass",
                "reason": "distinct pair",
                "moved_obj_id": 1,
                "query_obj_id": 2,
                "moved_obj_label": "bed",
                "query_obj_label": "pillow",
                "raw_response": '{"decision":"pass","reason":"distinct pair"}',
            },
        }

        def fake_review(_review_fn, *, question_index: int, question: dict[str, object], **kwargs):
            self.assertEqual(question_index, 0)
            self.assertEqual(question["question"], questions[0]["question"])
            return reviewed_non_self

        with (
            patch.object(run_pipeline_module, "_resolve_question_review_vlm", return_value=(Mock(), "fake-vlm")),
            patch.object(run_pipeline_module, "_make_question_presence_reviewer", return_value=("fake-vlm", Mock())),
            patch.object(run_pipeline_module, "_make_attachment_pair_reviewer", return_value=("fake-vlm", Mock())),
            patch.object(run_pipeline_module, "_prebuild_question_review_frame_contexts", return_value={}),
            patch.object(run_pipeline_module, "_review_question_attachment_pair", side_effect=fake_review),
            patch("scripts.make_viewer.build_viewer_html", return_value="<html>flagged</html>"),
        ):
            result = run_pipeline_module._run_question_presence_review(
                questions=questions,
                data_root=Path("."),
                output_dir=output_dir,
                vlm_url="http://fake-vlm.local",
                vlm_model="fake-vlm",
                workers=1,
            )

        self.assertEqual(result["model"], "fake-vlm")
        self.assertEqual(result["reviewed_question_count"], 1)
        self.assertEqual(result["manual_review_count"], 0)
        self.assertTrue(result["review_json_path"].exists())

        payload = json.loads(result["review_json_path"].read_text(encoding="utf-8"))
        self.assertEqual(payload["reviewed_question_count"], 1)
        self.assertEqual(payload["manual_review_count"], 0)
        self.assertEqual(len(payload["questions"]), 1)
        self.assertEqual(payload["questions"][0]["question"], questions[0]["question"])
        self.assertEqual(
            payload["questions"][0]["question_attachment_pair_review"]["query_obj_id"],
            2,
        )

    def test_run_question_presence_review_skips_vlm_when_no_targets(self) -> None:
        root = make_case_dir("question_presence_review_no_targets")
        self.addCleanup(shutil.rmtree, root, True)
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(
                run_pipeline_module,
                "_resolve_question_review_vlm",
                side_effect=AssertionError("VLM resolver should not be called when there are no review targets"),
            ),
            patch("scripts.make_viewer.build_viewer_html", return_value="<html>empty</html>"),
        ):
            result = run_pipeline_module._run_question_presence_review(
                questions=[],
                data_root=Path("."),
                output_dir=output_dir,
                vlm_url="http://fake-vlm.local",
                vlm_model=None,
                workers=1,
            )

        self.assertEqual(result["reviewed_question_count"], 0)
        self.assertEqual(result["manual_review_count"], 0)
        self.assertEqual(result["questions"], [])
        self.assertTrue((output_dir / "question_presence_review.json").exists())
        self.assertTrue((output_dir / "question_presence_review_flagged.json").exists())
        self.assertTrue((output_dir / "question_presence_review_flagged.html").exists())

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

    def test_apply_question_referability_filter_uses_role_specific_attachment_pools(self) -> None:
        kept, audited = run_pipeline_module._apply_question_referability_filter(
            [
                {
                    "scene_id": "scene0000_00",
                    "image_name": "000123.jpg",
                    "type": "attachment_chain",
                    "question": "If the table moves, which objects move with it?",
                    "grandparent_id": 1,
                    "grandparent_label": "table",
                    "parent_id": 2,
                    "parent_label": "box",
                    "grandchild_id": 3,
                    "grandchild_label": "cup",
                    "neighbor_id": 4,
                    "neighbor_label": "lamp",
                    "mentioned_objects": [
                        {"role": "grandparent", "label": "table", "obj_id": 1},
                        {"role": "parent", "label": "box", "obj_id": 2},
                        {"role": "grandchild", "label": "cup", "obj_id": 3},
                        {"role": "neighbor", "label": "lamp", "obj_id": 4},
                    ],
                }
            ],
            objects_by_id={
                1: make_object(1, "table"),
                2: make_object(2, "box"),
                3: make_object(3, "cup"),
                4: make_object(4, "lamp"),
            },
            referability_entry={
                "label_statuses": {
                    "table": "unique",
                    "box": "unique",
                    "cup": "unique",
                    "lamp": "unique",
                },
                "label_to_object_ids": {
                    "table": [1],
                    "box": [2],
                    "cup": [3],
                    "lamp": [4],
                },
            },
            frame_referable_ids=[1, 4],
            attachment_frame_referable_ids=[1, 2, 3],
            attachment_frame_referable_pairs=[(1, 2), (2, 3)],
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(audited), 1)
        self.assertEqual(audited[0]["question_referability_audit"]["decision"], "pass")
        self.assertEqual(
            audited[0]["question_referability_audit"]["frame_referable_object_ids"],
            [1, 4],
        )
        self.assertEqual(
            audited[0]["question_referability_audit"]["frame_attachment_referable_object_ids"],
            [1, 2, 3],
        )
        pools = {
            mention["role"]: mention["required_referability_pool"]
            for mention in audited[0]["question_referability_audit"]["mentioned_objects"]
        }
        self.assertEqual(pools["grandparent"], "attachment")
        self.assertEqual(pools["parent"], "attachment")
        self.assertEqual(pools["grandchild"], "attachment")
        self.assertEqual(pools["neighbor"], "ordinary")

    def test_attachment_question_requires_ordinary_reference_to_pass_ordinary_pool(self) -> None:
        question = {
            "type": "object_move_agent",
            "attachment_remapped": True,
            "moved_obj_id": 1,
            "moved_obj_label": "table",
            "query_obj_id": 2,
            "query_obj_label": "box",
            "obj_c_id": 3,
            "obj_c_label": "chair",
            "mentioned_objects": [
                {"role": "moved_object", "label": "table", "obj_id": 1},
                {"role": "query_object", "label": "box", "obj_id": 2},
                {"role": "reference_object", "label": "chair", "obj_id": 3},
            ],
        }
        objects_by_id = {
            1: make_object(1, "table"),
            2: make_object(2, "box"),
            3: make_object(3, "chair"),
        }
        referability_entry = {
            "label_statuses": {
                "table": "unique",
                "box": "unique",
                "chair": "unique",
            },
            "label_to_object_ids": {
                "table": [1],
                "box": [2],
                "chair": [3],
            },
        }

        passing = run_pipeline_module._build_question_referability_audit(
            question,
            objects_by_id=objects_by_id,
            referability_entry=referability_entry,
            frame_referable_ids=[3],
            attachment_frame_referable_ids=[1, 2],
            attachment_frame_referable_pairs=[(1, 2)],
        )
        failing = run_pipeline_module._build_question_referability_audit(
            question,
            objects_by_id=objects_by_id,
            referability_entry=referability_entry,
            frame_referable_ids=[],
            attachment_frame_referable_ids=[1, 2, 3],
            attachment_frame_referable_pairs=[(1, 2)],
        )

        self.assertEqual(passing["decision"], "pass")
        self.assertEqual(failing["decision"], "drop")
        reference_audit = next(
            mention
            for mention in failing["mentioned_objects"]
            if mention["role"] == "reference_object"
        )
        self.assertEqual(reference_audit["required_referability_pool"], "ordinary")
        self.assertIn("mentioned_nonreferable_object", reference_audit["reason_codes"])

    def test_attachment_question_rejects_unreviewed_pair_path(self) -> None:
        audit = run_pipeline_module._build_question_referability_audit(
            {
                "type": "object_move_agent",
                "attachment_remapped": True,
                "moved_obj_id": 1,
                "moved_obj_label": "table",
                "query_obj_id": 3,
                "query_obj_label": "cup",
                "obj_c_id": 4,
                "obj_c_label": "chair",
            },
            objects_by_id={
                1: make_object(1, "table"),
                2: make_object(2, "box"),
                3: make_object(3, "cup"),
                4: make_object(4, "chair"),
            },
            referability_entry={
                "label_statuses": {
                    "table": "unique",
                    "box": "unique",
                    "cup": "unique",
                    "chair": "unique",
                },
                "label_to_object_ids": {
                    "table": [1],
                    "box": [2],
                    "cup": [3],
                    "chair": [4],
                },
            },
            frame_referable_ids=[4],
            attachment_frame_referable_ids=[1, 2, 3],
            attachment_frame_referable_pairs=[(1, 2)],
        )

        self.assertEqual(audit["decision"], "drop")
        self.assertIn("attachment_pair_not_referable", audit["reason_codes"])

    def test_frame_attachment_pairs_treats_explicit_empty_as_authoritative(self) -> None:
        kwargs = {
            "attachment_graph": {1: [2]},
            "attachment_referable_ids": [1, 2],
            "visible_object_ids": [1, 2],
        }
        self.assertEqual(
            run_pipeline_module._frame_attachment_referable_pairs(
                referability_entry={"attachment_referable_pairs": []},
                **kwargs,
            ),
            [],
        )
        self.assertEqual(
            run_pipeline_module._frame_attachment_referable_pairs(
                referability_entry={},
                **kwargs,
            ),
            [(1, 2)],
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
        write_neighbor_edited_html(cache_path)

        with self.assertRaisesRegex(ValueError, "expected 20.0"):
            run_pipeline_module._load_referability_cache(cache_path)

    def test_load_referability_cache_loads_without_legacy_edited_html(self) -> None:
        case_dir = make_case_dir("cache_missing_edited_html")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "20.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        cache = run_pipeline_module._load_referability_cache(cache_path)

        self.assertEqual(cache["version"], "20.0")

    def test_load_referability_cache_accepts_unique_prefixed_legacy_edited_html(self) -> None:
        case_dir = make_case_dir("cache_prefixed_edited_html")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "20.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )
        write_neighbor_edited_html(cache_path, filename="edited_review.html")

        cache = run_pipeline_module._load_referability_cache(cache_path)

        self.assertEqual(cache["version"], "20.0")

    def test_load_referability_cache_rejects_multiple_legacy_edited_html_candidates(self) -> None:
        case_dir = make_case_dir("cache_multiple_edited_html")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps({"version": "20.0", "frames": {}}, ensure_ascii=False),
            encoding="utf-8",
        )
        write_neighbor_edited_html(cache_path, filename="edited.html")
        write_neighbor_edited_html(cache_path, filename="edited_review.html")

        with self.assertRaises(ValueError) as exc_info:
            run_pipeline_module._load_referability_cache(cache_path)

        message = str(exc_info.exception)
        self.assertIn("multiple candidates", message.lower())
        self.assertIn("edited*.html", message)
        self.assertIn(str((case_dir / "edited.html").resolve()), message)
        self.assertIn(str((case_dir / "edited_review.html").resolve()), message)

    def test_load_referability_cache_prefers_legacy_edited_html_over_scene_scoped_html(self) -> None:
        case_dir = make_case_dir("cache_scene_html_preferred")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "flash_batch.json"
        scene_id = "scene0001_00"
        image_name = "000001.jpg"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "frames": {
                        scene_id: {
                            image_name: {
                                "frame_usable": True,
                                "frame_quality_clear": True,
                                "frame_quality_score": 82,
                                "frame_quality_reason": "clear enough",
                                "frame_selection_score": 82001,
                                "attachment_referable_pairs": [],
                                "attachment_referable_pair_count": 0,
                                "attachment_referable_object_ids": [],
                                "attachment_final_referability": {
                                    "referable_object_ids": [],
                                    "pairs": [],
                                    "pair_count": 0,
                                },
                                "final_selection_rank": 0,
                                "candidate_visible_object_ids": [1, 2],
                                "candidate_visibility_source": "mesh_ray_refined",
                                "candidate_labels": ["cup", "table"],
                                "label_to_object_ids": {"cup": [1], "table": [2]},
                                "selector_visible_object_ids": [1, 2],
                                "selector_visible_label_counts": {"cup": 1, "table": 1},
                                "visibility_audit_by_object_id": {},
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "cup",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    },
                                    "2": {
                                        "obj_id": 2,
                                        "label": "table",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
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
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1, 2],
                                "vlm_unique_object_ids": [1, 2],
                            }
                        }
                    },
                    "scene_grouping": {
                        scene_id: {
                            "scene_id": scene_id,
                            "pipeline_outcome": "processed",
                        }
                    },
                    "scene_status": {
                        scene_id: {
                            "scene_id": scene_id,
                            "processed": True,
                            "pipeline_outcome": "processed",
                            "split": "train",
                            "has_cache_frames": True,
                            "final_cacheable_frame_count": 1,
                            "scene_skip_reason": None,
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_neighbor_edited_html(cache_path)
        write_scene_edited_html(
            cache_path,
            scene_id,
            make_attachment_pair_review_html(
                scene_id=scene_id,
                image_name=image_name,
                parent_id=2,
                parent_label="table",
                parent_surface_text="wooden table",
                child_id=1,
                child_label="cup",
                child_surface_text="blue cup",
            ),
        )

        cache = run_pipeline_module._load_referability_cache(
            cache_path,
            repair_inconsistent_entries=True,
        )

        loaded_entry = cache["frames"][scene_id][image_name]
        self.assertEqual(loaded_entry["attachment_referable_pairs"], [])
        self.assertEqual(loaded_entry["attachment_referable_pair_count"], 0)

    def test_load_referability_cache_uses_scene_scoped_html_when_legacy_is_absent(self) -> None:
        case_dir = make_case_dir("cache_scene_html_only")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "flash_batch.json"
        scene_id = "scene0001_00"
        image_name = "000001.jpg"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "frames": {
                        scene_id: {
                            image_name: {
                                "frame_usable": True,
                                "frame_quality_clear": True,
                                "frame_quality_score": 82,
                                "frame_quality_reason": "clear enough",
                                "frame_selection_score": 82001,
                                "attachment_referable_pairs": [],
                                "attachment_referable_pair_count": 0,
                                "attachment_referable_object_ids": [],
                                "attachment_final_referability": {
                                    "referable_object_ids": [],
                                    "pairs": [],
                                    "pair_count": 0,
                                },
                                "final_selection_rank": 0,
                                "candidate_visible_object_ids": [1, 2],
                                "candidate_visibility_source": "mesh_ray_refined",
                                "candidate_labels": ["cup", "table"],
                                "label_to_object_ids": {"cup": [1], "table": [2]},
                                "selector_visible_object_ids": [1, 2],
                                "selector_visible_label_counts": {"cup": 1, "table": 1},
                                "visibility_audit_by_object_id": {},
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "cup",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    },
                                    "2": {
                                        "obj_id": 2,
                                        "label": "table",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
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
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1, 2],
                                "vlm_unique_object_ids": [1, 2],
                            }
                        }
                    },
                    "scene_grouping": {
                        scene_id: {
                            "scene_id": scene_id,
                            "pipeline_outcome": "processed",
                        }
                    },
                    "scene_status": {
                        scene_id: {
                            "scene_id": scene_id,
                            "processed": True,
                            "pipeline_outcome": "processed",
                            "split": "train",
                            "has_cache_frames": True,
                            "final_cacheable_frame_count": 1,
                            "scene_skip_reason": None,
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_scene_edited_html(
            cache_path,
            scene_id,
            make_attachment_pair_review_html(
                scene_id=scene_id,
                image_name=image_name,
                parent_id=2,
                parent_label="table",
                parent_surface_text="wooden table",
                child_id=1,
                child_label="cup",
                child_surface_text="blue cup",
            ),
        )

        cache = run_pipeline_module._load_referability_cache(cache_path)

        loaded_entry = cache["frames"][scene_id][image_name]
        self.assertEqual(loaded_entry["attachment_referable_pairs"], [[2, 1]])
        self.assertEqual(loaded_entry["attachment_referable_object_ids"], [1, 2])

    def test_load_referability_cache_warns_incomplete_scene_scoped_html_set(self) -> None:
        case_dir = make_case_dir("cache_scene_html_incomplete")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "flash_batch.json"
        cache_doc = {
            "version": "20.0",
            "frames": {},
            "scene_grouping": {
                "scene0000_00": {"scene_id": "scene0000_00"},
                "scene0001_00": {"scene_id": "scene0001_00"},
            },
            "scene_status": {
                "scene0000_00": {"scene_id": "scene0000_00"},
                "scene0001_00": {"scene_id": "scene0001_00"},
            },
        }
        cache_path.write_text(json.dumps(cache_doc, ensure_ascii=False), encoding="utf-8")
        write_scene_edited_html(cache_path, "scene0000_00")

        with self.assertLogs(run_pipeline_module.logger, level="WARNING") as log_ctx:
            cache = run_pipeline_module._load_referability_cache(cache_path)

        self.assertEqual(cache["version"], "20.0")
        warning_text = "\n".join(log_ctx.output)
        self.assertIn("scene0001_00", warning_text)
        self.assertIn("缺少按 scene 划分的人工审核文件", warning_text)

    def test_load_referability_cache_scene_scoped_html_warns_missing_zero_frame_scenes(self) -> None:
        case_dir = make_case_dir("cache_scene_html_zero_frames")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "flash_batch.json"
        cache_doc = {
            "version": "20.0",
            "frames": {},
            "scene_grouping": {
                "scene0000_00": {"scene_id": "scene0000_00"},
            },
            "scene_status": {
                "scene0000_00": {"scene_id": "scene0000_00"},
                "scene0002_00": {
                    "scene_id": "scene0002_00",
                    "has_cache_frames": False,
                    "final_cacheable_frame_count": 0,
                },
            },
        }
        cache_path.write_text(json.dumps(cache_doc, ensure_ascii=False), encoding="utf-8")
        write_scene_edited_html(cache_path, "scene0000_00")

        with self.assertLogs(run_pipeline_module.logger, level="WARNING") as log_ctx:
            cache = run_pipeline_module._load_referability_cache(cache_path)

        self.assertEqual(cache["version"], "20.0")
        warning_text = "\n".join(log_ctx.output)
        self.assertIn("scene0002_00", warning_text)
        self.assertIn("缺少按 scene 划分的人工审核文件", warning_text)

    def test_load_referability_cache_rejects_inconsistent_entry_without_repair_flag(self) -> None:
        case_dir = make_case_dir("cache_inconsistent")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "frames": {
                        "scene0000_00": {
                            "000123.jpg": {
                                "frame_usable": True,
                                "label_to_object_ids": {"lamp": [1]},
                                "selector_visible_label_counts": {"lamp": 1},
                                "crop_label_statuses": {"lamp": "unique"},
                                "crop_label_counts": {"lamp": 1},
                                "crop_referable_object_ids": [1],
                                "full_frame_label_reviews": [{"label": "lamp", "status": "absent"}],
                                "full_frame_label_statuses": {"lamp": "absent"},
                                "full_frame_label_counts": {"lamp": 0},
                                "label_statuses": {"lamp": "unique"},
                                "label_counts": {"lamp": 1},
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1],
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "lamp",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    }
                                },
                            }
                        }
                    },
                },
                ensure_ascii=False,
                ),
            encoding="utf-8",
        )
        write_neighbor_edited_html(cache_path)

        with self.assertRaisesRegex(ValueError, "Rerun with --repair_referability_cache"):
            run_pipeline_module._load_referability_cache(cache_path)

    def test_load_referability_cache_validates_merged_cache_after_human_salvage(self) -> None:
        case_dir = make_case_dir("cache_inconsistent_after_merge")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "frames": {
                        scene_id: {
                            image_name: {
                                "frame_usable": True,
                                "label_to_object_ids": {"cup": [1], "table": [2]},
                                "selector_visible_label_counts": {"cup": 1, "table": 1},
                                "crop_label_statuses": {"cup": "unique", "table": "unique"},
                                "crop_label_counts": {"cup": 1, "table": 1},
                                "crop_referable_object_ids": [1, 2],
                                "full_frame_label_reviews": [{"label": "cup", "status": "absent"}],
                                "full_frame_label_statuses": {"cup": "absent"},
                                "full_frame_label_counts": {"cup": 0},
                                "label_statuses": {"cup": "unique", "table": "unique"},
                                "label_counts": {"cup": 1, "table": 1},
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1, 2],
                                "attachment_referable_pairs": [],
                                "attachment_referable_object_ids": [],
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "cup",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    },
                                    "2": {
                                        "obj_id": 2,
                                        "label": "table",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    },
                                },
                            }
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_neighbor_edited_html(
            cache_path,
            make_attachment_pair_review_html(
                scene_id=scene_id,
                image_name=image_name,
                parent_id=2,
                parent_label="table",
                parent_surface_text="wooden table",
                child_id=1,
                child_label="cup",
                child_surface_text="blue cup",
            ),
        )

        with self.assertRaisesRegex(ValueError, "Rerun with --repair_referability_cache"):
            run_pipeline_module._load_referability_cache(cache_path)

    def test_load_referability_cache_can_repair_inconsistent_entry(self) -> None:
        case_dir = make_case_dir("cache_repair")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "frames": {
                        scene_id: {
                            image_name: {
                                "frame_usable": True,
                                "frame_quality_clear": True,
                                "frame_quality_score": 82,
                                "frame_quality_reason": "clear enough",
                                "frame_selection_score": 82001,
                                "attachment_referable_pairs": [],
                                "attachment_referable_pair_count": 0,
                                "final_selection_rank": 0,
                                "candidate_visible_object_ids": [1],
                                "candidate_visibility_source": "mesh_ray_refined",
                                "candidate_labels": ["lamp"],
                                "label_to_object_ids": {"lamp": [1]},
                                "selector_visible_object_ids": [1],
                                "selector_visible_label_counts": {"lamp": 1},
                                "visibility_audit_by_object_id": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "lamp",
                                        "candidate_considered": True,
                                        "candidate_passed": True,
                                        "candidate_rejection_reasons": [],
                                    }
                                },
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "lamp",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    }
                                },
                                "crop_label_statuses": {"lamp": "unique"},
                                "crop_label_counts": {"lamp": 1},
                                "crop_referable_object_ids": [1],
                                "full_frame_label_reviews": [{"label": "lamp", "status": "absent"}],
                                "full_frame_label_statuses": {"lamp": "absent"},
                                "full_frame_label_counts": {"lamp": 0},
                                "label_statuses": {"lamp": "unique"},
                                "label_counts": {"lamp": 1},
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1],
                            }
                        }
                    },
                },
                ensure_ascii=False,
                ),
            encoding="utf-8",
        )
        write_neighbor_edited_html(cache_path)

        cache = run_pipeline_module._load_referability_cache(
            cache_path,
            repair_inconsistent_entries=True,
            persist_repaired_entries=True,
        )

        repaired_entry = cache["frames"][scene_id][image_name]
        self.assertEqual(repaired_entry["label_statuses"], {"lamp": "absent"})
        self.assertEqual(repaired_entry["label_counts"], {"lamp": 0})
        self.assertEqual(repaired_entry["attachment_referable_object_ids"], [])
        self.assertEqual(repaired_entry["referable_object_ids"], [])
        self.assertEqual(repaired_entry["vlm_unique_object_ids"], [])

        persisted_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        persisted_entry = persisted_cache["frames"][scene_id][image_name]
        self.assertEqual(persisted_entry["label_statuses"], {"lamp": "absent"})
        self.assertEqual(persisted_entry["referable_object_ids"], [])

    def test_load_referability_cache_applies_human_salvage_html_without_rewriting_source_json(self) -> None:
        case_dir = make_case_dir("cache_salvage_merge")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        scene_id = "scene0001_00"
        image_name = "000001.jpg"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "frames": {
                        scene_id: {
                            image_name: {
                                "frame_usable": True,
                                "frame_quality_clear": True,
                                "frame_quality_score": 82,
                                "frame_quality_reason": "clear enough",
                                "frame_selection_score": 82001,
                                "attachment_referable_pairs": [],
                                "attachment_referable_pair_count": 0,
                                "attachment_referable_object_ids": [],
                                "attachment_final_referability": {
                                    "referable_object_ids": [],
                                    "pairs": [],
                                    "pair_count": 0,
                                },
                                "final_selection_rank": 0,
                                "candidate_visible_object_ids": [1, 2],
                                "candidate_visibility_source": "mesh_ray_refined",
                                "candidate_labels": ["cup", "table"],
                                "label_to_object_ids": {"cup": [1], "table": [2]},
                                "selector_visible_object_ids": [1, 2],
                                "selector_visible_label_counts": {"cup": 1, "table": 1},
                                "visibility_audit_by_object_id": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "cup",
                                        "candidate_considered": True,
                                        "candidate_passed": True,
                                        "candidate_rejection_reasons": [],
                                    },
                                    "2": {
                                        "obj_id": 2,
                                        "label": "table",
                                        "candidate_considered": True,
                                        "candidate_passed": True,
                                        "candidate_rejection_reasons": [],
                                    },
                                },
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "cup",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    },
                                    "2": {
                                        "obj_id": 2,
                                        "label": "table",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    },
                                },
                                "crop_label_statuses": {"cup": "unique", "table": "unique"},
                                "crop_label_counts": {"cup": 1, "table": 1},
                                "crop_referable_object_ids": [1, 2],
                                "full_frame_label_reviews": [
                                    {"label": "cup", "status": "unique"},
                                    {"label": "table", "status": "unique"},
                                ],
                                "full_frame_label_statuses": {"cup": "unique", "table": "unique"},
                                "full_frame_label_counts": {"cup": 1, "table": 1},
                                "label_statuses": {"cup": "unique", "table": "unique"},
                                "label_counts": {"cup": 1, "table": 1},
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1, 2],
                                "vlm_unique_object_ids": [1, 2],
                            }
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        original_cache_text = cache_path.read_text(encoding="utf-8")
        write_neighbor_edited_html(
            cache_path,
            make_attachment_pair_review_html(
                scene_id=scene_id,
                image_name=image_name,
                parent_id=2,
                parent_label="table",
                parent_surface_text="wooden table",
                child_id=1,
                child_label="cup",
                child_surface_text="blue cup",
            ),
        )

        cache = run_pipeline_module._load_referability_cache(cache_path)

        loaded_entry = cache["frames"][scene_id][image_name]
        self.assertEqual(loaded_entry["attachment_referable_pairs"], [[2, 1]])
        self.assertEqual(loaded_entry["attachment_referable_object_ids"], [1, 2])
        self.assertEqual(
            loaded_entry["attachment_human_review_cards"],
            [
                {
                    "pair_id": "2->1",
                    "parent_id": 2,
                    "parent_label": "table",
                    "parent_surface_text": "wooden table",
                    "child_id": 1,
                    "child_label": "cup",
                    "child_surface_text": "blue cup",
                    "source": "human_salvage_html",
                }
            ],
        )
        self.assertEqual(
            run_pipeline_module._attachment_human_review_surface_text_by_object_id(
                loaded_entry["attachment_human_review_cards"]
            ),
            {1: "blue cup", 2: "wooden table"},
        )
        self.assertEqual(cache_path.read_text(encoding="utf-8"), original_cache_text)

    def test_load_referability_cache_restores_missing_frame_from_sidecar_html_review(self) -> None:
        case_dir = make_case_dir("cache_sidecar_restore")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        scene_id = "scene0000_00"
        image_name = "000124.jpg"
        cache_doc = make_referability_batch_doc(
            scene_id=scene_id,
            model="fake-vlm",
            alias_config_version=referability_module.ALIAS_CONFIG_VERSION,
        )
        cache_doc["scene_grouping"][scene_id]["final_cacheable_frame_image_names"] = ["000123.jpg"]
        cache_doc["scene_grouping"][scene_id]["final_cacheable_frame_count"] = 1
        cache_doc["scene_status"][scene_id]["has_cache_frames"] = True
        cache_doc["scene_status"][scene_id]["final_cacheable_frame_count"] = 1
        cache_doc["frames"][scene_id] = {
            "000123.jpg": {
                "frame_usable": True,
                "frame_quality_clear": True,
                "frame_quality_score": 82,
                "frame_quality_reason": "clear enough",
                "frame_selection_score": 82001,
                "attachment_referable_pairs": [],
                "attachment_referable_pair_count": 0,
                "attachment_final_referability": {
                    "object_ids": [],
                    "pairs": [],
                    "pair_count": 0,
                },
                "final_selection_rank": 0,
                "candidate_visible_object_ids": [1],
                "candidate_visibility_source": "mesh_ray_refined",
                "candidate_labels": ["cup"],
                "label_to_object_ids": {"cup": [1]},
                "selector_visible_object_ids": [1],
                "selector_visible_label_counts": {"cup": 1},
                "visibility_audit_by_object_id": {
                    "1": {
                        "obj_id": 1,
                        "label": "cup",
                        "projected_area_px": 1200.0,
                        "bbox_in_frame_ratio": 0.95,
                    }
                },
                "object_reviews": {},
                "crop_label_statuses": {"cup": "unique"},
                "crop_label_counts": {"cup": 1},
                "crop_referable_object_ids": [1],
                "full_frame_label_reviews": [],
                "full_frame_label_statuses": {"cup": "unique"},
                "full_frame_label_counts": {"cup": 1},
                "label_statuses": {"cup": "unique"},
                "label_counts": {"cup": 1},
                "out_of_frame_label_reviews": [],
                "out_of_frame_not_visible_labels": [],
                "out_of_frame_label_to_object_ids": {},
                "out_of_frame_vlm_early_stop": False,
                "referable_object_ids": [1],
                "vlm_unique_object_ids": [1],
            }
        }
        cache_path.write_text(json.dumps(cache_doc, ensure_ascii=False), encoding="utf-8")

        sidecar_dir = cache_path.parent / referability_module.FRAME_CACHE_SIDECAR_DIR_NAME
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_doc = {
            "scene_id": scene_id,
            "version": run_pipeline_module.EXPECTED_REFERABILITY_CACHE_VERSION,
            "alias_config_version": referability_module.ALIAS_CONFIG_VERSION,
            "referability_backend": "crop_vlm_with_mesh_ray",
            "vlm_model": "fake-vlm",
            "frames": {
                image_name: {
                    "frame_info": {
                        "clear": True,
                        "clarity_score": 88,
                        "frame_usable": True,
                        "reason": "clear",
                    },
                    "frame_selection_score": 82002,
                    "referability_entry": {
                        "frame_usable": True,
                        "frame_quality_clear": True,
                        "frame_quality_score": 88,
                        "frame_quality_reason": "clear",
                        "frame_selection_score": 82002,
                        "attachment_referable_pairs": [],
                        "attachment_referable_pair_count": 0,
                        "attachment_final_referability": {
                            "object_ids": [],
                            "pairs": [],
                            "pair_count": 0,
                        },
                        "final_selection_rank": 1,
                        "candidate_visible_object_ids": [1, 2],
                        "candidate_visibility_source": "mesh_ray_refined",
                        "candidate_labels": ["cup", "table"],
                        "label_to_object_ids": {"cup": [1], "table": [2]},
                        "selector_visible_object_ids": [1, 2],
                        "selector_visible_label_counts": {"cup": 1, "table": 1},
                        "visibility_audit_by_object_id": {
                            "1": {
                                "obj_id": 1,
                                "label": "cup",
                                "projected_area_px": 1200.0,
                                "bbox_in_frame_ratio": 0.95,
                            },
                            "2": {
                                "obj_id": 2,
                                "label": "table",
                                "projected_area_px": 2400.0,
                                "bbox_in_frame_ratio": 0.98,
                            },
                        },
                        "object_reviews": {},
                        "crop_label_statuses": {"cup": "unique", "table": "unique"},
                        "crop_label_counts": {"cup": 1, "table": 1},
                        "crop_referable_object_ids": [1, 2],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {"cup": "unique", "table": "unique"},
                        "full_frame_label_counts": {"cup": 1, "table": 1},
                        "label_statuses": {"cup": "unique", "table": "unique"},
                        "label_counts": {"cup": 1, "table": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "referable_object_ids": [1, 2],
                        "vlm_unique_object_ids": [1, 2],
                    },
                }
            },
        }
        (sidecar_dir / f"{scene_id}.json").write_text(
            json.dumps(sidecar_doc, ensure_ascii=False),
            encoding="utf-8",
        )
        write_neighbor_edited_html(
            cache_path,
            make_attachment_pair_review_html(
                scene_id=scene_id,
                image_name=image_name,
                parent_id=2,
                parent_label="table",
                parent_surface_text="wooden table",
                child_id=1,
                child_label="cup",
                child_surface_text="blue cup",
            ),
        )

        cache = run_pipeline_module._load_referability_cache(
            cache_path,
            repair_inconsistent_entries=True,
        )

        loaded_entry = cache["frames"][scene_id][image_name]
        self.assertEqual(loaded_entry["attachment_referable_pairs"], [[2, 1]])
        self.assertEqual(loaded_entry["attachment_referable_object_ids"], [1, 2])
        self.assertEqual(
            cache["scene_grouping"][scene_id]["final_cacheable_frame_image_names"],
            ["000123.jpg", "000124.jpg"],
        )
        self.assertEqual(cache["scene_grouping"][scene_id]["final_cacheable_frame_count"], 2)
        self.assertTrue(cache["scene_status"][scene_id]["has_cache_frames"])
        self.assertEqual(cache["scene_status"][scene_id]["final_cacheable_frame_count"], 2)

    def test_load_single_referability_cache_passes_cache_path_to_html_review(self) -> None:
        case_dir = make_case_dir("cache_path_forward")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        scene_id = "scene0000_00"
        cache_path.write_text(
            json.dumps(make_referability_batch_doc(scene_id=scene_id), ensure_ascii=False),
            encoding="utf-8",
        )
        write_scene_edited_html(cache_path, scene_id)

        captured_args: list[Path | None] = []

        def fake_apply_attachment_pair_salvage_html_review(
            *,
            html_text: str,
            cache_doc: dict,
            cache_path: Path | None = None,
        ) -> dict:
            captured_args.append(cache_path)
            return cache_doc

        with patch.object(
            run_pipeline_module,
            "_apply_attachment_pair_salvage_html_review",
            side_effect=fake_apply_attachment_pair_salvage_html_review,
        ):
            run_pipeline_module._load_single_referability_cache(cache_path)

        self.assertEqual(captured_args, [cache_path])

    def test_load_referability_cache_accepts_scene_grouping_metadata(self) -> None:
        case_dir = make_case_dir("cache_scene_grouping")
        self.addCleanup(shutil.rmtree, case_dir, True)
        cache_path = case_dir / "referability_cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "20.0",
                    "scene_grouping": {
                        "scene0000_00": {
                            "scene_id": "scene0000_00",
                            "pipeline_outcome": "processed",
                            "grouping_available": True,
                            "groups": [],
                        }
                    },
                    "frames": {
                        "scene0000_00": {
                            "000123.jpg": {
                                "frame_usable": True,
                                "frame_quality_clear": True,
                                "frame_quality_score": 82,
                                "frame_quality_reason": "clear enough",
                                "frame_selection_score": 82001,
                                "attachment_referable_pairs": [],
                                "attachment_referable_pair_count": 0,
                                "final_selection_rank": 0,
                                "candidate_visible_object_ids": [1],
                                "candidate_visibility_source": "mesh_ray_refined",
                                "candidate_labels": ["lamp"],
                                "label_to_object_ids": {"lamp": [1]},
                                "selector_visible_object_ids": [1],
                                "selector_visible_label_counts": {"lamp": 1},
                                "visibility_audit_by_object_id": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "lamp",
                                        "candidate_considered": True,
                                        "candidate_passed": True,
                                        "candidate_rejection_reasons": [],
                                    }
                                },
                                "object_reviews": {
                                    "1": {
                                        "obj_id": 1,
                                        "label": "lamp",
                                        "local_outcome": "reviewed",
                                        "vlm_status": "clear",
                                    }
                                },
                                "crop_label_statuses": {"lamp": "unique"},
                                "crop_label_counts": {"lamp": 1},
                                "crop_referable_object_ids": [1],
                                "full_frame_label_reviews": [],
                                "full_frame_label_statuses": {},
                                "full_frame_label_counts": {},
                                "label_statuses": {"lamp": "unique"},
                                "label_counts": {"lamp": 1},
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1],
                            }
                        }
                    },
                },
                ensure_ascii=False,
                ),
            encoding="utf-8",
        )
        write_neighbor_edited_html(cache_path)

        cache = run_pipeline_module._load_referability_cache(cache_path)
        scene_frames = run_pipeline_module._get_referability_scene_frames(cache, "scene0000_00")
        frames = run_pipeline_module._frames_from_referability_cache(scene_frames)

        self.assertIn("scene_grouping", cache)
        self.assertEqual(cache["scene_grouping"]["scene0000_00"]["pipeline_outcome"], "processed")
        self.assertEqual([frame["image_name"] for frame in frames], ["000123.jpg"])
        self.assertEqual(frames[0]["visible_object_ids"], [1])

    def test_load_referability_cache_merges_globbed_batches(self) -> None:
        case_dir = make_case_dir("cache_glob_merge")
        self.addCleanup(shutil.rmtree, case_dir, True)
        batch_a = case_dir / "flash_a.json"
        batch_b = case_dir / "flash_b.json"
        batch_a.write_text(
            json.dumps(make_referability_batch_doc(scene_id="scene0000_00"), ensure_ascii=False),
            encoding="utf-8",
        )
        batch_b.write_text(
            json.dumps(make_referability_batch_doc(scene_id="scene0001_00"), ensure_ascii=False),
            encoding="utf-8",
        )
        write_neighbor_edited_html(batch_a)

        merged = run_pipeline_module._load_referability_cache(str(case_dir / "flash*.json"))

        self.assertEqual(sorted(merged["scene_grouping"].keys()), ["scene0000_00", "scene0001_00"])
        self.assertEqual(sorted(merged["scene_status"].keys()), ["scene0000_00", "scene0001_00"])
        self.assertEqual(merged["model"], "fake-vlm")
        self.assertEqual(merged["alias_config_version"], "test-alias")

    def test_load_referability_cache_glob_reads_each_batch_scene_html_without_cross_batch_pollution(self) -> None:
        case_dir = make_case_dir("cache_glob_scene_html_isolated")
        self.addCleanup(shutil.rmtree, case_dir, True)
        batch_a = case_dir / "flash_a.json"
        batch_b = case_dir / "flash_b.json"
        scene_a = "scene0000_00"
        scene_b = "scene0001_00"
        image_name = "000001.jpg"
        batch_a.write_text(
            json.dumps(
                {
                    **make_referability_batch_doc(scene_id=scene_a),
                    "frames": {
                        scene_a: {
                            image_name: {
                                "frame_usable": True,
                                "frame_quality_clear": True,
                                "frame_quality_score": 82,
                                "frame_quality_reason": "clear enough",
                                "frame_selection_score": 82001,
                                "attachment_referable_pairs": [],
                                "attachment_referable_pair_count": 0,
                                "attachment_referable_object_ids": [],
                                "attachment_final_referability": {
                                    "referable_object_ids": [],
                                    "pairs": [],
                                    "pair_count": 0,
                                },
                                "final_selection_rank": 0,
                                "candidate_visible_object_ids": [1, 2],
                                "candidate_visibility_source": "mesh_ray_refined",
                                "candidate_labels": ["cup", "table"],
                                "label_to_object_ids": {"cup": [1], "table": [2]},
                                "selector_visible_object_ids": [1, 2],
                                "selector_visible_label_counts": {"cup": 1, "table": 1},
                                "visibility_audit_by_object_id": {},
                                "object_reviews": {},
                                "crop_label_statuses": {"cup": "unique", "table": "unique"},
                                "crop_label_counts": {"cup": 1, "table": 1},
                                "crop_referable_object_ids": [1, 2],
                                "full_frame_label_reviews": [],
                                "full_frame_label_statuses": {},
                                "full_frame_label_counts": {},
                                "label_statuses": {"cup": "unique", "table": "unique"},
                                "label_counts": {"cup": 1, "table": 1},
                                "out_of_frame_label_reviews": [],
                                "out_of_frame_not_visible_labels": [],
                                "out_of_frame_label_to_object_ids": {},
                                "out_of_frame_vlm_early_stop": False,
                                "referable_object_ids": [1, 2],
                                "vlm_unique_object_ids": [1, 2],
                            }
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        batch_b.write_text(
            json.dumps(make_referability_batch_doc(scene_id=scene_b), ensure_ascii=False),
            encoding="utf-8",
        )
        write_scene_edited_html(
            batch_a,
            scene_a,
            make_attachment_pair_review_html(
                scene_id=scene_a,
                image_name=image_name,
                parent_id=2,
                parent_label="table",
                parent_surface_text="wooden table",
                child_id=1,
                child_label="cup",
                child_surface_text="blue cup",
            ),
        )
        write_scene_edited_html(batch_b, scene_b)

        merged = run_pipeline_module._load_referability_cache(str(case_dir / "flash*.json"))

        self.assertEqual(
            merged["frames"][scene_a][image_name]["attachment_referable_pairs"],
            [[2, 1]],
        )
        self.assertNotIn("scene0000_00", merged["frames"].get(scene_b, {}))

    def test_load_referability_cache_glob_rejects_duplicate_scene(self) -> None:
        case_dir = make_case_dir("cache_glob_duplicate")
        self.addCleanup(shutil.rmtree, case_dir, True)
        batch_a = case_dir / "flash_a.json"
        batch_b = case_dir / "flash_b.json"
        duplicate_doc = make_referability_batch_doc(scene_id="scene0000_00")
        batch_a.write_text(json.dumps(duplicate_doc, ensure_ascii=False), encoding="utf-8")
        batch_b.write_text(json.dumps(duplicate_doc, ensure_ascii=False), encoding="utf-8")
        write_neighbor_edited_html(batch_a)

        with self.assertRaisesRegex(ValueError, "Duplicate referability cache scene"):
            run_pipeline_module._load_referability_cache(str(case_dir / "flash*.json"))

    def test_load_referability_cache_glob_rejects_metadata_mismatch(self) -> None:
        case_dir = make_case_dir("cache_glob_metadata")
        self.addCleanup(shutil.rmtree, case_dir, True)
        batch_a = case_dir / "flash_a.json"
        batch_b = case_dir / "flash_b.json"
        batch_a.write_text(
            json.dumps(make_referability_batch_doc(scene_id="scene0000_00", model="model-a"), ensure_ascii=False),
            encoding="utf-8",
        )
        batch_b.write_text(
            json.dumps(make_referability_batch_doc(scene_id="scene0001_00", model="model-b"), ensure_ascii=False),
            encoding="utf-8",
        )
        write_neighbor_edited_html(batch_a)

        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            run_pipeline_module._load_referability_cache(str(case_dir / "flash*.json"))

    def test_load_referability_cache_glob_requires_matches(self) -> None:
        case_dir = make_case_dir("cache_glob_missing")
        self.addCleanup(shutil.rmtree, case_dir, True)

        with self.assertRaisesRegex(ValueError, "matched no files"):
            run_pipeline_module._load_referability_cache(str(case_dir / "flash*.json"))

    def test_has_l1_visibility_candidates_only_keeps_vlm_out_of_frame_labels(self) -> None:
        self.assertTrue(
            run_pipeline_module._has_l1_visibility_candidates(
                {"lamp": "absent"},
                ["lamp"],
            )
        )
        self.assertFalse(
            run_pipeline_module._has_l1_visibility_candidates(
                {"chair": "unique", "table": "multiple", "sofa": "unsure"},
                [],
            )
        )

    def test_frames_from_referability_cache_prefers_reranked_scores_over_image_name(self) -> None:
        frames = run_pipeline_module._frames_from_referability_cache(
            {
                "000900.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [1],
                    "frame_selection_score": 10,
                    "selector_score": 10,
                },
                "000100.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [2],
                    "frame_selection_score": 100,
                    "selector_score": 50,
                },
                "000500.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [3],
                    "frame_selection_score": 100,
                    "selector_score": 40,
                },
            }
        )

        self.assertEqual(
            [frame["image_name"] for frame in frames],
            ["000100.jpg", "000500.jpg", "000900.jpg"],
        )
        self.assertEqual(
            [frame["visible_object_ids"] for frame in frames],
            [[2], [3], [1]],
        )

    def test_frames_from_referability_cache_prefers_final_attachment_selection_rank(self) -> None:
        frames = run_pipeline_module._frames_from_referability_cache(
            {
                "000900.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [1],
                    "attachment_referable_pair_count": 3,
                    "final_selection_rank": 1,
                    "frame_selection_score": 10,
                    "selector_score": 10,
                },
                "000100.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [2],
                    "attachment_referable_pair_count": 1,
                    "final_selection_rank": 0,
                    "frame_selection_score": 100,
                    "selector_score": 50,
                },
                "000500.jpg": {
                    "frame_usable": True,
                    "candidate_visible_object_ids": [3],
                    "attachment_referable_pair_count": 2,
                    "final_selection_rank": 2,
                    "frame_selection_score": 1000,
                    "selector_score": 999,
                },
            }
        )

        self.assertEqual(
            [frame["image_name"] for frame in frames],
            ["000100.jpg", "000900.jpg", "000500.jpg"],
        )

    def test_support_chain_graph_has_two_hop_chain_requires_depth_two(self) -> None:
        self.assertFalse(run_pipeline_module._support_chain_graph_has_two_hop_chain({}))
        self.assertFalse(run_pipeline_module._support_chain_graph_has_two_hop_chain({1: [2]}))
        self.assertTrue(run_pipeline_module._support_chain_graph_has_two_hop_chain({1: [2], 2: [3]}))

    def test_attachment_graph_has_two_hop_chain_requires_depth_two(self) -> None:
        self.assertFalse(run_pipeline_module._attachment_graph_has_two_hop_chain({}))
        self.assertFalse(run_pipeline_module._attachment_graph_has_two_hop_chain({1: [2]}))
        self.assertTrue(run_pipeline_module._attachment_graph_has_two_hop_chain({1: [2], 2: [3]}))

    def test_frame_has_l3_attachment_chain_requires_visible_attachment_referable_two_hop(self) -> None:
        support_chain_graph = {1: [2], 2: [3]}
        frame = {"image_name": "chain.jpg", "visible_object_ids": [1, 2, 3, 4]}

        self.assertTrue(
            run_pipeline_module._frame_has_l3_attachment_chain(
                frame,
                {"attachment_referable_object_ids": [1, 2, 3]},
                support_chain_graph,
            )
        )
        self.assertFalse(
            run_pipeline_module._frame_has_l3_attachment_chain(
                frame,
                {"attachment_referable_object_ids": [1, 2]},
                support_chain_graph,
            )
        )
        self.assertFalse(
            run_pipeline_module._frame_has_l3_attachment_chain(
                {"image_name": "one-hop.jpg", "visible_object_ids": [1, 2, 4]},
                {"attachment_referable_object_ids": [1, 2, 3]},
                support_chain_graph,
            )
        )

    def test_l3_attachment_chain_filter_runs_before_max_frame_limit(self) -> None:
        scene_frames = {
            "000001.jpg": {
                "frame_usable": True,
                "candidate_visible_object_ids": [1, 2],
                "attachment_referable_object_ids": [1, 2],
                "final_selection_rank": 1,
            },
            "000002.jpg": {
                "frame_usable": True,
                "candidate_visible_object_ids": [1, 2, 3],
                "attachment_referable_object_ids": [1, 2, 3],
                "final_selection_rank": 2,
            },
        }
        frames = run_pipeline_module._frames_from_referability_cache(scene_frames)
        eligible_frames = [
            frame
            for frame in frames
            if run_pipeline_module._frame_has_l3_attachment_chain(
                frame,
                scene_frames[str(frame["image_name"])],
                {1: [2], 2: [3]},
            )
        ]

        self.assertEqual([frame["image_name"] for frame in frames], ["000001.jpg", "000002.jpg"])
        self.assertEqual([frame["image_name"] for frame in eligible_frames[:1]], ["000002.jpg"])

    def test_l3_attachment_move_scene_filter_skips_non_two_hop_scenes_before_generation(self) -> None:
        root = make_case_dir("pipeline_l3_attachment_move_scene_skip")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [1, 2, 3],
                        "crop_label_statuses": {"chair": "unique", "table": "unique", "lamp": "unique"},
                        "crop_label_counts": {"chair": 1, "table": 1, "lamp": 1},
                        "crop_referable_object_ids": [1, 2, 3],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
                        "attachment_referable_object_ids": [1, 2, 3],
                        "referable_object_ids": [1, 2, 3],
                        "label_statuses": {"chair": "unique", "table": "unique", "lamp": "unique"},
                        "label_counts": {"chair": 1, "table": 1, "lamp": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["chair", "table", "lamp"],
                        "label_to_object_ids": {"chair": [1], "table": [2], "lamp": [3]},
                    }
                }
            },
        }
        scene = {
            "scene_id": scene_id,
            "objects": [
                make_object(1, "chair"),
                make_object(2, "table"),
                make_object(3, "lamp"),
            ],
            "attachment_edges": [
                {"parent_id": 3, "child_id": 2, "type": "attachment"},
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
                side_effect=AssertionError("should skip non-two-hop attachment_move scene before generation"),
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
                only_question_types=["L3_attachment_move"],
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertEqual(questions, [])

    def test_run_pipeline_rejects_stale_cache_when_full_frame_marks_label_absent(self) -> None:
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
            "version": "20.0",
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
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
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
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                side_effect=AssertionError(
                    "should reject stale referability cache before question generation"
                ),
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            with self.assertRaisesRegex(ValueError, "inconsistent with cache version 20.0"):
                run_pipeline_module.run_pipeline(
                    data_root=data_root,
                    output_dir=output_dir,
                    max_scenes=10,
                    max_frames=10,
                    use_occlusion=False,
                    referability_cache=referability_cache,
                    run_question_presence_review=False,
                    write_frame_debug=False,
                )

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

    def test_in_frame_ratio_map_uses_visibility_audit_without_fallback(self) -> None:
        referability_entry = {
            "object_reviews": {
                "1": {
                    "obj_id": 1,
                    "bbox_in_frame_ratio": 0.95,
                },
            },
            "visibility_audit_by_object_id": {
                "2": {
                    "obj_id": 2,
                    "bbox_in_frame_ratio": 0.85,
                },
            },
        }

        with patch.object(
            run_pipeline_module,
            "compute_frame_object_visibility",
            side_effect=AssertionError("should use cached visibility audit"),
        ):
            ratios = run_pipeline_module._build_visible_object_in_frame_ratio_map(
                visible_object_ids=[1, 2],
                referability_entry=referability_entry,
                scene_objects=[make_object(1, "cup"), make_object(2, "table")],
                camera_pose=make_camera_pose("000123.jpg"),
                color_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual(ratios, {1: 0.95, 2: 0.85})

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
            "version": "20.0",
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
                        "attachment_referable_object_ids": [1, 2],
                        "referable_object_ids": [1, 2],
                        "label_statuses": {"cup": "unique", "table": "unique"},
                        "label_counts": {"cup": 1, "table": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["cup", "table"],
                        "label_to_object_ids": {"cup": [1], "table": [2]},
                        "attachment_human_review_cards": [
                            {
                                "pair_id": "2->1",
                                "parent_id": 2,
                                "parent_label": "table",
                                "parent_surface_text": "wooden table",
                                "child_id": 1,
                                "child_label": "cup",
                                "child_surface_text": "blue cup",
                                "source": "human_salvage_html",
                            }
                        ],
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
            captured["attachment_referable_object_ids"] = list(
                kwargs.get("attachment_referable_object_ids") or []
            )
            captured["attachment_referable_pairs"] = list(
                kwargs.get("attachment_referable_pairs") or []
            )
            captured["attachment_object_surface_text_by_id"] = dict(
                kwargs.get("attachment_object_surface_text_by_id") or {}
            )
            captured["attachment_priority_pairs"] = list(
                kwargs.get("attachment_priority_pairs") or []
            )
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
        self.assertEqual(captured["attachment_referable_object_ids"], [1, 2])
        self.assertEqual(captured["attachment_referable_pairs"], [(2, 1)])
        self.assertEqual(captured["attachment_object_surface_text_by_id"], {1: "blue cup", 2: "wooden table"})
        self.assertEqual(captured["attachment_priority_pairs"], [(2, 1)])
        self.assertEqual(captured["occlusion_eligible_object_ids"], [1, 2])
        self.assertEqual(captured["mention_in_frame_ratio_by_obj_id"], {1: 0.95, 2: 0.85})
        self.assertEqual(captured["label_statuses"], {"cup": "unique", "table": "unique"})
        self.assertEqual(captured["label_counts"], {"cup": 1, "table": 1})
        self.assertEqual(captured["label_to_object_ids"], {"cup": [1], "table": [2]})
        self.assertEqual(len(questions), 1)
        self.assertNotIn("question_dinox_audit", questions[0])
        self.assertNotIn("question_mesh_audit", questions[0])
        self.assertNotIn("question_post_generation_review", questions[0])

    def test_run_pipeline_only_question_types_accepts_l3_attachment_move(self) -> None:
        root = make_case_dir("pipeline_only_l3_attachment_move")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    # lamp (id 3) IS listed as referable here: the referability
                    # "backstop" filter (_apply_question_referability_filter) checks
                    # every mentioned object against THIS frame's referable ids
                    # regardless of where the two-frame split ends up putting them,
                    # since it's a generation-time precondition (obj_ref_id must be
                    # referable in the frame a question is generated for) that's
                    # independent of which real frame the split later assigns as
                    # frame_1. image_name's own camera is placed far from chair/
                    # table/lamp (see load_scannet_poses below) so it never becomes
                    # a two-frame-split candidate at all -- frame_1 ends up being
                    # "anchor.jpg", which has no referability_cache entry, so the
                    # mutual-exclusivity referability check can't fire against it.
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [1, 2, 3],
                        "crop_label_statuses": {"chair": "unique", "table": "unique", "lamp": "unique"},
                        "crop_label_counts": {"chair": 1, "table": 1, "lamp": 1},
                        "crop_referable_object_ids": [1, 2, 3],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
                        "attachment_referable_object_ids": [1, 2, 3],
                        "referable_object_ids": [1, 2, 3],
                        "label_statuses": {"chair": "unique", "table": "unique", "lamp": "unique"},
                        "label_counts": {"chair": 1, "table": 1, "lamp": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["chair", "table", "lamp"],
                        "label_to_object_ids": {"chair": [1], "table": [2], "lamp": [3]},
                    }
                }
            },
        }
        def make_positioned_object(obj_id: int, label: str, x: float) -> dict:
            # attachment_move requires a valid two-frame split (root/query in one
            # frame, ref in another): chair/table sit near x=0, lamp far enough away
            # (x=1.6, past the ~0.64 half-FOV window at z=1 with this test's
            # intrinsics) that no single pose fully frames both groups.
            return {
                "id": obj_id,
                "label": label,
                "center": [x, 0.0, 1.0],
                "bbox_min": [x - 0.1, -0.1, 0.9],
                "bbox_max": [x + 0.1, 0.1, 1.1],
            }

        scene = {
            "scene_id": scene_id,
            "objects": [
                make_positioned_object(1, "chair", x=0.0),
                make_positioned_object(2, "table", x=0.1),
                make_positioned_object(3, "lamp", x=1.6),
            ],
            "attachment_edges": [
                {"parent_id": 3, "child_id": 2, "type": "attachment"},
                {"parent_id": 2, "child_id": 1, "type": "attachment"},
            ],
            "room_bounds": None,
            "wall_objects": [],
        }
        captured: dict[str, object] = {}

        def fake_generate_all_questions(**kwargs):
            captured["only_question_types"] = list(kwargs.get("only_question_types") or [])
            return [
                {
                    "question": "If the table moves, where is the chair relative to the lamp?",
                    "answer": "A",
                    "options": ["left", "right", "front", "back"],
                    "type": "attachment_move",
                    "level": "L3",
                    "reference_frame": "agent",
                    "root_id": 2,
                    "query_obj_id": 1,
                    "obj_ref_id": 3,
                }
            ]

        with (
            patch.object(run_pipeline_module, "parse_scene", return_value=scene),
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={3: [2], 2: [1]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={1: [2], 2: [3]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={3: [2], 2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={1: [2], 2: [3]}),
            patch.object(run_pipeline_module, "has_nontrivial_attachment", return_value=True),
            patch.object(run_pipeline_module, "_load_scene_geometry", return_value=None),
            patch.object(run_pipeline_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                run_pipeline_module,
                "load_scannet_poses",
                return_value={
                    # image_name's own camera is placed far away (x=100) so it does
                    # NOT geometrically frame chair/table at all -- this deliberately
                    # decouples it from the two-frame-split's frame_1 choice. Its
                    # referability_cache entry (all 3 objects "referable") reflects a
                    # generation-time precondition on the ORIGINAL processing frame,
                    # independent of geometry or of which real frame the split later
                    # picks -- see the long comment on the referability_cache literal
                    # above. anchor.jpg is the real geometric stand-in for frame_1
                    # (frames chair/table, has no referability_cache entry, so the
                    # mutual-exclusivity referability check can't fire against it).
                    image_name: make_camera_pose_at(image_name, 100.0),
                    "anchor.jpg": make_camera_pose_at("anchor.jpg", 0.3),
                    # Bridge poses providing route-continuity between anchor.jpg and
                    # a real frame that fully frames the far-off lamp (x=1.6), so
                    # _apply_two_frame_split finds a valid split instead of dropping
                    # this question (verified in isolation: with just these poses,
                    # find_two_frame_split_v2 returns
                    # ("anchor.jpg", "000456.jpg", ["bridge1.jpg", "bridge2.jpg"])).
                    "bridge1.jpg": make_camera_pose_at("bridge1.jpg", 0.5),
                    "bridge2.jpg": make_camera_pose_at("bridge2.jpg", 1.0),
                    "000456.jpg": make_camera_pose_at("000456.jpg", 1.6),
                },
            ),
            patch.object(run_pipeline_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(run_pipeline_module, "load_instance_mesh_data", return_value=object()),
            patch.object(
                run_pipeline_module,
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                    2: {"bbox_in_frame_ratio": 0.85},
                    3: {"bbox_in_frame_ratio": 0.8},
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
                only_question_types=["L3_attachment_move"],
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertEqual(captured["only_question_types"], ["L3_attachment_move"])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "attachment_move")

    def test_run_pipeline_skips_dinox_post_generation_audit_when_disabled(self) -> None:
        root = make_case_dir("pipeline_skip_dinox_audit")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
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
                        "attachment_referable_object_ids": [1, 2],
                        "referable_object_ids": [1, 2],
                        "label_statuses": {"cup": "unique", "table": "unique"},
                        "label_counts": {"cup": 1, "table": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
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
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                    2: {"bbox_in_frame_ratio": 0.85},
                },
            ),
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                return_value=[
                    {
                        "question": "Is the cup on the table?",
                        "answer": "A",
                        "options": ["yes", "no"],
                        "type": "attachment",
                        "level": "L1",
                        "mentioned_objects": [
                            {"role": "target", "label": "cup", "obj_id": 1},
                            {"role": "reference", "label": "table", "obj_id": 2},
                        ],
                    }
                ],
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(
                run_pipeline_module,
                "_prebuild_question_review_frame_contexts",
                side_effect=AssertionError("should not prebuild review frame contexts when all post reviews are disabled"),
            ),
            patch.object(
                run_pipeline_module,
                "_apply_question_post_generation_audit",
                side_effect=AssertionError("should not run DINO-X post-generation audit when disabled"),
            ),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_dinox_audit=False,
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertEqual(len(questions), 1)
        self.assertNotIn("question_dinox_audit", questions[0])
        self.assertNotIn("question_mesh_audit", questions[0])
        self.assertNotIn("question_post_generation_review", questions[0])

    def test_run_pipeline_derives_relaxed_attachment_ids_when_cache_entry_omits_them(self) -> None:
        root = make_case_dir("pipeline_attachment_legacy_cache")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [2, 1],
                        "crop_label_statuses": {"cup": "unique", "table": "unique"},
                        "crop_label_counts": {"cup": 1, "table": 1},
                        "crop_referable_object_ids": [1, 2],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {"cup": "unique", "table": "unique"},
                        "full_frame_label_counts": {"cup": 1, "table": 1},
                        "referable_object_ids": [2],
                        "label_statuses": {"cup": "unique", "table": "unique"},
                        "label_counts": {"cup": 1, "table": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["cup", "table"],
                        "label_to_object_ids": {"cup": [2], "table": [1]},
                        "object_reviews": {
                            "1": {
                                "obj_id": 1,
                                "label": "table",
                                "local_outcome": "reviewed",
                                "vlm_status": "clear",
                                "bbox_in_frame_ratio": 0.55,
                            },
                            "2": {
                                "obj_id": 2,
                                "label": "cup",
                                "local_outcome": "reviewed",
                                "vlm_status": "clear",
                                "bbox_in_frame_ratio": 0.95,
                            },
                        },
                    }
                }
            },
        }

        scene = {
            "scene_id": scene_id,
            "objects": [
                make_object(1, "table"),
                make_object(2, "cup"),
            ],
            "attachment_edges": [
                {"parent_id": 1, "child_id": 2, "type": "attachment"},
            ],
            "room_bounds": None,
            "wall_objects": [],
        }

        captured: dict[str, object] = {}

        def fake_generate_all_questions(**kwargs):
            captured["referable_object_ids"] = list(kwargs["referable_object_ids"] or [])
            captured["attachment_referable_object_ids"] = list(
                kwargs.get("attachment_referable_object_ids") or []
            )
            return []

        with (
            patch.object(run_pipeline_module, "parse_scene", return_value=scene),
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={2: [1]}),
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
                    1: {"bbox_in_frame_ratio": 0.55},
                    2: {"bbox_in_frame_ratio": 0.95},
                },
            ),
            patch.object(run_pipeline_module, "generate_all_questions", side_effect=fake_generate_all_questions),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertEqual(captured["referable_object_ids"], [2])
        self.assertEqual(captured["attachment_referable_object_ids"], [1, 2])

    def test_run_pipeline_keeps_benchmark_raw_after_question_presence_review(self) -> None:
        root = make_case_dir("pipeline_review_raw_benchmark")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
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
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
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

        def fake_run_question_presence_review(**kwargs):
            reviewed_questions = []
            for question in kwargs["questions"]:
                reviewed_questions.append(
                    {
                        **question,
                        "manual_review_reason": "Post review flagged this question.",
                        "question_presence_review": {
                            "review_mode": "instance",
                            "decision": "manual_review",
                            "flagged_labels": ["cup"],
                            "flagged_object_ids": [1],
                            "object_reviews": [],
                            "raw_response": "",
                        },
                    }
                )
            return {"questions": reviewed_questions}

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
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                return_value=[
                    {
                        "question": "Is the cup on the table?",
                        "answer": "A",
                        "options": ["yes", "no"],
                        "type": "attachment",
                        "level": "L1",
                        "mentioned_objects": [
                            {"role": "target", "label": "cup", "obj_id": 1},
                            {"role": "reference", "label": "table", "obj_id": 2},
                        ],
                    }
                ],
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module, "_prebuild_question_review_frame_contexts", return_value={}),
            patch.object(run_pipeline_module, "_apply_question_post_generation_audit", side_effect=lambda **kwargs: kwargs["questions"]),
            patch.object(run_pipeline_module, "_run_question_presence_review", side_effect=fake_run_question_presence_review),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=True,
                write_frame_debug=False,
            )

        self.assertNotIn("manual_review_reason", questions[0])
        self.assertNotIn("question_presence_review", questions[0])
        self.assertNotIn("question_attachment_pair_review", questions[0])
        benchmark_path = output_dir / "benchmark.json"
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        self.assertNotIn("manual_review_reason", benchmark["questions"][0])
        self.assertNotIn("question_presence_review", benchmark["questions"][0])
        self.assertNotIn("question_attachment_pair_review", benchmark["questions"][0])

    def test_run_pipeline_question_presence_review_also_runs_attachment_pair_review_without_mutating_benchmark(self) -> None:
        root = make_case_dir("pipeline_combined_question_review")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [3, 2, 1],
                        "crop_label_statuses": {"bed": "unique", "pillow": "unique", "chair": "unique"},
                        "crop_label_counts": {"bed": 1, "pillow": 1, "chair": 1},
                        "crop_referable_object_ids": [1, 2, 3],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
                        "referable_object_ids": [1, 2, 3],
                        "attachment_referable_object_ids": [1, 2, 3],
                        "label_statuses": {"bed": "unique", "pillow": "unique", "chair": "unique"},
                        "label_counts": {"bed": 1, "pillow": 1, "chair": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["bed", "pillow", "chair"],
                        "label_to_object_ids": {"bed": [1], "pillow": [2], "chair": [3]},
                    }
                }
            },
        }

        scene = {
            "scene_id": scene_id,
            "objects": [
                make_object(1, "bed"),
                make_object(2, "pillow"),
                make_object(3, "chair"),
            ],
            "attachment_edges": [
                {"parent_id": 1, "child_id": 2, "type": "attachment"},
            ],
            "room_bounds": None,
            "wall_objects": [],
        }

        generated_questions = [
            {
                "question": "If the bed moves, where is the pillow relative to the chair?",
                "answer": "A",
                "options": ["left", "right", "front", "back"],
                "type": "object_move_agent",
                "level": "L2",
                "moved_obj_id": 1,
                "moved_obj_label": "bed",
                "query_obj_id": 2,
                "query_obj_label": "pillow",
                "attachment_remapped": True,
                "mentioned_objects": [
                    {"role": "moved_object", "label": "bed", "obj_id": 1},
                    {"role": "query_object", "label": "pillow", "obj_id": 2},
                    {"role": "reference_object", "label": "chair", "obj_id": 3},
                ],
            },
        ]

        question_review_calls: list[dict[str, object]] = []

        def fake_run_question_presence_review(**kwargs):
            question_review_calls.append(kwargs)
            return {
                "questions": [
                    {
                        **generated_questions[0],
                        "question_attachment_pair_review": {
                            "decision": "pass",
                            "reason": "distinct pair",
                            "moved_obj_id": 1,
                            "query_obj_id": 2,
                            "moved_obj_label": "bed",
                            "query_obj_label": "pillow",
                            "raw_response": "",
                        },
                    }
                ]
            }

        with (
            patch.object(run_pipeline_module, "parse_scene", return_value=scene),
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={2: [1]}),
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
                    3: {"bbox_in_frame_ratio": 0.80},
                },
            ),
            patch.object(run_pipeline_module, "generate_all_questions", return_value=generated_questions),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module, "_prebuild_question_review_frame_contexts", return_value={("scene0000_00", "000123.jpg"): {}}) as prebuild_mock,
            patch.object(run_pipeline_module, "_run_question_presence_review", side_effect=fake_run_question_presence_review),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=True,
                write_frame_debug=False,
            )

        self.assertEqual(len(question_review_calls), 1)
        self.assertEqual(
            [q["question"] for q in question_review_calls[0]["questions"]],
            [q["question"] for q in generated_questions],
        )
        self.assertEqual(question_review_calls[0]["frame_context_by_key"], {("scene0000_00", "000123.jpg"): {}})
        prebuild_mock.assert_called_once()
        self.assertEqual(len(questions), 1)
        self.assertTrue(all("question_attachment_pair_review" not in q for q in questions))
        self.assertTrue(all("question_presence_review" not in q for q in questions))

        benchmark_path = output_dir / "benchmark.json"
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        self.assertEqual(len(benchmark["questions"]), 1)
        self.assertTrue(all("question_attachment_pair_review" not in q for q in benchmark["questions"]))
        self.assertTrue(all("question_presence_review" not in q for q in benchmark["questions"]))

    def test_run_pipeline_applies_referable_occlusion_veto_before_generation(self) -> None:
        root = make_case_dir("pipeline_occlusion_veto")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
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
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
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
            return []

        veto_result = {
            "raw_object_ids": [1, 2],
            "filtered_object_ids": [1],
            "low_visible_object_ids": [2],
            "not_visible_object_ids": [],
            "skipped_object_ids": [],
            "audit_by_object_id": {
                "1": {"status": "visible_enough", "keep_for_generation": True},
                "2": {"status": "low_visible", "keep_for_generation": False},
            },
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
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95, "projected_area_px": 900.0},
                    2: {"bbox_in_frame_ratio": 0.85, "projected_area_px": 1200.0},
                },
            ),
            patch.object(
                run_pipeline_module,
                "_filter_referable_object_ids_with_occlusion_veto",
                return_value=veto_result,
            ) as veto_mock,
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

        veto_mock.assert_called_once()
        self.assertEqual(captured["visible_object_ids"], [1, 2])
        self.assertEqual(captured["referable_object_ids"], [1])
        self.assertEqual(questions, [])

    def test_evaluate_referable_occlusion_veto_uses_valid_count_ratio_threshold(self) -> None:
        obj = make_object(1, "cup")
        sample_points = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (8, 1))
        sample_triangle_ids = np.zeros(8, dtype=np.int64)
        sample_barycentrics = np.tile(
            np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
            (8, 1),
        )
        with (
            patch.object(
                run_pipeline_module,
                "_resample_instance_surface_probe_points",
                return_value=(sample_points, sample_triangle_ids, sample_barycentrics),
            ),
            patch.object(
                run_pipeline_module,
                "_in_frame_surface_sample_subset",
                return_value=(
                    100.0,
                    1.0,
                    sample_points,
                    sample_triangle_ids,
                    sample_barycentrics,
                ),
            ),
            patch.object(run_pipeline_module, "_instance_triangle_id_set", return_value={0}),
            patch.object(
                run_pipeline_module,
                "_mesh_visibility_counts_with_early_stop",
                return_value={
                    "visible_count": 1,
                    "valid_count": 4,
                    "processed_count": 8,
                    "stopped_early": False,
                    "stop_reason": "completed",
                },
            ),
        ):
            audit = run_pipeline_module._evaluate_referable_occlusion_veto_for_object(
                obj=obj,
                obj_id=1,
                scene_id="scene0000_00",
                image_name="000123.jpg",
                projected_area_px=800.0,
                camera_pose=make_camera_pose("000123.jpg"),
                color_intrinsics=make_camera_intrinsics(),
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(audit["status"], "low_visible")
        self.assertFalse(audit["keep_for_generation"])
        self.assertEqual(audit["dense_in_frame_sample_count"], 8)
        self.assertEqual(audit["dense_valid_count"], 4)
        self.assertAlmostEqual(float(audit["dense_visible_ratio"]), 0.25)
        self.assertAlmostEqual(float(audit["dense_visible_ratio_threshold"]), 0.35)
        self.assertEqual(audit["dense_visible_ratio_denominator"], "valid_count")
        self.assertEqual(audit["reason"], "dense_visible_ratio_below_threshold")

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
            "version": "20.0",
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
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
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

    def test_run_pipeline_respects_max_scene_and_frame_limits_with_referability_cache(self) -> None:
        root = make_case_dir("pipeline_cache_limits")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_ids = ["scene0000_00", "scene0001_00"]
        scene_frames = {
            "scene0000_00": ["000001.jpg", "000002.jpg"],
            "scene0001_00": ["000003.jpg", "000004.jpg"],
        }

        for scene_id in scene_ids:
            scene_dir = data_root / scene_id
            (scene_dir / "pose").mkdir(parents=True)
            (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [1, 2],
                        "crop_label_statuses": {"chair": "unique", "table": "unique"},
                        "crop_label_counts": {"chair": 1, "table": 1},
                        "crop_referable_object_ids": [1, 2],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
                        "referable_object_ids": [1, 2],
                        "label_statuses": {"chair": "unique", "table": "unique"},
                        "label_counts": {"chair": 1, "table": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["chair", "table"],
                        "label_to_object_ids": {"chair": [1], "table": [2]},
                    }
                    for image_name in image_names
                }
                for scene_id, image_names in scene_frames.items()
            },
        }

        scenes = {
            scene_id: {
                "scene_id": scene_id,
                "objects": [
                    make_object(1, "chair"),
                    make_object(2, "table"),
                ],
                "attachment_edges": [
                    {"parent_id": 2, "child_id": 1, "type": "attachment"},
                ],
                "room_bounds": None,
                "wall_objects": [],
            }
            for scene_id in scene_ids
        }
        generate_calls: list[str] = []

        def fake_parse_scene(scene_dir: Path, preloaded_geometry=None):
            return scenes[scene_dir.name]

        def fake_load_scannet_poses(scene_dir: Path, axis_alignment=None):
            return {
                image_name: make_camera_pose(image_name)
                for image_name in scene_frames[scene_dir.name]
            }

        def fake_generate_all_questions(**kwargs):
            generate_calls.append(kwargs["camera_pose"].image_name)
            return [
                {
                    "question": f"Is {kwargs['camera_pose'].image_name} valid?",
                    "answer": "A",
                    "options": ["yes", "no"],
                    "type": "attachment",
                    "level": "L1",
                }
            ]

        with (
            patch.object(run_pipeline_module, "parse_scene", side_effect=fake_parse_scene),
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "has_nontrivial_attachment", return_value=True),
            patch.object(run_pipeline_module, "_load_scene_geometry", return_value=None),
            patch.object(run_pipeline_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(run_pipeline_module, "load_scannet_poses", side_effect=fake_load_scannet_poses),
            patch.object(run_pipeline_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(run_pipeline_module, "load_instance_mesh_data", return_value=object()),
            patch.object(
                run_pipeline_module,
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                    2: {"bbox_in_frame_ratio": 0.90},
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
                max_scenes=1,
                max_frames=1,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        self.assertEqual(generate_calls, ["000001.jpg"])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["scene_id"], "scene0000_00")
        self.assertEqual(questions[0]["image_name"], "000001.jpg")

    def test_run_pipeline_defers_non_occlusion_by_source_and_keeps_occlusion_per_pair(self) -> None:
        root = make_case_dir("pipeline_cross_frame_orchestration")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")
        frame_names = ["000001.jpg", "000002.jpg", "000003.jpg"]

        objects = [
            make_object(1, "table"),
            make_object(2, "chair"),
            make_object(3, "cabinet"),
            make_object(5, "book"),
            make_object(6, "lamp"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in objects}
        scene = {
            "scene_id": scene_id,
            "objects": objects,
            "attachment_edges": [
                {"parent_id": 1, "child_id": 5, "type": "attachment"},
                {"parent_id": 1, "child_id": 6, "type": "attachment"},
            ],
            "room_bounds": None,
            "wall_objects": [],
        }

        def cache_entry(regular_ids: list[int], attachment_ids: list[int], rank: int) -> dict:
            visible_ids = sorted(set(regular_ids) | set(attachment_ids))
            regular_id_set = set(regular_ids)
            crop_label_statuses = {
                objects_by_id[obj_id]["label"]: (
                    "unique" if obj_id in regular_id_set else "absent"
                )
                for obj_id in visible_ids
            }
            attachment_only_ids = sorted(set(attachment_ids) - regular_id_set)
            attachment_cards = []
            if attachment_only_ids and regular_ids:
                attachment_cards.append({
                    "pair_id": f"{attachment_only_ids[0]}->{regular_ids[0]}",
                    "parent_id": attachment_only_ids[0],
                    "parent_label": objects_by_id[attachment_only_ids[0]]["label"],
                    "parent_surface_text": f"the {objects_by_id[attachment_only_ids[0]]['label']}",
                    "child_id": regular_ids[0],
                    "child_label": objects_by_id[regular_ids[0]]["label"],
                    "child_surface_text": f"the {objects_by_id[regular_ids[0]]['label']}",
                    "source": "human_salvage_html",
                })
            entry = {
                "frame_usable": True,
                "final_selection_rank": rank,
                "candidate_visibility_source": "mesh_ray_refined",
                "candidate_visible_object_ids": visible_ids,
                "visibility_audit_by_object_id": {
                    str(obj_id): {
                        "obj_id": obj_id,
                        "bbox_in_frame_ratio": 1.0,
                    }
                    for obj_id in visible_ids
                },
                "label_to_object_ids": {
                    objects_by_id[obj_id]["label"]: [obj_id] for obj_id in visible_ids
                },
                "crop_label_statuses": crop_label_statuses,
                "crop_label_counts": {
                    label: 1 if status == "unique" else 0
                    for label, status in crop_label_statuses.items()
                },
                "crop_referable_object_ids": regular_ids,
                "full_frame_label_statuses": {},
                "full_frame_label_counts": {},
                "attachment_human_review_cards": attachment_cards,
                "out_of_frame_label_reviews": [],
                "out_of_frame_not_visible_labels": [],
                "out_of_frame_label_to_object_ids": {},
                "out_of_frame_vlm_early_stop": False,
            }
            return referability_module._repair_final_referability_fields(entry)

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    frame_names[0]: cache_entry([5], [1], 1),
                    frame_names[1]: cache_entry([2, 6], [], 2),
                    frame_names[2]: cache_entry([3, 6], [], 3),
                }
            },
        }
        poses = {
            name: make_camera_pose_at(name, float(index) * 0.2)
            for index, name in enumerate(frame_names)
        }
        non_occlusion_calls: list[tuple[str, str, tuple[str, ...]]] = []
        occlusion_calls: list[tuple[str, str]] = []
        occlusion_veto_call_count = 0

        class FakeDepthRoute:
            auxiliary_image_names = ("depth_bridge.jpg",)
            cost = 0.9
            edge_count = 2
            route_sample_count = 10
            frame_a_coverage_end = 0.3
            frame_b_coverage_start = 0.7
            auxiliary_responsibility_fraction = 0.4
            transition_overlap_fraction = 0.2
            search_method = "dijkstra_depth_corridor"
            min_progress_fraction = 0.05
            min_depth_valid_fraction = 0.8
            min_depth_visible_fraction = 0.7
            max_local_perpendicular_m = 0.2
            max_global_perpendicular_m = 0.3
            max_height_change_m = 0.1
            max_parallel_change_m = 0.5
            max_forward_angle_deg = 15.0
            depth_sources = ("test_depth",)
            pre_prune_auxiliary_count = 2
            pruned_auxiliary_frame_count = 1
            visual_pruned_auxiliary_frame_count = 1
            visual_duplicate_candidate_count = 1
            visual_prune_relaxed_angle_edge_count = 1
            visual_redundancy_metric_version = 1
            semantic_rejected_frame_count = 0

        class FakeRoute:
            auxiliary_image_names = ("bridge.jpg",)
            cost = 1.0
            edge_count = 2
            min_inliers = 40
            min_inlier_ratio = 0.5

        class FakeLegacyRoute:
            auxiliary_image_names = ("legacy_bridge.jpg",)
            cost = 1.25
            edge_count = 2
            route_sample_count = 10
            frame_a_coverage_end = 0.3
            frame_b_coverage_start = 0.7
            auxiliary_responsibility_fraction = 0.4
            transition_overlap_fraction = 0.2

        class FakeHybridRoute:
            auxiliary_image_names = ("hybrid_bridge.jpg",)
            cost = 1.1
            edge_count = 2
            route_sample_count = 10
            frame_a_coverage_end = 0.3
            frame_b_coverage_start = 0.7
            auxiliary_responsibility_fraction = 0.4
            transition_overlap_fraction = 0.2
            min_mutual_matches = 32
            min_inliers = 24
            min_inlier_ratio = 0.6
            min_grid_fraction = 0.5
            visual_models = ("fundamental", "homography")
            semantic_rejected_frames = 1

        depth_route_call_count = 0
        visual_graph_init_count = 0
        legacy_route_call_count = 0
        hybrid_route_call_count = 0

        class FakeVisualPoseGraph:
            def __init__(self, **_kwargs):
                nonlocal visual_graph_init_count
                visual_graph_init_count += 1

            def load_cache(self, _cache_path):
                return False

            def build(self):
                pass

            def save_cache(self, _cache_path):
                pass

            def diagnostics(self):
                return {
                    "pose_count": 3,
                    "readable_count": 3,
                    "graph_node_count": 3,
                    "graph_edge_count": 3,
                    "rejected_edge_counts": {},
                }

            def find_route(self, start, end, **_kwargs):
                return None if start == end else FakeRoute()

        class FakeHybridAuxiliaryRouter:
            def __init__(self, **_kwargs):
                pass

            def load_cache(self, _cache_path):
                return False

            def save_cache(self, _cache_path):
                pass

            def diagnostics(self):
                return {
                    "pose_count": 3,
                    "route_count": hybrid_route_call_count,
                    "visual_edge_cache_count": 2,
                    "feature_frame_count": 3,
                    "visual_counts": {"passed": 2},
                    "route_rejection_counts": {},
                }

            def visual_continuity(self, _left, _right):
                return Mock(
                    passed=True,
                    inliers=40,
                    inlier_ratio=0.7,
                    min_grid_fraction=0.5,
                )

            def find_route(self, **_kwargs):
                nonlocal hybrid_route_call_count
                hybrid_route_call_count += 1
                return FakeHybridRoute()

        def fake_generate_cross_frame_questions(**kwargs):
            frame_1 = kwargs["frame_1"]
            frame_2 = kwargs["frame_2"]
            requested = tuple(kwargs.get("only_question_types") or [])
            if requested == ("L2_object_move_occlusion",):
                occlusion_calls.append((frame_1.image_name, frame_2.image_name))
                raw = {
                    "level": "L2",
                    "type": "object_move_occlusion",
                    "question": "occlusion question",
                    "options": ["visible", "occluded", "not visible"],
                    "answer": "A",
                    "correct_value": "visible",
                    "moved_obj_id": 1,
                    "target_obj_id": 6,
                    "query_obj_id": 6,
                }
                return run_pipeline_module._annotate_cross_frame_questions(
                    [raw],
                    frame_1=frame_1,
                    frame_2=frame_2,
                    objects_by_id=objects_by_id,
                )

            non_occlusion_calls.append((frame_1.image_name, frame_2.image_name, requested))
            if frame_1.image_name != frame_names[0]:
                return []
            questions = [
                {
                    "level": "L2",
                    "type": "object_move_agent",
                    "reference_frame": "agent",
                    "question": f"agent question {ref_id}",
                    "options": ["left", "right", "front", "back"],
                    "answer": "A",
                    "correct_value": "left",
                    "moved_obj_id": 1,
                    "query_obj_id": 5,
                    "obj_b_id": 5,
                    "obj_c_id": ref_id,
                    "delta": [0.5, 0.0, 0.0],
                }
                for ref_id in (2, 3)
            ]
            questions.append({
                "level": "L2",
                "type": "object_move_occlusion",
                "question": "occlusion question",
                "options": ["visible", "occluded", "not visible"],
                "answer": "A",
                "correct_value": "visible",
                "moved_obj_id": 1,
                "target_obj_id": 6,
                "query_obj_id": 6,
            })
            return questions

        def fake_occlusion_veto(**kwargs):
            nonlocal occlusion_veto_call_count
            occlusion_veto_call_count += 1
            referable_ids = list(kwargs["referable_object_ids"])
            return {
                "raw_object_ids": referable_ids,
                "filtered_object_ids": referable_ids,
                "low_visible_object_ids": [],
                "not_visible_object_ids": [],
                "skipped_object_ids": [],
                "audit_by_object_id": {},
            }

        def fake_legacy_route(**_kwargs):
            nonlocal legacy_route_call_count
            legacy_route_call_count += 1
            return FakeLegacyRoute()

        def fake_depth_route(**_kwargs):
            nonlocal depth_route_call_count
            depth_route_call_count += 1
            return FakeDepthRoute()

        with (
            patch.object(run_pipeline_module, "parse_scene", return_value=scene),
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda _scene: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={1: [5, 6]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={5: 1, 6: 1}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={1: [5, 6]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={5: 1, 6: 1}),
            patch.object(run_pipeline_module, "has_nontrivial_attachment", return_value=True),
            patch.object(run_pipeline_module, "_load_scene_geometry", return_value=None),
            patch.object(run_pipeline_module, "load_axis_alignment", return_value=np.eye(4)),
            patch.object(run_pipeline_module, "load_scannet_poses", return_value=poses),
            patch.object(run_pipeline_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(run_pipeline_module, "load_instance_mesh_data", return_value=object()),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=object()),
            patch.object(
                run_pipeline_module,
                "_filter_referable_object_ids_with_occlusion_veto",
                side_effect=fake_occlusion_veto,
            ),
            patch.object(run_pipeline_module, "VisualPoseGraph", FakeVisualPoseGraph),
            patch.object(
                run_pipeline_module,
                "find_depth_corridor_auxiliary_route",
                side_effect=fake_depth_route,
            ),
            patch.object(
                run_pipeline_module,
                "HybridAuxiliaryRouter",
                FakeHybridAuxiliaryRouter,
            ),
            patch.object(
                run_pipeline_module,
                "find_geometric_auxiliary_route",
                side_effect=fake_legacy_route,
            ),
            patch.object(
                run_pipeline_module,
                "generate_cross_frame_questions",
                side_effect=fake_generate_cross_frame_questions,
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(
                run_pipeline_module,
                "compute_statistics",
                side_effect=lambda questions: {"total": len(questions)},
            ),
        ):
            questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=1,
                max_frames=3,
                use_occlusion=False,
                referability_cache=referability_cache,
                only_question_types=[
                    "L2_object_move_agent",
                    "L2_object_move_occlusion",
                ],
                scene_type_cap=0,
                frame_type_cap=0,
                frame_type_object_cap=0,
                run_question_presence_review=False,
                write_frame_debug=False,
            )
            default_non_occlusion_calls = list(non_occlusion_calls)
            default_occlusion_calls = list(occlusion_calls)
            visual_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=root / "visual_output",
                max_scenes=1,
                max_frames=3,
                use_occlusion=False,
                referability_cache=referability_cache,
                only_question_types=[
                    "L2_object_move_agent",
                    "L2_object_move_occlusion",
                ],
                scene_type_cap=0,
                frame_type_cap=0,
                frame_type_object_cap=0,
                run_question_presence_review=False,
                write_frame_debug=False,
                auxiliary_route_method="visual_pose_graph",
            )
            legacy_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=root / "legacy_output",
                max_scenes=1,
                max_frames=3,
                use_occlusion=False,
                referability_cache=referability_cache,
                only_question_types=[
                    "L2_object_move_agent",
                    "L2_object_move_occlusion",
                ],
                scene_type_cap=0,
                frame_type_cap=0,
                frame_type_object_cap=0,
                run_question_presence_review=False,
                write_frame_debug=False,
                auxiliary_route_method="legacy_geometric",
            )
            hybrid_output_dir = root / "hybrid_output"
            hybrid_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=hybrid_output_dir,
                max_scenes=1,
                max_frames=3,
                use_occlusion=False,
                referability_cache=referability_cache,
                only_question_types=[
                    "L2_object_move_agent",
                    "L2_object_move_occlusion",
                ],
                scene_type_cap=0,
                frame_type_cap=0,
                frame_type_object_cap=0,
                run_question_presence_review=False,
                write_frame_debug=False,
                auxiliary_route_method="hybrid_geometric_visual",
            )

        self.assertEqual(
            [call[0] for call in default_non_occlusion_calls],
            frame_names,
        )
        self.assertEqual(occlusion_veto_call_count, 0)
        self.assertTrue(
            all(call[1] == "__deferred_frame_2__" for call in default_non_occlusion_calls)
        )
        self.assertEqual(default_occlusion_calls, [])
        self.assertEqual(Counter(question["type"] for question in questions), {
            "object_move_agent": 2,
            "object_move_occlusion": 2,
        })
        self.assertEqual(
            {question.get("reasoning_frame_2") for question in questions},
            {frame_names[1], frame_names[2]},
        )
        funnel = json.loads(
            (output_dir / "cross_frame_funnel" / f"{scene_id}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(funnel["question_type_generated_counts"], {
            "object_move_agent": 2,
            "object_move_occlusion": 2,
        })
        self.assertFalse(funnel["auxiliary_graph"]["cache_hit"])
        self.assertEqual(funnel["auxiliary_graph"]["method"], "depth_corridor_geometric")
        self.assertGreater(depth_route_call_count, 0)
        self.assertTrue(questions)
        self.assertTrue(all(
            question["auxiliary_route"]["method"] == "depth_corridor_geometric"
            for question in questions
        ))
        self.assertTrue(all(
            question["auxiliary_image_names"] == ["depth_bridge.jpg"]
            for question in questions
        ))
        self.assertTrue(all(
            question["auxiliary_route"]["min_depth_valid_fraction"] == 0.8
            for question in questions
        ))
        self.assertTrue(all(
            question["auxiliary_route"]["visual_pruned_auxiliary_frame_count"] == 1
            for question in questions
        ))
        self.assertTrue(all(
            question["auxiliary_route"]["visual_prune_relaxed_angle_edge_count"] == 1
            for question in questions
        ))
        self.assertEqual(visual_graph_init_count, 1)
        self.assertTrue(visual_questions)
        self.assertTrue(all(
            question["auxiliary_route"]["method"] == "visual_pose_graph"
            for question in visual_questions
        ))
        self.assertGreater(legacy_route_call_count, 0)
        self.assertTrue(legacy_questions)
        self.assertTrue(all(
            question["auxiliary_route"]["method"] == "legacy_geometric"
            for question in legacy_questions
        ))
        self.assertTrue(all(
            question["auxiliary_image_names"] == ["legacy_bridge.jpg"]
            for question in legacy_questions
        ))
        self.assertGreater(hybrid_route_call_count, 0)
        self.assertTrue(hybrid_questions)
        self.assertTrue(all(
            question["auxiliary_route"]["method"] == "hybrid_geometric_visual"
            for question in hybrid_questions
        ))
        self.assertTrue(all(
            question["auxiliary_image_names"] == ["hybrid_bridge.jpg"]
            for question in hybrid_questions
        ))
        self.assertTrue(all(
            question["auxiliary_route"]["visual_models"]
            == ["fundamental", "homography"]
            for question in hybrid_questions
        ))
        hybrid_funnel = json.loads(
            (hybrid_output_dir / "cross_frame_funnel" / f"{scene_id}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            hybrid_funnel["auxiliary_graph"]["method"],
            "hybrid_geometric_visual",
        )
        with self.assertRaisesRegex(RuntimeError, "Cannot resume with auxiliary_route_method"):
            run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=1,
                max_frames=3,
                referability_cache=referability_cache,
                only_question_types=["L2_object_move_agent"],
                run_question_presence_review=False,
                write_frame_debug=False,
                resume=True,
                auxiliary_route_method="legacy_geometric",
            )

    def test_run_pipeline_finalizes_frame_debug_after_scene_level_flush(self) -> None:
        root = make_case_dir("pipeline_frame_debug_finalize")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        referability_cache = {
            "version": "20.0",
            "frames": {
                scene_id: {
                    image_name: {
                        "frame_usable": True,
                        "candidate_visible_object_ids": [1, 2],
                        "crop_label_statuses": {"lamp": "unique", "table": "unique"},
                        "crop_label_counts": {"lamp": 1, "table": 1},
                        "crop_referable_object_ids": [1, 2],
                        "full_frame_label_reviews": [],
                        "full_frame_label_statuses": {},
                        "full_frame_label_counts": {},
                        "referable_object_ids": [1, 2],
                        "label_statuses": {"lamp": "unique", "table": "unique"},
                        "label_counts": {"lamp": 1, "table": 1},
                        "out_of_frame_label_reviews": [],
                        "out_of_frame_not_visible_labels": [],
                        "out_of_frame_label_to_object_ids": {},
                        "out_of_frame_vlm_early_stop": False,
                        "candidate_labels": ["lamp", "table"],
                        "label_to_object_ids": {"lamp": [1], "table": [2]},
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
                    1: {"bbox_in_frame_ratio": 0.92},
                    2: {"bbox_in_frame_ratio": 0.88},
                },
            ),
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                return_value=[
                    {
                        "question": "Is the lamp on the table?",
                        "answer": "A",
                        "options": ["yes", "no"],
                        "type": "attachment",
                        "level": "L1",
                        "mentioned_objects": [
                            {"role": "target", "label": "lamp", "obj_id": 1},
                            {"role": "reference", "label": "table", "obj_id": 2},
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
                run_question_presence_review=False,
                write_frame_debug=True,
            )

        self.assertEqual(len(questions), 1)
        debug_path = output_dir / "frame_debug" / f"{scene_id}.json"
        with open(debug_path, "r", encoding="utf-8") as f:
            debug_record = json.load(f)

        self.assertEqual(len(debug_record["frames"]), 1)
        frame_record = debug_record["frames"][0]
        self.assertEqual(len(frame_record["generated_questions"]), 1)
        self.assertEqual(len(frame_record["final_questions"]), 1)
        self.assertEqual(frame_record["final_question_count"], 1)
        self.assertEqual(debug_record["summary"]["generated_question_count"], 1)
        self.assertEqual(debug_record["summary"]["final_question_count"], 1)

    def test_run_pipeline_resume_processes_only_pending_scenes(self) -> None:
        root = make_case_dir("pipeline_resume_pending_only")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_frames = {
            "scene0000_00": ["000001.jpg"],
            "scene0001_00": ["000002.jpg"],
        }
        referability_cache = make_simple_referability_cache(scene_frames)
        scenes = {scene_id: make_simple_scene(scene_id) for scene_id in scene_frames}

        for scene_id in scene_frames:
            scene_dir = data_root / scene_id
            (scene_dir / "pose").mkdir(parents=True)
            (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        def fake_load_scannet_poses(scene_dir: Path, axis_alignment=None):
            return {
                image_name: make_camera_pose(image_name)
                for image_name in scene_frames[scene_dir.name]
            }

        def interrupted_generate_all_questions(**kwargs):
            image_name = kwargs["camera_pose"].image_name
            if image_name == "000002.jpg":
                raise KeyboardInterrupt("stop after first cached scene")
            return [
                {
                    "question": f"Question for {image_name}",
                    "answer": "A",
                    "options": ["yes", "no"],
                    "type": "attachment",
                    "level": "L1",
                }
            ]

        common_patches = (
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "has_nontrivial_attachment", return_value=True),
            patch.object(run_pipeline_module, "_load_scene_geometry", return_value=None),
            patch.object(run_pipeline_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(run_pipeline_module, "load_scannet_poses", side_effect=fake_load_scannet_poses),
            patch.object(run_pipeline_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(run_pipeline_module, "load_instance_mesh_data", return_value=object()),
            patch.object(
                run_pipeline_module,
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                    2: {"bbox_in_frame_ratio": 0.90},
                },
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "parse_scene",
                    side_effect=lambda scene_dir, preloaded_geometry=None: scenes[scene_dir.name],
                )
            )
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "generate_all_questions",
                    side_effect=interrupted_generate_all_questions,
                )
            )
            for ctx in common_patches:
                stack.enter_context(ctx)
            with self.assertRaisesRegex(KeyboardInterrupt, "stop after first cached scene"):
                run_pipeline_module.run_pipeline(
                    data_root=data_root,
                    output_dir=output_dir,
                    max_scenes=10,
                    max_frames=10,
                    use_occlusion=False,
                    referability_cache=referability_cache,
                    run_question_presence_review=False,
                    write_frame_debug=False,
                )

        scene_status_path = output_dir / "scene_status.json"
        with open(scene_status_path, "r", encoding="utf-8") as f:
            scene_status = json.load(f)
        self.assertEqual(sorted(scene_status["completed_scenes"].keys()), ["scene0000_00"])
        raw_cache_dir = run_pipeline_module._raw_scene_questions_cache_dir(output_dir)
        self.assertTrue((raw_cache_dir / "scene0000_00.json").exists())
        self.assertFalse((raw_cache_dir / "scene0001_00.json").exists())

        parse_calls: list[str] = []

        def resume_parse_scene(scene_dir: Path, preloaded_geometry=None):
            parse_calls.append(scene_dir.name)
            if scene_dir.name == "scene0000_00":
                raise AssertionError("resume should skip completed scene0000_00")
            return scenes[scene_dir.name]

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(run_pipeline_module, "parse_scene", side_effect=resume_parse_scene)
            )
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "generate_all_questions",
                    side_effect=lambda **kwargs: [
                        {
                            "question": f"Question for {kwargs['camera_pose'].image_name}",
                            "answer": "A",
                            "options": ["yes", "no"],
                            "type": "attachment",
                            "level": "L1",
                        }
                    ],
                )
            )
            for ctx in common_patches:
                stack.enter_context(ctx)
            questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
                resume=True,
            )

        self.assertEqual(parse_calls, ["scene0001_00"])
        self.assertEqual(sorted(question["scene_id"] for question in questions), ["scene0000_00", "scene0001_00"])
        benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(len(benchmark["questions"]), 2)

    def test_run_pipeline_resume_rebuilds_final_output_without_reprocessing_completed_scenes(self) -> None:
        root = make_case_dir("pipeline_resume_rebuild_only")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")
        referability_cache = make_simple_referability_cache({scene_id: [image_name]})
        scene = make_simple_scene(scene_id)

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
                    2: {"bbox_in_frame_ratio": 0.90},
                },
            ),
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                return_value=[
                    {
                        "question": "Initial cached question",
                        "answer": "A",
                        "options": ["yes", "no"],
                        "type": "attachment",
                        "level": "L1",
                    }
                ],
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        ):
            first_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        first_benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))

        with (
            patch.object(
                run_pipeline_module,
                "parse_scene",
                side_effect=AssertionError("resume should rebuild from cached raw scene questions only"),
            ),
            patch.object(
                run_pipeline_module,
                "generate_all_questions",
                side_effect=AssertionError("resume should not regenerate completed scenes"),
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
        ):
            resumed_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
                resume=True,
            )

        resumed_benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(resumed_questions, first_questions)
        self.assertEqual(resumed_benchmark, first_benchmark)

    def test_run_pipeline_resume_regenerates_scene_when_status_exists_but_raw_cache_missing(self) -> None:
        root = make_case_dir("pipeline_resume_missing_raw")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_id = "scene0000_00"
        image_name = "000123.jpg"
        scene_dir = data_root / scene_id
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")
        referability_cache = make_simple_referability_cache({scene_id: [image_name]})
        scene = make_simple_scene(scene_id)

        first_generate = [
            {
                "question": "Original question",
                "answer": "A",
                "options": ["yes", "no"],
                "type": "attachment",
                "level": "L1",
            }
        ]
        second_generate = [
            {
                "question": "Regenerated question",
                "answer": "A",
                "options": ["yes", "no"],
                "type": "attachment",
                "level": "L1",
            }
        ]

        common_patches = (
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
                    2: {"bbox_in_frame_ratio": 0.90},
                },
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(run_pipeline_module, "parse_scene", return_value=scene)
            )
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "generate_all_questions",
                    return_value=first_generate,
                )
            )
            for ctx in common_patches:
                stack.enter_context(ctx)
            run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        raw_cache_dir = run_pipeline_module._raw_scene_questions_cache_dir(output_dir)
        (raw_cache_dir / f"{scene_id}.json").unlink()
        parse_calls: list[str] = []

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "parse_scene",
                    side_effect=lambda scene_dir, preloaded_geometry=None: parse_calls.append(scene_dir.name) or scene,
                )
            )
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "generate_all_questions",
                    return_value=second_generate,
                )
            )
            for ctx in common_patches:
                stack.enter_context(ctx)
            resumed_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
                resume=True,
            )

        self.assertEqual(parse_calls, [scene_id])
        self.assertEqual([question["question"] for question in resumed_questions], ["Regenerated question"])
        self.assertTrue((raw_cache_dir / f"{scene_id}.json").exists())

    def test_run_pipeline_reset_reprocesses_only_most_recent_completed_scene(self) -> None:
        root = make_case_dir("pipeline_resume_reset")
        self.addCleanup(shutil.rmtree, root, True)
        data_root = root / "data"
        output_dir = root / "output"
        scene_frames = {
            "scene0000_00": ["000001.jpg"],
            "scene0001_00": ["000002.jpg"],
        }
        referability_cache = make_simple_referability_cache(scene_frames)
        scenes = {scene_id: make_simple_scene(scene_id) for scene_id in scene_frames}

        for scene_id in scene_frames:
            scene_dir = data_root / scene_id
            (scene_dir / "pose").mkdir(parents=True)
            (scene_dir / f"{scene_id}_vh_clean.ply").write_text("ply\n", encoding="utf-8")

        def fake_load_scannet_poses(scene_dir: Path, axis_alignment=None):
            return {
                image_name: make_camera_pose(image_name)
                for image_name in scene_frames[scene_dir.name]
            }

        common_patches = (
            patch.object(run_pipeline_module, "enrich_scene_with_attachment", side_effect=lambda scene_dict: None),
            patch.object(run_pipeline_module, "get_scene_attachment_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_attached_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_graph", return_value={2: [1]}),
            patch.object(run_pipeline_module, "get_scene_support_chain_by", return_value={1: [2]}),
            patch.object(run_pipeline_module, "has_nontrivial_attachment", return_value=True),
            patch.object(run_pipeline_module, "_load_scene_geometry", return_value=None),
            patch.object(run_pipeline_module, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(run_pipeline_module, "load_scannet_poses", side_effect=fake_load_scannet_poses),
            patch.object(run_pipeline_module, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(run_pipeline_module, "load_instance_mesh_data", return_value=object()),
            patch.object(
                run_pipeline_module,
                "compute_frame_object_visibility",
                return_value={
                    1: {"bbox_in_frame_ratio": 0.95},
                    2: {"bbox_in_frame_ratio": 0.90},
                },
            ),
            patch.object(run_pipeline_module, "full_quality_pipeline", side_effect=lambda questions: questions),
            patch.object(run_pipeline_module, "compute_statistics", side_effect=lambda questions: {"total": len(questions)}),
            patch.object(run_pipeline_module.RayCaster, "from_ply", return_value=Mock()),
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "parse_scene",
                    side_effect=lambda scene_dir, preloaded_geometry=None: scenes[scene_dir.name],
                )
            )
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "generate_all_questions",
                    side_effect=lambda **kwargs: [
                        {
                            "question": f"Initial {kwargs['camera_pose'].image_name}",
                            "answer": "A",
                            "options": ["yes", "no"],
                            "type": "attachment",
                            "level": "L1",
                        }
                    ],
                )
            )
            for ctx in common_patches:
                stack.enter_context(ctx)
            run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
            )

        parse_calls: list[str] = []

        def reset_parse_scene(scene_dir: Path, preloaded_geometry=None):
            parse_calls.append(scene_dir.name)
            if scene_dir.name == "scene0000_00":
                raise AssertionError("reset=1 resume should keep the older completed scene cached")
            return scenes[scene_dir.name]

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(run_pipeline_module, "parse_scene", side_effect=reset_parse_scene)
            )
            stack.enter_context(
                patch.object(
                    run_pipeline_module,
                    "generate_all_questions",
                    side_effect=lambda **kwargs: [
                        {
                            "question": f"Reset {kwargs['camera_pose'].image_name}",
                            "answer": "A",
                            "options": ["yes", "no"],
                            "type": "attachment",
                            "level": "L1",
                        }
                    ],
                )
            )
            for ctx in common_patches:
                stack.enter_context(ctx)
            resumed_questions = run_pipeline_module.run_pipeline(
                data_root=data_root,
                output_dir=output_dir,
                max_scenes=10,
                max_frames=10,
                use_occlusion=False,
                referability_cache=referability_cache,
                run_question_presence_review=False,
                write_frame_debug=False,
                resume=True,
                reset=1,
            )

        self.assertEqual(parse_calls, ["scene0001_00"])
        questions_by_scene = {
            question["scene_id"]: question["question"]
            for question in resumed_questions
        }
        self.assertEqual(questions_by_scene["scene0000_00"], "Initial 000001.jpg")
        self.assertEqual(questions_by_scene["scene0001_00"], "Reset 000002.jpg")


class VerticalObjectRotateQuestionFilterTests(unittest.TestCase):
    @staticmethod
    def _box_object(
        obj_id: int,
        center: tuple[float, float, float],
        *,
        half_extent: float = 0.1,
    ) -> dict:
        center_arr = np.asarray(center, dtype=float)
        extent = np.full(3, half_extent, dtype=float)
        return {
            "id": obj_id,
            "label": f"object-{obj_id}",
            "center": center_arr.tolist(),
            "bbox_min": (center_arr - extent).tolist(),
            "bbox_max": (center_arr + extent).tolist(),
        }

    @staticmethod
    def _question(**overrides) -> dict:
        question = {
            "level": "L2",
            "type": "object_rotate_object_centric",
            "moved_obj_id": 10,
            "query_obj_id": 1,
            "obj_ref_id": 2,
            "obj_face_id": 3,
            "rotation_angle": 90,
            "rotation_direction": "counterclockwise",
        }
        question.update(overrides)
        return question

    def _filter(self, question: dict, objects: list[dict]) -> list[dict]:
        return run_pipeline_module._filter_vertical_object_rotate_questions(
            [question],
            scene_objects=objects,
            attachment_graph={10: [1]},
        )

    def test_drops_initial_vertical_query_ref_pair(self) -> None:
        objects = [
            self._box_object(10, (1.0, 0.0, 0.2)),
            self._box_object(1, (1.0, 0.0, 1.1)),
            self._box_object(2, (1.0, 0.0, 0.2)),
            self._box_object(3, (0.0, 0.0, 0.2)),
        ]

        self.assertEqual(self._filter(self._question(), objects), [])

    def test_drops_initial_vertical_query_face_pair(self) -> None:
        objects = [
            self._box_object(10, (1.0, 0.0, 0.2)),
            self._box_object(1, (1.0, 0.0, 1.1)),
            self._box_object(2, (2.0, 0.0, 0.2)),
            self._box_object(3, (1.0, 0.0, 0.2)),
        ]

        self.assertEqual(self._filter(self._question(), objects), [])

    def test_drops_pair_that_becomes_vertical_after_orbit_rotation(self) -> None:
        objects = [
            self._box_object(10, (1.0, 0.0, 0.2)),
            self._box_object(1, (1.0, 0.0, 1.1)),
            self._box_object(2, (0.0, 1.0, 0.2)),
            self._box_object(3, (0.0, 0.0, 0.2)),
        ]

        self.assertEqual(self._filter(self._question(), objects), [])

    def test_keeps_horizontal_pairs_before_and_after_rotation(self) -> None:
        question = self._question()
        objects = [
            self._box_object(10, (1.0, 0.0, 0.2)),
            self._box_object(1, (1.0, 0.0, 1.1)),
            self._box_object(2, (2.0, 0.0, 0.2)),
            self._box_object(3, (0.0, 0.0, 0.2)),
        ]

        self.assertEqual(self._filter(question, objects), [question])

    def test_does_not_filter_other_question_types(self) -> None:
        question = self._question(type="object_move_object_centric")
        objects = [
            self._box_object(10, (1.0, 0.0, 0.2)),
            self._box_object(1, (1.0, 0.0, 1.1)),
            self._box_object(2, (1.0, 0.0, 0.2)),
            self._box_object(3, (1.0, 0.0, 0.2)),
        ]

        self.assertEqual(self._filter(question, objects), [question])

    def test_keeps_question_with_incomplete_rotation_metadata(self) -> None:
        question = self._question()
        question.pop("rotation_angle")
        objects = [
            self._box_object(10, (1.0, 0.0, 0.2)),
            self._box_object(1, (1.0, 0.0, 1.1)),
            self._box_object(2, (2.0, 0.0, 0.2)),
            self._box_object(3, (0.0, 0.0, 0.2)),
        ]

        with self.assertLogs(run_pipeline_module.logger, level="WARNING"):
            self.assertEqual(self._filter(question, objects), [question])

if __name__ == "__main__":
    unittest.main()
