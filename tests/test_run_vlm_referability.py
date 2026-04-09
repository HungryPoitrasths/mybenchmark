import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import scripts.run_vlm_referability as referability_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="000000.jpg",
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
    base = float(obj_id)
    return {
        "id": obj_id,
        "label": label,
        "center": [base, 0.0, 1.0],
        "bbox_min": [base - 0.1, -0.1, 0.9],
        "bbox_max": [base + 0.1, 0.1, 1.1],
    }


def make_visibility_meta(
    *,
    roi_bounds_px: list[int] | None = None,
    projected_area_px: float = 900.0,
    bbox_in_frame_ratio: float = 0.2,
) -> dict:
    return {
        "roi_bounds_px": roi_bounds_px if roi_bounds_px is not None else [20, 60, 30, 90],
        "projected_area_px": projected_area_px,
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "edge_margin_px": 5.0,
    }


def make_instance_mesh_data(
    *,
    obj_id: int,
    sample_count: int = 8,
    point: np.ndarray | None = None,
):
    base_point = (
        np.asarray(point, dtype=np.float64)
        if point is not None
        else np.array([0.0, 0.0, 2.0], dtype=np.float64)
    )
    surface_points = np.tile(base_point[None, :], (int(sample_count), 1))
    triangle_ids = np.zeros(int(sample_count), dtype=np.int64)
    barycentrics = np.tile(
        np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
        (int(sample_count), 1),
    )
    return SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 2.0],
                [0.5, 0.0, 2.0],
                [0.0, 0.5, 2.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        triangle_ids_by_instance={int(obj_id): np.array([0], dtype=np.int64)},
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={int(obj_id): surface_points},
        surface_triangle_ids_by_instance={int(obj_id): triangle_ids},
        surface_barycentrics_by_instance={int(obj_id): barycentrics},
    )


class _SequenceVisibilityCaster:
    def __init__(self, responses: list[tuple[int, int]]) -> None:
        self._responses = [tuple((int(visible), int(valid))) for visible, valid in responses]
        self.calls: list[dict[str, object]] = []

    def mesh_visibility_stats(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        **kwargs,
    ):
        self.calls.append(
            {
                "camera_pos": np.asarray(camera_pos, dtype=np.float64),
                "target_points": np.asarray(target_points, dtype=np.float64),
                "target_tri_ids": set(int(tri_id) for tri_id in target_tri_ids),
            }
        )
        if not self._responses:
            raise AssertionError("mesh_visibility_stats called more times than expected")
        return self._responses.pop(0)


class RunVlmReferabilityTests(unittest.TestCase):
    def test_frame_prompt_requests_clarity_scoring_with_lenient_blur_handling(self) -> None:
        prompt = referability_module._frame_prompt().lower()

        self.assertIn("clarity_score", prompt)
        self.assertIn("slight softness is acceptable", prompt)
        self.assertIn("severely_out_of_focus", prompt)
        self.assertIn("usable_for_spatial_reasoning", prompt)
        self.assertIn("prioritize image clarity", prompt)

    def test_object_review_prompt_uses_full_image_plus_crop(self) -> None:
        prompt = referability_module._object_review_prompt("chair").lower()

        self.assertIn("full scene image", prompt)
        self.assertIn("crop", prompt)
        self.assertIn("clear", prompt)
        self.assertIn("absent", prompt)
        self.assertIn("unsure", prompt)
        self.assertNotIn("obj_id", prompt)

    def test_full_frame_label_review_prompt_requests_uniqueness_judgment(self) -> None:
        prompt = referability_module._full_frame_label_review_prompt("table").lower()

        self.assertIn("full frame", prompt)
        self.assertIn("unique", prompt)
        self.assertIn("multiple", prompt)
        self.assertIn("absent", prompt)
        self.assertIn("how many objects", prompt)

    def test_frame_decision_propagates_clarity_review_result(self) -> None:
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=(
                {
                    "clarity_score": 28,
                    "severely_out_of_focus": True,
                    "usable_for_spatial_reasoning": False,
                    "reason": "severely blurry",
                },
                "",
            ),
        ):
            decision = referability_module._frame_decision(
                client=object(),
                model="fake-model",
                image=np.zeros((2, 2, 3), dtype=np.uint8),
            )

        self.assertEqual(
            decision,
            {
                "clarity_score": 28,
                "severely_out_of_focus": True,
                "usable_for_spatial_reasoning": False,
                "frame_usable": False,
                "reason": "severely blurry",
            },
        )

    def test_frame_selection_score_prioritizes_usable_and_clarity(self) -> None:
        clear_usable = referability_module._frame_selection_score(
            10,
            {
                "clarity_score": 70,
                "frame_usable": True,
            },
        )
        sharper_but_unusable = referability_module._frame_selection_score(
            999,
            {
                "clarity_score": 95,
                "frame_usable": False,
            },
        )
        slightly_clearer = referability_module._frame_selection_score(
            1,
            {
                "clarity_score": 71,
                "frame_usable": True,
            },
        )

        self.assertGreater(clear_usable, sharper_but_unusable)
        self.assertGreater(slightly_clearer, clear_usable)

    def test_normalize_frame_review_backfills_from_partial_entry(self) -> None:
        normalized = referability_module._normalize_frame_review(
            {
                "frame_usable": False,
                "reason": "too blurry for geometry",
            }
        )

        self.assertEqual(
            normalized,
            {
                "clarity_score": 60,
                "severely_out_of_focus": False,
                "usable_for_spatial_reasoning": False,
                "frame_usable": False,
                "reason": "too blurry for geometry",
            },
        )

    def test_object_review_decision_normalizes_status(self) -> None:
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=({"status": "visible"}, '{"status":"visible"}'),
        ):
            status, raw = referability_module._object_review_decision(
                client=object(),
                model="fake-model",
                image_b64="ZnVsbA==",
                crop_b64="Y3JvcA==",
                label="chair",
            )

        self.assertEqual(status, "clear")
        self.assertEqual(raw, '{"status":"visible"}')

    def test_full_frame_label_review_decision_normalizes_multiple(self) -> None:
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=({"status": "two_or_more"}, '{"status":"two_or_more"}'),
        ):
            status, raw = referability_module._full_frame_label_review_decision(
                client=object(),
                model="fake-model",
                image_b64="ZnVsbA==",
                label="table",
            )

        self.assertEqual(status, "multiple")
        self.assertEqual(raw, '{"status":"two_or_more"}')

    def test_full_frame_label_review_decision_falls_back_to_unsure_for_invalid_status(self) -> None:
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=({"status": "banana"}, '{"status":"banana"}'),
        ):
            status, raw = referability_module._full_frame_label_review_decision(
                client=object(),
                model="fake-model",
                image_b64="ZnVsbA==",
                label="table",
            )

        self.assertEqual(status, "unsure")
        self.assertEqual(raw, '{"status":"banana"}')

    def test_refine_candidate_visible_object_ids_uses_depth_when_available(self) -> None:
        with patch.object(
            referability_module,
            "refine_visible_ids_with_depth",
            return_value=[2],
        ) as refine_mock:
            refined_ids, source = referability_module._refine_candidate_visible_object_ids(
                visible_object_ids=[1, 2],
                objects=[make_object(1, "cup"), make_object(2, "cup")],
                camera_pose=make_camera_pose(),
                depth_image=np.ones((4, 4), dtype=np.float32),
                depth_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual(refined_ids, [2])
        self.assertEqual(source, "depth_refined")
        refine_mock.assert_called_once()

    def test_build_object_review_crop_treats_missing_projection_as_out_of_frame(self) -> None:
        crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            {"projected_area_px": 0.0},
        )

        self.assertFalse(crop["valid"])
        self.assertEqual(crop["local_outcome"], "out_of_frame")
        self.assertEqual(crop["reason"], "missing_projection")

    def test_build_object_review_crop_excludes_small_projection_but_does_not_gate_on_in_frame_ratio(self) -> None:
        tiny_crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=399.0, bbox_in_frame_ratio=0.1),
        )
        valid_crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.1),
        )

        self.assertEqual(tiny_crop["local_outcome"], "excluded")
        self.assertEqual(tiny_crop["reason"], "projected_area_too_small")
        self.assertTrue(valid_crop["valid"])
        self.assertEqual(valid_crop["local_outcome"], "reviewed")

    def test_aggregate_label_reviews_uses_strict_policy(self) -> None:
        label_to_ids = {
            "chair": [1, 2],
            "lamp": [3, 4],
            "plant": [5, 6],
            "table": [7],
            "sofa": [8, 9],
        }
        object_reviews = {
            1: {"obj_id": 1, "local_outcome": "reviewed", "vlm_status": "clear"},
            2: {"obj_id": 2, "local_outcome": "out_of_frame", "vlm_status": None},
            3: {"obj_id": 3, "local_outcome": "reviewed", "vlm_status": "clear"},
            4: {"obj_id": 4, "local_outcome": "reviewed", "vlm_status": "clear"},
            5: {"obj_id": 5, "local_outcome": "reviewed", "vlm_status": "absent"},
            6: {"obj_id": 6, "local_outcome": "excluded", "vlm_status": None},
            7: {"obj_id": 7, "local_outcome": "reviewed", "vlm_status": "unsure"},
            8: {"obj_id": 8, "local_outcome": "reviewed", "vlm_status": "clear"},
            9: {"obj_id": 9, "local_outcome": "reviewed", "vlm_status": "unsure"},
        }

        label_statuses, label_counts, referable_ids = referability_module._aggregate_label_reviews(
            label_to_ids,
            object_reviews,
        )

        self.assertEqual(
            label_statuses,
            {
                "chair": "unique",
                "lamp": "multiple",
                "plant": "absent",
                "sofa": "unsure",
                "table": "unsure",
            },
        )
        self.assertEqual(
            label_counts,
            {
                "chair": 1,
                "lamp": 2,
                "plant": 0,
                "sofa": 1,
                "table": 0,
            },
        )
        self.assertEqual(referable_ids, [1])

    def test_compute_frame_referability_entry_builds_v10_object_reviews(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
            make_object(3, "lamp"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: {"projected_area_px": 0.0},
            3: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clarity_score": 82,
                    "severely_out_of_focus": False,
                    "usable_for_spatial_reasoning": True,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2, 3], "depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                side_effect=[("clear", '{"status":"clear"}'), ("absent", '{"status":"absent"}')],
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
                return_value=("unique", '{"status":"unique"}'),
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_ray_review",
                side_effect=lambda **_kwargs: None,
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2, 3],
            )

        self.assertEqual(frame_entry["frame_usable"], True)
        self.assertEqual(frame_entry["frame_quality_score"], 82)
        self.assertEqual(frame_entry["frame_quality_severely_out_of_focus"], False)
        self.assertEqual(frame_entry["frame_quality_usable_for_spatial_reasoning"], True)
        self.assertEqual(frame_entry["frame_quality_reason"], "clear enough")
        self.assertEqual(frame_entry["candidate_visibility_source"], "depth_refined")
        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "unique", "lamp": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 1, "lamp": 0})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["full_frame_label_counts"], {"chair": 1})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique", "lamp": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 1, "lamp": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_reviews"][0]["crop_referable_object_id"], 1)
        self.assertEqual(frame_entry["object_reviews"]["1"]["vlm_status"], "clear")
        self.assertEqual(frame_entry["object_reviews"]["2"]["local_outcome"], "out_of_frame")
        self.assertEqual(frame_entry["object_reviews"]["3"]["vlm_status"], "absent")
        self.assertEqual(review_mock.call_count, 2)
        full_frame_mock.assert_called_once_with(
            unittest.mock.ANY,
            "fake-vlm",
            unittest.mock.ANY,
            "chair",
        )

    def test_compute_frame_referability_entry_keeps_crop_unique_label_when_full_frame_is_multiple(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: {"projected_area_px": 0.0},
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clarity_score": 82,
                    "severely_out_of_focus": False,
                    "usable_for_spatial_reasoning": True,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2], "depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
                return_value=("multiple", '{"status":"multiple"}'),
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_ray_review",
                side_effect=lambda **_kwargs: None,
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"chair": "multiple"})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 1})
        self.assertEqual(frame_entry["referable_object_ids"], [1])

    def test_compute_frame_referability_entry_keeps_crop_unique_label_when_full_frame_is_absent(self) -> None:
        scene_objects = [make_object(1, "lamp")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clarity_score": 82,
                    "severely_out_of_focus": False,
                    "usable_for_spatial_reasoning": True,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
                return_value=("absent", '{"status":"absent"}'),
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_ray_review",
                side_effect=lambda **_kwargs: None,
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"lamp": "unique"})
        self.assertEqual(frame_entry["crop_label_counts"], {"lamp": 1})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"lamp": "absent"})
        self.assertEqual(frame_entry["full_frame_label_counts"], {"lamp": 0})
        self.assertEqual(frame_entry["label_statuses"], {"lamp": "unique"})
        self.assertEqual(frame_entry["label_counts"], {"lamp": 1})
        self.assertEqual(frame_entry["referable_object_ids"], [1])

    def test_apply_crop_unique_mesh_ray_review_passes_after_stage2_visible_evidence(self) -> None:
        object_reviews = {
            1: {
                "obj_id": 1,
                "label": "chair",
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
            }
        }
        caster = _SequenceVisibilityCaster([(0, 4), (2, 4)])
        instance_mesh_data = make_instance_mesh_data(obj_id=1, sample_count=8)

        referability_module._apply_crop_unique_mesh_ray_review(
            crop_unique_label_object_ids={"chair": 1},
            object_reviews=object_reviews,
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            ray_caster_getter=lambda: caster,
            instance_mesh_data_getter=lambda _base: instance_mesh_data,
        )

        review = object_reviews[1]["ray_visibility_review"]
        self.assertEqual(review["decision"], "pass")
        self.assertEqual(review["reason"], "stage2_visible_evidence")
        self.assertEqual(review["stage1"]["base_sample_count"], 64)
        self.assertEqual(review["stage1"]["visible_count"], 0)
        self.assertEqual(review["stage2"]["base_sample_count"], 512)
        self.assertEqual(review["stage2"]["visible_count"], 2)

    def test_apply_crop_unique_mesh_ray_review_drops_when_stage2_finds_no_visible_rays(self) -> None:
        object_reviews = {
            1: {
                "obj_id": 1,
                "label": "chair",
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
            }
        }
        caster = _SequenceVisibilityCaster([(0, 3), (0, 6)])
        instance_mesh_data = make_instance_mesh_data(obj_id=1, sample_count=8)

        referability_module._apply_crop_unique_mesh_ray_review(
            crop_unique_label_object_ids={"chair": 1},
            object_reviews=object_reviews,
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            ray_caster_getter=lambda: caster,
            instance_mesh_data_getter=lambda _base: instance_mesh_data,
        )

        review = object_reviews[1]["ray_visibility_review"]
        self.assertEqual(review["decision"], "drop")
        self.assertEqual(review["reason"], "fully_occluded_after_stage2")
        self.assertEqual(review["stage2"]["valid_count"], 6)
        self.assertEqual(review["stage2"]["visible_count"], 0)

    def test_apply_crop_unique_mesh_ray_review_drops_when_stage2_has_no_valid_rays(self) -> None:
        object_reviews = {
            1: {
                "obj_id": 1,
                "label": "chair",
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
            }
        }
        caster = _SequenceVisibilityCaster([(0, 0), (0, 0)])
        instance_mesh_data = make_instance_mesh_data(obj_id=1, sample_count=8)

        referability_module._apply_crop_unique_mesh_ray_review(
            crop_unique_label_object_ids={"chair": 1},
            object_reviews=object_reviews,
            camera_pose=make_camera_pose(),
            color_intrinsics=make_camera_intrinsics(),
            ray_caster_getter=lambda: caster,
            instance_mesh_data_getter=lambda _base: instance_mesh_data,
        )

        review = object_reviews[1]["ray_visibility_review"]
        self.assertEqual(review["decision"], "drop")
        self.assertEqual(review["reason"], "no_valid_rays_after_stage2")
        self.assertEqual(review["stage2"]["valid_count"], 0)

    def test_aggregate_label_reviews_treats_ray_drop_as_absent_like(self) -> None:
        label_statuses, label_counts, referable_ids = referability_module._aggregate_label_reviews(
            {"chair": [1, 2]},
            {
                1: {
                    "obj_id": 1,
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                    "ray_visibility_review": {
                        "applied": True,
                        "decision": "drop",
                        "reason": "fully_occluded_after_stage2",
                        "stage1": {"visible_count": 0, "valid_count": 1},
                        "stage2": {"visible_count": 0, "valid_count": 2},
                    },
                },
                2: {"obj_id": 2, "local_outcome": "out_of_frame", "vlm_status": None},
            },
        )

        self.assertEqual(label_statuses, {"chair": "absent"})
        self.assertEqual(label_counts, {"chair": 0})
        self.assertEqual(referable_ids, [])

    def test_compute_frame_referability_entry_drops_crop_unique_object_when_mesh_ray_finds_no_evidence(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: {"projected_area_px": 0.0},
        }
        low_mesh = make_instance_mesh_data(obj_id=1, sample_count=8)
        high_mesh = make_instance_mesh_data(obj_id=1, sample_count=16)
        caster = _SequenceVisibilityCaster([(0, 2), (0, 5)])

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clarity_score": 82,
                    "severely_out_of_focus": False,
                    "usable_for_spatial_reasoning": True,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2], "depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
                side_effect=AssertionError("full-frame review should not run after ray drop"),
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="fake-vlm",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("image.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2],
                ray_caster_getter=lambda: caster,
                instance_mesh_data_getter=lambda base: low_mesh if int(base) == 64 else high_mesh,
            )

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [])
        self.assertEqual(frame_entry["label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["referable_object_ids"], [])
        self.assertEqual(frame_entry["full_frame_label_reviews"], [])
        self.assertEqual(
            frame_entry["object_reviews"]["1"]["ray_visibility_review"]["reason"],
            "fully_occluded_after_stage2",
        )


if __name__ == "__main__":
    unittest.main()
