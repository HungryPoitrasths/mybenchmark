import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import numpy as np

import scripts.run_vlm_referability as referability_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="000000.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics(width: int = 120, height: int = 120) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def make_object(
    obj_id: int,
    label: str,
    *,
    alias_group: str | None = None,
    alias_variants: list[str] | None = None,
) -> dict:
    alias_group_name = alias_group or f"{label.replace(' ', '_')}_family"
    variants = alias_variants or [label]
    base = float(obj_id)
    return {
        "id": obj_id,
        "label": label,
        "raw_label": label,
        "canonical_label": label,
        "alias_group": alias_group_name,
        "alias_variants": list(variants),
        "center": [base, 0.0, 2.0],
        "bbox_min": [base - 0.2, -0.2, 1.8],
        "bbox_max": [base + 0.2, 0.2, 2.2],
    }


def make_visibility_meta(
    *,
    projected_area_px: float = 900.0,
    bbox_in_frame_ratio: float = 0.9,
    zbuffer_mask_area_px: float | None = None,
    has_zbuffer_mask_area: bool | None = None,
) -> dict:
    zbuffer_area = 0.0 if zbuffer_mask_area_px is None else float(zbuffer_mask_area_px)
    has_zbuffer = (
        bool(zbuffer_mask_area_px is not None)
        if has_zbuffer_mask_area is None
        else bool(has_zbuffer_mask_area)
    )
    return {
        "roi_bounds_px": [20, 60, 20, 60],
        "projected_area_px": projected_area_px,
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "edge_margin_px": 10.0,
        "zbuffer_mask_area_px": zbuffer_area,
        "has_zbuffer_mask_area": has_zbuffer,
    }


def make_detection(
    *,
    bbox: tuple[int, int, int, int],
    score: float,
    image_shape: tuple[int, int] = (120, 120),
    category: str = "object",
) -> dict:
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "mask": mask,
        "score": float(score),
        "area_px": int(mask.sum()),
        "category": category,
    }


def make_strip_instance_mesh_data(*, obj_id: int = 1, cells: int = 10) -> SimpleNamespace:
    vertices: list[list[float]] = []
    for x in range(cells + 1):
        for y in range(2):
            vertices.append([float(x), float(y), 2.0])

    def vertex_id(x: int, y: int) -> int:
        return x * 2 + y

    faces: list[list[int]] = []
    for x in range(cells):
        v00 = vertex_id(x, 0)
        v01 = vertex_id(x, 1)
        v10 = vertex_id(x + 1, 0)
        v11 = vertex_id(x + 1, 1)
        faces.append([v00, v01, v11])
        faces.append([v00, v11, v10])

    face_array = np.asarray(faces, dtype=np.int64)
    return SimpleNamespace(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=face_array,
        triangle_ids_by_instance={int(obj_id): np.arange(len(face_array), dtype=np.int64)},
        boundary_triangle_ids_by_instance={},
        surface_points_by_instance={},
        surface_triangle_ids_by_instance={},
        surface_barycentrics_by_instance={},
    )


def make_topology_quality(obj_id: int, status: str = "pass") -> dict:
    return {
        "obj_id": int(obj_id),
        "triangle_count": 32,
        "connected_component_count": 1,
        "largest_component_triangle_share": 1.0,
        "boundary_edge_ratio": 0.20,
        "num_boundary_loops": 1,
        "largest_boundary_loop_edge_share": 0.20,
        "status": status,
        "reason_codes": [] if status == "pass" else ["warn_flag"],
    }


def make_mesh_quality(obj_id: int, status: str = "pass", *, reason_codes: list[str] | None = None) -> dict:
    return {
        "obj_id": int(obj_id),
        "status": status,
        "profile": "topology_pass_base",
        "image_mask_area_px": 900,
        "mesh_mask_area_px": 880,
        "intersection_px": 800,
        "union_px": 980,
        "iou": 0.82 if status == "pass" else 0.20,
        "under_coverage": 0.11 if status == "pass" else 0.60,
        "over_coverage": 0.09 if status == "pass" else 0.32,
        "area_ratio": 0.98 if status == "pass" else 2.10,
        "depth_bad_ratio": 0.05 if status == "pass" else 0.40,
        "reason_codes": list(reason_codes or ([] if status == "pass" else ["low_iou"])),
        "thresholds": referability_module._mesh_quality_thresholds_for_topology_status("pass"),
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

    def mesh_visibility_stats(
        self,
        camera_pos,
        target_points,
        target_tri_ids,
        **kwargs,
    ):
        _ = np.asarray(camera_pos, dtype=np.float64)
        _ = np.asarray(target_points, dtype=np.float64)
        _ = set(int(tri_id) for tri_id in target_tri_ids)
        if not self._responses:
            raise AssertionError("mesh_visibility_stats called more times than expected")
        return self._responses.pop(0)


class RunVlmReferabilityTests(unittest.TestCase):
    def test_topology_warn_triggers_when_any_warn_condition_is_met(self) -> None:
        quality = referability_module._compute_topology_quality_for_object(
            obj_id=1,
            instance_mesh_data=make_strip_instance_mesh_data(obj_id=1, cells=10),
        )

        self.assertEqual(quality["status"], "warn")
        self.assertIn("boundary_edge_ratio_warn", quality["reason_codes"])
        self.assertNotIn("component_count_warn", quality["reason_codes"])

    def test_mesh_mask_quality_requires_iou_and_under_coverage(self) -> None:
        detection_mask = np.zeros((20, 20), dtype=bool)
        detection_mask[0:10, 0:10] = True
        rendered_mask = np.zeros((20, 20), dtype=bool)
        rendered_mask[8:18, 8:18] = True
        rendered_depth = np.where(rendered_mask, 2.0, np.inf).astype(np.float32)

        with (
            patch.object(
                referability_module,
                "_rasterize_instance_depth_map",
                return_value={"mask": rendered_mask, "depth": rendered_depth, "triangle_count": 20},
            ),
            patch.object(
                referability_module,
                "_compute_depth_bad_ratio",
                return_value=0.0,
            ),
        ):
            quality = referability_module._compute_mesh_mask_quality_for_object(
                obj_id=1,
                detection_mask=detection_mask,
                topology_status="pass",
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(20, 20),
                depth_image=None,
                depth_intrinsics=None,
                instance_mesh_data=None,
            )

        self.assertEqual(quality["status"], "fail")
        self.assertIn("low_iou", quality["reason_codes"])
        self.assertIn("high_under_coverage", quality["reason_codes"])

    def test_build_object_review_crop_requires_projected_area_of_at_least_800px(self) -> None:
        tiny_crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=799.0, bbox_in_frame_ratio=0.1),
        )
        valid_crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=800.0, bbox_in_frame_ratio=0.1),
        )

        self.assertEqual(tiny_crop["local_outcome"], "excluded")
        self.assertEqual(tiny_crop["reason"], "projected_area_too_small")
        self.assertTrue(valid_crop["valid"])
        self.assertEqual(valid_crop["local_outcome"], "reviewed")

    def test_build_object_review_crop_does_not_gate_on_zbuffer_mask_area(self) -> None:
        crop = referability_module._build_object_review_crop(
            np.zeros((120, 120, 3), dtype=np.uint8),
            make_visibility_meta(projected_area_px=900.0, zbuffer_mask_area_px=1.0),
        )

        self.assertTrue(crop["valid"])
        self.assertEqual(crop["local_outcome"], "reviewed")

    def test_refine_candidate_visible_object_ids_requires_both_mesh_ray_and_depth(self) -> None:
        with patch.object(
            referability_module,
            "refine_visible_ids_with_depth",
            return_value=[1],
        ) as depth_mock:
            candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
                [1],
                [make_object(1, "chair")],
                make_camera_pose(),
                make_camera_intrinsics(),
                np.ones((4, 4), dtype=np.float32),
                make_camera_intrinsics(),
                ray_caster_getter=lambda: _SequenceVisibilityCaster([(1, 4)]),
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
            )

        self.assertEqual(candidate_ids, [1])
        self.assertEqual(source, "mesh_ray_depth_refined")
        depth_mock.assert_called_once()

    def test_refine_candidate_visible_object_ids_drops_when_depth_rejects(self) -> None:
        with patch.object(
            referability_module,
            "refine_visible_ids_with_depth",
            return_value=[],
        ):
            candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
                [1],
                [make_object(1, "chair")],
                make_camera_pose(),
                make_camera_intrinsics(),
                np.ones((4, 4), dtype=np.float32),
                make_camera_intrinsics(),
                ray_caster_getter=lambda: _SequenceVisibilityCaster([(1, 4)]),
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
            )

        self.assertEqual(candidate_ids, [])
        self.assertEqual(source, "mesh_ray_depth_refined")

    def test_refine_candidate_visible_object_ids_uses_stage2_when_stage1_ratio_is_too_low(self) -> None:
        caster = _SequenceVisibilityCaster([(1, 20), (2, 8)])
        with patch.object(
            referability_module,
            "refine_visible_ids_with_depth",
            return_value=[1],
        ):
            candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
                [1],
                [make_object(1, "chair")],
                make_camera_pose(),
                make_camera_intrinsics(),
                np.ones((4, 4), dtype=np.float32),
                make_camera_intrinsics(),
                ray_caster_getter=lambda: caster,
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
            )

        self.assertEqual(candidate_ids, [1])
        self.assertEqual(source, "mesh_ray_depth_refined")
        self.assertEqual(caster._responses, [])

    def test_refine_candidate_visible_object_ids_drops_when_stage2_ratio_is_too_low(self) -> None:
        caster = _SequenceVisibilityCaster([(1, 20), (1, 20)])
        with patch.object(
            referability_module,
            "refine_visible_ids_with_depth",
            return_value=[1],
        ):
            candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
                [1],
                [make_object(1, "chair")],
                make_camera_pose(),
                make_camera_intrinsics(),
                np.ones((4, 4), dtype=np.float32),
                make_camera_intrinsics(),
                ray_caster_getter=lambda: caster,
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
            )

        self.assertEqual(candidate_ids, [])
        self.assertEqual(source, "mesh_ray_depth_refined")
        self.assertEqual(caster._responses, [])

    def test_refine_candidate_visible_object_ids_falls_back_to_projection_when_mesh_ray_fails(self) -> None:
        with patch.object(
            referability_module,
            "refine_visible_ids_with_depth",
            return_value=[1],
        ):
            candidate_ids, source = referability_module._refine_candidate_visible_object_ids(
                [1],
                [make_object(1, "chair")],
                make_camera_pose(),
                make_camera_intrinsics(),
                np.ones((4, 4), dtype=np.float32),
                make_camera_intrinsics(),
                ray_caster_getter=lambda: (_ for _ in ()).throw(RuntimeError("ray failed")),
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1, sample_count=8),
            )

        self.assertEqual(candidate_ids, [1])
        self.assertEqual(source, "projection_fallback")

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

    def test_normalize_frame_review_uses_clear_output_for_frame_gate(self) -> None:
        normalized = referability_module._normalize_frame_review(
            {
                "clear": False,
                "clarity_score": 28,
                "reason": "obviously blurry overall",
            }
        )

        self.assertEqual(
            normalized,
            {
                "clear": False,
                "clarity_score": 28,
                "frame_usable": False,
                "reason": "obviously blurry overall",
            },
        )

    def test_normalize_frame_review_accepts_legacy_frame_quality_fields(self) -> None:
        normalized = referability_module._normalize_frame_review(
            {
                "clarity_score": 82,
                "severely_out_of_focus": False,
                "usable_for_spatial_reasoning": True,
                "reason": "clear enough",
            }
        )

        self.assertEqual(normalized["clear"], True)
        self.assertEqual(normalized["frame_usable"], True)
        self.assertEqual(normalized["clarity_score"], 82)

    def test_compute_frame_referability_entry_builds_crop_vlm_reviews(self) -> None:
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
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2, 3], "mesh_ray_depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                side_effect=[("absent", '{"status":"absent"}')],
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
                return_value=("unique", '{"status":"unique"}'),
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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
        self.assertEqual(frame_entry["frame_quality_clear"], True)
        self.assertEqual(frame_entry["frame_quality_score"], 82)
        self.assertEqual(frame_entry["candidate_visibility_source"], "mesh_ray_depth_refined")
        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "multiple", "lamp": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 2, "lamp": 0})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {})
        self.assertEqual(frame_entry["full_frame_label_counts"], {})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "multiple", "lamp": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 2, "lamp": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [])
        self.assertEqual(frame_entry["full_frame_label_reviews"], [])
        self.assertEqual(frame_entry["object_reviews"]["1"]["review_mode"], "selector_duplicate_shortcut")
        self.assertEqual(frame_entry["object_reviews"]["1"]["review_skip_reason"], "selector_visible_label_multiple")
        self.assertIsNone(frame_entry["object_reviews"]["1"]["vlm_status"])
        self.assertEqual(frame_entry["object_reviews"]["2"]["local_outcome"], "out_of_frame")
        self.assertEqual(frame_entry["object_reviews"]["3"]["vlm_status"], "absent")
        self.assertEqual(frame_entry["referability_reason_by_alias_group"]["chair_family"], "selector_duplicate_shortcut")
        self.assertEqual(review_mock.call_count, 1)
        full_frame_mock.assert_not_called()

    def test_compute_frame_referability_entry_uses_earlier_quantity_veto(self) -> None:
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
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1, 2], "mesh_ray_depth_refined"),
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
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "multiple"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 2})
        self.assertEqual(frame_entry["full_frame_label_statuses"], {})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "multiple"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 2})
        self.assertEqual(frame_entry["referable_object_ids"], [])

    def test_compute_frame_referability_entry_duplicate_selector_label_without_candidates_becomes_absent(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
            2: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([], "mesh_ray_depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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

        self.assertEqual(frame_entry["candidate_visible_object_ids"], [])
        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["full_frame_label_statuses"], {})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [])
        review_mock.assert_not_called()
        full_frame_mock.assert_not_called()

    def test_compute_frame_referability_entry_full_frame_absent_vetoes_crop_unique(self) -> None:
        scene_objects = [make_object(1, "shelves")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_depth_refined"),
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
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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

        self.assertEqual(frame_entry["crop_label_statuses"], {"shelves": "unique"})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"shelves": "absent"})
        self.assertEqual(frame_entry["label_statuses"], {"shelves": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"shelves": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [])

    def test_compute_frame_referability_entry_excludes_unique_object_below_projected_area_threshold(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=799.0, zbuffer_mask_area_px=1.0),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
            ) as review_mock,
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
            ) as full_frame_mock,
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "absent"})
        self.assertEqual(frame_entry["crop_label_counts"], {"chair": 0})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [])
        self.assertEqual(frame_entry["referable_object_ids"], [])
        self.assertEqual(frame_entry["object_reviews"]["1"]["local_outcome"], "excluded")
        self.assertEqual(frame_entry["object_reviews"]["1"]["local_reason"], "projected_area_too_small")
        review_mock.assert_not_called()
        full_frame_mock.assert_not_called()

    def test_compute_frame_referability_entry_passes_instance_mesh_data_to_visibility(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        captured: dict[str, object] = {}
        visibility_meta = make_visibility_meta(
            projected_area_px=900.0,
            zbuffer_mask_area_px=900.0,
        )
        sentinel_instance_mesh_data = object()

        def fake_compute_frame_object_visibility(*args, **kwargs):
            captured["instance_mesh_data"] = kwargs.get("instance_mesh_data")
            return {1: visibility_meta}

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                side_effect=fake_compute_frame_object_visibility,
            ),
            patch.object(
                referability_module,
                "_object_review_decision",
                return_value=("clear", '{"status":"clear"}'),
            ),
            patch.object(
                referability_module,
                "_full_frame_label_review_decision",
                return_value=("unique", '{"status":"unique"}'),
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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
                instance_mesh_data_getter=lambda _base: sentinel_instance_mesh_data,
            )

        self.assertIs(captured["instance_mesh_data"], sentinel_instance_mesh_data)
        self.assertEqual(frame_entry["referable_object_ids"], [1])

    def test_compute_frame_referability_entry_applies_bbox_ratio_gate_to_final_referable_ids(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {
            1: make_visibility_meta(projected_area_px=900.0, bbox_in_frame_ratio=0.69),
        }

        with (
            patch.object(
                referability_module,
                "_frame_decision",
                return_value={
                    "clear": True,
                    "clarity_score": 82,
                    "frame_usable": True,
                    "reason": "clear enough",
                },
            ),
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([1], "mesh_ray_depth_refined"),
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
                return_value=("unique", '{"status":"unique"}'),
            ),
            patch.object(
                referability_module,
                "_apply_crop_unique_mesh_quality_review",
                return_value={},
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

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["crop_referable_object_ids"], [1])
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["attachment_referable_object_ids"], [1])
        self.assertEqual(frame_entry["referable_object_ids"], [])

    def test_apply_crop_unique_mesh_quality_review_drops_unique_object_on_mesh_mismatch(self) -> None:
        crop_entry = {
            "local_outcome": "reviewed",
            "reason": "",
            "roi_bounds_px": [20, 60, 20, 60],
            "crop_bounds_px": [16, 64, 16, 64],
            "projected_area_px": 900.0,
            "bbox_in_frame_ratio": 0.9,
            "edge_margin_px": 10.0,
        }
        object_reviews = {
            1: referability_module._build_object_review_entry(
                obj_id=1,
                label="chair",
                crop_entry=crop_entry,
            )
        }
        objects_by_id = {1: make_object(1, "chair")}
        topology_quality_by_obj_id: dict[int, dict] = {}
        mesh_mask_quality_by_obj_id: dict[int, dict] = {}

        with (
            patch.object(
                referability_module,
                "_compute_topology_quality_for_object",
                return_value=make_topology_quality(1, "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[make_detection(bbox=(22, 22, 58, 58), score=0.95, category="chair")],
            ),
            patch.object(
                referability_module,
                "_compute_mesh_mask_quality_for_object",
                return_value=make_mesh_quality(1, "fail", reason_codes=["low_iou", "high_under_coverage"]),
            ),
        ):
            failed = referability_module._apply_crop_unique_mesh_quality_review(
                crop_unique_label_object_ids={"chair": 1},
                object_reviews=object_reviews,
                objects_by_id=objects_by_id,
                image_path=Path("image.jpg"),
                image_shape=(120, 120, 3),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                instance_mesh_data_getter=lambda _base: make_instance_mesh_data(obj_id=1),
                topology_quality_by_obj_id=topology_quality_by_obj_id,
                mesh_mask_quality_by_obj_id=mesh_mask_quality_by_obj_id,
                client=object(),
            )

        self.assertEqual(failed, {"chair": "mesh_mask_mismatch"})
        self.assertEqual(topology_quality_by_obj_id[1]["status"], "pass")
        self.assertEqual(mesh_mask_quality_by_obj_id[1]["status"], "fail")
        review = object_reviews[1]
        self.assertEqual(review["topology_status"], "pass")
        self.assertEqual(review["mesh_mask_status"], "fail")
        self.assertEqual(review["mesh_quality_review"]["decision"], "drop")
        self.assertEqual(review["mesh_quality_review"]["reason"], "mesh_mask_mismatch")
        self.assertIsNotNone(review["mesh_quality_review"]["matched_detection"])

    def test_repair_final_referability_fields_recomputes_stale_final_fields(self) -> None:
        stale_entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1],
            "candidate_visibility_source": "mesh_ray_depth_refined",
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
            "referable_object_ids": [1],
        }

        repaired = referability_module._repair_final_referability_fields(stale_entry)

        self.assertEqual(repaired["crop_label_statuses"], {"lamp": "unique"})
        self.assertEqual(repaired["full_frame_label_statuses"], {"lamp": "absent"})
        self.assertEqual(repaired["label_statuses"], {"lamp": "absent"})
        self.assertEqual(repaired["label_counts"], {"lamp": 0})
        self.assertEqual(repaired["attachment_referable_object_ids"], [])
        self.assertEqual(repaired["referable_object_ids"], [])
        self.assertEqual(repaired["vlm_unique_object_ids"], [])

    def test_derive_final_referability_fields_recovers_relaxed_attachment_ids_from_legacy_entry(self) -> None:
        legacy_entry = {
            "label_to_object_ids": {
                "table": [1],
                "cup": [2],
            },
            "crop_label_statuses": {
                "table": "unique",
                "cup": "unique",
            },
            "crop_label_counts": {
                "table": 1,
                "cup": 1,
            },
            "crop_referable_object_ids": [1, 2],
            "full_frame_label_statuses": {
                "table": "unique",
                "cup": "unique",
            },
            "full_frame_label_counts": {
                "table": 1,
                "cup": 1,
            },
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
            "referable_object_ids": [2],
        }

        derived = referability_module._derive_final_referability_fields(legacy_entry)

        self.assertEqual(derived["label_statuses"], {"cup": "unique", "table": "unique"})
        self.assertEqual(derived["referable_object_ids"], [2])
        self.assertEqual(derived["attachment_referable_object_ids"], [1, 2])

    def test_frame_entry_has_debug_fields_rejects_stale_final_field_mismatch(self) -> None:
        stale_entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1],
            "candidate_visibility_source": "mesh_ray_depth_refined",
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
            "referable_object_ids": [1],
        }

        consistent_entry = dict(stale_entry)
        consistent_entry["label_statuses"] = {"lamp": "absent"}
        consistent_entry["label_counts"] = {"lamp": 0}
        consistent_entry["referable_object_ids"] = []
        consistent_entry["vlm_unique_object_ids"] = []

        self.assertFalse(referability_module._frame_entry_has_debug_fields(stale_entry))
        self.assertTrue(referability_module._frame_entry_has_debug_fields(consistent_entry))

    def test_frame_entry_has_debug_fields_accepts_exact_crop_label_counts_above_two(self) -> None:
        entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1, 2, 3, 4],
            "candidate_visibility_source": "mesh_ray_depth_refined",
            "candidate_labels": ["stool"],
            "label_to_object_ids": {"stool": [1, 2, 3, 4]},
            "selector_visible_object_ids": [1, 2, 3, 4],
            "selector_visible_label_counts": {"stool": 4},
            "visibility_audit_by_object_id": {
                str(obj_id): {
                    "obj_id": obj_id,
                    "label": "stool",
                    "candidate_considered": True,
                    "candidate_passed": True,
                    "candidate_rejection_reasons": [],
                    "bbox_in_frame_ratio": 0.9,
                }
                for obj_id in (1, 2, 3, 4)
            },
            "object_reviews": {
                str(obj_id): {
                    "obj_id": obj_id,
                    "label": "stool",
                    "local_outcome": "reviewed",
                    "vlm_status": "clear",
                    "bbox_in_frame_ratio": 0.9,
                }
                for obj_id in (1, 2, 3, 4)
            },
            "crop_label_statuses": {"stool": "multiple"},
            "crop_label_counts": {"stool": 4},
            "crop_referable_object_ids": [],
            "full_frame_label_reviews": [],
            "full_frame_label_statuses": {},
            "full_frame_label_counts": {},
            "label_statuses": {"stool": "multiple"},
            "label_counts": {"stool": 2},
            "referable_object_ids": [],
            "vlm_unique_object_ids": [],
        }

        self.assertTrue(referability_module._frame_entry_has_debug_fields(entry))
        repaired = referability_module._repair_final_referability_fields(entry)
        self.assertEqual(repaired["crop_label_counts"], {"stool": 4})

    def test_frame_entry_has_debug_fields_accepts_selector_duplicate_shortcut_counts(self) -> None:
        entry = {
            "frame_usable": True,
            "frame_quality_clear": True,
            "frame_quality_score": 82,
            "frame_quality_reason": "clear enough",
            "frame_selection_score": 82001,
            "attachment_referable_pairs": [],
            "attachment_referable_pair_count": 0,
            "final_selection_rank": 0,
            "candidate_visible_object_ids": [1],
            "candidate_visibility_source": "mesh_ray_depth_refined",
            "candidate_labels": ["chair"],
            "label_to_object_ids": {"chair": [1]},
            "selector_visible_object_ids": [1, 2],
            "selector_visible_label_counts": {"chair": 2},
            "visibility_audit_by_object_id": {
                "1": {
                    "obj_id": 1,
                    "label": "chair",
                    "candidate_considered": True,
                    "candidate_passed": True,
                    "candidate_rejection_reasons": [],
                    "bbox_in_frame_ratio": 0.9,
                }
            },
            "object_reviews": {
                "1": {
                    "obj_id": 1,
                    "label": "chair",
                    "review_mode": "selector_duplicate_shortcut",
                    "review_skip_reason": "selector_visible_label_multiple",
                    "local_outcome": "reviewed",
                    "vlm_status": None,
                    "bbox_in_frame_ratio": 0.9,
                }
            },
            "crop_label_statuses": {"chair": "multiple"},
            "crop_label_counts": {"chair": 1},
            "crop_referable_object_ids": [],
            "full_frame_label_reviews": [],
            "full_frame_label_statuses": {},
            "full_frame_label_counts": {},
            "label_statuses": {"chair": "multiple"},
            "label_counts": {"chair": 2},
            "referable_object_ids": [],
            "vlm_unique_object_ids": [],
        }

        self.assertTrue(referability_module._frame_entry_has_debug_fields(entry))
        repaired = referability_module._repair_final_referability_fields(entry)
        self.assertEqual(repaired["crop_label_counts"], {"chair": 1})

    def test_compress_attachment_group_frames_keeps_distinct_referable_pairs(self) -> None:
        frames = [
            {
                "image_name": "000003.jpg",
                "crop_ge_70_count": 3,
                "attachment_referable_pairs": [[2, 1]],
                "attachment_referable_pair_count": 1,
            },
            {
                "image_name": "000006.jpg",
                "crop_ge_70_count": 2,
                "attachment_referable_pairs": [[3, 1]],
                "attachment_referable_pair_count": 1,
            },
            {
                "image_name": "000009.jpg",
                "crop_ge_70_count": 1,
                "attachment_referable_pairs": [[2, 1]],
                "attachment_referable_pair_count": 1,
            },
        ]

        kept = referability_module._compress_attachment_group_frames(frames)

        self.assertEqual(
            [frame["image_name"] for frame in kept],
            ["000003.jpg", "000006.jpg"],
        )

    def test_select_attachment_frames_by_global_pair_coverage_prefers_new_pairs(self) -> None:
        frames = [
            {
                "image_name": "000003.jpg",
                "crop_ge_70_count": 3,
                "attachment_referable_pairs": [[2, 1], [3, 1]],
                "attachment_referable_pair_count": 2,
            },
            {
                "image_name": "000006.jpg",
                "crop_ge_70_count": 2,
                "attachment_referable_pairs": [[2, 1]],
                "attachment_referable_pair_count": 1,
            },
            {
                "image_name": "000009.jpg",
                "crop_ge_70_count": 1,
                "attachment_referable_pairs": [[4, 1]],
                "attachment_referable_pair_count": 1,
            },
        ]

        selected = referability_module._select_attachment_frames_by_global_pair_coverage(
            frames,
            max_frames=2,
        )

        self.assertEqual(
            [frame["image_name"] for frame in selected],
            ["000003.jpg", "000009.jpg"],
        )

    def test_select_and_rerank_frames_filters_unusable_frames_then_prefers_selector_score(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5},
            {"image_name": "000030.jpg", "score": 9, "n_visible": 3},
            {"image_name": "000060.jpg", "score": 7, "n_visible": 4},
        ]
        frame_decisions = [
            {
                "clear": False,
                "clarity_score": 99,
                "frame_usable": False,
                "reason": "overall blurry",
            },
            {
                "clear": True,
                "clarity_score": 10,
                "frame_usable": True,
                "reason": "barely clear but acceptable",
            },
            {
                "clear": True,
                "clarity_score": 95,
                "frame_usable": True,
                "reason": "sharp",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ),
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
            )

        self.assertEqual([entry["image_name"] for entry in selected], ["000030.jpg", "000060.jpg"])
        self.assertTrue(all(entry["frame_info"]["frame_usable"] for entry in selected))
        self.assertEqual(selected[0]["frame_selection_score"], 100009)
        self.assertEqual(selected[1]["frame_selection_score"], 100007)

    def test_select_and_rerank_frames_stops_reviewing_group_after_first_high_quality_hit(self) -> None:
        frame_candidates = [
            {"image_name": "000000.jpg", "score": 20, "n_visible": 5},
            {"image_name": "000030.jpg", "score": 19, "n_visible": 4},
            {"image_name": "000060.jpg", "score": 18, "n_visible": 4},
            {"image_name": "000090.jpg", "score": 17, "n_visible": 4},
            {"image_name": "000120.jpg", "score": 16, "n_visible": 3},
            {"image_name": "000150.jpg", "score": 15, "n_visible": 3},
        ]
        frame_decisions = [
            {
                "clear": True,
                "clarity_score": 80,
                "frame_usable": True,
                "reason": "sharp",
            },
            {
                "clear": True,
                "clarity_score": 72,
                "frame_usable": True,
                "reason": "clear enough",
            },
        ]

        root = Path(__file__).resolve().parent / "_tmp" / f"rerank_group_stop_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"

        with (
            patch.object(
                referability_module.cv2,
                "imread",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
            patch.object(
                referability_module,
                "_frame_decision",
                side_effect=frame_decisions,
            ) as frame_decision_mock,
        ):
            selected = referability_module._select_and_rerank_frames(
                client=object(),
                model_name="fake-vlm",
                scene_dir=scene_dir,
                frame_candidates=frame_candidates,
                max_frames=2,
            )

        self.assertEqual(frame_decision_mock.call_count, 2)
        self.assertEqual([entry["image_name"] for entry in selected], ["000000.jpg", "000090.jpg"])
        self.assertEqual([entry["frame_info"]["clarity_score"] for entry in selected], [80, 72])


if __name__ == "__main__":
    unittest.main()
