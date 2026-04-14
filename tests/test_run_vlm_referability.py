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
) -> dict:
    return {
        "roi_bounds_px": [20, 60, 20, 60],
        "projected_area_px": projected_area_px,
        "bbox_in_frame_ratio": bbox_in_frame_ratio,
        "edge_margin_px": 10.0,
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
        self.assertEqual(frame_entry["frame_quality_score"], 82)
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
        self.assertEqual(frame_entry["referability_reason_by_alias_group"]["chair_family"], "derived_from_crop_vlm")
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

        self.assertEqual(frame_entry["crop_label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["full_frame_label_statuses"], {"chair": "multiple"})
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 1})
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


if __name__ == "__main__":
    unittest.main()
