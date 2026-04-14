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

    def test_alias_group_backfills_label_statuses_and_object_ids(self) -> None:
        scene_objects = [
            make_object(
                12,
                "night stand",
                alias_group="bedside_table_family",
                alias_variants=["night stand", "nightstand", "bedside table"],
            ),
            make_object(
                34,
                "bedside table",
                alias_group="bedside_table_family",
                alias_variants=["night stand", "nightstand", "bedside table"],
            ),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {12: make_visibility_meta(), 34: make_visibility_meta()}

        with (
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([12, 34], "depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_compute_topology_quality_for_object",
                side_effect=lambda **kwargs: make_topology_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[
                    make_detection(bbox=(5, 5, 35, 35), score=0.95, category="night stand"),
                    make_detection(bbox=(60, 5, 90, 35), score=0.92, category="bedside table"),
                ],
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="unused",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("frame.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[12, 34],
                frame_info={"clarity_score": 80, "frame_usable": True, "reason": "ok"},
                frame_selection_score=1,
            )

        self.assertEqual(
            frame_entry["label_statuses"],
            {
                "bedside table": "multiple",
                "night stand": "multiple",
            },
        )
        self.assertEqual(frame_entry["label_to_object_ids"]["night stand"], [12, 34])
        self.assertEqual(frame_entry["label_to_object_ids"]["bedside table"], [12, 34])
        self.assertEqual(
            frame_entry["referability_reason_by_alias_group"]["bedside_table_family"],
            "multiple_visible_instances",
        )

    def test_stage1_only_filters_extreme_noise(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {1: scene_objects[0]}
        visibility = {1: make_visibility_meta()}

        with (
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
                "_compute_topology_quality_for_object",
                side_effect=lambda **kwargs: make_topology_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[
                    make_detection(bbox=(5, 5, 35, 35), score=0.95, category="chair"),
                    make_detection(bbox=(50, 5, 62, 15), score=0.20, category="chair"),
                ],
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="unused",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("frame.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
                frame_info={"clarity_score": 80, "frame_usable": True, "reason": "ok"},
                frame_selection_score=1,
            )

        self.assertEqual(frame_entry["label_statuses"], {"chair": "multiple"})
        self.assertEqual(
            frame_entry["referability_reason_by_alias_group"]["chair_family"],
            "multiple_visible_instances",
        )

    def test_no_3d_anchor_found_is_reported_separately(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {1: scene_objects[0]}
        visibility = {1: make_visibility_meta()}

        with (
            patch.object(
                referability_module,
                "_refine_candidate_visible_object_ids",
                return_value=([], "depth_refined"),
            ),
            patch.object(
                referability_module,
                "compute_frame_object_visibility",
                return_value=visibility,
            ),
            patch.object(
                referability_module,
                "_compute_topology_quality_for_object",
                side_effect=lambda **kwargs: make_topology_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[make_detection(bbox=(5, 5, 35, 35), score=0.95, category="chair")],
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="unused",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("frame.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
                frame_info={"clarity_score": 80, "frame_usable": True, "reason": "ok"},
                frame_selection_score=1,
            )

        self.assertEqual(frame_entry["label_statuses"], {"chair": "unsure"})
        self.assertEqual(
            frame_entry["referability_reason_by_alias_group"]["chair_family"],
            "no_3d_anchor_found",
        )

    def test_multiple_3d_anchors_found_is_reported_separately(self) -> None:
        scene_objects = [
            make_object(1, "chair"),
            make_object(2, "chair"),
        ]
        objects_by_id = {int(obj["id"]): obj for obj in scene_objects}
        visibility = {1: make_visibility_meta(), 2: make_visibility_meta()}

        with (
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
                "_compute_topology_quality_for_object",
                side_effect=lambda **kwargs: make_topology_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_compute_mesh_mask_quality_for_object",
                side_effect=lambda **kwargs: make_mesh_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[make_detection(bbox=(5, 5, 35, 35), score=0.95, category="chair")],
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="unused",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("frame.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1, 2],
                frame_info={"clarity_score": 80, "frame_usable": True, "reason": "ok"},
                frame_selection_score=1,
            )

        self.assertEqual(frame_entry["label_statuses"], {"chair": "multiple"})
        self.assertEqual(
            frame_entry["referability_reason_by_alias_group"]["chair_family"],
            "multiple_3d_anchors_found",
        )

    def test_mesh_quality_failure_blocks_unique_referability(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {1: scene_objects[0]}
        visibility = {1: make_visibility_meta()}

        with (
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
                "_compute_topology_quality_for_object",
                side_effect=lambda **kwargs: make_topology_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_compute_mesh_mask_quality_for_object",
                side_effect=lambda **kwargs: make_mesh_quality(
                    kwargs["obj_id"],
                    "fail",
                    reason_codes=["low_iou", "high_under_coverage"],
                ),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[make_detection(bbox=(5, 5, 35, 35), score=0.95, category="chair")],
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="unused",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("frame.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
                frame_info={"clarity_score": 80, "frame_usable": True, "reason": "ok"},
                frame_selection_score=1,
            )

        self.assertEqual(frame_entry["label_statuses"], {"chair": "unsure"})
        self.assertEqual(
            frame_entry["referability_reason_by_alias_group"]["chair_family"],
            "mesh_mask_mismatch",
        )
        self.assertEqual(
            frame_entry["mesh_mask_quality_by_obj_id"]["1"]["reason_codes"],
            ["low_iou", "high_under_coverage"],
        )

    def test_unique_detection_with_single_good_anchor_is_referable(self) -> None:
        scene_objects = [make_object(1, "chair")]
        objects_by_id = {1: scene_objects[0]}
        visibility = {1: make_visibility_meta()}

        with (
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
                "_compute_topology_quality_for_object",
                side_effect=lambda **kwargs: make_topology_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_compute_mesh_mask_quality_for_object",
                side_effect=lambda **kwargs: make_mesh_quality(kwargs["obj_id"], "pass"),
            ),
            patch.object(
                referability_module,
                "_call_dinox_joint_detection",
                return_value=[make_detection(bbox=(5, 5, 35, 35), score=0.95, category="chair")],
            ),
        ):
            frame_entry = referability_module._compute_frame_referability_entry(
                client=object(),
                model_name="unused",
                scene_objects=scene_objects,
                objects_by_id=objects_by_id,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
                image_path=Path("frame.jpg"),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                depth_image=None,
                depth_intrinsics=None,
                selector_visible_object_ids=[1],
                frame_info={"clarity_score": 80, "frame_usable": True, "reason": "ok"},
                frame_selection_score=1,
            )

        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique"})
        self.assertEqual(frame_entry["referable_object_ids"], [1])
        self.assertEqual(
            frame_entry["referability_reason_by_alias_group"]["chair_family"],
            "referable",
        )


if __name__ == "__main__":
    unittest.main()
