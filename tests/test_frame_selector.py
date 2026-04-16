import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

import src.frame_selector as frame_selector
from src.scene_parser import InstanceMeshData
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
        "center": [float(obj_id), 0.0, 1.0],
        "bbox_min": [float(obj_id), -0.1, 0.9],
        "bbox_max": [float(obj_id) + 0.2, 0.1, 1.1],
    }


class FrameSelectorTests(unittest.TestCase):
    def test_selector_visibility_audit_accepts_zbuffer_roi_fallback_at_800px_projected_area(self) -> None:
        audit = frame_selector.build_selector_visibility_audit_from_meta(
            {
                "center_uv_px": [10.0, 10.0],
                "depth_m": 2.0,
                "bbox_in_frame_ratio": 0.05,
                "zbuffer_mask_area_px": 400.0,
                "has_zbuffer_mask_area": True,
                "projected_area_px": 800.0,
            },
            make_camera_intrinsics(),
        )

        self.assertTrue(audit["selector_passed"])
        self.assertEqual(audit["selector_decision"], "selected_roi_fallback")
        self.assertEqual(audit["selector_roi_ratio_source"], "zbuffer_mask_area")

    def test_selector_visibility_audit_rejects_zbuffer_roi_fallback_below_400px_mask_area(self) -> None:
        audit = frame_selector.build_selector_visibility_audit_from_meta(
            {
                "center_uv_px": [10.0, 10.0],
                "depth_m": 2.0,
                "bbox_in_frame_ratio": 0.90,
                "zbuffer_mask_area_px": 399.0,
                "has_zbuffer_mask_area": True,
                "projected_area_px": 9999.0,
            },
            make_camera_intrinsics(),
        )

        self.assertFalse(audit["selector_passed"])
        self.assertIn("zbuffer_mask_area_below_threshold", audit["selector_rejection_reasons"])

    def test_selector_visibility_audit_requires_projected_area_at_800px_when_zbuffer_area_exists(self) -> None:
        audit = frame_selector.build_selector_visibility_audit_from_meta(
            {
                "center_uv_px": [10.0, 10.0],
                "depth_m": 2.0,
                "bbox_in_frame_ratio": 0.01,
                "zbuffer_mask_area_px": 400.0,
                "has_zbuffer_mask_area": True,
                "projected_area_px": 799.0,
            },
            make_camera_intrinsics(),
        )

        self.assertFalse(audit["selector_passed"])
        self.assertIn("projected_area_below_threshold", audit["selector_rejection_reasons"])

    def test_selector_visibility_audit_falls_back_to_bbox_projection_without_mesh_mask(self) -> None:
        audit = frame_selector.build_selector_visibility_audit_from_meta(
            {
                "center_uv_px": [10.0, 10.0],
                "depth_m": 2.0,
                "bbox_in_frame_ratio": 0.35,
                "projected_area_px": 800.0,
            },
            make_camera_intrinsics(),
        )

        self.assertTrue(audit["selector_passed"])
        self.assertEqual(audit["selector_decision"], "selected_roi_fallback")
        self.assertEqual(audit["selector_roi_ratio_source"], "bbox_projection")

    def test_selector_visibility_audit_accepts_depth_up_to_8m(self) -> None:
        audit = frame_selector.build_selector_visibility_audit_from_meta(
            {
                "center_uv_px": [320.0, 240.0],
                "depth_m": 7.5,
                "bbox_in_frame_ratio": 0.0,
                "zbuffer_mask_area_px": 0.0,
                "has_zbuffer_mask_area": True,
                "projected_area_px": 0.0,
            },
            make_camera_intrinsics(),
        )

        self.assertTrue(audit["selector_passed"])
        self.assertEqual(audit["selector_decision"], "selected_center")

    def test_build_selector_visibility_meta_skips_mask_projection_below_projected_area_threshold(self) -> None:
        obj = make_object(1, "cup")

        with (
            patch.object(
                frame_selector,
                "_project_object_roi",
                return_value={
                    "bbox_in_frame_ratio": 0.9,
                    "projected_area_px": 799.0,
                    "roi_bounds": (0, 20, 0, 20),
                },
            ),
            patch.object(
                frame_selector,
                "_project_object_mask_stats",
                side_effect=AssertionError("mask stats should not run below projected area threshold"),
            ),
        ):
            meta = frame_selector._build_selector_visibility_meta(
                obj,
                make_camera_pose("000000.jpg"),
                make_camera_intrinsics(),
                instance_mesh_data=object(),
            )

        self.assertEqual(meta["projected_area_px"], 799.0)
        self.assertEqual(meta["zbuffer_mask_area_px"], 0.0)
        self.assertFalse(meta["has_zbuffer_mask_area"])

    def test_build_selector_visibility_meta_skips_mask_projection_for_center_visible_audits(self) -> None:
        obj = make_object(0, "cup")

        with (
            patch.object(
                frame_selector,
                "_project_object_roi",
                return_value={
                    "bbox_in_frame_ratio": 0.9,
                    "projected_area_px": 900.0,
                    "roi_bounds": (0, 20, 0, 20),
                },
            ),
            patch.object(
                frame_selector,
                "_project_object_mask_stats",
                side_effect=AssertionError("mask stats should not run when center is already visible"),
            ),
        ):
            meta = frame_selector._build_selector_visibility_meta(
                obj,
                make_camera_pose("000000.jpg"),
                make_camera_intrinsics(),
                instance_mesh_data=object(),
                include_roi_metrics=True,
            )

        self.assertEqual(meta["bbox_in_frame_ratio"], 0.9)
        self.assertEqual(meta["projected_area_px"], 900.0)
        self.assertEqual(meta["zbuffer_mask_area_px"], 0.0)
        self.assertFalse(meta["has_zbuffer_mask_area"])

    def test_build_selector_visibility_meta_skips_mask_projection_for_roi_fallback_candidates(self) -> None:
        obj = make_object(1, "cup")

        with (
            patch.object(
                frame_selector,
                "project_to_image",
                return_value=(np.array([10.0, 10.0], dtype=np.float64), 2.0),
            ),
            patch.object(
                frame_selector,
                "_project_object_roi",
                return_value={
                    "bbox_in_frame_ratio": 0.9,
                    "projected_area_px": 900.0,
                    "roi_bounds": (0, 20, 0, 20),
                },
            ),
            patch.object(
                frame_selector,
                "_project_object_mask_stats",
                side_effect=AssertionError("selector visibility should not use zbuffer mask stats"),
            ),
        ):
            meta = frame_selector._build_selector_visibility_meta(
                obj,
                make_camera_pose("000000.jpg"),
                make_camera_intrinsics(),
                instance_mesh_data=object(),
            )

        self.assertEqual(meta["bbox_in_frame_ratio"], 0.9)
        self.assertEqual(meta["projected_area_px"], 900.0)
        self.assertEqual(meta["zbuffer_mask_area_px"], 0.0)
        self.assertFalse(meta["has_zbuffer_mask_area"])

    def test_project_object_mask_stats_only_reports_in_frame_area(self) -> None:
        intrinsics = make_camera_intrinsics()
        pose = make_camera_pose("000000.jpg")
        obj = {
            "id": 1,
            "label": "picture",
            "center": [333.0, 333.0, 0.051],
            "bbox_min": [0.0, 0.0, 0.051],
            "bbox_max": [1000.0, 1000.0, 0.051],
        }
        instance_mesh_data = InstanceMeshData(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.051],
                    [1000.0, 0.0, 0.051],
                    [0.0, 1000.0, 0.051],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            triangle_ids_by_instance={1: np.array([0], dtype=np.int64)},
            boundary_triangle_ids_by_instance={},
            surface_points_by_instance={},
        )

        stats = frame_selector._project_object_mask_stats(
            obj,
            pose,
            intrinsics,
            instance_mesh_data,
        )

        self.assertEqual(stats["zbuffer_mask_in_frame_ratio"], 0.0)
        self.assertEqual(stats["zbuffer_full_mask_area_px"], 0.0)
        self.assertGreaterEqual(stats["zbuffer_mask_area_px"], 0.0)

    def test_count_well_cropped_visible_objects_uses_80_percent_threshold(self) -> None:
        visible = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]

        with patch.object(
            frame_selector,
            "_project_object_roi",
            side_effect=[
                {"bbox_in_frame_ratio": 0.80},
                {"bbox_in_frame_ratio": 0.79},
                {"bbox_in_frame_ratio": 0.95},
            ],
        ):
            count = frame_selector._count_well_cropped_visible_objects(
                visible,
                make_camera_pose("000000.jpg"),
                make_camera_intrinsics(),
            )

        self.assertEqual(count, 2)

    def test_count_well_cropped_visible_objects_reuses_precomputed_audits(self) -> None:
        visible = [make_object(1, "cup"), make_object(2, "table")]
        audits = {
            1: {"bbox_in_frame_ratio": 0.80},
            2: {"bbox_in_frame_ratio": 0.79},
        }

        with patch.object(
            frame_selector,
            "_project_object_roi",
            side_effect=AssertionError("should use cached audit ratios"),
        ):
            count = frame_selector._count_well_cropped_visible_objects(
                visible,
                visibility_audits_by_obj_id=audits,
            )

        self.assertEqual(count, 1)

    def test_count_well_cropped_attachment_pairs_uses_50_percent_threshold(self) -> None:
        visible = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]
        audits = {
            1: {"bbox_in_frame_ratio": 0.50},
            2: {"bbox_in_frame_ratio": 0.49},
            3: {"bbox_in_frame_ratio": 0.90},
        }

        count = frame_selector._count_well_cropped_attachment_pairs(
            visible,
            {3: [1], 2: [1]},
            visibility_audits_by_obj_id=audits,
        )

        self.assertEqual(count, 1)

    def test_frame_candidate_score_does_not_use_visible_object_count(self) -> None:
        base_a, score_a = frame_selector._frame_candidate_score(
            n_visible=10,
            n_attachment=2,
            crop_ge_80_count=1,
            attachment_pair_ge_50_count=1,
        )
        base_b, score_b = frame_selector._frame_candidate_score(
            n_visible=3,
            n_attachment=2,
            crop_ge_80_count=1,
            attachment_pair_ge_50_count=1,
        )

        self.assertEqual(base_a, base_b)
        self.assertEqual(score_a, score_b)

    def test_select_frames_adds_crop_bonus_to_score(self) -> None:
        root = make_case_dir("frame_selector")
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / "color").mkdir(parents=True)
        (scene_dir / "intrinsic_color.txt").write_text("stub", encoding="utf-8")
        image_names = [f"{idx:06d}.jpg" for idx in range(6)]
        for image_name in image_names:
            (scene_dir / "color" / image_name).write_bytes(b"jpg")

        objects = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]

        with (
            patch.object(frame_selector, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(frame_selector, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                frame_selector,
                "load_scannet_poses",
                return_value={image_name: make_camera_pose(image_name) for image_name in image_names},
            ),
            patch.object(frame_selector, "get_visible_objects", side_effect=[objects, objects]),
            patch.object(frame_selector, "passes_image_quality", return_value=True),
            patch.object(frame_selector, "_count_attachment_objects", return_value=0),
            patch.object(frame_selector, "_count_well_cropped_visible_objects", side_effect=[0, 2]),
        ):
            results = frame_selector.select_frames(scene_dir, objects, max_frames=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["image_name"], "000005.jpg")
        self.assertEqual(results[0]["base_score"], 0)
        self.assertEqual(results[0]["crop_ge_80_count"], 2)
        self.assertEqual(results[0]["score"], 20)

    def test_select_frames_adds_attachment_pair_bonus_to_score(self) -> None:
        root = make_case_dir("frame_selector_attachment_bonus")
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / "color").mkdir(parents=True)
        (scene_dir / "intrinsic_color.txt").write_text("stub", encoding="utf-8")
        image_names = [f"{idx:06d}.jpg" for idx in range(6)]
        for image_name in image_names:
            (scene_dir / "color" / image_name).write_bytes(b"jpg")

        objects = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]

        with (
            patch.object(frame_selector, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(frame_selector, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                frame_selector,
                "load_scannet_poses",
                return_value={image_name: make_camera_pose(image_name) for image_name in image_names},
            ),
            patch.object(frame_selector, "get_visible_objects", side_effect=[objects, objects]),
            patch.object(frame_selector, "passes_image_quality", return_value=True),
            patch.object(frame_selector, "_count_attachment_objects", return_value=2),
            patch.object(frame_selector, "_count_well_cropped_visible_objects", return_value=0),
            patch.object(frame_selector, "_count_well_cropped_attachment_pairs", side_effect=[0, 1]),
        ):
            results = frame_selector.select_frames(
                scene_dir,
                objects,
                attachment_graph={2: [1]},
                max_frames=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["image_name"], "000005.jpg")
        self.assertEqual(results[0]["base_score"], 2)
        self.assertEqual(results[0]["attachment_pair_ge_50_count"], 1)
        self.assertEqual(results[0]["score"], 17)

    def test_select_frames_prefers_frames_with_well_cropped_objects(self) -> None:
        root = make_case_dir("frame_selector_prefers_ge80")
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / "color").mkdir(parents=True)
        (scene_dir / "intrinsic_color.txt").write_text("stub", encoding="utf-8")
        image_names = [f"{idx:06d}.jpg" for idx in range(6)]
        for image_name in image_names:
            (scene_dir / "color" / image_name).write_bytes(b"jpg")

        objects = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]

        with (
            patch.object(frame_selector, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(frame_selector, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                frame_selector,
                "load_scannet_poses",
                return_value={image_name: make_camera_pose(image_name) for image_name in image_names},
            ),
            patch.object(frame_selector, "get_visible_objects", side_effect=[objects, objects]),
            patch.object(frame_selector, "passes_image_quality", return_value=True),
            patch.object(frame_selector, "_count_attachment_objects", side_effect=[10, 0]),
            patch.object(frame_selector, "_count_well_cropped_visible_objects", side_effect=[0, 1]),
        ):
            results = frame_selector.select_frames(scene_dir, objects, max_frames=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["image_name"], "000005.jpg")
        self.assertEqual(results[0]["base_score"], 0)
        self.assertEqual(results[0]["crop_ge_80_count"], 1)
        self.assertEqual(results[0]["score"], 10)

    def test_select_frames_checks_image_quality_before_visibility(self) -> None:
        root = make_case_dir("frame_selector_quality_first")
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / "color").mkdir(parents=True)
        (scene_dir / "intrinsic_color.txt").write_text("stub", encoding="utf-8")
        image_name = "000000.jpg"
        (scene_dir / "color" / image_name).write_bytes(b"jpg")

        objects = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]

        with (
            patch.object(frame_selector, "load_scannet_intrinsics", return_value=make_camera_intrinsics()),
            patch.object(frame_selector, "load_axis_alignment", return_value=np.eye(4, dtype=np.float64)),
            patch.object(
                frame_selector,
                "load_scannet_poses",
                return_value={image_name: make_camera_pose(image_name)},
            ),
            patch.object(frame_selector, "passes_image_quality", return_value=False),
            patch.object(
                frame_selector,
                "get_visible_objects",
                side_effect=AssertionError("visibility should not run on rejected frames"),
            ),
        ):
            results = frame_selector.select_frames(scene_dir, objects, max_frames=1)

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
