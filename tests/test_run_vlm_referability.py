import unittest
from pathlib import Path
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


class RunVlmReferabilityTests(unittest.TestCase):
    def test_frame_prompt_focuses_on_focus_quality_only(self) -> None:
        prompt = referability_module._frame_prompt().lower()

        self.assertIn("focus", prompt)
        self.assertIn("out of focus", prompt)
        self.assertIn("only about image focus quality", prompt)
        self.assertNotIn("candidate label list", prompt)

    def test_object_review_prompt_uses_full_image_plus_crop(self) -> None:
        prompt = referability_module._object_review_prompt("chair").lower()

        self.assertIn("full scene image", prompt)
        self.assertIn("crop", prompt)
        self.assertIn("clear", prompt)
        self.assertIn("absent", prompt)
        self.assertIn("unsure", prompt)
        self.assertNotIn("obj_id", prompt)

    def test_frame_decision_propagates_focus_result(self) -> None:
        with patch.object(
            referability_module,
            "_call_vlm_json",
            return_value=({"frame_usable": False, "reason": "out_of_focus"}, ""),
        ):
            decision = referability_module._frame_decision(
                client=object(),
                model="fake-model",
                image=np.zeros((2, 2, 3), dtype=np.uint8),
            )

        self.assertEqual(
            decision,
            {
                "frame_usable": False,
                "reason": "out_of_focus",
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

    def test_compute_frame_referability_entry_builds_v8_object_reviews(self) -> None:
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
                return_value={"frame_usable": True, "reason": "in_focus"},
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
        self.assertEqual(frame_entry["candidate_visibility_source"], "depth_refined")
        self.assertEqual(frame_entry["label_statuses"], {"chair": "unique", "lamp": "absent"})
        self.assertEqual(frame_entry["label_counts"], {"chair": 1, "lamp": 0})
        self.assertEqual(frame_entry["referable_object_ids"], [1])
        self.assertEqual(frame_entry["object_reviews"]["1"]["vlm_status"], "clear")
        self.assertEqual(frame_entry["object_reviews"]["2"]["local_outcome"], "out_of_frame")
        self.assertEqual(frame_entry["object_reviews"]["3"]["vlm_status"], "absent")
        self.assertEqual(review_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
