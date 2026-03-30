import unittest
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
        "bbox_min": [base, 0.0, 0.5],
        "bbox_max": [base + 0.2, 0.2, 1.5],
    }


class RunVlmReferabilityTests(unittest.TestCase):
    def test_resolve_referable_object_ids_separates_ambiguous_labels(self) -> None:
        referable_ids, ambiguous = referability_module._resolve_referable_object_ids(
            {"lamp": 1, "cup": 1, "chair": 2},
            {"lamp": [8], "cup": [5, 3], "chair": [1, 2]},
        )

        self.assertEqual(referable_ids, [8])
        self.assertEqual(ambiguous, {"cup": [5, 3]})

    def test_disambiguate_by_depth_selects_clear_winner(self) -> None:
        objects_by_id = {
            1: make_object(1, "cup"),
            2: make_object(2, "cup"),
        }
        with patch.object(
            referability_module,
            "compute_depth_occlusion",
            side_effect=[("partially occluded", 0.58), ("partially occluded", 0.29)],
        ):
            best_id, meta = referability_module._disambiguate_by_depth(
                obj_ids=[1, 2],
                objects_by_id=objects_by_id,
                camera_pose=make_camera_pose(),
                depth_image=np.ones((4, 4), dtype=np.float32),
                depth_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual(best_id, 1)
        self.assertEqual(meta["decision"], "selected")
        self.assertEqual(meta["selected_object_id"], 1)
        self.assertEqual(
            meta["candidate_scores"],
            [
                {"object_id": 1, "visible_ratio": 0.58},
                {"object_id": 2, "visible_ratio": 0.29},
            ],
        )

    def test_disambiguate_by_depth_rejects_low_visible_ratio(self) -> None:
        objects_by_id = {
            1: make_object(1, "cup"),
            2: make_object(2, "cup"),
        }
        with patch.object(
            referability_module,
            "compute_depth_occlusion",
            side_effect=[("not visible", 0.19), ("not visible", 0.02)],
        ):
            best_id, meta = referability_module._disambiguate_by_depth(
                obj_ids=[1, 2],
                objects_by_id=objects_by_id,
                camera_pose=make_camera_pose(),
                depth_image=np.ones((4, 4), dtype=np.float32),
                depth_intrinsics=make_camera_intrinsics(),
            )

        self.assertIsNone(best_id)
        self.assertEqual(meta["decision"], "winner_below_min_ratio")
        self.assertIsNone(meta["selected_object_id"])

    def test_disambiguate_by_depth_rejects_small_gap(self) -> None:
        objects_by_id = {
            1: make_object(1, "cup"),
            2: make_object(2, "cup"),
        }
        with patch.object(
            referability_module,
            "compute_depth_occlusion",
            side_effect=[("partially occluded", 0.41), ("partially occluded", 0.33)],
        ):
            best_id, meta = referability_module._disambiguate_by_depth(
                obj_ids=[1, 2],
                objects_by_id=objects_by_id,
                camera_pose=make_camera_pose(),
                depth_image=np.ones((4, 4), dtype=np.float32),
                depth_intrinsics=make_camera_intrinsics(),
            )

        self.assertIsNone(best_id)
        self.assertEqual(meta["decision"], "gap_too_small")
        self.assertIsNone(meta["selected_object_id"])

    def test_augment_with_depth_disambiguation_reports_missing_depth(self) -> None:
        extra_ids, meta = referability_module._augment_with_depth_disambiguation(
            ambiguous_labels_to_ids={"cup": [1, 2]},
            objects_by_id={1: make_object(1, "cup"), 2: make_object(2, "cup")},
            camera_pose=make_camera_pose(),
            depth_image=None,
            depth_intrinsics=None,
        )

        self.assertEqual(extra_ids, [])
        self.assertEqual(
            meta,
            {
                "cup": {
                    "decision": "missing_depth",
                    "selected_object_id": None,
                    "candidate_scores": [],
                }
            },
        )

    def test_augment_with_depth_disambiguation_collects_extra_ids_and_meta(self) -> None:
        def fake_disambiguate(**kwargs):
            label_ids = tuple(kwargs["obj_ids"])
            if label_ids == (1, 2):
                return 2, {
                    "decision": "selected",
                    "selected_object_id": 2,
                    "candidate_scores": [
                        {"object_id": 2, "visible_ratio": 0.47},
                        {"object_id": 1, "visible_ratio": 0.20},
                    ],
                }
            return None, {
                "decision": "gap_too_small",
                "selected_object_id": None,
                "candidate_scores": [
                    {"object_id": 3, "visible_ratio": 0.42},
                    {"object_id": 4, "visible_ratio": 0.35},
                ],
            }

        with patch.object(referability_module, "_disambiguate_by_depth", side_effect=fake_disambiguate):
            extra_ids, meta = referability_module._augment_with_depth_disambiguation(
                ambiguous_labels_to_ids={"cup": [1, 2], "lamp": [3, 4]},
                objects_by_id={
                    1: make_object(1, "cup"),
                    2: make_object(2, "cup"),
                    3: make_object(3, "lamp"),
                    4: make_object(4, "lamp"),
                },
                camera_pose=make_camera_pose(),
                depth_image=np.ones((4, 4), dtype=np.float32),
                depth_intrinsics=make_camera_intrinsics(),
            )

        self.assertEqual(extra_ids, [2])
        self.assertEqual(meta["cup"]["decision"], "selected")
        self.assertEqual(meta["cup"]["selected_object_id"], 2)
        self.assertEqual(meta["lamp"]["decision"], "gap_too_small")
        self.assertIsNone(meta["lamp"]["selected_object_id"])


if __name__ == "__main__":
    unittest.main()
