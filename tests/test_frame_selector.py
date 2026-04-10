import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

import src.frame_selector as frame_selector
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
    def test_count_well_cropped_visible_objects_uses_60_percent_threshold(self) -> None:
        visible = [make_object(1, "cup"), make_object(2, "table"), make_object(3, "lamp")]

        with patch.object(
            frame_selector,
            "_project_object_roi",
            side_effect=[
                {"bbox_in_frame_ratio": 0.60},
                {"bbox_in_frame_ratio": 0.59},
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
            1: {"bbox_in_frame_ratio": 0.60},
            2: {"bbox_in_frame_ratio": 0.59},
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

    def test_select_frames_adds_crop_bonus_to_score(self) -> None:
        root = make_case_dir("frame_selector")
        self.addCleanup(shutil.rmtree, root, True)
        scene_dir = root / "scene0000_00"
        (scene_dir / "pose").mkdir(parents=True)
        (scene_dir / "color").mkdir(parents=True)
        (scene_dir / "intrinsic_color.txt").write_text("stub", encoding="utf-8")
        image_names = [f"{idx:06d}.jpg" for idx in range(31)]
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
            patch.object(frame_selector, "get_visible_objects", side_effect=[(objects, {}), (objects, {})]),
            patch.object(frame_selector, "passes_image_quality", return_value=True),
            patch.object(frame_selector, "_count_attachment_objects", return_value=0),
            patch.object(frame_selector, "_count_well_cropped_visible_objects", side_effect=[0, 2]),
        ):
            results = frame_selector.select_frames(scene_dir, objects, max_frames=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["image_name"], "000030.jpg")
        self.assertEqual(results[0]["base_score"], 3)
        self.assertEqual(results[0]["crop_ge_60_count"], 2)
        self.assertEqual(results[0]["score"], 23)

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
