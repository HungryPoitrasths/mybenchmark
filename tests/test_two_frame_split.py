import unittest

import numpy as np

from src.qa_generator import find_two_frame_split_v2
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=320, height=240, fx=200.0, fy=200.0, cx=160.0, cy=120.0)


def make_pose(name: str, position_x: float) -> CameraPose:
    # Identity rotation (forward = +z for all poses), translated along world x only.
    # translation = -position for R = I (see CameraPose.position property).
    return CameraPose(
        image_name=name,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.array([-position_x, 0.0, 0.0], dtype=np.float64),
    )


def make_object(obj_id: int, x: float, z: float = 2.0) -> dict:
    return {
        "id": obj_id,
        "label": f"obj{obj_id}",
        "center": [x, 0.0, z],
        "bbox_min": [x - 0.1, -0.1, z - 0.1],
        "bbox_max": [x + 0.1, 0.1, z + 0.1],
    }


class TestFindTwoFrameSplitV2(unittest.TestCase):
    def test_direct_split_no_bridge_needed(self):
        color_intrinsics = make_intrinsics()
        cam_orig = make_pose("orig.jpg", position_x=0.0)
        cam_far = make_pose("far.jpg", position_x=1.0)
        all_poses = {"orig.jpg": cam_orig, "far.jpg": cam_far}

        obj_a = make_object(1, x=0.0)
        obj_b = make_object(2, x=2.0)

        # obj_b is out of frame from cam_orig (u ~ 360, width=320), so cam_orig
        # cannot itself serve as frame_b -- a different real frame must be found.
        result = find_two_frame_split_v2(
            group_a_objects=[obj_a],
            group_b_objects=[obj_b],
            all_poses=all_poses,
            color_intrinsics=color_intrinsics,
            preferred_camera_pose=cam_orig,
        )
        self.assertIsNotNone(result)
        frame_a_name, frame_b_name, chain = result
        self.assertEqual(frame_a_name, "orig.jpg")
        self.assertEqual(frame_b_name, "far.jpg")
        self.assertEqual(chain, [])

    def test_returns_none_when_group_b_never_visible(self):
        color_intrinsics = make_intrinsics()
        cam_orig = make_pose("orig.jpg", position_x=0.0)
        all_poses = {"orig.jpg": cam_orig}

        obj_a = make_object(1, x=0.0)
        obj_b = make_object(2, x=50.0)  # far outside any candidate frame

        result = find_two_frame_split_v2(
            group_a_objects=[obj_a],
            group_b_objects=[obj_b],
            all_poses=all_poses,
            color_intrinsics=color_intrinsics,
            preferred_camera_pose=cam_orig,
        )
        self.assertIsNone(result)

    def test_multi_object_group_requires_every_member_in_frame(self):
        color_intrinsics = make_intrinsics()
        cam_orig = make_pose("orig.jpg", position_x=0.0)
        cam_far = make_pose("far.jpg", position_x=1.0)
        all_poses = {"orig.jpg": cam_orig, "far.jpg": cam_far}

        # group_a has two objects; obj_a2 is only visible from cam_far, not cam_orig,
        # so cam_orig should be rejected as a frame_a candidate despite obj_a1 fitting.
        obj_a1 = make_object(1, x=0.0)
        obj_a2 = make_object(3, x=2.0)
        obj_b = make_object(2, x=1.0)

        result = find_two_frame_split_v2(
            group_a_objects=[obj_a1, obj_a2],
            group_b_objects=[obj_b],
            all_poses=all_poses,
            color_intrinsics=color_intrinsics,
            preferred_camera_pose=cam_orig,
        )
        self.assertIsNotNone(result)
        frame_a_name, frame_b_name, _chain = result
        # cam_orig doesn't see obj_a2, so frame_a can't be "orig.jpg" here even though
        # it was preferred; cam_far sees both obj_a1 and obj_a2 fully.
        self.assertEqual(frame_a_name, "far.jpg")
        self.assertNotEqual(frame_b_name, frame_a_name)


if __name__ == "__main__":
    unittest.main()
