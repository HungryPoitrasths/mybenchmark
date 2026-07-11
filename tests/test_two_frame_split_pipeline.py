import unittest

import numpy as np

import scripts.run_pipeline as run_pipeline_module
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(width=320, height=240, fx=200.0, fy=200.0, cx=160.0, cy=120.0)


def make_pose(name: str, position_x: float) -> CameraPose:
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


class TestApplyTwoFrameSplit(unittest.TestCase):
    def setUp(self):
        self.color_intrinsics = make_intrinsics()
        self.cam_orig = make_pose("orig.jpg", position_x=0.0)
        self.cam_far = make_pose("far.jpg", position_x=1.0)
        self.all_poses = {"orig.jpg": self.cam_orig, "far.jpg": self.cam_far}
        # obj 1/2 near x=0 (visible from orig), obj 3 near x=2 (only visible from far)
        self.objects_by_id = {
            1: make_object(1, x=0.0),
            2: make_object(2, x=0.1),
            3: make_object(3, x=2.0),
        }

    def _apply(self, q: dict) -> dict:
        q = dict(q)
        run_pipeline_module._apply_two_frame_split(
            q,
            objects_by_id=self.objects_by_id,
            all_poses=self.all_poses,
            color_intrinsics=self.color_intrinsics,
            camera_pose=self.cam_orig,
        )
        return q

    def test_object_move_agent_splits_move_pair_from_ref(self):
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result = self._apply(q)
        self.assertEqual(result["image_name"], "orig.jpg")
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")
        self.assertEqual(result["auxiliary_image_names"], [])
        self.assertEqual(result["object_frame_groups"], {"frame_1": [1], "frame_2": [3]})

    def test_object_move_agent_prepends_multi_frame_note_to_question_text(self):
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "question": "What is the relative position of obj1 to obj3?",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 3,
        }
        result = self._apply(q)
        self.assertTrue(result["question"].endswith(q["question"]))
        self.assertTrue(result["question"].startswith("A series of photos is provided"))
        self.assertIn("obj1", result["question"])
        self.assertIn("obj3", result["question"])

    def test_coordinate_rotation_agent_also_gets_the_note(self):
        q = {
            "type": "coordinate_rotation_agent",
            "image_name": "orig.jpg",
            "question": "In which direction is obj1 relative to obj3?",
            "obj_a_id": 1,
            "obj_b_id": 3,
        }
        result = self._apply(q)
        # Its own template only names the rotation-pivot camera, not which object
        # ends up in which frame, so it needs the note just like every other type.
        self.assertTrue(result["question"].startswith("A series of photos is provided"))
        self.assertTrue(result["question"].endswith(q["question"]))
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")

    def test_attachment_move_pairs_root_and_query_against_ref(self):
        q = {
            "type": "attachment_move",
            "image_name": "orig.jpg",
            "root_id": 1,
            "query_obj_id": 2,
            "obj_ref_id": 3,
        }
        result = self._apply(q)
        self.assertEqual(result["reasoning_frame_2"], "far.jpg")
        self.assertEqual(sorted(result["object_frame_groups"]["frame_1"]), [1, 2])
        self.assertEqual(result["object_frame_groups"]["frame_2"], [3])

    def test_object_remove_is_excluded_and_stays_single_frame(self):
        q = {
            "type": "object_remove",
            "image_name": "orig.jpg",
            "removed_obj_id": 1,
            "obj_b_id": 3,
        }
        result = self._apply(q)
        # object_remove was decided to stay single-frame like attachment_chain,
        # even though removed_obj/obj_b are both real objects that could split.
        self.assertEqual(result, q)
        self.assertNotIn("reasoning_frame_2", result)

    def test_unmapped_type_is_a_no_op(self):
        q = {"type": "L1_occlusion", "image_name": "orig.jpg"}
        result = self._apply(q)
        self.assertEqual(result, {"type": "L1_occlusion", "image_name": "orig.jpg"})
        self.assertNotIn("reasoning_frame_2", result)

    def test_no_valid_split_leaves_question_single_frame(self):
        q = {
            "type": "object_move_agent",
            "image_name": "orig.jpg",
            "moved_obj_id": 1,
            "query_obj_id": 1,
            "obj_c_id": 999,  # unresolvable id
        }
        result = self._apply(q)
        self.assertEqual(result["image_name"], "orig.jpg")
        self.assertNotIn("reasoning_frame_2", result)
        self.assertNotIn("object_frame_groups", result)


if __name__ == "__main__":
    unittest.main()
