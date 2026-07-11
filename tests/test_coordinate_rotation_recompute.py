import unittest

import numpy as np

import scripts.run_pipeline as run_pipeline_module
from src.qa_generator import (
    _invert_direction,
    recompute_coordinate_rotation_agent_answer,
)
from src.relation_engine import compute_pairwise_direction
from src.utils.colmap_loader import CameraPose
from src.virtual_ops import apply_coordinate_rotation


def make_object(obj_id: int, x: float, y: float, z: float = 0.0) -> dict:
    return {
        "id": obj_id,
        "label": f"obj{obj_id}",
        "center": [x, y, z],
        "bbox_min": [x - 0.1, y - 0.1, z - 0.1],
        "bbox_max": [x + 0.1, y + 0.1, z + 0.1],
    }


def make_pose(name: str, rotation: np.ndarray, translation: np.ndarray) -> CameraPose:
    return CameraPose(image_name=name, rotation=rotation, translation=translation)


class TestRecomputeCoordinateRotationAgentAnswer(unittest.TestCase):
    def setUp(self):
        self.obj_a = make_object(1, x=1.0, y=0.0)
        self.obj_b = make_object(2, x=-1.0, y=0.0)
        self.identity_camera = make_pose(
            "cam_identity.jpg", np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        )

    def test_matches_room_center_pivot_when_observer_camera_is_the_same(self):
        """Pivot choice must not change the answer when the observer camera is identical."""
        angle = 90.0
        room_center = np.mean(
            [np.array(self.obj_a["center"]), np.array(self.obj_b["center"])], axis=0
        )
        rotated_room = apply_coordinate_rotation(
            [self.obj_a, self.obj_b], -angle, rotation_center=room_center
        )
        dir_label, _amb = compute_pairwise_direction(
            rotated_room[0], rotated_room[1], self.identity_camera
        )
        expected_new_dir = _invert_direction(dir_label)

        new_dir, options, answer_letter = recompute_coordinate_rotation_agent_answer(
            self.obj_a, self.obj_b, angle, self.identity_camera
        )

        self.assertEqual(new_dir, expected_new_dir)
        self.assertEqual(options[ord(answer_letter) - ord("A")], new_dir)

    def test_different_observer_orientation_can_change_the_answer(self):
        """A 90-degree-rotated observer camera bins directions differently."""
        angle = 180.0
        # cam_b faces world +x instead of +z (a genuine orientation change, not just
        # a translated position -- translation cancels out of the relative-direction
        # computation, so only rotation differences can move the answer).
        cam_rotated = make_pose(
            "cam_rotated.jpg",
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

        new_dir_identity, _opts1, _ans1 = recompute_coordinate_rotation_agent_answer(
            self.obj_a, self.obj_b, angle, self.identity_camera
        )
        new_dir_rotated, _opts2, _ans2 = recompute_coordinate_rotation_agent_answer(
            self.obj_a, self.obj_b, angle, cam_rotated
        )
        self.assertNotEqual(new_dir_identity, new_dir_rotated)


class TestRecomputeCoordinateRotationGtForNewAnchor(unittest.TestCase):
    def setUp(self):
        self.objects_by_id = {
            1: make_object(1, x=1.0, y=0.0),
            2: make_object(2, x=-1.0, y=0.0),
        }
        self.cam_rotated = make_pose(
            "rotated.jpg",
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        self.all_poses = {"rotated.jpg": self.cam_rotated}

    def test_agent_type_updates_answer_options_and_correct_value(self):
        q = {
            "type": "coordinate_rotation_agent",
            "image_name": "rotated.jpg",
            "obj_a_id": 1,
            "obj_b_id": 2,
            "rotation_angle": 90,
            "old_direction": "left",
        }
        run_pipeline_module._recompute_coordinate_rotation_gt_for_new_anchor(
            q, objects_by_id=self.objects_by_id, all_poses=self.all_poses
        )
        self.assertIn("correct_value", q)
        self.assertEqual(q["new_direction"], q["correct_value"])
        self.assertEqual(q["options"][ord(q["answer"]) - ord("A")], q["correct_value"])
        self.assertEqual(q["relation_unchanged"], (q["old_direction"] == q["correct_value"]))

    def test_allocentric_type_updates_camera_cardinal_and_question_text(self):
        q = {
            "type": "coordinate_rotation_allocentric",
            "image_name": "rotated.jpg",
            "camera_cardinal": "north",
            "question": "The camera, facing north, remains in place. Where is obj1?",
        }
        run_pipeline_module._recompute_coordinate_rotation_gt_for_new_anchor(
            q, objects_by_id=self.objects_by_id, all_poses=self.all_poses
        )
        self.assertNotEqual(q["camera_cardinal"], "north")
        self.assertIn(q["camera_cardinal"], q["question"])
        self.assertNotIn("facing north", q["question"])

    def test_unknown_frame_is_a_no_op(self):
        q = {
            "type": "coordinate_rotation_agent",
            "image_name": "missing.jpg",
            "obj_a_id": 1,
            "obj_b_id": 2,
            "rotation_angle": 90,
        }
        before = dict(q)
        run_pipeline_module._recompute_coordinate_rotation_gt_for_new_anchor(
            q, objects_by_id=self.objects_by_id, all_poses=self.all_poses
        )
        self.assertEqual(q, before)


if __name__ == "__main__":
    unittest.main()
