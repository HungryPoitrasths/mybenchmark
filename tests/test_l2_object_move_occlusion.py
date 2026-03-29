import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import generate_l2_object_move
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="test.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=320,
        height=240,
        fx=200.0,
        fy=200.0,
        cx=160.0,
        cy=120.0,
    )


def make_object(
    obj_id: int,
    label: str,
    center: tuple[float, float, float],
) -> dict:
    x, y, z = center
    return {
        "id": obj_id,
        "label": label,
        "center": [x, y, z],
        "bbox_min": [x - 0.1, y - 0.1, z - 0.1],
        "bbox_max": [x + 0.1, y + 0.1, z + 0.1],
    }


class L2ObjectMoveOcclusionTests(unittest.TestCase):
    def test_generate_l2_object_move_includes_occlusion_questions_from_visibility_diff(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 2.0)),
            make_object(2, "chair", (1.0, 0.0, 2.0)),
        ]
        templates = {
            "L2_object_move_agent": [
                "move {obj_a} {direction_with_camera_hint} by {distance}: where is {obj_b} relative to {obj_c}?"
            ],
            "L2_object_move_occlusion": [
                "move {obj_a} {direction_with_camera_hint} by {distance}: visibility of {obj_b} relative to {obj_c}?"
            ],
        }
        changed = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "changes": {
                    "direction_b_rel_a": {"old": "right", "new": "front-right"},
                },
                "old": {},
                "new": {},
            }
        ]
        occlusion_changes = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "changes": {
                    "occlusion_b": {
                        "old": "fully visible",
                        "new": "partially occluded",
                    },
                },
                "old": {},
                "new": {},
            }
        ]

        with (
            patch(
                "src.qa_generator.find_meaningful_movement",
                return_value=(np.array([0.5, 0.0, 0.0], dtype=np.float64), changed),
            ),
            patch(
                "src.qa_generator._find_object_move_occlusion_changes",
                return_value=occlusion_changes,
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={},
                attached_by={},
                camera_pose=make_camera_pose(),
                templates=templates,
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        occlusion_questions = [
            q for q in questions
            if q.get("type") == "object_move_occlusion"
        ]
        self.assertTrue(occlusion_questions)
        self.assertEqual(
            {q["correct_value"] for q in occlusion_questions},
            {"partially occluded"},
        )


if __name__ == "__main__":
    unittest.main()
