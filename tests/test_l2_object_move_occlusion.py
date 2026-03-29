import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _find_object_move_occlusion_changes,
    generate_l2_object_move,
)
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
    def test_find_object_move_occlusion_changes_updates_old_and_new_relation_payloads(self) -> None:
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "direction_b_rel_a": "right",
            "distance_bin": "close (1.5-3m)",
            "occlusion_a": "unknown",
            "occlusion_b": "unknown",
        }
        with (
            patch(
                "src.qa_generator._counterfactual_occlusion_backend",
                return_value="mesh_ray",
            ),
            patch(
                "src.qa_generator._compute_visibility_status_per_object",
                return_value={
                    1: ("fully visible", 1.0),
                    2: ("fully visible", 1.0),
                },
            ),
            patch(
                "src.qa_generator._compute_movement_visibility_status_per_object",
                return_value={
                    1: ("partially occluded", 0.5),
                    2: ("not visible", 0.0),
                },
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[relation],
            ),
        ):
            changes = _find_object_move_occlusion_changes(
                original_objects=[make_object(1, "box", (0.0, 0.0, 2.0)), make_object(2, "chair", (1.0, 0.0, 2.0))],
                moved_objects=[make_object(1, "box", (0.5, 0.0, 2.0)), make_object(2, "chair", (1.0, 0.0, 2.0))],
                moved_ids={1},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["old"]["occlusion_a"], "fully visible")
        self.assertEqual(changes[0]["old"]["occlusion_b"], "fully visible")
        self.assertEqual(changes[0]["new"]["occlusion_a"], "partially occluded")
        self.assertEqual(changes[0]["new"]["occlusion_b"], "not visible")

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

    def test_generate_l2_object_move_can_use_occlusion_only_delta_when_relation_delta_is_missing(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 2.0)),
            make_object(2, "chair", (1.0, 0.0, 2.0)),
        ]
        templates = {
            "L2_object_move_occlusion": [
                "move {obj_a} {direction_with_camera_hint} by {distance}: visibility of {obj_b} relative to {obj_c}?"
            ],
        }
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
                return_value=(None, []),
            ),
            patch(
                "src.qa_generator._iter_valid_object_move_states",
                return_value=[
                    (
                        np.array([0.5, 0.0, 0.0], dtype=np.float64),
                        objects,
                        {1},
                    )
                ],
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

        self.assertTrue(any(q.get("type") == "object_move_occlusion" for q in questions))

    def test_generate_l2_object_move_does_not_reuse_occlusion_changes_across_different_deltas(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 2.0)),
            make_object(2, "cup", (0.2, 0.0, 2.0)),
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
        helper_side_effect = [
            (
                np.array([0.5, 0.0, 0.0], dtype=np.float64),
                changed,
            ),
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                changed,
            ),
        ]

        with patch(
            "src.qa_generator._find_object_move_delta_and_changes",
            side_effect=helper_side_effect,
        ) as mocked_helper:
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates=templates,
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(mocked_helper.call_count, 2)
        self.assertEqual(
            [tuple(q["delta"]) for q in questions],
            [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )
        self.assertTrue(all(args.args[2] == 1 for args in mocked_helper.call_args_list))


if __name__ == "__main__":
    unittest.main()
