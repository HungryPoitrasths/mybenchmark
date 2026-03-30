import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _find_stable_distance_move_for_relation,
    generate_l2_object_move,
)
from src.quality_control import quality_filter
from src.relation_engine import compute_distance
from src.utils.colmap_loader import CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="test.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
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


class DistanceRelationTests(unittest.TestCase):
    def test_compute_distance_uses_new_bins_and_boundary_margin(self) -> None:
        label, _dist, near_boundary = compute_distance(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.95, 0.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(label, "very close (<1.0m)")
        self.assertTrue(near_boundary)

        label, _dist, near_boundary = compute_distance(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([2.4, 0.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(label, "moderate (2.0-3.3m)")
        self.assertFalse(near_boundary)

        label, _dist, near_boundary = compute_distance(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([3.35, 0.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(label, "far (>3.3m)")
        self.assertTrue(near_boundary)

    def test_quality_filter_keeps_distance_questions_flagged_near_boundary(self) -> None:
        questions = [
            {
                "type": "distance",
                "question": "How far apart are the chair and the table?",
                "scene_id": "scene",
                "image_name": "frame.jpg",
                "near_boundary": True,
            }
        ]
        filtered = quality_filter(questions)
        self.assertEqual(len(filtered), 1)
        self.assertTrue(filtered[0]["near_boundary"])


class DistanceMovementSearchTests(unittest.TestCase):
    def test_find_stable_distance_move_picks_smallest_non_boundary_crossing(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 0.0)),
            make_object(2, "chair", (1.15, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "close (1.0-2.0m)",
        }

        delta, new_label = _find_stable_distance_move_for_relation(
            objects,
            attachment_graph={},
            target_id=1,
            relation=relation,
            room_bounds={
                "bbox_min": [-2.0, -1.0, -1.0],
                "bbox_max": [2.0, 1.0, 1.0],
            },
        )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.tolist(), [0.3, 0.0, 0.0])
        self.assertEqual(new_label, "very close (<1.0m)")

    def test_find_stable_distance_move_skips_room_invalid_candidates(self) -> None:
        objects = [
            make_object(1, "box", (0.85, 0.0, 0.0)),
            make_object(2, "chair", (0.0, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "very close (<1.0m)",
        }

        delta, new_label = _find_stable_distance_move_for_relation(
            objects,
            attachment_graph={},
            target_id=1,
            relation=relation,
            room_bounds={
                "bbox_min": [-1.0, -1.0, -1.0],
                "bbox_max": [1.2, 1.0, 1.0],
            },
        )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.tolist(), [0.0, 0.7, 0.0])
        self.assertEqual(new_label, "close (1.0-2.0m)")

    def test_generate_l2_object_move_uses_distance_specific_search_when_generic_move_is_missing(self) -> None:
        mover = make_object(1, "box", (0.0, 0.0, 0.0))
        ref = make_object(2, "chair", (1.15, 0.0, 0.0))
        templates = {
            "L2_object_move_distance": [
                "move {obj_a} {direction_with_camera_hint} by {distance}: distance of {obj_b} and {obj_c}?"
            ]
        }

        with patch(
            "src.qa_generator._find_object_move_delta_and_changes",
            return_value=(None, []),
        ):
            questions = generate_l2_object_move(
                objects=[mover],
                attachment_graph={},
                attached_by={},
                camera_pose=make_camera_pose(),
                templates=templates,
                movement_objects=[mover, ref],
                object_map={1: mover, 2: ref},
                room_bounds={
                    "bbox_min": [-2.0, -1.0, -1.0],
                    "bbox_max": [2.0, 1.0, 1.0],
                },
            )

        distance_questions = [q for q in questions if q.get("type") == "object_move_distance"]
        self.assertEqual(len(distance_questions), 1)
        self.assertEqual(distance_questions[0]["correct_value"], "very close (<1.0m)")
        self.assertEqual(distance_questions[0]["delta"], [0.3, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
