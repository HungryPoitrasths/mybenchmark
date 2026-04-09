import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _default_templates,
    _find_stable_distance_move_for_relation,
    _generate_l2_distance_questions_for_object,
    generate_l1_distance,
    generate_l2_object_move,
)
from src.quality_control import quality_filter
from src.relation_engine import compute_distance, compute_distance_details
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
    def test_compute_distance_uses_aabb_closest_point_bins_and_boundary_margin(self) -> None:
        obj_a = make_object(1, "box", (0.0, 0.0, 0.0))
        obj_b = make_object(2, "chair", (1.15, 0.0, 0.0))
        label, _dist, near_boundary = compute_distance(obj_a, obj_b)
        details = compute_distance_details(obj_a, obj_b)
        self.assertEqual(details["distance_definition"], "aabb_closest_point_approx")
        self.assertEqual(label, "very close (<1.0m)")
        self.assertTrue(near_boundary)

        obj_b = make_object(2, "chair", (2.6, 0.0, 0.0))
        label, _dist, near_boundary = compute_distance(obj_a, obj_b)
        self.assertEqual(label, "moderate (2.0-3.3m)")
        self.assertFalse(near_boundary)

        obj_b = make_object(2, "chair", (3.55, 0.0, 0.0))
        label, _dist, near_boundary = compute_distance(obj_a, obj_b)
        self.assertEqual(label, "far (>3.3m)")
        self.assertTrue(near_boundary)

    def test_generate_l1_distance_uses_closest_points_wording(self) -> None:
        question = generate_l1_distance(
            {
                "obj_a_id": 1,
                "obj_a_label": "chair",
                "obj_b_id": 2,
                "obj_b_label": "table",
                "distance_bin": "close (1.0-2.0m)",
                "distance_bin_id": "close",
                "distance_m": 1.4,
                "near_boundary": False,
                "distance_definition": "surface_sample_min_euclidean",
            },
            _default_templates(),
        )
        self.assertIsNotNone(question)
        assert question is not None
        self.assertIn("closest points", question["question"])

    def test_generate_l1_distance_uses_unrounded_distance_threshold(self) -> None:
        question = generate_l1_distance(
            {
                "obj_a_id": 1,
                "obj_a_label": "chair",
                "obj_b_id": 2,
                "obj_b_label": "table",
                "distance_bin": "very close (<1.0m)",
                "distance_bin_id": "very_close",
                "distance_m": 0.2,
                "distance_m_raw": 0.195,
                "near_boundary": False,
                "distance_definition": "surface_sample_min_euclidean",
            },
            _default_templates(),
        )
        self.assertIsNone(question)

    def test_generate_l1_distance_allows_pairs_at_threshold(self) -> None:
        question = generate_l1_distance(
            {
                "obj_a_id": 1,
                "obj_a_label": "chair",
                "obj_b_id": 2,
                "obj_b_label": "table",
                "distance_bin": "very close (<1.0m)",
                "distance_bin_id": "very_close",
                "distance_m": 0.2,
                "distance_m_raw": 0.2,
                "near_boundary": False,
                "distance_definition": "surface_sample_min_euclidean",
            },
            _default_templates(),
        )
        self.assertIsNotNone(question)

    def test_generate_l1_distance_rejects_touching_pairs(self) -> None:
        question = generate_l1_distance(
            {
                "obj_a_id": 1,
                "obj_a_label": "chair",
                "obj_b_id": 2,
                "obj_b_label": "table",
                "distance_bin": "very close (<1.0m)",
                "distance_bin_id": "very_close",
                "distance_m": 0.0,
                "distance_m_raw": 0.0,
                "near_boundary": False,
                "distance_definition": "surface_sample_min_euclidean",
            },
            _default_templates(),
        )
        self.assertIsNone(question)

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
            make_object(2, "chair", (1.35, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "close (1.0-2.0m)",
            "distance_bin_id": "close",
        }

        delta, old_label, new_label, relation_unchanged = _find_stable_distance_move_for_relation(
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
        self.assertEqual(old_label, "close (1.0-2.0m)")
        self.assertEqual(new_label, "very close (<1.0m)")
        self.assertFalse(relation_unchanged)

    def test_find_stable_distance_move_skips_room_invalid_candidates(self) -> None:
        objects = [
            make_object(1, "box", (1.15, 0.0, 0.0)),
            make_object(2, "chair", (0.0, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "very close (<1.0m)",
            "distance_bin_id": "very_close",
        }

        delta, old_label, new_label, relation_unchanged = _find_stable_distance_move_for_relation(
            objects,
            attachment_graph={},
            target_id=1,
            relation=relation,
            room_bounds={
                "bbox_min": [-1.0, -1.0, -1.0],
                "bbox_max": [1.2, 1.0, 1.0],
            },
        )

        self.assertIsNone(delta)
        self.assertEqual(old_label, "very close (<1.0m)")
        self.assertIsNone(new_label)
        self.assertFalse(relation_unchanged)

    def test_find_stable_distance_move_rejects_initial_pairs_below_threshold(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 0.0)),
            make_object(2, "chair", (0.35, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "very close (<1.0m)",
            "distance_bin_id": "very_close",
            "distance_m": 0.15,
            "distance_m_raw": 0.15,
        }

        delta, old_label, new_label, relation_unchanged = _find_stable_distance_move_for_relation(
            objects,
            attachment_graph={},
            target_id=1,
            relation=relation,
            room_bounds={
                "bbox_min": [-1.0, -1.0, -1.0],
                "bbox_max": [1.0, 1.0, 1.0],
            },
        )

        self.assertIsNone(delta)
        self.assertEqual(old_label, "very close (<1.0m)")
        self.assertIsNone(new_label)
        self.assertFalse(relation_unchanged)

    def test_find_stable_distance_move_unchanged_fallback_skips_too_close_candidates(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 0.0)),
            make_object(2, "chair", (0.5, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "very close (<1.0m)",
            "distance_bin_id": "very_close",
            "distance_m": 0.3,
            "distance_m_raw": 0.3,
        }
        fallback_states = [
            (np.array([0.1, 0.0, 0.0]), objects, {1}),
            (np.array([0.2, 0.0, 0.0]), objects, {1}),
        ]

        with patch("src.qa_generator._iter_distance_move_deltas", return_value=[]), patch(
            "src.qa_generator._iter_valid_object_move_states",
            return_value=fallback_states,
        ), patch(
            "src.qa_generator.compute_distance_details",
            side_effect=[
                {
                    "distance_m": 0.15,
                    "distance_bin": "very close (<1.0m)",
                    "distance_bin_id": "very_close",
                    "near_boundary": False,
                },
                {
                    "distance_m": 0.35,
                    "distance_bin": "very close (<1.0m)",
                    "distance_bin_id": "very_close",
                    "near_boundary": False,
                },
                {
                    "distance_m": 0.15,
                    "distance_bin": "very close (<1.0m)",
                    "distance_bin_id": "very_close",
                    "near_boundary": False,
                },
                {
                    "distance_m": 0.35,
                    "distance_bin": "very close (<1.0m)",
                    "distance_bin_id": "very_close",
                    "near_boundary": False,
                },
            ],
        ):
            delta, old_label, new_label, relation_unchanged = _find_stable_distance_move_for_relation(
                objects,
                attachment_graph={},
                target_id=1,
                relation=relation,
                room_bounds={
                    "bbox_min": [-1.0, -1.0, -1.0],
                    "bbox_max": [1.0, 1.0, 1.0],
                },
                allow_unchanged_fallback=True,
            )

        self.assertIsNotNone(delta)
        assert delta is not None
        self.assertEqual(delta.tolist(), [0.2, 0.0, 0.0])
        self.assertEqual(old_label, "very close (<1.0m)")
        self.assertEqual(new_label, "very close (<1.0m)")
        self.assertTrue(relation_unchanged)

    def test_find_stable_distance_move_can_use_valid_object_move_candidates_after_axis_search_fails(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 0.0)),
            make_object(2, "chair", (2.35, 0.0, 0.0)),
        ]
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "moderate (2.0-3.3m)",
            "distance_bin_id": "moderate",
            "distance_m": 2.15,
            "distance_m_raw": 2.15,
        }
        valid_states = [
            (np.array([0.7, 0.7, 0.0], dtype=np.float64), objects, {1}),
        ]

        with patch("src.qa_generator._iter_distance_move_deltas", return_value=[]), patch(
            "src.qa_generator._iter_valid_object_move_states",
            return_value=valid_states,
        ), patch(
            "src.qa_generator.compute_distance_details",
            return_value={
                "distance_m": 1.8,
                "distance_bin": "close (1.0-2.0m)",
                "distance_bin_id": "close",
                "near_boundary": False,
            },
        ):
            delta, old_label, new_label, relation_unchanged = _find_stable_distance_move_for_relation(
                objects,
                attachment_graph={},
                target_id=1,
                relation=relation,
                room_bounds={
                    "bbox_min": [-3.0, -3.0, -1.0],
                    "bbox_max": [3.0, 3.0, 1.0],
                },
            )

        self.assertIsNotNone(delta)
        assert delta is not None
        self.assertEqual(delta.tolist(), [0.7, 0.7, 0.0])
        self.assertEqual(old_label, "moderate (2.0-3.3m)")
        self.assertEqual(new_label, "close (1.0-2.0m)")
        self.assertFalse(relation_unchanged)

    def test_generate_l2_object_move_uses_distance_specific_search_when_generic_move_is_missing(self) -> None:
        mover = make_object(1, "box", (0.0, 0.0, 0.0))
        ref = make_object(2, "chair", (1.35, 0.0, 0.0))
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

    def test_generate_l2_distance_questions_skip_initial_pairs_below_threshold(self) -> None:
        mover = make_object(1, "box", (0.0, 0.0, 0.0))
        ref = make_object(2, "chair", (0.35, 0.0, 0.0))
        relation = {
            "obj_a_id": 1,
            "obj_b_id": 2,
            "distance_bin": "very close (<1.0m)",
            "distance_bin_id": "very_close",
            "distance_m": 0.15,
            "distance_m_raw": 0.15,
        }

        questions = _generate_l2_distance_questions_for_object(
            query_obj=mover,
            move_source=mover,
            move_source_id=1,
            attachment_remapped=False,
            relations=[relation],
            movement_scene_objects=[mover, ref],
            attachment_graph={},
            camera_pose=make_camera_pose(),
            templates={
                "L2_object_move_distance": [
                    "move {obj_a} {direction_with_camera_hint} by {distance}: distance of {obj_b} and {obj_c}?"
                ]
            },
            obj_map={1: mover, 2: ref},
            room_bounds={
                "bbox_min": [-1.0, -1.0, -1.0],
                "bbox_max": [1.0, 1.0, 1.0],
            },
        )

        self.assertEqual(questions, [])


if __name__ == "__main__":
    unittest.main()
