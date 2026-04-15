import unittest

import numpy as np

from src.qa_generator import (
    generate_l1_direction,
    generate_l1_direction_allocentric,
    generate_l1_direction_object_centric,
)
from src.relation_engine import compute_all_relations
from src.utils.colmap_loader import CameraPose


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ]


def make_object(
    obj_id: int,
    label: str,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    *,
    bottom_hull_xy: list[list[float]] | None = None,
) -> dict:
    center = [
        0.5 * (bbox_min[0] + bbox_max[0]),
        0.5 * (bbox_min[1] + bbox_max[1]),
        0.5 * (bbox_min[2] + bbox_max[2]),
    ]
    return {
        "id": obj_id,
        "label": label,
        "center": center,
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "support_geom": {
            "bottom_hull_xy": bottom_hull_xy or _rect(bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]),
            "top_hull_xy": [],
            "top_surface_candidates": [],
        },
    }


def make_floorplan_camera_pose() -> CameraPose:
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    return CameraPose(
        image_name="test.jpg",
        rotation=rotation,
        translation=np.zeros(3, dtype=np.float64),
    )


class L1DirectionOptionTests(unittest.TestCase):
    def test_l1_direction_agent_vertical_answer_excludes_horizontal_cluster(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        bed = make_object(
            1,
            "bed",
            (0.0, 0.0, 0.0),
            (2.0, 4.0, 1.0),
        )
        lamp = make_object(
            2,
            "lamp",
            (0.0, 3.4, 1.7),
            (0.4, 3.8, 2.5),
        )

        relation = compute_all_relations([bed, lamp], camera_pose)[0]

        self.assertEqual(relation["direction_b_rel_a"], "above")
        self.assertEqual(relation["horizontal_direction_b_rel_a"], "front-left")

        question = generate_l1_direction(relation, templates={})
        self.assertIsNotNone(question)
        self.assertEqual(question["correct_value"], "above")
        self.assertNotIn("front", question["options"])
        self.assertNotIn("front-left", question["options"])
        self.assertNotIn("left", question["options"])

    def test_l1_direction_agent_non_overlapping_vertical_pair_falls_back_to_horizontal(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        bed = make_object(
            1,
            "bed",
            (0.0, 0.0, 0.0),
            (2.0, 4.0, 1.0),
        )
        lamp = make_object(
            2,
            "lamp",
            (-0.5, 3.4, 1.7),
            (-0.1, 3.8, 2.5),
        )

        relation = compute_all_relations([bed, lamp], camera_pose)[0]

        self.assertEqual(relation["direction_b_rel_a"], "front-left")
        self.assertEqual(relation["horizontal_direction_b_rel_a"], "front-left")

        self.assertGreaterEqual(relation["ambiguity_score"], 0.0)

    def test_l1_direction_agent_suppresses_attached_non_vertical_pair(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        stand = make_object(
            1,
            "stand",
            (0.0, 0.0, 0.0),
            (0.6, 0.6, 0.8),
        )
        lamp = make_object(
            2,
            "lamp",
            (0.05, 0.65, 0.0),
            (0.25, 0.85, 0.6),
        )

        relation = compute_all_relations([stand, lamp], camera_pose)[0]

        self.assertIn(relation["direction_b_rel_a"], {"front", "front-left", "front-right", "right", "left"})
        question = generate_l1_direction(
            relation,
            templates={},
            obj_a=stand,
            obj_b=lamp,
            attachment_edge_lookup={
                frozenset({1, 2}): {"parent_id": 1, "child_id": 2, "type": "supported_by"},
            },
        )
        self.assertIsNone(question)

    def test_l1_direction_agent_keeps_attached_vertical_pair(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        stand = make_object(
            1,
            "stand",
            (0.0, 0.0, 0.0),
            (0.6, 0.6, 0.8),
        )
        lamp = make_object(
            2,
            "lamp",
            (0.05, 0.05, 1.4),
            (0.25, 0.25, 2.0),
        )

        relation = compute_all_relations([stand, lamp], camera_pose)[0]

        self.assertEqual(relation["direction_b_rel_a"], "above")
        question = generate_l1_direction(
            relation,
            templates={},
            obj_a=stand,
            obj_b=lamp,
            attachment_edge_lookup={
                frozenset({1, 2}): {"parent_id": 1, "child_id": 2, "type": "supported_by"},
            },
        )
        self.assertIsNotNone(question)
        self.assertEqual(question["correct_value"], "above")

    def test_l1_direction_agent_suppresses_horizontal_overlap_without_attachment(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        table = make_object(
            1,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.8),
        )
        chair = make_object(
            2,
            "chair",
            (0.6, 0.8, 0.0),
            (1.2, 1.4, 0.8),
        )

        relation = compute_all_relations([table, chair], camera_pose)[0]

        self.assertNotIn(relation["direction_b_rel_a"], {"above", "below"})
        question = generate_l1_direction(
            relation,
            templates={},
            obj_a=table,
            obj_b=chair,
            attachment_edge_lookup={},
        )
        self.assertIsNone(question)

    def test_l1_direction_object_centric_excludes_adjacent_horizontal_directions(self) -> None:
        objects = [
            make_object(1, "chair", (-0.1, -0.1, 0.0), (0.1, 0.1, 1.0)),
            make_object(2, "lamp", (-0.1, 0.9, 0.0), (0.1, 1.1, 1.0)),
            make_object(3, "table", (-1.1, 0.9, 0.0), (-0.9, 1.1, 1.0)),
        ]

        questions = generate_l1_direction_object_centric(objects, templates={}, max_questions=20)
        question = next(
            q for q in questions
            if q["obj_ref_id"] == 1 and q["obj_face_id"] == 2 and q["obj_target_id"] == 3
        )

        self.assertEqual(question["correct_value"], "front-left")
        self.assertNotIn("front", question["options"])
        self.assertNotIn("left", question["options"])

    def test_l1_direction_object_centric_suppresses_overlapping_horizontal_target(self) -> None:
        objects = [
            make_object(1, "chair", (0.0, 0.0, 0.0), (0.3, 0.3, 1.0)),
            make_object(2, "lamp", (0.0, 1.0, 0.0), (0.3, 1.3, 1.0)),
            make_object(3, "table", (0.15, 0.2, 0.0), (0.55, 0.6, 1.0)),
        ]

        questions = generate_l1_direction_object_centric(
            objects,
            templates={},
            max_questions=20,
            attachment_edge_lookup={},
        )

        self.assertFalse(
            any(
                q["obj_ref_id"] == 1 and q["obj_face_id"] == 2 and q["obj_target_id"] == 3
                for q in questions
            )
        )

    def test_l1_direction_allocentric_excludes_adjacent_cardinal_directions(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        north_obj = make_object(1, "sofa", (-0.1, 0.9, 0.0), (0.1, 1.1, 1.0))
        south_obj = make_object(2, "desk", (-0.1, -0.1, 0.0), (0.1, 0.1, 1.0))

        questions = generate_l1_direction_allocentric(
            [north_obj, south_obj],
            camera_pose,
            templates={},
            max_questions=20,
        )

        self.assertEqual(len(questions), 1)
        question = questions[0]
        self.assertEqual(question["correct_value"], "north")
        self.assertNotIn("northwest", question["options"])
        self.assertNotIn("northeast", question["options"])

    def test_l1_direction_allocentric_suppresses_attached_non_vertical_pair(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        desk = make_object(1, "desk", (0.0, 0.0, 0.0), (0.6, 0.6, 1.0))
        lamp = make_object(2, "lamp", (0.0, 1.0, 0.0), (0.2, 1.2, 0.6))

        questions = generate_l1_direction_allocentric(
            [desk, lamp],
            camera_pose,
            templates={},
            max_questions=20,
            attachment_edge_lookup={
                frozenset({1, 2}): {"parent_id": 1, "child_id": 2, "type": "supported_by"},
            },
        )

        self.assertEqual(questions, [])


if __name__ == "__main__":
    unittest.main()
