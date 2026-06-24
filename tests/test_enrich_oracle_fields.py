import unittest

import numpy as np

from scripts.enrich_oracle_fields import (
    _frame_kind_for_question,
    _make_allocentric_oracle_prefix,
    _reference_ids_for_object_frame,
    agent_oracle_coords,
    object_centric_coords,
)
from src.utils.colmap_loader import CameraPose


class EnrichOracleFieldsTests(unittest.TestCase):
    def test_agent_oracle_coords_use_forward_right_up_axes(self) -> None:
        pose = CameraPose(
            image_name="test.jpg",
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )

        coords = agent_oracle_coords(np.array([2.0, 3.0, 5.0]), pose)

        np.testing.assert_allclose(coords, [5.0, 2.0, -3.0])

    def test_object_centric_coords_match_relation_engine_right_axis(self) -> None:
        ref = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        face = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        target = np.array([1.0, 1.0, 0.5], dtype=np.float64)

        coords = object_centric_coords(target, ref, face)

        self.assertIsNotNone(coords)
        np.testing.assert_allclose(coords, [1.0, 1.0, 0.5])

    def test_allocentric_oracle_declares_current_benchmark_axes(self) -> None:
        question = {
            "type": "direction_allocentric",
            "camera_cardinal": "north",
            "mentioned_objects": [
                {"obj_id": 1, "label": "chair", "role": "obj_a"},
            ],
        }
        scene_objects = {
            1: {"id": 1, "label": "chair", "center": [2.0, 3.0, 0.5]},
        }

        text = _make_allocentric_oracle_prefix(question, scene_objects)

        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("+X = east; +Y = north; +Z = up", text)
        self.assertIn("chair (obj_a): [X=2.000, Y=3.000, Z=0.500]", text)

    def test_attachment_move_frame_kind_uses_question_text(self) -> None:
        self.assertEqual(
            _frame_kind_for_question(
                {
                    "type": "attachment_move",
                    "question": "from the camera's perspective, what is the position?",
                }
            ),
            "agent",
        )
        self.assertEqual(
            _frame_kind_for_question(
                {
                    "type": "attachment_move",
                    "question": "Imagine you are the sink and initially facing the camera.",
                }
            ),
            "object_centric",
        )
        self.assertEqual(
            _frame_kind_for_question(
                {
                    "type": "attachment_move",
                    "camera_cardinal": "north",
                    "question": "on the floor plan, in which cardinal direction?",
                }
            ),
            "allocentric",
        )

    def test_object_move_object_centric_uses_query_object_as_origin(self) -> None:
        self.assertEqual(
            _reference_ids_for_object_frame(
                {
                    "type": "object_move_object_centric",
                    "query_obj_id": 24,
                    "obj_ref_id": 12,
                }
            ),
            (24, None),
        )


if __name__ == "__main__":
    unittest.main()
