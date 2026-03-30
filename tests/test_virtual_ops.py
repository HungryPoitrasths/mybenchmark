import unittest

import numpy as np

from src.utils.colmap_loader import CameraPose
from src.virtual_ops import (
    find_meaningful_movement,
    find_meaningful_orbit_rotation,
    has_terminal_bbox_collision,
    is_within_room,
)


def make_object(
    obj_id: int,
    center: tuple[float, float, float],
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    label: str = "object",
) -> dict:
    return {
        "id": obj_id,
        "label": label,
        "center": list(center),
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
    }


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="test.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


class VirtualOpsRoomAndCollisionTests(unittest.TestCase):
    def test_is_within_room_rejects_bbox_overrun_even_when_center_inside(self) -> None:
        room_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        room_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        obj = make_object(
            1,
            center=(0.95, 0.5, 0.5),
            bbox_min=(0.75, 0.25, 0.25),
            bbox_max=(1.15, 0.75, 0.75),
        )

        self.assertFalse(is_within_room([obj], room_min, room_max))

    def test_is_within_room_allows_touching_boundary(self) -> None:
        room_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        room_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        obj = make_object(
            1,
            center=(0.8, 0.5, 0.5),
            bbox_min=(0.6, 0.25, 0.25),
            bbox_max=(1.0, 0.75, 0.75),
        )

        self.assertTrue(is_within_room([obj], room_min, room_max))

    def test_has_terminal_bbox_collision_uses_explicit_collision_set(self) -> None:
        original_objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.25, -0.1, -0.1), (0.25, 0.1, 0.1)),
        ]
        moved_objects = [
            make_object(1, (0.5, 0.0, 0.0), (0.25, -0.1, -0.1), (0.75, 0.1, 0.1)),
        ]
        obstacle = make_object(2, (0.6, 0.0, 0.0), (0.45, -0.1, -0.1), (0.85, 0.1, 0.1))

        self.assertTrue(
            has_terminal_bbox_collision(
                original_objects,
                moved_objects,
                {1},
                collision_objects=[obstacle],
            )
        )

    def test_has_terminal_bbox_collision_allows_face_touching(self) -> None:
        original_objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.25, -0.1, -0.1), (0.25, 0.1, 0.1)),
        ]
        moved_objects = [
            make_object(1, (0.5, 0.0, 0.0), (0.25, -0.1, -0.1), (0.75, 0.1, 0.1)),
        ]
        obstacle = make_object(2, (1.0, 0.0, 0.0), (0.75, -0.1, -0.1), (1.25, 0.1, 0.1))

        self.assertFalse(
            has_terminal_bbox_collision(
                original_objects,
                moved_objects,
                {1},
                collision_objects=[obstacle],
            )
        )


class VirtualOpsIntegrationTests(unittest.TestCase):
    def test_find_meaningful_movement_skips_bbox_out_of_room_candidate(self) -> None:
        objects = [
            make_object(1, (0.75, 0.0, 0.0), (0.5, -0.1, -0.1), (1.0, 0.1, 0.1), label="mover"),
            make_object(2, (-1.0, 0.0, 0.0), (-1.2, -0.1, -0.1), (-0.8, 0.1, 0.1), label="ref"),
        ]
        room_bounds = {
            "bbox_min": [-2.0, -1.0, -1.0],
            "bbox_max": [1.0, 1.0, 1.0],
        }

        delta, changed = find_meaningful_movement(
            objects,
            attachment_graph={},
            target_id=1,
            camera_pose=make_camera_pose(),
            room_bounds=room_bounds,
        )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.tolist(), [-1.0, 0.0, 0.0])
        self.assertTrue(changed)

    def test_find_meaningful_movement_uses_collision_objects(self) -> None:
        objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.25, -0.1, -0.1), (0.25, 0.1, 0.1), label="mover"),
            make_object(2, (1.6, 0.0, 0.0), (1.4, -0.1, -0.1), (1.8, 0.1, 0.1), label="ref_a"),
            make_object(3, (-1.6, 0.0, 0.0), (-1.8, -0.1, -0.1), (-1.4, 0.1, 0.1), label="ref_b"),
        ]
        collision_objects = [
            make_object(4, (0.6, 0.0, 0.0), (0.45, -0.1, -0.1), (0.85, 0.1, 0.1), label="obstacle"),
        ]

        delta_without_collision, _ = find_meaningful_movement(
            objects,
            attachment_graph={},
            target_id=1,
            camera_pose=make_camera_pose(),
        )
        delta_with_collision, _ = find_meaningful_movement(
            objects,
            attachment_graph={},
            target_id=1,
            camera_pose=make_camera_pose(),
            collision_objects=collision_objects,
        )

        self.assertEqual(delta_without_collision.tolist(), [0.5, 0.0, 0.0])
        self.assertEqual(delta_with_collision.tolist(), [-0.5, 0.0, 0.0])

    def test_find_meaningful_orbit_rotation_filters_invalid_candidates(self) -> None:
        objects = [
            make_object(1, (0.75, 0.0, 0.0), (0.5, -0.1, -0.1), (1.0, 0.1, 0.1), label="mover"),
            make_object(2, (0.0, 0.0, 0.0), (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), label="pivot"),
        ]
        room_bounds = {
            "bbox_min": [-1.0, -0.7, -1.0],
            "bbox_max": [1.0, 1.0, 1.0],
        }
        collision_objects = [
            make_object(3, (0.0, 0.75, 0.0), (-0.1, 0.65, -0.1), (0.1, 0.85, 0.1), label="obstacle"),
        ]

        rotations_without_collision = find_meaningful_orbit_rotation(
            objects,
            attachment_graph={},
            target_id=1,
            pivot_id=2,
            room_bounds=room_bounds,
        )
        rotations_with_collision = find_meaningful_orbit_rotation(
            objects,
            attachment_graph={},
            target_id=1,
            pivot_id=2,
            room_bounds=room_bounds,
            collision_objects=collision_objects,
        )

        without_collision_keys = {
            (entry["angle"], entry["rotation_direction"])
            for entry in rotations_without_collision
        }
        with_collision_keys = {
            (entry["angle"], entry["rotation_direction"])
            for entry in rotations_with_collision
        }

        self.assertEqual(
            without_collision_keys,
            {(90, "counterclockwise"), (180, "clockwise")},
        )
        self.assertEqual(with_collision_keys, {(180, "clockwise")})


if __name__ == "__main__":
    unittest.main()
