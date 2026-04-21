import unittest
from unittest.mock import patch

import numpy as np

from src.relation_engine import (
    DISTANCE_SURFACE_POINTS_KEY,
    DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY,
)
from src.utils.colmap_loader import CameraPose
from src.virtual_ops import (
    apply_movement,
    apply_removal,
    apply_coordinate_rotation,
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


def make_direction_relation(
    direction: str,
    *,
    distance_bin: str = "near",
    distance_bin_id: str = "near",
) -> dict:
    return {
        "obj_a_id": 1,
        "obj_b_id": 2,
        "direction_b_rel_a": direction,
        "distance_bin": distance_bin,
        "distance_bin_id": distance_bin_id,
    }


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
    def test_apply_removal_only_removes_target_object(self) -> None:
        objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
            make_object(2, (1.0, 0.0, 0.0), (0.9, -0.1, -0.1), (1.1, 0.1, 0.1)),
        ]

        remaining = apply_removal(objects, 1)

        self.assertEqual([obj["id"] for obj in remaining], [2])
        self.assertIsNot(remaining[0], objects[1])

    def test_apply_coordinate_rotation_handles_empty_object_list(self) -> None:
        rotated = apply_coordinate_rotation([], 90.0)
        self.assertEqual(rotated, [])

    def test_apply_movement_translates_runtime_distance_surface_geometry(self) -> None:
        obj = make_object(
            1,
            (0.0, 0.0, 0.0),
            (-0.1, -0.1, -0.1),
            (0.1, 0.1, 0.1),
            label="mover",
        )
        obj[DISTANCE_SURFACE_POINTS_KEY] = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float64,
        )
        obj[DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY] = np.array(
            [[
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ]],
            dtype=np.float64,
        )

        moved = apply_movement(
            [obj],
            attachment_graph={},
            target_obj_id=1,
            delta_position=np.array([1.0, 2.0, 0.5], dtype=np.float64),
        )

        np.testing.assert_allclose(
            moved[0][DISTANCE_SURFACE_POINTS_KEY],
            np.array([[1.0, 2.0, 0.5], [1.1, 2.0, 0.5]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            moved[0][DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY],
            np.array(
                [[
                    [1.0, 2.0, 0.5],
                    [1.1, 2.0, 0.5],
                    [1.0, 2.1, 0.5],
                ]],
                dtype=np.float64,
            ),
        )
        np.testing.assert_allclose(
            obj[DISTANCE_SURFACE_POINTS_KEY],
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64),
        )

    def test_apply_movement_translates_support_surface_candidates_xy_and_z(self) -> None:
        obj = make_object(
            1,
            (0.0, 0.0, 0.5),
            (-0.2, -0.2, 0.0),
            (0.2, 0.2, 1.0),
            label="mover",
        )
        obj["support_geom"] = {
            "bottom_hull_xy": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
            "top_hull_xy": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
            "bottom_surface_candidates": [{
                "z": 0.0,
                "hull_xy": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
                "area": 0.16,
                "score": 0.8,
            }],
            "top_surface_candidates": [{
                "z": 1.0,
                "hull_xy": [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]],
                "area": 0.16,
                "score": 0.9,
            }],
        }

        moved = apply_movement(
            [obj],
            attachment_graph={},
            target_obj_id=1,
            delta_position=np.array([0.5, -0.25, 0.75], dtype=np.float64),
        )

        moved_support = moved[0]["support_geom"]
        np.testing.assert_allclose(
            moved_support["bottom_surface_candidates"][0]["hull_xy"],
            np.array([[0.3, -0.45], [0.7, -0.45], [0.7, -0.05], [0.3, -0.05]], dtype=np.float64),
        )
        self.assertAlmostEqual(moved_support["bottom_surface_candidates"][0]["z"], 0.75)
        self.assertAlmostEqual(moved_support["top_surface_candidates"][0]["z"], 1.75)

    def test_apply_coordinate_rotation_rotates_runtime_distance_surface_geometry(self) -> None:
        obj = make_object(
            1,
            (1.0, 0.0, 0.0),
            (0.9, -0.1, -0.1),
            (1.1, 0.1, 0.1),
            label="mover",
        )
        obj[DISTANCE_SURFACE_POINTS_KEY] = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        obj[DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY] = np.array(
            [[
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.0, 0.1, 0.0],
            ]],
            dtype=np.float64,
        )

        rotated = apply_coordinate_rotation([obj], 90.0)

        np.testing.assert_allclose(
            rotated[0][DISTANCE_SURFACE_POINTS_KEY],
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            rotated[0][DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY],
            np.array(
                [[
                    [1.0, 0.0, 0.0],
                    [1.0, 0.1, 0.0],
                    [0.9, 0.0, 0.0],
                ]],
                dtype=np.float64,
            ),
            atol=1e-6,
        )

    def test_find_meaningful_movement_prefers_larger_90_degree_candidate_first(self) -> None:
        objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), label="mover"),
            make_object(2, (5.0, 5.0, 0.0), (4.9, 4.9, -0.1), (5.1, 5.1, 0.1), label="ref"),
        ]
        candidates = [
            np.array([3.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.5, 0.0, 0.0], dtype=np.float64),
        ]

        with (
            patch("src.virtual_ops.MOVEMENT_CANDIDATES", candidates),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front")],
                    [make_direction_relation("right")],
                ],
            ),
        ):
            delta, changed = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds={"bbox_min": [-10.0, -10.0, -10.0], "bbox_max": [10.0, 10.0, 10.0]},
            )

        np.testing.assert_allclose(delta, candidates[0])
        self.assertEqual(changed[0]["changes"]["direction_b_rel_a"]["new"], "right")

    def test_find_meaningful_movement_prefers_90_degree_over_larger_45_degree_change(self) -> None:
        objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), label="mover"),
            make_object(2, (5.0, 5.0, 0.0), (4.9, 4.9, -0.1), (5.1, 5.1, 0.1), label="ref"),
        ]
        candidates = [
            np.array([3.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.5, 0.0], dtype=np.float64),
        ]

        with (
            patch("src.virtual_ops.MOVEMENT_CANDIDATES", candidates),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front")],
                    [make_direction_relation("front-right")],
                    [make_direction_relation("right")],
                ],
            ),
        ):
            delta, changed = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds={"bbox_min": [-10.0, -10.0, -10.0], "bbox_max": [10.0, 10.0, 10.0]},
            )

        np.testing.assert_allclose(delta, candidates[1])
        self.assertEqual(changed[0]["changes"]["direction_b_rel_a"]["new"], "right")

    def test_find_meaningful_movement_falls_back_to_first_45_degree_change(self) -> None:
        objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), label="mover"),
            make_object(2, (5.0, 5.0, 0.0), (4.9, 4.9, -0.1), (5.1, 5.1, 0.1), label="ref"),
        ]
        candidates = [
            np.array([0.5, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.5, 0.0], dtype=np.float64),
        ]

        with (
            patch("src.virtual_ops.MOVEMENT_CANDIDATES", candidates),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front")],
                    [make_direction_relation("front-right")],
                    [make_direction_relation("front-left")],
                ],
            ),
        ):
            delta, changed = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds={"bbox_min": [-10.0, -10.0, -10.0], "bbox_max": [10.0, 10.0, 10.0]},
            )

        np.testing.assert_allclose(delta, candidates[0])
        self.assertEqual(changed[0]["changes"]["direction_b_rel_a"]["new"], "front-right")

    def test_find_meaningful_movement_ignores_non_horizontal_relation_changes(self) -> None:
        objects = [
            make_object(1, (0.0, 0.0, 0.0), (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1), label="mover"),
            make_object(2, (5.0, 5.0, 0.0), (4.9, 4.9, -0.1), (5.1, 5.1, 0.1), label="ref"),
        ]
        candidates = [
            np.array([0.5, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.5, 0.0], dtype=np.float64),
        ]

        with (
            patch("src.virtual_ops.MOVEMENT_CANDIDATES", candidates),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front", distance_bin="near", distance_bin_id="near")],
                    [make_direction_relation("front", distance_bin="moderate", distance_bin_id="moderate")],
                    [make_direction_relation("above", distance_bin="near", distance_bin_id="near")],
                ],
            ),
        ):
            delta, changed = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds={"bbox_min": [-10.0, -10.0, -10.0], "bbox_max": [10.0, 10.0, 10.0]},
            )

        self.assertIsNone(delta)
        self.assertEqual(changed, [])

    def test_find_meaningful_movement_skips_bbox_out_of_room_candidate(self) -> None:
        objects = [
            make_object(1, (0.75, 0.0, 0.0), (0.5, -0.1, -0.1), (1.0, 0.1, 0.1), label="mover"),
            make_object(2, (-1.0, 0.0, 0.0), (-1.2, -0.1, -0.1), (-0.8, 0.1, 0.1), label="ref"),
        ]
        room_bounds = {
            "bbox_min": [-2.0, -1.0, -1.0],
            "bbox_max": [1.0, 1.0, 1.0],
        }

        with (
            patch(
                "src.virtual_ops.MOVEMENT_CANDIDATES",
                [
                    np.array([0.5, 0.0, 0.0], dtype=np.float64),
                    np.array([-0.5, 0.0, 0.0], dtype=np.float64),
                ],
            ),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front")],
                    [make_direction_relation("right")],
                ],
            ),
        ):
            delta, changed = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds=room_bounds,
            )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.tolist(), [-0.5, 0.0, 0.0])
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

        candidates = [
            np.array([0.5, 0.0, 0.0], dtype=np.float64),
            np.array([-0.5, 0.0, 0.0], dtype=np.float64),
        ]

        with (
            patch("src.virtual_ops.MOVEMENT_CANDIDATES", candidates),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front")],
                    [make_direction_relation("front-right")],
                    [make_direction_relation("front-left")],
                ],
            ),
        ):
            delta_without_collision, _ = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds={"bbox_min": [-3.0, -1.0, -1.0], "bbox_max": [3.0, 1.0, 1.0]},
            )

        with (
            patch("src.virtual_ops.MOVEMENT_CANDIDATES", candidates),
            patch(
                "src.virtual_ops.compute_all_relations",
                side_effect=[
                    [make_direction_relation("front")],
                    [make_direction_relation("front-left")],
                ],
            ),
        ):
            delta_with_collision, _ = find_meaningful_movement(
                objects,
                attachment_graph={},
                target_id=1,
                camera_pose=make_camera_pose(),
                room_bounds={"bbox_min": [-3.0, -1.0, -1.0], "bbox_max": [3.0, 1.0, 1.0]},
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
            {
                (180, "clockwise"),
                (135, "clockwise"),
                (135, "counterclockwise"),
                (90, "counterclockwise"),
                (45, "clockwise"),
                (45, "counterclockwise"),
            },
        )
        self.assertEqual(
            with_collision_keys,
            {
                (180, "clockwise"),
                (135, "clockwise"),
                (135, "counterclockwise"),
                (45, "clockwise"),
                (45, "counterclockwise"),
            },
        )
        self.assertEqual(
            [
                (entry["angle"], entry["rotation_direction"])
                for entry in rotations_with_collision
            ],
            [
                (180, "clockwise"),
                (135, "clockwise"),
                (135, "counterclockwise"),
                (45, "clockwise"),
                (45, "counterclockwise"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
