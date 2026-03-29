import unittest

import numpy as np

from src.relation_engine import (
    compute_all_relations,
    primary_direction,
    primary_direction_allocentric,
)
from src.utils.colmap_loader import CameraPose
from src.virtual_ops import apply_movement


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


class RelationEngineDirectionTests(unittest.TestCase):
    def test_compute_all_relations_uses_footprint_edges_for_bedside_table(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        bed = make_object(
            1,
            "bed",
            (0.0, 0.0, 0.0),
            (2.0, 4.0, 1.0),
            bottom_hull_xy=_rect(0.0, 0.0, 2.0, 4.0),
        )
        stand = make_object(
            2,
            "stand",
            (-0.5, 3.4, 0.0),
            (-0.1, 3.8, 0.8),
            bottom_hull_xy=_rect(-0.5, 3.4, -0.1, 3.8),
        )

        relations = compute_all_relations([bed, stand], camera_pose)

        self.assertEqual(relations[0]["direction_b_rel_a"], "left")

    def test_primary_direction_horizontal_only_ignores_camera_pitch_mixing(self) -> None:
        angle_deg = 20.0
        rad = np.radians(angle_deg)
        pose = CameraPose(
            image_name="pitched.jpg",
            rotation=np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rad), -np.sin(rad)],
                [0.0, np.sin(rad), np.cos(rad)],
            ], dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )

        a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        full_dir, _ = primary_direction(a, b, pose)
        horiz_dir, _ = primary_direction(a, b, pose, horizontal_only=True)

        self.assertEqual(full_dir, "below")
        self.assertEqual(horiz_dir, "front")

    def test_allocentric_uses_footprint_pair_for_parallel_side_edges(self) -> None:
        obj_a = make_object(
            1,
            "sofa",
            (0.0, 0.0, 0.0),
            (4.0, 1.0, 1.0),
            bottom_hull_xy=_rect(0.0, 0.0, 4.0, 1.0),
        )
        obj_b = make_object(
            2,
            "bench",
            (1.0, 1.2, 0.0),
            (5.0, 2.2, 1.0),
            bottom_hull_xy=_rect(1.0, 1.2, 5.0, 2.2),
        )

        direction, _ = primary_direction_allocentric(
            np.array(obj_a["center"], dtype=float),
            np.array(obj_b["center"], dtype=float),
            obj_a_hull_xy=np.array(obj_a["support_geom"]["bottom_hull_xy"], dtype=float),
            obj_b_hull_xy=np.array(obj_b["support_geom"]["bottom_hull_xy"], dtype=float),
            obj_a_bbox_min=np.array(obj_a["bbox_min"], dtype=float),
            obj_a_bbox_max=np.array(obj_a["bbox_max"], dtype=float),
            obj_b_bbox_min=np.array(obj_b["bbox_min"], dtype=float),
            obj_b_bbox_max=np.array(obj_b["bbox_max"], dtype=float),
        )

        self.assertEqual(direction, "south")

    def test_apply_movement_translates_support_geom(self) -> None:
        mover = make_object(
            1,
            "chair",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            bottom_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
        )

        moved = apply_movement([mover], attachment_graph={}, target_obj_id=1, delta_position=np.array([1.0, 2.0, 0.0]))

        self.assertEqual(
            moved[0]["support_geom"]["bottom_hull_xy"],
            _rect(1.0, 2.0, 2.0, 3.0),
        )


if __name__ == "__main__":
    unittest.main()
