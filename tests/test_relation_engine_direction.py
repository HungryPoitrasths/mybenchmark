import unittest

import numpy as np

from src.relation_engine import (
    compute_all_relations,
    compute_pairwise_horizontal_direction,
    primary_direction,
    primary_direction_allocentric,
    primary_direction_object_centric,
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


def _rotated_rect(cx: float, cy: float, length: float, width: float, angle_deg: float) -> list[list[float]]:
    theta = np.radians(angle_deg)
    major = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    minor = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    center = np.array([cx, cy], dtype=float)
    half_l = 0.5 * length
    half_w = 0.5 * width
    corners = [
        center - half_l * major - half_w * minor,
        center + half_l * major - half_w * minor,
        center + half_l * major + half_w * minor,
        center - half_l * major + half_w * minor,
    ]
    return [corner.tolist() for corner in corners]


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
    def test_pairwise_horizontal_direction_uses_bbox_centers_not_bottom_hulls(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        night_stand = make_object(
            1,
            "night stand",
            (-0.2, -0.2, 0.0),
            (0.2, 0.2, 0.8),
            bottom_hull_xy=_rect(-0.2, -0.2, 0.2, 0.2),
        )
        pillow = make_object(
            2,
            "pillow",
            (-0.2, 1.8, 0.0),
            (0.2, 2.2, 0.4),
            bottom_hull_xy=_rect(1.4, 1.8, 2.0, 2.2),
        )

        direction, _ = compute_pairwise_horizontal_direction(night_stand, pillow, camera_pose)
        relations = compute_all_relations([night_stand, pillow], camera_pose)

        self.assertEqual(direction, "front")
        self.assertEqual(relations[0]["direction_b_rel_a"], "front")
        self.assertEqual(relations[0]["horizontal_direction_b_rel_a"], "front")

    def test_compute_all_relations_uses_bbox_center_mid_section_for_bedside_table(self) -> None:
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
            (-0.5, 1.8, 0.0),
            (-0.1, 2.2, 0.8),
            bottom_hull_xy=_rect(-0.5, 1.8, -0.1, 2.2),
        )

        relations = compute_all_relations([bed, stand], camera_pose)

        self.assertEqual(relations[0]["direction_b_rel_a"], "left")

    def test_compute_all_relations_uses_bbox_center_for_bed_corner_case(self) -> None:
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

        self.assertEqual(relations[0]["direction_b_rel_a"], "front-left")

    def test_allocentric_uses_bbox_centers_not_footprints(self) -> None:
        direction, _ = primary_direction_allocentric(
            np.array([0.0, 2.0, 0.5], dtype=float),
            np.array([0.0, 0.0, 0.4], dtype=float),
            obj_a_hull_xy=np.array(_rect(1.0, 2.0, 2.0, 3.0), dtype=float),
            obj_b_hull_xy=np.array(_rect(0.0, 0.0, 0.4, 0.4), dtype=float),
            obj_a_bbox_min=np.array([-0.2, 1.8, 0.0], dtype=float),
            obj_a_bbox_max=np.array([0.2, 2.2, 1.0], dtype=float),
            obj_b_bbox_min=np.array([-0.2, -0.2, 0.0], dtype=float),
            obj_b_bbox_max=np.array([0.2, 0.2, 0.8], dtype=float),
        )

        self.assertEqual(direction, "north")

    def test_object_centric_horizontal_direction_uses_bbox_centers_not_footprints(self) -> None:
        direction, _ = primary_direction_object_centric(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([0.0, 1.0, 0.5], dtype=float),
            np.array([0.0, 2.0, 0.5], dtype=float),
            anchor_hull_xy=np.array(_rect(-0.2, -0.2, 0.2, 0.2), dtype=float),
            target_hull_xy=np.array(_rect(1.4, 1.8, 2.0, 2.2), dtype=float),
            anchor_bbox_min=np.array([-0.2, -0.2, 0.0], dtype=float),
            anchor_bbox_max=np.array([0.2, 0.2, 1.0], dtype=float),
            target_bbox_min=np.array([-0.2, 1.8, 0.0], dtype=float),
            target_bbox_max=np.array([0.2, 2.2, 1.0], dtype=float),
        )

        self.assertEqual(direction, "front")

    def test_overlap_keeps_center_based_horizontal_direction(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        bed = make_object(
            1,
            "bed",
            (0.0, 0.0, 0.0),
            (2.0, 4.0, 1.0),
            bottom_hull_xy=_rect(0.0, 0.0, 2.0, 4.0),
        )
        chair = make_object(
            2,
            "chair",
            (-0.2, 3.2, 0.0),
            (0.2, 3.6, 0.8),
            bottom_hull_xy=_rect(-0.2, 3.2, 0.2, 3.6),
        )

        relations = compute_all_relations([bed, chair], camera_pose)

        self.assertEqual(relations[0]["direction_b_rel_a"], "front-left")

    def test_rotated_elongated_anchor_uses_bbox_centers_for_horizontal_direction(self) -> None:
        camera_pose = make_floorplan_camera_pose()
        bed_hull = _rotated_rect(0.0, 0.0, 4.0, 1.0, 45.0)
        bed = {
            "id": 1,
            "label": "bed",
            "center": [0.0, 0.0, 0.5],
            "bbox_min": [-1.8, -1.8, 0.0],
            "bbox_max": [1.8, 1.8, 1.0],
            "support_geom": {
                "bottom_hull_xy": bed_hull,
                "top_hull_xy": [],
                "top_surface_candidates": [],
            },
        }
        stand = make_object(
            2,
            "stand",
            (-1.2, 1.0, 0.0),
            (-0.8, 1.4, 0.8),
            bottom_hull_xy=_rect(-1.2, 1.0, -0.8, 1.4),
        )

        relations = compute_all_relations([bed, stand], camera_pose)

        self.assertEqual(relations[0]["direction_b_rel_a"], "front-left")

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

    def test_allocentric_parallel_side_edges_use_bbox_centers(self) -> None:
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

        self.assertEqual(direction, "southwest")

    def test_allocentric_vertical_requires_positive_footprint_overlap(self) -> None:
        direction, _ = primary_direction_allocentric(
            np.array([0.2, 1.2, 1.85], dtype=float),
            np.array([0.2, 0.2, 0.5], dtype=float),
            obj_a_hull_xy=np.array(_rect(0.0, 1.0, 0.4, 1.4), dtype=float),
            obj_b_hull_xy=np.array(_rect(0.0, 0.0, 0.4, 0.4), dtype=float),
            obj_a_bbox_min=np.array([0.0, 1.0, 1.5], dtype=float),
            obj_a_bbox_max=np.array([0.4, 1.4, 2.2], dtype=float),
            obj_b_bbox_min=np.array([0.0, 0.0, 0.0], dtype=float),
            obj_b_bbox_max=np.array([0.4, 0.4, 1.0], dtype=float),
        )

        self.assertEqual(direction, "north")

    def test_allocentric_vertical_keeps_above_when_footprints_overlap(self) -> None:
        direction, _ = primary_direction_allocentric(
            np.array([0.2, 0.2, 1.85], dtype=float),
            np.array([0.2, 0.2, 0.5], dtype=float),
            obj_a_hull_xy=np.array(_rect(0.0, 0.0, 0.4, 0.4), dtype=float),
            obj_b_hull_xy=np.array(_rect(0.0, 0.0, 0.4, 0.4), dtype=float),
            obj_a_bbox_min=np.array([0.0, 0.0, 1.5], dtype=float),
            obj_a_bbox_max=np.array([0.4, 0.4, 2.2], dtype=float),
            obj_b_bbox_min=np.array([0.0, 0.0, 0.0], dtype=float),
            obj_b_bbox_max=np.array([0.4, 0.4, 1.0], dtype=float),
        )

        self.assertEqual(direction, "above")

    def test_object_centric_vertical_requires_positive_footprint_overlap(self) -> None:
        direction, _ = primary_direction_object_centric(
            np.array([0.2, 0.2, 0.5], dtype=float),
            np.array([0.2, 1.2, 0.5], dtype=float),
            np.array([-0.8, 1.2, 1.85], dtype=float),
            anchor_hull_xy=np.array(_rect(0.0, 0.0, 0.4, 0.4), dtype=float),
            target_hull_xy=np.array(_rect(-1.0, 1.0, -0.6, 1.4), dtype=float),
            anchor_bbox_min=np.array([0.0, 0.0, 0.0], dtype=float),
            anchor_bbox_max=np.array([0.4, 0.4, 1.0], dtype=float),
            target_bbox_min=np.array([-1.0, 1.0, 1.5], dtype=float),
            target_bbox_max=np.array([-0.6, 1.4, 2.2], dtype=float),
        )

        self.assertEqual(direction, "front-left")

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
