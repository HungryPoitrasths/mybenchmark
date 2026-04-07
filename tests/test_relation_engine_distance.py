import unittest

import numpy as np

from src.relation_engine import (
    DISTANCE_SURFACE_BARYCENTRICS_KEY,
    DISTANCE_SURFACE_POINTS_KEY,
    DISTANCE_SURFACE_TRIANGLE_IDS_KEY,
    DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY,
    compute_aabb_closest_point_distance,
    compute_all_relations,
    compute_distance_details,
    find_changed_relations,
)
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
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    *,
    surface_points: list[list[float]] | None = None,
) -> dict:
    obj = {
        "id": obj_id,
        "label": label,
        "center": list(center),
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "support_geom": {
            "bottom_hull_xy": [
                [bbox_min[0], bbox_min[1]],
                [bbox_max[0], bbox_min[1]],
                [bbox_max[0], bbox_max[1]],
                [bbox_min[0], bbox_max[1]],
            ],
        },
    }
    if surface_points:
        points = np.asarray(surface_points, dtype=np.float64)
        obj[DISTANCE_SURFACE_POINTS_KEY] = points
        obj[DISTANCE_SURFACE_TRIANGLE_IDS_KEY] = np.zeros(len(points), dtype=np.int64)
        obj[DISTANCE_SURFACE_BARYCENTRICS_KEY] = np.tile(
            np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
            (len(points), 1),
        )
        triangle_vertices = []
        for point in points:
            triangle_vertices.append(
                [
                    [point[0] - 0.01, point[1] - 0.01, point[2]],
                    [point[0] + 0.02, point[1] - 0.01, point[2]],
                    [point[0] - 0.01, point[1] + 0.02, point[2]],
                ]
            )
        obj[DISTANCE_SURFACE_TRIANGLE_VERTICES_KEY] = np.asarray(triangle_vertices, dtype=np.float64)
    return obj


class RelationEngineDistanceTests(unittest.TestCase):
    def test_compute_distance_details_prefers_surface_samples_when_available(self) -> None:
        obj_a = make_object(
            1,
            "sofa",
            center=(0.0, 0.0, 0.0),
            bbox_min=(-1.0, -0.5, -0.5),
            bbox_max=(1.0, 0.5, 0.5),
            surface_points=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        )
        obj_b = make_object(
            2,
            "curtain",
            center=(4.0, 0.0, 0.0),
            bbox_min=(3.9, -0.1, -0.1),
            bbox_max=(4.1, 0.1, 0.1),
            surface_points=[[0.95, 0.0, 0.0]],
        )

        details = compute_distance_details(obj_a, obj_b)

        self.assertEqual(details["distance_definition"], "surface_sample_min_euclidean")
        self.assertEqual(details["distance_bin_id"], "very_close")
        self.assertGreater(details["distance_m"], 0.84)
        self.assertLess(details["distance_m"], 0.85)

    def test_compute_distance_details_local_refinement_strictly_improves_coarse_min(self) -> None:
        obj_a = make_object(
            1,
            "sofa",
            center=(0.0, 0.0, 0.0),
            bbox_min=(-1.0, -0.5, -0.5),
            bbox_max=(1.0, 0.5, 0.5),
            surface_points=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        )
        obj_b = make_object(
            2,
            "curtain",
            center=(4.0, 0.0, 0.0),
            bbox_min=(3.9, -0.1, -0.1),
            bbox_max=(4.1, 0.1, 0.1),
            surface_points=[[0.95, 0.0, 0.0]],
        )

        coarse_min = float(
            np.linalg.norm(
                obj_a[DISTANCE_SURFACE_POINTS_KEY][:, None, :]
                - obj_b[DISTANCE_SURFACE_POINTS_KEY][None, :, :],
                axis=2,
            ).min()
        )
        details = compute_distance_details(obj_a, obj_b)

        self.assertAlmostEqual(coarse_min, 0.85, places=6)
        self.assertLess(details["distance_m"], coarse_min)

    def test_compute_distance_details_falls_back_to_aabb_when_samples_missing(self) -> None:
        obj_a = make_object(
            1,
            "box",
            center=(0.0, 0.0, 0.0),
            bbox_min=(-0.1, -0.1, -0.1),
            bbox_max=(0.1, 0.1, 0.1),
        )
        obj_b = make_object(
            2,
            "chair",
            center=(1.15, 0.0, 0.0),
            bbox_min=(1.05, -0.1, -0.1),
            bbox_max=(1.25, 0.1, 0.1),
        )

        details = compute_distance_details(obj_a, obj_b)

        self.assertEqual(details["distance_definition"], "aabb_closest_point_approx")
        self.assertEqual(details["distance_bin_id"], "very_close")
        self.assertAlmostEqual(details["distance_m"], 0.95, places=6)

    def test_compute_aabb_closest_point_distance_returns_zero_for_contact_or_overlap(self) -> None:
        obj_a = make_object(
            1,
            "box",
            center=(0.0, 0.0, 0.0),
            bbox_min=(-0.5, -0.5, -0.5),
            bbox_max=(0.5, 0.5, 0.5),
        )
        obj_overlap = make_object(
            2,
            "chair",
            center=(0.5, 0.0, 0.0),
            bbox_min=(0.0, -0.25, -0.25),
            bbox_max=(1.0, 0.25, 0.25),
        )
        obj_contact = make_object(
            3,
            "stool",
            center=(1.0, 0.0, 0.0),
            bbox_min=(0.5, -0.25, -0.25),
            bbox_max=(1.5, 0.25, 0.25),
        )

        for other in (obj_overlap, obj_contact):
            with self.subTest(obj_b_id=other["id"]):
                self.assertAlmostEqual(
                    compute_aabb_closest_point_distance(obj_a, other),
                    0.0,
                    places=6,
                )
                details = compute_distance_details(obj_a, other)
                self.assertEqual(details["distance_definition"], "aabb_closest_point_approx")
                self.assertEqual(details["distance_bin_id"], "very_close")
                self.assertAlmostEqual(details["distance_m"], 0.0, places=6)

    def test_compute_all_relations_includes_distance_metadata(self) -> None:
        relations = compute_all_relations(
            [
                make_object(
                    1,
                    "sofa",
                    center=(0.0, 0.0, 0.0),
                    bbox_min=(-0.1, -0.1, -0.1),
                    bbox_max=(0.1, 0.1, 0.1),
                    surface_points=[[0.0, 0.0, 0.0]],
                ),
                make_object(
                    2,
                    "curtain",
                    center=(2.0, 0.0, 0.0),
                    bbox_min=(1.9, -0.1, -0.1),
                    bbox_max=(2.1, 0.1, 0.1),
                    surface_points=[[0.9, 0.0, 0.0]],
                ),
            ],
            make_camera_pose(),
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["distance_bin_id"], "very_close")
        self.assertEqual(relations[0]["distance_definition"], "surface_sample_min_euclidean")
        self.assertIn("distance_m_raw", relations[0])

    def test_compute_all_relations_preserves_unrounded_distance_metadata(self) -> None:
        relations = compute_all_relations(
            [
                make_object(
                    1,
                    "sofa",
                    center=(0.0, 0.0, 0.0),
                    bbox_min=(-0.1, -0.1, -0.1),
                    bbox_max=(0.1, 0.1, 0.1),
                ),
                make_object(
                    2,
                    "curtain",
                    center=(0.395, 0.0, 0.0),
                    bbox_min=(0.295, -0.1, -0.1),
                    bbox_max=(0.495, 0.1, 0.1),
                ),
            ],
            make_camera_pose(),
        )

        self.assertEqual(len(relations), 1)
        self.assertAlmostEqual(relations[0]["distance_m_raw"], 0.195, places=6)
        self.assertEqual(relations[0]["distance_m"], round(relations[0]["distance_m_raw"], 2))

    def test_find_changed_relations_compares_distance_bin_ids(self) -> None:
        old_relations = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "left",
                "distance_bin": "close (1.0-2.0m)",
                "distance_bin_id": "close",
                "occlusion_a": "unknown",
                "occlusion_b": "unknown",
            }
        ]
        new_relations_same_bin = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "left",
                "distance_bin": "close (1.2-2.4m)",
                "distance_bin_id": "close",
                "occlusion_a": "unknown",
                "occlusion_b": "unknown",
            }
        ]
        new_relations_changed_bin = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "left",
                "distance_bin": "moderate (2.4-3.6m)",
                "distance_bin_id": "moderate",
                "occlusion_a": "unknown",
                "occlusion_b": "unknown",
            }
        ]

        self.assertEqual(find_changed_relations(old_relations, new_relations_same_bin), [])

        changed = find_changed_relations(old_relations, new_relations_changed_bin)
        self.assertEqual(len(changed), 1)
        self.assertIn("distance_bin", changed[0]["changes"])
        self.assertEqual(changed[0]["changes"]["distance_bin"]["old_bin_id"], "close")
        self.assertEqual(changed[0]["changes"]["distance_bin"]["new_bin_id"], "moderate")


if __name__ == "__main__":
    unittest.main()
