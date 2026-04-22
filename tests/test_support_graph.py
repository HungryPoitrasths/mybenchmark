import unittest
from unittest.mock import patch

from src.support_graph import (
    _attachment_candidate,
    _affixed_to_metrics,
    _contained_in_metrics,
    _resting_on_soft_surface_metrics,
    _supported_by_metrics,
    build_attachment_candidates,
    build_attachment_graph,
    compute_bottom_footprint_overlap_metrics,
)


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
    bottom_surface_candidates: list[dict] | None = None,
    top_hull_xy: list[list[float]] | None = None,
    top_surface_candidates: list[dict] | None = None,
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
            "bottom_hull_xy": bottom_hull_xy or [],
            "bottom_surface_candidates": bottom_surface_candidates or [],
            "top_hull_xy": top_hull_xy or [],
            "top_surface_candidates": top_surface_candidates or [],
        },
    }


class SupportGraphHeuristicTests(unittest.TestCase):
    def test_compute_bottom_footprint_overlap_metrics_prefers_bottom_hulls(self) -> None:
        obj_a = make_object(
            1,
            "book",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.2),
            bottom_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
        )
        obj_b = make_object(
            2,
            "box",
            (0.5, 0.5, 0.0),
            (1.5, 1.5, 0.3),
            bottom_hull_xy=_rect(0.5, 0.5, 1.5, 1.5),
        )

        metrics = compute_bottom_footprint_overlap_metrics(obj_a, obj_b)

        self.assertAlmostEqual(metrics["overlap_area"], 0.25)
        self.assertAlmostEqual(metrics["coverage_a"], 0.25)
        self.assertAlmostEqual(metrics["coverage_b"], 0.25)
        self.assertAlmostEqual(metrics["coverage_small"], 0.25)
        self.assertEqual(metrics["source_a"], "bottom_hull_xy")
        self.assertEqual(metrics["source_b"], "bottom_hull_xy")

    def test_compute_bottom_footprint_overlap_metrics_falls_back_to_bbox(self) -> None:
        obj_a = make_object(
            1,
            "book",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.2),
        )
        obj_b = make_object(
            2,
            "box",
            (0.8, 0.0, 0.0),
            (1.8, 1.0, 0.3),
        )

        metrics = compute_bottom_footprint_overlap_metrics(obj_a, obj_b)

        self.assertAlmostEqual(metrics["overlap_area"], 0.2)
        self.assertAlmostEqual(metrics["coverage_small"], 0.2)
        self.assertEqual(metrics["source_a"], "bbox")
        self.assertEqual(metrics["source_b"], "bbox")

    def test_supported_by_allows_small_penetration_but_rejects_large_below_surface_gap(self) -> None:
        parent = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child_small_penetration = make_object(
            1,
            "cup",
            (0.2, 0.2, 0.99),
            (0.4, 0.4, 1.09),
            bottom_hull_xy=_rect(0.2, 0.2, 0.4, 0.4),
        )
        child_far_below = make_object(
            2,
            "cup",
            (0.2, 0.2, 0.94),
            (0.4, 0.4, 1.04),
            bottom_hull_xy=_rect(0.2, 0.2, 0.4, 0.4),
        )

        self.assertIsNotNone(_supported_by_metrics(child_small_penetration, parent))
        self.assertIsNone(_supported_by_metrics(child_far_below, parent))

    def test_supported_by_uses_child_bottom_surface_candidates_to_ignore_low_outlier(self) -> None:
        parent = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.2, 0.2, 0.94),
            (0.7, 0.7, 1.08),
            bottom_hull_xy=_rect(0.2, 0.2, 0.32, 0.32),
            bottom_surface_candidates=[
                {
                    "z": 0.94,
                    "hull_xy": _rect(0.2, 0.2, 0.32, 0.32),
                    "area": 0.0144,
                    "score": 0.10,
                },
                {
                    "z": 1.0,
                    "hull_xy": _rect(0.2, 0.2, 0.7, 0.7),
                    "area": 0.25,
                    "score": 0.95,
                },
            ],
        )

        metrics = _supported_by_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["evidence"]["geometry_contact"]["contact_z_child"], 1.0)
        self.assertAlmostEqual(metrics["evidence"]["geometry_contact"]["signed_z_gap"], 0.0)

    def test_supported_by_shallow_penetration_requires_strong_overlap(self) -> None:
        parent = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child_pass = make_object(
            1,
            "book",
            (0.1, 0.1, 0.96),
            (0.7, 0.7, 1.08),
            bottom_surface_candidates=[{
                "z": 0.96,
                "hull_xy": _rect(0.1, 0.1, 0.7, 0.7),
                "area": 0.36,
                "score": 0.9,
            }],
        )
        child_fail = make_object(
            2,
            "book",
            (0.75, 0.75, 0.96),
            (1.35, 1.35, 1.08),
            bottom_surface_candidates=[{
                "z": 0.96,
                "hull_xy": _rect(0.75, 0.75, 1.35, 1.35),
                "area": 0.36,
                "score": 0.9,
            }],
        )

        self.assertIsNotNone(_supported_by_metrics(child_pass, parent))
        self.assertIsNone(_supported_by_metrics(child_fail, parent))

    def test_supported_by_rigid_prior_boosts_book_on_table_confidence(self) -> None:
        parent_table = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        parent_box = make_object(
            11,
            "box",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.2, 0.2, 1.0),
            (0.7, 0.7, 1.08),
            bottom_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.2, 0.2, 0.7, 0.7),
                "area": 0.25,
                "score": 0.95,
            }],
        )

        table_metrics = _supported_by_metrics(child, parent_table)
        box_metrics = _supported_by_metrics(child, parent_box)

        self.assertIsNotNone(table_metrics)
        self.assertIsNotNone(box_metrics)
        self.assertGreater(table_metrics["confidence"], box_metrics["confidence"])

    def test_resting_on_soft_surface_exact_prior_boosts_pillow_on_sofa_confidence(self) -> None:
        parent_sofa = make_object(
            10,
            "sofa",
            (0.0, 0.0, 0.0),
            (1.2, 1.2, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.2, 1.2),
                "area": 1.44,
                "score": 1.0,
            }],
        )
        parent_bench = make_object(
            11,
            "bench",
            (0.0, 0.0, 0.0),
            (1.2, 1.2, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.2, 1.2),
                "area": 1.44,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "pillow",
            (0.2, 0.2, 1.0),
            (0.8, 0.8, 1.15),
            bottom_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.2, 0.2, 0.8, 0.8),
                "area": 0.36,
                "score": 0.95,
            }],
        )

        sofa_metrics = _resting_on_soft_surface_metrics(child, parent_sofa)
        bench_metrics = _resting_on_soft_surface_metrics(child, parent_bench)

        self.assertIsNotNone(sofa_metrics)
        self.assertIsNotNone(bench_metrics)
        self.assertGreater(sofa_metrics["confidence"], bench_metrics["confidence"])

    def test_resting_on_soft_surface_marks_pillow_on_ottoman_as_exact_prior(self) -> None:
        parent = make_object(
            10,
            "ottoman",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.8),
            top_surface_candidates=[{
                "z": 0.8,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "pillow",
            (0.2, 0.2, 0.8),
            (0.8, 0.8, 0.95),
            bottom_surface_candidates=[{
                "z": 0.8,
                "hull_xy": _rect(0.2, 0.2, 0.8, 0.8),
                "area": 0.36,
                "score": 0.95,
            }],
        )

        metrics = _resting_on_soft_surface_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["evidence"]["semantic_prior"]["score"], 1.0)

    def test_contained_in_exact_prior_boosts_book_in_cabinet_confidence(self) -> None:
        parent = make_object(
            10,
            "cabinet",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        book = make_object(
            1,
            "book",
            (0.2, 0.2, 0.2),
            (0.8, 0.8, 0.8),
            bottom_surface_candidates=[{
                "z": 0.2,
                "hull_xy": _rect(0.2, 0.2, 0.8, 0.8),
                "area": 0.36,
                "score": 0.95,
            }],
        )
        apple = make_object(
            2,
            "apple",
            (0.2, 0.2, 0.2),
            (0.8, 0.8, 0.8),
            bottom_surface_candidates=[{
                "z": 0.2,
                "hull_xy": _rect(0.2, 0.2, 0.8, 0.8),
                "area": 0.36,
                "score": 0.95,
            }],
        )

        book_metrics = _contained_in_metrics(book, parent)
        apple_metrics = _contained_in_metrics(apple, parent)

        self.assertIsNotNone(book_metrics)
        self.assertIsNotNone(apple_metrics)
        self.assertGreater(book_metrics["confidence"], apple_metrics["confidence"])

    def test_contained_in_marks_towel_in_laundry_basket_as_exact_prior(self) -> None:
        parent = make_object(
            10,
            "laundry basket",
            (0.0, 0.0, 0.0),
            (1.2, 1.2, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.2, 1.2),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.2, 1.2),
                "area": 1.44,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "towel",
            (0.2, 0.2, 0.2),
            (0.8, 0.8, 0.8),
            bottom_surface_candidates=[{
                "z": 0.2,
                "hull_xy": _rect(0.2, 0.2, 0.8, 0.8),
                "area": 0.36,
                "score": 0.95,
            }],
        )

        metrics = _contained_in_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["evidence"]["semantic_prior"]["score"], 1.0)

    def test_affixed_to_detects_drawer_attached_to_cabinet(self) -> None:
        parent = make_object(
            10,
            "cabinet",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        child = make_object(
            1,
            "drawer",
            (0.05, 0.05, 0.10),
            (0.95, 0.95, 0.95),
        )

        metrics = _affixed_to_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["type"], "affixed_to")
        self.assertGreaterEqual(metrics["confidence"], 0.55)

    def test_affixed_to_marks_drawer_attached_to_dresser_as_exact_prior(self) -> None:
        parent = make_object(
            10,
            "dresser",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        child = make_object(
            1,
            "drawer",
            (0.05, 0.05, 0.10),
            (0.95, 0.95, 0.95),
        )

        metrics = _affixed_to_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["evidence"]["semantic_prior"]["score"], 1.0)

    def test_supported_by_enclosed_bbox_fallback_detects_boxed_in_book_on_table(self) -> None:
        parent = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 0.2, 0.2),
                "area": 0.04,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.55, 0.55, 1.0),
            (0.95, 0.95, 1.08),
            bottom_hull_xy=_rect(0.55, 0.55, 0.95, 0.95),
        )

        metrics = _supported_by_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["type"], "supported_by")
        self.assertEqual(metrics["evidence"]["geometry_contact"]["mode"], "enclosed_bbox_fallback")
        self.assertAlmostEqual(metrics["evidence"]["geometry_contact"]["bottom_lift"], 1.0)
        self.assertAlmostEqual(metrics["evidence"]["geometry_contact"]["bbox_top_gap"], 0.0)
        self.assertAlmostEqual(metrics["evidence"]["geometry_contact"]["z_tolerance"], 0.05)
        self.assertAlmostEqual(metrics["evidence"]["geometry_contact"]["under_tolerance"], 0.03)

    def test_supported_by_enclosed_bbox_fallback_rejects_container_parent(self) -> None:
        parent = make_object(
            10,
            "bin",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 0.2, 0.2),
                "area": 0.04,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.55, 0.55, 1.0),
            (0.95, 0.95, 1.08),
            bottom_hull_xy=_rect(0.55, 0.55, 0.95, 0.95),
        )

        self.assertIsNone(_supported_by_metrics(child, parent))

    def test_supported_by_enclosed_bbox_fallback_requires_minimum_bottom_lift(self) -> None:
        parent = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.55),
            top_surface_candidates=[{
                "z": 0.55,
                "hull_xy": _rect(0.0, 0.0, 0.2, 0.2),
                "area": 0.04,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.45, 0.45, 0.49),
            (0.85, 0.85, 0.57),
            bottom_hull_xy=_rect(0.45, 0.45, 0.85, 0.85),
        )

        self.assertIsNone(_supported_by_metrics(child, parent))

    def test_supported_by_enclosed_bbox_fallback_rejects_objects_too_deep_inside_parent(self) -> None:
        parent = make_object(
            10,
            "table",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.3),
            top_surface_candidates=[{
                "z": 1.3,
                "hull_xy": _rect(0.0, 0.0, 0.2, 0.2),
                "area": 0.04,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.45, 0.45, 0.95),
            (0.85, 0.85, 1.03),
            bottom_hull_xy=_rect(0.45, 0.45, 0.85, 0.85),
        )

        self.assertIsNone(_supported_by_metrics(child, parent))

    def test_contained_in_rejects_non_prior_pair_without_hull_containment(self) -> None:
        parent = make_object(
            10,
            "dresser",
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 1.2),
        )
        child = make_object(
            1,
            "shoe",
            (0.4, 0.4, 0.2),
            (0.8, 0.8, 0.5),
            bottom_hull_xy=_rect(0.4, 0.4, 0.8, 0.8),
        )

        self.assertIsNone(_contained_in_metrics(child, parent))

    def test_contained_in_rejects_when_child_hull_extends_outside_parent_hull(self) -> None:
        parent = make_object(
            10,
            "bin",
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 2.0, 2.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (1.3, 1.3, 0.2),
            (1.7, 1.7, 0.8),
            bottom_hull_xy=_rect(1.3, 1.3, 1.7, 1.7),
        )

        self.assertIsNone(_contained_in_metrics(child, parent))

    def test_contained_in_does_not_override_affixed_pair(self) -> None:
        parent = make_object(
            10,
            "dresser",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        child = make_object(
            1,
            "drawer",
            (0.05, 0.05, 0.10),
            (0.95, 0.95, 0.95),
            bottom_hull_xy=_rect(0.05, 0.05, 0.95, 0.95),
        )

        candidate = _attachment_candidate(child, parent)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["type"], "affixed_to")

    def test_contained_in_requires_complete_child_hull_coverage(self) -> None:
        parent = make_object(
            10,
            "bin",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.2, 0.2, 0.2),
            (1.05, 0.8, 0.8),
            bottom_hull_xy=_rect(0.2, 0.2, 1.05, 0.8),
        )

        self.assertIsNone(_contained_in_metrics(child, parent))

    def test_contained_in_marks_hull_fully_contained_in_evidence(self) -> None:
        parent = make_object(
            10,
            "bin",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.2, 0.2, 0.2),
            (0.8, 0.8, 0.8),
            bottom_hull_xy=_rect(0.2, 0.2, 0.8, 0.8),
        )

        metrics = _contained_in_metrics(child, parent)

        self.assertIsNotNone(metrics)
        self.assertTrue(metrics["evidence"]["containment"]["hull_fully_contained"])

    def test_contained_in_uses_z_overlap_ratio(self) -> None:
        parent = make_object(
            10,
            "bin",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child_pass = make_object(
            1,
            "book",
            (0.2, 0.2, 0.5),
            (0.8, 0.8, 1.5),
            bottom_hull_xy=_rect(0.2, 0.2, 0.8, 0.8),
        )
        child_fail = make_object(
            2,
            "book",
            (0.2, 0.2, 0.7),
            (0.8, 0.8, 1.7),
            bottom_hull_xy=_rect(0.2, 0.2, 0.8, 0.8),
        )

        self.assertIsNotNone(_contained_in_metrics(child_pass, parent))
        self.assertIsNone(_contained_in_metrics(child_fail, parent))

    def test_contained_in_center_inside_xyz_does_not_add_xy_tolerance(self) -> None:
        parent = make_object(
            10,
            "bin",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            top_hull_xy=_rect(0.0, 0.0, 1.0, 1.0),
            top_surface_candidates=[{
                "z": 1.0,
                "hull_xy": _rect(0.0, 0.0, 1.0, 1.0),
                "area": 1.0,
                "score": 1.0,
            }],
        )
        child = make_object(
            1,
            "book",
            (0.8, 0.8, 0.2),
            (1.2, 1.2, 0.8),
            bottom_hull_xy=_rect(0.8, 0.8, 1.2, 1.2),
        )

        metrics = _contained_in_metrics(child, parent)

        self.assertIsNone(metrics)

    def test_attachment_candidate_falls_through_weak_contained_in(self) -> None:
        obj_a = make_object(1, "book", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        obj_b = make_object(2, "box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        with (
            patch("src.support_graph._contained_in_metrics", return_value={
                "type": "contained_in",
                "confidence": 0.54,
                "evidence": {"containment": {"score": 0.54}},
            }),
            patch("src.support_graph._affixed_to_metrics", return_value=None),
            patch("src.support_graph._resting_on_soft_surface_metrics", return_value=None),
            patch("src.support_graph._supported_by_metrics", return_value={
                "type": "supported_by",
                "confidence": 0.91,
                "evidence": {"geometry_contact": {"z_gap": 0.0}},
            }),
        ):
            candidate = _attachment_candidate(obj_a, obj_b)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["type"], "supported_by")
        self.assertEqual(candidate["confidence"], 0.91)

    def test_attachment_candidate_rejects_weak_supported_by(self) -> None:
        obj_a = make_object(1, "book", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        obj_b = make_object(2, "box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        with (
            patch("src.support_graph._contained_in_metrics", return_value=None),
            patch("src.support_graph._affixed_to_metrics", return_value=None),
            patch("src.support_graph._resting_on_soft_surface_metrics", return_value=None),
            patch("src.support_graph._supported_by_metrics", return_value={
                "type": "supported_by",
                "confidence": 0.44,
                "evidence": {"geometry_contact": {"z_gap": 0.0, "contact_z_parent": 1.0}},
            }),
        ):
            candidate = _attachment_candidate(obj_a, obj_b)

        self.assertIsNone(candidate)

    def test_build_attachment_graph_keeps_root_graph_for_movement_and_immediate_graph_for_support_chain(self) -> None:
        objects = [
            make_object(1, "table", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            make_object(2, "box", (0.0, 0.0, 1.0), (1.0, 1.0, 2.0)),
            make_object(3, "cup", (0.0, 0.0, 2.0), (1.0, 1.0, 2.2)),
        ]

        candidate_map = {
            (2, 1): {
                "parent_id": 1,
                "child_id": 2,
                "type": "supported_by",
                "confidence": 0.80,
                "evidence": {
                    "geometry_contact": {"z_gap": 0.0, "contact_z_parent": 1.0},
                    "xy_overlap": {"child_coverage": 1.0},
                },
            },
            (3, 2): {
                "parent_id": 2,
                "child_id": 3,
                "type": "supported_by",
                "confidence": 0.82,
                "evidence": {
                    "geometry_contact": {"z_gap": 0.0, "contact_z_parent": 2.0},
                    "xy_overlap": {"child_coverage": 1.0},
                },
            },
            (3, 1): {
                "parent_id": 1,
                "child_id": 3,
                "type": "supported_by",
                "confidence": 0.95,
                "evidence": {
                    "geometry_contact": {"z_gap": 0.0, "contact_z_parent": 1.0},
                    "xy_overlap": {"child_coverage": 1.0},
                },
            },
        }

        def fake_candidate(obj_a: dict, obj_b: dict, z_threshold=None):
            return candidate_map.get((int(obj_a["id"]), int(obj_b["id"])))

        with patch("src.support_graph._attachment_candidate", side_effect=fake_candidate):
            (
                attachment_graph,
                attached_by,
                _attachment_edges,
                support_chain_graph,
                support_chain_by,
            ) = build_attachment_graph(objects)

        self.assertEqual(attachment_graph, {1: [3, 2]})
        self.assertEqual(attached_by, {3: 1, 2: 1})
        self.assertEqual(support_chain_graph, {2: [3], 1: [2]})
        self.assertEqual(support_chain_by, {3: 2, 2: 1})

    def test_build_attachment_candidates_returns_all_raw_candidates_per_child(self) -> None:
        objects = [
            make_object(1, "table", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            make_object(2, "book", (0.0, 0.0, 1.0), (0.5, 0.5, 1.2)),
            make_object(3, "shelf", (0.0, 0.0, 1.3), (1.0, 1.0, 1.6)),
        ]

        candidate_map = {
            (2, 1): {
                "parent_id": 1,
                "child_id": 2,
                "type": "supported_by",
                "confidence": 0.70,
                "evidence": {
                    "geometry_contact": {"z_gap": 0.0, "contact_z_parent": 1.0},
                    "xy_overlap": {"child_coverage": 0.80},
                },
            },
            (2, 3): {
                "parent_id": 3,
                "child_id": 2,
                "type": "supported_by",
                "confidence": 0.92,
                "evidence": {
                    "geometry_contact": {"z_gap": 0.0, "contact_z_parent": 1.3},
                    "xy_overlap": {"child_coverage": 0.95},
                },
            },
        }

        def fake_candidate(obj_a: dict, obj_b: dict, z_threshold=None):
            return candidate_map.get((int(obj_a["id"]), int(obj_b["id"])))

        with patch("src.support_graph._attachment_candidate", side_effect=fake_candidate):
            raw_candidates = build_attachment_candidates(objects)
            attachment_graph, attached_by, final_edges, _support_chain_graph, _support_chain_by = (
                build_attachment_graph(objects)
            )

        self.assertEqual(
            [(edge["parent_id"], edge["child_id"], edge["confidence"]) for edge in raw_candidates],
            [(3, 2, 0.92), (1, 2, 0.70)],
        )
        self.assertEqual(attachment_graph, {3: [2]})
        self.assertEqual(attached_by, {2: 3})
        self.assertEqual([(edge["parent_id"], edge["child_id"]) for edge in final_edges], [(3, 2)])

    def test_build_attachment_graph_includes_containment_and_affixed_edges_in_support_chain(self) -> None:
        objects = [
            make_object(1, "cabinet", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
            make_object(2, "drawer", (0.2, 0.2, 0.6), (1.8, 1.8, 1.2)),
            make_object(3, "book", (0.4, 0.4, 0.7), (1.0, 1.0, 1.0)),
        ]

        candidate_map = {
            (2, 1): {
                "parent_id": 1,
                "child_id": 2,
                "type": "affixed_to",
                "confidence": 0.88,
                "evidence": {
                    "geometry_contact": {"z_gap": 0.0, "contact_z_parent": 1.0},
                },
            },
            (3, 2): {
                "parent_id": 2,
                "child_id": 3,
                "type": "contained_in",
                "confidence": 0.92,
                "evidence": {
                    "containment": {"score": 0.92},
                },
            },
        }

        def fake_candidate(obj_a: dict, obj_b: dict, z_threshold=None):
            return candidate_map.get((int(obj_a["id"]), int(obj_b["id"])))

        with patch("src.support_graph._attachment_candidate", side_effect=fake_candidate):
            (
                attachment_graph,
                attached_by,
                _attachment_edges,
                support_chain_graph,
                support_chain_by,
            ) = build_attachment_graph(objects)

        self.assertEqual(attachment_graph, {1: [2], 2: [3]})
        self.assertEqual(attached_by, {2: 1, 3: 2})
        self.assertEqual(support_chain_graph, {1: [2], 2: [3]})
        self.assertEqual(support_chain_by, {2: 1, 3: 2})


if __name__ == "__main__":
    unittest.main()
