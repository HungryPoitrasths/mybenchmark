import unittest
from unittest.mock import patch

from src.support_graph import (
    _attachment_candidate,
    _contained_in_metrics,
    _supported_by_metrics,
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
            "top_hull_xy": top_hull_xy or [],
            "top_surface_candidates": top_surface_candidates or [],
        },
    }


class SupportGraphHeuristicTests(unittest.TestCase):
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

    def test_contained_in_uses_parent_opening_polygon_instead_of_bbox(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
