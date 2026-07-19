import unittest
from pathlib import Path
from unittest import mock

from src.utils.colmap_loader import CameraPose

from scripts.filter_object_move_occlusion_in_frame import (
    DROP_COUNTERFACTUAL_NOT_ENOUGH_IN_FRAME,
    DROP_ORIGINAL_NOT_FULLY_IN_FRAME,
    DROP_REFERENCE_NOT_ENOUGH_IN_FRAME,
    filter_object_move_occlusion_questions,
)


class FilterObjectMoveOcclusionInFrameTests(unittest.TestCase):
    def _scene_context(self) -> dict:
        return {
            "dataset": "scannetpp",
            "scene_dir": Path("/fake/0d2ee665be"),
            "objects": [
                {"id": 10, "label": "table", "center": [0.0, 0.0, 0.5], "bbox_min": [-0.5, -0.5, 0.0], "bbox_max": [0.5, 0.5, 1.0]},
                {"id": 11, "label": "cup", "center": [0.0, 0.0, 1.1], "bbox_min": [-0.1, -0.1, 1.0], "bbox_max": [0.1, 0.1, 1.2]},
            ],
            "obj_map": {
                10: {"id": 10, "label": "table", "center": [0.0, 0.0, 0.5], "bbox_min": [-0.5, -0.5, 0.0], "bbox_max": [0.5, 0.5, 1.0]},
                11: {"id": 11, "label": "cup", "center": [0.0, 0.0, 1.1], "bbox_min": [-0.1, -0.1, 1.0], "bbox_max": [0.1, 0.1, 1.2]},
            },
            "attachment_graph": {10: [11]},
            "intrinsics": object(),
            "poses": {
                "frame_000000.jpg": CameraPose(
                    image_name="frame_000000.jpg",
                    rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    translation=[0.0, 0.0, 0.0],
                )
            },
        }

    def _question(self) -> dict:
        return {
            "scene_id": "0d2ee665be",
            "image_name": "frame_000000.jpg",
            "level": "L2",
            "type": "object_move_occlusion",
            "trace_question_id": "q1",
            "moved_obj_id": 10,
            "target_obj_id": 11,
            "query_obj_id": 11,
            "delta": [3.0, 0.0, 0.0],
            "_source_benchmark": "output/scannetpp_polit/0-9/benchmark.json",
        }

    def test_filter_drops_when_original_target_not_fully_in_frame(self) -> None:
        payload = {"questions": [self._question()]}
        with (
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._load_scene_context",
                return_value=self._scene_context(),
            ),
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._bbox_fully_in_frame",
                return_value=False,
            ),
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._bbox_in_frame_corner_count",
                return_value=(7, 8),
            ),
        ):
            filtered_payload, report = filter_object_move_occlusion_questions(
                payload,
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
                progress_every=1,
            )

        self.assertEqual(filtered_payload["questions"], [])
        self.assertEqual(report["dropped_count"], 1)
        self.assertEqual(report["dropped"][0]["drop_reason"], DROP_ORIGINAL_NOT_FULLY_IN_FRAME)

    def test_filter_drops_when_counterfactual_target_has_fewer_than_six_corners(self) -> None:
        payload = {"questions": [self._question()]}
        moved_objects = [
            {"id": 10, "label": "table", "center": [3.0, 0.0, 0.5], "bbox_min": [2.5, -0.5, 0.0], "bbox_max": [3.5, 0.5, 1.0]},
            {"id": 11, "label": "cup", "center": [3.0, 0.0, 1.1], "bbox_min": [2.9, -0.1, 1.0], "bbox_max": [3.1, 0.1, 1.2]},
        ]
        with (
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._load_scene_context",
                return_value=self._scene_context(),
            ),
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._bbox_fully_in_frame",
                return_value=True,
            ),
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame.apply_movement",
                return_value=moved_objects,
            ),
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._bbox_in_frame_corner_count",
                side_effect=[(8, 8), (5, 8)],
            ),
        ):
            filtered_payload, report = filter_object_move_occlusion_questions(
                payload,
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
                progress_every=1,
            )

        self.assertEqual(filtered_payload["questions"], [])
        self.assertEqual(report["dropped_count"], 1)
        self.assertEqual(report["dropped"][0]["drop_reason"], DROP_COUNTERFACTUAL_NOT_ENOUGH_IN_FRAME)

    def test_v2_filter_checks_query_and_reference_from_frame_1(self) -> None:
        question = {
            **self._question(),
            "occlusion_semantics_version": 2,
            "obj_ref_id": 12,
        }
        context = self._scene_context()
        reference = {
            "id": 12,
            "label": "sofa",
            "center": [0.4, 0.0, 1.0],
            "bbox_min": [0.2, -0.2, 0.8],
            "bbox_max": [0.6, 0.2, 1.2],
        }
        context["objects"].append(reference)
        context["obj_map"][12] = reference

        with (
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._load_scene_context",
                return_value=context,
            ),
            mock.patch(
                "scripts.filter_object_move_occlusion_in_frame._bbox_in_frame_corner_count",
                side_effect=[(8, 8), (5, 8)],
            ),
        ):
            filtered_payload, report = filter_object_move_occlusion_questions(
                {"questions": [question]},
                scannet_root=Path("/unused/scannet"),
                scannetpp_root=Path("/unused/scannetpp"),
                progress_every=1,
            )

        self.assertEqual(filtered_payload["questions"], [])
        self.assertEqual(report["dropped"][0]["drop_reason"], DROP_REFERENCE_NOT_ENOUGH_IN_FRAME)


if __name__ == "__main__":
    unittest.main()
