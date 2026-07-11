import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    DISTANCE_MOVE_DIRECTIONS,
    MOVEMENT_CANDIDATES,
    _bbox_has_min_in_frame_corners,
    _bbox_fully_in_frame,
    _counterfactual_occlusion_backend,
    _find_object_move_occlusion_changes,
    _find_occlusion_directed_delta_for_occluder,
    _iter_occlusion_directed_object_move_states,
    _make_l1_occlusion_metrics,
    _match_world_xy_direction_bin,
    _occluder_blocks_translated_query_object,
    _select_l2_object_move_occlusion_records,
    _select_object_move_state,
    _select_occlusion_directed_occluder_candidates,
    generate_l2_object_move,
)
from src.utils.colmap_loader import CameraIntrinsics, CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="test.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=320,
        height=240,
        fx=200.0,
        fy=200.0,
        cx=160.0,
        cy=120.0,
    )


def make_object(
    obj_id: int,
    label: str,
    center: tuple[float, float, float],
) -> dict:
    x, y, z = center
    return {
        "id": obj_id,
        "label": label,
        "center": [x, y, z],
        "bbox_min": [x - 0.1, y - 0.1, z - 0.1],
        "bbox_max": [x + 0.1, y + 0.1, z + 0.1],
    }


def make_l1_metrics(status: str):
    if status == "not occluded":
        return _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.0,
            valid_in_frame_count=512,
            sampled_point_count=512,
            in_frame_sample_count=512,
            backend="mesh_ray",
        )
    if status == "occluded":
        return _make_l1_occlusion_metrics(
            projected_area=500.0,
            in_frame_ratio=1.0,
            occlusion_ratio_in_frame=0.2,
            valid_in_frame_count=512,
            sampled_point_count=512,
            in_frame_sample_count=512,
            backend="mesh_ray",
        )
    if status == "not visible":
        return _make_l1_occlusion_metrics(
            projected_area=0.0,
            in_frame_ratio=0.0,
            occlusion_ratio_in_frame=1.0,
            valid_in_frame_count=0,
            sampled_point_count=512,
            in_frame_sample_count=0,
            backend="mesh_ray",
        )
    raise ValueError(f"Unsupported test status: {status}")


class L2ObjectMoveOcclusionTests(unittest.TestCase):
    def test_bbox_fully_in_frame_accepts_all_corners_in_frame(self) -> None:
        obj = make_object(1, "box", (0.0, 0.0, 2.0))

        self.assertTrue(
            _bbox_fully_in_frame(obj, make_camera_pose(), make_camera_intrinsics())
        )

    def test_bbox_fully_in_frame_rejects_bbox_straddling_image_edge(self) -> None:
        obj = {
            "id": 1,
            "label": "box",
            "center": [1.6, 0.0, 2.0],
            "bbox_min": [1.5, -0.1, 1.9],
            "bbox_max": [1.7, 0.1, 2.1],
        }

        self.assertFalse(
            _bbox_fully_in_frame(obj, make_camera_pose(), make_camera_intrinsics())
        )

    def test_bbox_fully_in_frame_rejects_bbox_partly_behind_camera(self) -> None:
        obj = {
            "id": 1,
            "label": "box",
            "center": [0.0, 0.0, 0.05],
            "bbox_min": [-0.01, -0.01, -0.05],
            "bbox_max": [0.01, 0.01, 0.15],
        }

        self.assertFalse(
            _bbox_fully_in_frame(obj, make_camera_pose(), make_camera_intrinsics())
        )

    def test_bbox_fully_in_frame_uses_distorted_projection(self) -> None:
        obj = make_object(1, "box", (0.0, 0.0, 2.0))
        intrinsics = CameraIntrinsics(
            width=320,
            height=240,
            fx=200.0,
            fy=200.0,
            cx=160.0,
            cy=120.0,
            distortion_model="OPENCV_FISHEYE",
            distortion_params=np.array(
                [-0.0313, -0.0037, -0.0024, -7.6e-7],
                dtype=np.float64,
            ),
        )

        self.assertTrue(_bbox_fully_in_frame(obj, make_camera_pose(), intrinsics))

    def test_bbox_has_min_in_frame_corners_accepts_six_visible_corners(self) -> None:
        obj = make_object(1, "box", (0.0, 0.0, 2.0))
        projected_records = [
            {"in_frame": True},
            {"in_frame": True},
            {"in_frame": True},
            {"in_frame": True},
            {"in_frame": True},
            {"in_frame": True},
            {"in_frame": False},
            {"in_frame": False},
        ]

        with patch(
            "src.qa_generator._project_sample_point_records",
            return_value=projected_records,
        ):
            self.assertTrue(
                _bbox_has_min_in_frame_corners(
                    obj,
                    make_camera_pose(),
                    make_camera_intrinsics(),
                    min_corners=6,
                )
            )
            self.assertFalse(
                _bbox_has_min_in_frame_corners(
                    obj,
                    make_camera_pose(),
                    make_camera_intrinsics(),
                    min_corners=7,
                )
            )

    def test_counterfactual_occlusion_backend_rejects_unsupported_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy_backend"):
            _counterfactual_occlusion_backend(
                "legacy_backend",
                ray_caster=object(),
                instance_mesh_data=SimpleNamespace(),
            )

    def test_select_l2_object_move_occlusion_records_caps_unchanged_at_changed_quarter(self) -> None:
        records = [
            {"candidate_index": 0, "relation_unchanged": False},
            {"candidate_index": 1, "relation_unchanged": False},
            {"candidate_index": 2, "relation_unchanged": False},
            {"candidate_index": 3, "relation_unchanged": False},
            {"candidate_index": 4, "relation_unchanged": True},
            {"candidate_index": 5, "relation_unchanged": True},
            {"candidate_index": 6, "relation_unchanged": False},
        ]

        selected = _select_l2_object_move_occlusion_records(records)

        self.assertEqual(
            [record["candidate_index"] for record in selected],
            [0, 1, 2, 3, 4, 6],
        )

    def test_find_object_move_occlusion_changes_tracks_l1_style_changes_for_moved_targets_only(self) -> None:
        objects = [
            make_object(1, "sofa", (0.0, 0.0, 2.0)),
            make_object(2, "cushion", (0.2, 0.0, 2.0)),
            make_object(3, "television", (1.0, 0.0, 2.0)),
            make_object(4, "lamp", (0.4, 0.0, 2.0)),
        ]

        with (
            patch(
                "src.qa_generator._counterfactual_occlusion_backend",
                return_value="mesh_ray",
            ),
            patch(
                "src.qa_generator._build_modified_scene",
                return_value=object(),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not visible"), "mesh_ray"),
                    (make_l1_metrics("occluded"), "mesh_ray"),
                ],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
                side_effect=[
                    (make_l1_metrics("not visible"), "mesh_ray"),
                    (make_l1_metrics("occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ),
        ):
            changes = _find_object_move_occlusion_changes(
                original_objects=objects,
                moved_objects=objects,
                moved_ids={1, 2, 4},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0]["target_obj_id"], 1)
        self.assertEqual(changes[0]["target_obj_label"], "sofa")
        self.assertEqual(changes[0]["old"]["visibility_status"], "not occluded")
        self.assertEqual(changes[0]["new"]["visibility_status"], "not visible")
        self.assertEqual(changes[1]["target_obj_id"], 4)
        self.assertEqual(changes[1]["target_obj_label"], "lamp")
        self.assertEqual(changes[1]["old"]["visibility_status"], "occluded")
        self.assertEqual(changes[1]["new"]["visibility_status"], "not occluded")

    def test_find_object_move_occlusion_changes_skips_tiny_projected_moved_bbox(self) -> None:
        objects = [make_object(1, "pin", (0.0, 0.0, 2.0))]
        moved_objects = [dict(objects[0])]
        moved_objects[0]["bbox_min"] = [-0.005, -0.005, 2.0]
        moved_objects[0]["bbox_max"] = [0.005, 0.005, 2.01]

        with (
            patch(
                "src.qa_generator._counterfactual_occlusion_backend",
                return_value="mesh_ray",
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
            ) as static_mock,
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
            ) as moved_mock,
        ):
            changes = _find_object_move_occlusion_changes(
                original_objects=objects,
                moved_objects=moved_objects,
                moved_ids={1},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                precomputed_original_visibility={
                    1: (
                        "not occluded",
                        "mesh_ray",
                        "resolved_visibility",
                        make_l1_metrics("not occluded"),
                    ),
                },
            )

        self.assertEqual(changes, [])
        static_mock.assert_not_called()
        moved_mock.assert_not_called()

    def test_select_object_move_state_keeps_attachment_fallback_when_no_meaningful_delta(self) -> None:
        objects = [
            make_object(1, "bed", (0.0, 0.0, 2.0)),
            make_object(2, "pillow", (0.2, 0.0, 2.1)),
        ]
        fallback_state = object()

        with (
            patch("src.qa_generator._find_object_move_delta_and_changes", return_value=(None, [])),
            patch("src.qa_generator._first_valid_object_move_state", return_value=fallback_state) as fallback_mock,
        ):
            selected_state = _select_object_move_state(
                objects,
                attachment_graph={1: [2]},
                target_id=1,
                camera_pose=make_camera_pose(),
                allow_unchanged_attachment=True,
            )

        self.assertIs(selected_state, fallback_state)
        fallback_mock.assert_called_once()

    def test_generate_l2_object_move_agent_distance_filter_skips_occlusion_helpers(self) -> None:
        objects = [
            make_object(1, "sofa", (0.0, 0.0, 2.0)),
            make_object(2, "cushion", (0.2, 0.0, 2.0)),
        ]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator._find_object_move_occlusion_changes") as find_mock,
            patch("src.qa_generator._query_visibility_for_object_move_state") as query_mock,
            patch("src.qa_generator._compute_l1_style_visibility_metrics_for_static_target") as static_mock,
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={},
                attached_by={},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={
                    "object_move_agent",
                    "object_move_distance",
                },
            )

        self.assertEqual(questions, [])
        find_mock.assert_not_called()
        query_mock.assert_not_called()
        static_mock.assert_not_called()

    def test_generate_l2_object_move_distance_filter_skips_agent_and_occlusion(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 2.0)),
            make_object(2, "chair", (1.0, 0.0, 2.0)),
        ]
        distance_question = {
            "level": "L2",
            "type": "object_move_distance",
            "question": "distance?",
            "options": ["near", "far"],
            "answer": "near",
        }

        def fake_distance_questions(**kwargs):
            if kwargs["move_source_id"] == 1:
                return [dict(distance_question)]
            return []

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator._select_object_move_state") as select_mock,
            patch("src.qa_generator._find_object_move_occlusion_changes") as find_mock,
            patch("src.qa_generator._query_visibility_for_object_move_state") as query_mock,
            patch("src.qa_generator._compute_l1_style_visibility_metrics_for_static_target") as static_mock,
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                side_effect=fake_distance_questions,
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={"object_move_distance"},
            )

        self.assertEqual([q["type"] for q in questions], ["object_move_distance"])
        select_mock.assert_not_called()
        find_mock.assert_not_called()
        query_mock.assert_not_called()
        static_mock.assert_not_called()

    def test_generate_l2_object_move_emits_single_target_l1_style_occlusion_question(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                    "not visible",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not visible"),
                ),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        occlusion_questions = [
            q for q in questions
            if q.get("type") == "object_move_occlusion"
        ]
        self.assertEqual(len(occlusion_questions), 1)
        question = occlusion_questions[0]
        self.assertEqual(question["correct_value"], "not in the frame")
        self.assertEqual(set(question["options"]), {"not occluded", "occluded", "not in the frame"})
        self.assertEqual(len(question["options"]), 3)
        self.assertEqual(question["moved_obj_label"], "sofa")
        self.assertTrue(question["attachment_remapped"])
        self.assertEqual(question["query_obj_label"], "cushion")
        self.assertEqual(question["target_obj_label"], "cushion")
        self.assertEqual(question["obj_b_label"], "cushion")
        self.assertNotIn("obj_c_id", question)
        self.assertNotIn("relative to", question["question"])
        self.assertNotIn("attachment", question["question"].lower())
        self.assertIn("blocked by another object", question["question"])
        self.assertIn("does not count as occlusion", question["question"])

    def test_generate_l2_object_move_selects_shared_state_without_occlusion_precompute(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
            changed_relations=[],
            used_changed_delta=True,
        )

        def select_state_side_effect(*args, **kwargs):
            self.assertNotIn("color_intrinsics", kwargs)
            self.assertNotIn("ray_caster", kwargs)
            self.assertNotIn("instance_mesh_data", kwargs)
            self.assertNotIn("precomputed_original_visibility", kwargs)
            self.assertNotIn("occlusion_backend", kwargs)
            return selected_state

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                side_effect=select_state_side_effect,
            ) as select_mock,
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                    "not visible",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not visible"),
                ),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        self.assertEqual(select_mock.call_count, 1)
        self.assertEqual(
            [q.get("type") for q in questions if q.get("type") == "object_move_occlusion"],
            ["object_move_occlusion"],
        )

    def test_generate_l2_object_move_skips_occlusion_when_target_bbox_not_fully_in_frame(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = {
            "id": 2,
            "label": "cushion",
            "center": [1.6, 0.0, 2.0],
            "bbox_min": [1.5, -0.1, 1.9],
            "bbox_max": [1.7, 0.1, 2.1],
        }
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )
        trace_events: list[dict] = []

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
                return_value=(make_l1_metrics("not visible"), "mesh_ray"),
            ) as moved_visibility_mock,
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                attachment_referable_object_ids=[1],
                attachment_query_objects=[cushion],
                trace_recorder=trace_events.append,
                trace_detail="full",
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        self.assertFalse(any(q.get("type") == "object_move_occlusion" for q in questions))
        moved_visibility_mock.assert_not_called()
        self.assertTrue(
            any(
                event.get("event") == "generator_candidate"
                and event.get("candidate_kind") == "object_move_occlusion_target"
                and event.get("candidate_key") == "1:2"
                and event.get("reason_code") == "occlusion_target_not_fully_in_frame"
                for event in trace_events
            )
        )

    def test_generate_l2_object_move_skips_occlusion_when_target_moves_out_of_frame(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        moved_objects = [
            sofa,
            {
                **cushion,
                "center": [1.4, 0.0, 2.0],
                "bbox_min": [1.3, -0.1, 1.9],
                "bbox_max": [1.5, 0.1, 2.1],
            },
        ]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=moved_objects,
            moved_ids={1, 2},
            changed_relations=[],
            used_changed_delta=True,
        )
        trace_events: list[dict] = []

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
            patch(
                "src.qa_generator._bbox_has_min_in_frame_corners",
                return_value=True,
            ),
            patch(
                "src.qa_generator._bbox_in_frame_corner_count",
                side_effect=[(5, 8)],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
                return_value=(make_l1_metrics("not visible"), "mesh_ray"),
            ) as moved_visibility_mock,
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                trace_recorder=trace_events.append,
                trace_detail="full",
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        self.assertFalse(any(q.get("type") == "object_move_occlusion" for q in questions))
        moved_visibility_mock.assert_not_called()
        self.assertTrue(
            any(
                event.get("event") == "generator_candidate"
                and event.get("candidate_kind") == "object_move_occlusion_target"
                and event.get("candidate_key") == "1:2"
                and event.get("reason_code") == "occlusion_target_not_enough_in_frame_after_move"
                for event in trace_events
            )
        )

    def test_generate_l2_object_move_skips_unchanged_attachment_occlusion_state(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                side_effect=[None, selected_state],
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                side_effect=[
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                    (make_l1_metrics("not occluded"), "mesh_ray"),
                ],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertFalse(any(q.get("type") == "object_move_occlusion" for q in questions))

    def test_generate_l2_object_move_uses_query_specific_occlusion_fallback_state(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        shared_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )
        fallback_state = SimpleNamespace(
            delta=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
            used_changed_delta=False,
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=shared_state,
            ),
            patch(
                "src.qa_generator._iter_additional_object_move_states",
                return_value=[fallback_state],
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                side_effect=[
                    (
                        "not occluded",
                        "mesh_ray",
                        "static_visible",
                        make_l1_metrics("not occluded"),
                        "not occluded",
                        "mesh_ray",
                        "counterfactual_visible",
                        make_l1_metrics("not occluded"),
                    ),
                    (
                        "not occluded",
                        "mesh_ray",
                        "static_visible",
                        make_l1_metrics("not occluded"),
                        "not visible",
                        "mesh_ray",
                        "counterfactual_not_visible",
                        make_l1_metrics("not visible"),
                    ),
                ],
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        occlusion_questions = [q for q in questions if q.get("type") == "object_move_occlusion"]
        self.assertEqual(len(occlusion_questions), 1)
        self.assertEqual(occlusion_questions[0]["delta"], [1.0, 0.0, 0.0])
        self.assertEqual(occlusion_questions[0]["correct_value"], "not in the frame")
        self.assertFalse(occlusion_questions[0]["relation_unchanged"])

    def test_generate_l2_object_move_skips_fallback_visibility_when_fallback_target_fails_corner_gate(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        shared_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
            changed_relations=[],
            used_changed_delta=True,
        )
        fallback_state = SimpleNamespace(
            delta=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
            changed_relations=[],
            used_changed_delta=False,
        )
        trace_events: list[dict] = []

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=shared_state,
            ),
            patch(
                "src.qa_generator._iter_additional_object_move_states",
                return_value=[fallback_state],
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._bbox_fully_in_frame",
                return_value=True,
            ),
            patch(
                "src.qa_generator._bbox_in_frame_corner_count",
                side_effect=[(8, 8), (5, 8)],
            ),
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                ),
            ) as visibility_mock,
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                trace_recorder=trace_events.append,
                trace_detail="full",
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        self.assertFalse(any(q.get("type") == "object_move_occlusion" for q in questions))
        self.assertEqual(visibility_mock.call_count, 1)
        self.assertIs(visibility_mock.call_args_list[0].kwargs["selected_state"], shared_state)

    def test_generate_l2_object_move_emits_visible_to_occluded_transition(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                    "occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("occluded"),
                ),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        occlusion_questions = [q for q in questions if q.get("type") == "object_move_occlusion"]
        self.assertEqual(len(occlusion_questions), 1)
        self.assertEqual(occlusion_questions[0]["old_visibility"], "not occluded")
        self.assertEqual(occlusion_questions[0]["new_visibility"], "occluded")
        self.assertEqual(occlusion_questions[0]["correct_value"], "occluded")
        self.assertFalse(occlusion_questions[0]["relation_unchanged"])

    def test_generate_l2_object_move_emits_occluded_to_visible_transition(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("occluded"),
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                ),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        occlusion_questions = [q for q in questions if q.get("type") == "object_move_occlusion"]
        self.assertEqual(len(occlusion_questions), 1)
        self.assertEqual(occlusion_questions[0]["old_visibility"], "occluded")
        self.assertEqual(occlusion_questions[0]["new_visibility"], "not occluded")
        self.assertEqual(occlusion_questions[0]["correct_value"], "not occluded")
        self.assertFalse(occlusion_questions[0]["relation_unchanged"])

    def test_generate_l2_object_move_does_not_reuse_occlusion_changes_across_different_deltas(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 2.0)),
            make_object(2, "cup", (0.2, 0.0, 2.0)),
            make_object(3, "shelf", (1.2, 0.0, 2.0)),
            make_object(4, "book", (1.4, 0.0, 2.0)),
        ]
        base_relations = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "right",
            },
            {
                "obj_a_id": 3,
                "obj_b_id": 4,
                "direction_b_rel_a": "right",
            },
        ]
        moved_relations_box = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "front-right",
            },
            {
                "obj_a_id": 3,
                "obj_b_id": 4,
                "direction_b_rel_a": "right",
            },
        ]
        moved_relations_cup = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "right",
            },
            {
                "obj_a_id": 3,
                "obj_b_id": 4,
                "direction_b_rel_a": "front-right",
            },
        ]
        changed = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "changes": {
                    "direction_b_rel_a": {"old": "right", "new": "front-right"},
                },
                "old": {},
                "new": {},
            }
        ]
        helper_side_effect = [
            (
                np.array([0.5, 0.0, 0.0], dtype=np.float64),
                changed,
            ),
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                changed,
            ),
        ]

        with patch(
            "src.qa_generator._find_object_move_delta_and_changes",
            side_effect=helper_side_effect,
        ) as mocked_helper, patch(
            "src.qa_generator.compute_all_relations",
            side_effect=[
                base_relations,
                moved_relations_box,
                moved_relations_cup,
            ],
        ), patch(
            "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
            side_effect=[
                (make_l1_metrics("not occluded"), "mesh_ray"),
                (make_l1_metrics("not occluded"), "mesh_ray"),
            ],
        ), patch(
            "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
            return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
        ), patch(
            "src.qa_generator._generate_l2_distance_questions_for_object",
            return_value=[],
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2], 3: [4]},
                attached_by={2: 1, 4: 3},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_agent": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: where is {obj_b} relative to {obj_c}?"
                    ],
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ],
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={"object_move_agent"},
            )

        self.assertEqual(mocked_helper.call_count, 2)
        self.assertEqual(
            [tuple(q["delta"]) for q in questions if q.get("type") == "object_move_agent"],
            [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )
        self.assertEqual([args.args[2] for args in mocked_helper.call_args_list], [1, 3])

    def test_generate_l2_object_move_uses_relation_specific_agent_fallback_state(self) -> None:
        table = make_object(1, "table", (0.0, 0.0, 2.0))
        laptop = make_object(2, "laptop", (0.2, 0.0, 2.0))
        chair = make_object(3, "chair", (1.0, 0.0, 2.0))
        objects = [table, laptop, chair]
        shared_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )
        fallback_state = (np.array([1.0, 0.0, 0.0], dtype=np.float64), objects, {1, 2})
        base_relations = [
            {
                "obj_a_id": 3,
                "obj_b_id": 2,
                "direction_b_rel_a": "left",
            }
        ]
        fallback_relations = [
            {
                "obj_a_id": 3,
                "obj_b_id": 2,
                "direction_b_rel_a": "front-left",
            }
        ]

        def select_state_for_source(*args, **kwargs):
            return shared_state if args[2] == 2 else None

        def fallback_states_for_source(*args, **kwargs):
            return [fallback_state] if args[2] == 1 else []

        with patch(
            "src.qa_generator._select_object_move_state",
            side_effect=select_state_for_source,
        ), patch(
            "src.qa_generator._iter_valid_object_move_states",
            side_effect=fallback_states_for_source,
        ), patch(
            "src.qa_generator.compute_all_relations",
            side_effect=[
                base_relations,
                fallback_relations,
                [],
            ],
        ), patch(
            "src.qa_generator._generate_l2_distance_questions_for_object",
            return_value=[],
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_agent": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: where is {obj_b} relative to {obj_c}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
            )

        agent_questions = [q for q in questions if q.get("type") == "object_move_agent"]
        self.assertEqual(len(agent_questions), 1)
        self.assertTrue(agent_questions[0]["attachment_remapped"])
        self.assertEqual(agent_questions[0]["moved_obj_id"], 1)
        self.assertEqual(agent_questions[0]["query_obj_id"], 2)
        self.assertEqual(agent_questions[0]["delta"], [1.0, 0.0, 0.0])
        self.assertEqual(agent_questions[0]["correct_value"], "front-left")


class OcclusionDirectedSearchTests(unittest.TestCase):
    """Tests for the Phase 1.5 occluder-directed candidate search that aims a
    moved object at a specific nearby occluder instead of blindly walking the
    fixed movement grid."""

    def test_match_world_xy_direction_bin_accepts_vector_within_15_degrees(self) -> None:
        # +X axis is DISTANCE_MOVE_DIRECTIONS[0] (see MOVEMENT_CANDIDATES[:8]).
        vector = np.array([1.0, math.tan(math.radians(10.0)), 0.0], dtype=np.float64)

        match = _match_world_xy_direction_bin(vector)

        self.assertIsNotNone(match)
        bin_idx, unit_direction = match
        self.assertEqual(bin_idx, 0)
        np.testing.assert_allclose(unit_direction, DISTANCE_MOVE_DIRECTIONS[0])

    def test_match_world_xy_direction_bin_rejects_vector_between_bins(self) -> None:
        # Exactly at the 22.5-degree bin boundary: the OLD (incorrect) 22.5deg
        # threshold would accept this (cos(22.5deg) is an identity, never
        # rejects for 8 bins spaced 45deg apart); the 15deg threshold used
        # here must reject it. This is a regression guard against
        # accidentally reverting to 22.5deg.
        angle = math.radians(22.5)
        vector = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)

        match = _match_world_xy_direction_bin(vector)

        self.assertIsNone(match)
        # Sanity-check the claim above: cos(22.5deg) would not reject this.
        best_dot = max(
            float(np.dot(vector[:2], np.asarray(direction)[:2]))
            for direction in DISTANCE_MOVE_DIRECTIONS
        )
        self.assertGreaterEqual(best_dot, math.cos(math.radians(22.5)) - 1e-9)

    def test_match_world_xy_direction_bin_rejects_zero_vector(self) -> None:
        self.assertIsNone(_match_world_xy_direction_bin(np.zeros(3)))

    def test_select_occlusion_directed_occluder_candidates_filters_and_sorts(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        near_big = make_object(2, "shelf", (0.5, 0.0, 2.0))
        far_big = make_object(3, "wardrobe", (2.0, 0.0, 2.0))
        small = make_object(4, "pin", (0.3, 0.0, 2.0))
        moved_sibling = make_object(5, "pillow", (0.4, 0.0, 2.0))
        original_visibility = {
            2: (None, "mesh_ray", "", make_l1_metrics("occluded")),
            3: (None, "mesh_ray", "", make_l1_metrics("occluded")),
            4: (
                None,
                "mesh_ray",
                "",
                _make_l1_occlusion_metrics(
                    projected_area=10.0,
                    in_frame_ratio=1.0,
                    occlusion_ratio_in_frame=0.0,
                    valid_in_frame_count=512,
                    sampled_point_count=512,
                    in_frame_sample_count=512,
                    backend="mesh_ray",
                ),
            ),
            5: (None, "mesh_ray", "", make_l1_metrics("occluded")),
        }

        candidates = _select_occlusion_directed_occluder_candidates(
            query_obj=query_obj,
            moved_ids={1, 5},
            occlusion_source_objects=[near_big, far_big, small, moved_sibling],
            original_visibility=original_visibility,
        )

        self.assertEqual([int(obj["id"]) for obj in candidates], [2, 3])

    def test_select_occlusion_directed_occluder_candidates_caps_at_max_candidates(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        occluders = [
            make_object(idx, f"obj{idx}", (float(idx) * 0.5, 0.0, 2.0))
            for idx in range(2, 12)
        ]
        original_visibility = {
            int(obj["id"]): (None, "mesh_ray", "", make_l1_metrics("occluded"))
            for obj in occluders
        }

        candidates = _select_occlusion_directed_occluder_candidates(
            query_obj=query_obj,
            moved_ids=set(),
            occlusion_source_objects=occluders,
            original_visibility=original_visibility,
            max_candidates=3,
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual([int(obj["id"]) for obj in candidates], [2, 3, 4])

    def test_select_occlusion_directed_occluder_candidates_filters_by_max_distance(self) -> None:
        # obj_ref and obj_move need not share a frame (a separate
        # auxiliary-frame mechanism can supplement a second image), so this
        # is a generous distance cutoff, not a "same image" restriction.
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        near = make_object(2, "shelf", (3.0, 0.0, 2.0))
        far = make_object(3, "wardrobe", (7.0, 0.0, 2.0))
        original_visibility = {
            2: (None, "mesh_ray", "", make_l1_metrics("occluded")),
            3: (None, "mesh_ray", "", make_l1_metrics("occluded")),
        }

        candidates = _select_occlusion_directed_occluder_candidates(
            query_obj=query_obj,
            moved_ids=set(),
            occlusion_source_objects=[near, far],
            original_visibility=original_visibility,
            max_distance_m=5.0,
        )

        self.assertEqual([int(obj["id"]) for obj in candidates], [2])

    def test_occluder_blocks_translated_query_object_true_when_occluder_is_closer(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (0.5, 0.0, 1.0))

        with (
            patch(
                "src.qa_generator._get_instance_intersector",
                return_value=object(),
            ),
            patch(
                "src.qa_generator._instance_surface_samples",
                return_value=np.array([[0.5, 0.0, 2.0]], dtype=np.float64),
            ),
            patch(
                "src.qa_generator._batch_first_hit_distances_compat",
                return_value=np.array([1.0], dtype=np.float64),
            ),
        ):
            blocked = _occluder_blocks_translated_query_object(
                obj_ref=obj_ref,
                query_obj=query_obj,
                target_delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
            )

        self.assertTrue(blocked)

    def test_occluder_blocks_translated_query_object_false_when_occluder_is_farther(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (0.5, 0.0, 1.0))

        with (
            patch(
                "src.qa_generator._get_instance_intersector",
                return_value=object(),
            ),
            patch(
                "src.qa_generator._instance_surface_samples",
                return_value=np.array([[0.5, 0.0, 2.0]], dtype=np.float64),
            ),
            patch(
                "src.qa_generator._batch_first_hit_distances_compat",
                return_value=np.array([np.inf], dtype=np.float64),
            ),
        ):
            blocked = _occluder_blocks_translated_query_object(
                obj_ref=obj_ref,
                query_obj=query_obj,
                target_delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
            )

        self.assertFalse(blocked)

    def test_occluder_blocks_translated_query_object_false_when_no_instance_caster(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (0.5, 0.0, 1.0))

        with patch("src.qa_generator._get_instance_intersector", return_value=None):
            blocked = _occluder_blocks_translated_query_object(
                obj_ref=obj_ref,
                query_obj=query_obj,
                target_delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
            )

        self.assertFalse(blocked)

    def test_find_occlusion_directed_delta_for_occluder_picks_smallest_passing_magnitude(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (1.0, 0.0, 2.0))
        unit_direction = DISTANCE_MOVE_DIRECTIONS[0]
        moved_objects = [query_obj, obj_ref]

        # obj_ref is 1.0m away along the matched direction, so the scan now
        # starts at max(0.2, 1.0 - reach_margin(0.3)) = 0.7, not at
        # min_magnitude_m -- 0.7 and 1.0 must fail here to actually exercise
        # "picks the smallest PASSING magnitude" rather than just the first
        # one tried.
        def fake_blocks(*, target_delta, **kwargs):
            return float(np.linalg.norm(target_delta)) >= 1.3 - 1e-9

        with (
            patch(
                "src.qa_generator._occluder_blocks_translated_query_object",
                side_effect=fake_blocks,
            ) as blocks_mock,
            patch("src.qa_generator.apply_movement", return_value=moved_objects),
            patch("src.qa_generator.is_within_room", return_value=True),
            patch("src.qa_generator.has_terminal_bbox_collision", return_value=False),
        ):
            selected_state = _find_occlusion_directed_delta_for_occluder(
                query_obj=query_obj,
                obj_ref=obj_ref,
                unit_direction=unit_direction,
                move_source_id=1,
                moved_ids={1},
                movement_scene_objects=[query_obj, obj_ref],
                attachment_graph={},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
                room_min=np.array([-10.0, -10.0, -10.0]),
                room_max=np.array([10.0, 10.0, 10.0]),
                collision_objects=None,
            )

        self.assertIsNotNone(selected_state)
        self.assertAlmostEqual(float(np.linalg.norm(selected_state.delta)), 1.3, places=6)
        self.assertEqual(blocks_mock.call_count, 3)
        self.assertGreater(blocks_mock.call_count, 0)

    def test_find_occlusion_directed_delta_for_occluder_returns_none_when_no_magnitude_passes(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (1.0, 0.0, 2.0))

        with patch(
            "src.qa_generator._occluder_blocks_translated_query_object",
            return_value=False,
        ):
            selected_state = _find_occlusion_directed_delta_for_occluder(
                query_obj=query_obj,
                obj_ref=obj_ref,
                unit_direction=DISTANCE_MOVE_DIRECTIONS[0],
                move_source_id=1,
                moved_ids={1},
                movement_scene_objects=[query_obj, obj_ref],
                attachment_graph={},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
                room_min=np.array([-10.0, -10.0, -10.0]),
                room_max=np.array([10.0, 10.0, 10.0]),
                collision_objects=None,
            )

        self.assertIsNone(selected_state)

    def test_find_occlusion_directed_delta_for_occluder_skips_without_ray_casting_when_reach_exceeds_max_magnitude(self) -> None:
        # obj_ref is 6m away along the matched direction, past max_magnitude_m
        # (5.0 default) -- query_obj can never move far enough to get behind
        # obj_ref within the search budget, so this must be rejected up front
        # without a single ray-cast call.
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (6.0, 0.0, 2.0))

        with patch(
            "src.qa_generator._occluder_blocks_translated_query_object",
        ) as blocks_mock:
            selected_state = _find_occlusion_directed_delta_for_occluder(
                query_obj=query_obj,
                obj_ref=obj_ref,
                unit_direction=DISTANCE_MOVE_DIRECTIONS[0],
                move_source_id=1,
                moved_ids={1},
                movement_scene_objects=[query_obj, obj_ref],
                attachment_graph={},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
                room_min=np.array([-10.0, -10.0, -10.0]),
                room_max=np.array([10.0, 10.0, 10.0]),
                collision_objects=None,
            )

        self.assertIsNone(selected_state)
        blocks_mock.assert_not_called()

    def test_find_occlusion_directed_delta_for_occluder_uses_snapped_direction_not_raw_vector_for_reach(self) -> None:
        # obj_ref's raw offset from query_obj is (4.9, 2.0, 0) -- within
        # 5.0m euclidean distance, but its projection onto the SNAPPED +X
        # direction is only 4.9m (< 5.0m max), so the reach check must use
        # that projection, not the ~5.29m raw euclidean distance, to decide
        # whether to proceed.
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (4.9, 2.0, 2.0))
        raw_vector = np.asarray(obj_ref["center"]) - np.asarray(query_obj["center"])
        self.assertGreater(float(np.linalg.norm(raw_vector)), 5.0)

        with patch(
            "src.qa_generator._occluder_blocks_translated_query_object",
            return_value=False,
        ) as blocks_mock:
            selected_state = _find_occlusion_directed_delta_for_occluder(
                query_obj=query_obj,
                obj_ref=obj_ref,
                unit_direction=DISTANCE_MOVE_DIRECTIONS[0],
                move_source_id=1,
                moved_ids={1},
                movement_scene_objects=[query_obj, obj_ref],
                attachment_graph={},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
                room_min=np.array([-10.0, -10.0, -10.0]),
                room_max=np.array([10.0, 10.0, 10.0]),
                collision_objects=None,
            )

        self.assertIsNone(selected_state)
        # The projected reach (4.9m) is within budget, so the scan must have
        # actually run (not been skipped by the raw-distance check).
        blocks_mock.assert_called()

    def test_find_occlusion_directed_delta_for_occluder_rejects_delta_failing_room_bounds(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        obj_ref = make_object(2, "shelf", (1.0, 0.0, 2.0))

        with (
            patch(
                "src.qa_generator._occluder_blocks_translated_query_object",
                return_value=True,
            ),
            patch("src.qa_generator.apply_movement", return_value=[query_obj, obj_ref]),
            patch("src.qa_generator.is_within_room", return_value=False),
        ):
            selected_state = _find_occlusion_directed_delta_for_occluder(
                query_obj=query_obj,
                obj_ref=obj_ref,
                unit_direction=DISTANCE_MOVE_DIRECTIONS[0],
                move_source_id=1,
                moved_ids={1},
                movement_scene_objects=[query_obj, obj_ref],
                attachment_graph={},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=object(),
                room_min=np.array([-10.0, -10.0, -10.0]),
                room_max=np.array([10.0, 10.0, 10.0]),
                collision_objects=None,
            )

        self.assertIsNone(selected_state)

    def test_iter_occlusion_directed_object_move_states_excludes_moved_ids_and_direction_mismatches(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        # Directly "north" (+Y direction) of query_obj: matches a canonical bin.
        aligned_occluder = make_object(2, "shelf", (0.0, 1.0, 2.0))
        # 20 degrees off the nearest canonical bin (bins are 45deg apart, so
        # this is also 25deg off the next-nearest bin): should be rejected by
        # the direction filter and never reach the (mocked) delta search.
        misaligned_occluder = make_object(3, "lamp", (
            math.cos(math.radians(20.0)),
            math.sin(math.radians(20.0)),
            2.0,
        ))
        moved_sibling = make_object(4, "pillow", (0.0, 1.0, 2.0))
        original_visibility = {
            2: (None, "mesh_ray", "", make_l1_metrics("occluded")),
            3: (None, "mesh_ray", "", make_l1_metrics("occluded")),
            4: (None, "mesh_ray", "", make_l1_metrics("occluded")),
        }
        fake_state = SimpleNamespace(delta=np.array([0.0, 0.8, 0.0]))

        with patch(
            "src.qa_generator._find_occlusion_directed_delta_for_occluder",
            return_value=fake_state,
        ) as delta_mock:
            states = list(
                _iter_occlusion_directed_object_move_states(
                    query_obj=query_obj,
                    move_source_id=1,
                    moved_ids={1, 4},
                    movement_scene_objects=[query_obj, aligned_occluder, misaligned_occluder, moved_sibling],
                    occlusion_source_objects=[aligned_occluder, misaligned_occluder, moved_sibling],
                    original_visibility=original_visibility,
                    attachment_graph={},
                    camera_pose=make_camera_pose(),
                    color_intrinsics=make_camera_intrinsics(),
                    instance_mesh_data=object(),
                    room_min=np.array([-10.0, -10.0, -10.0]),
                    room_max=np.array([10.0, 10.0, 10.0]),
                    collision_objects=None,
                )
            )

        self.assertEqual(len(states), 1)
        self.assertIs(states[0], fake_state)
        # Only the aligned occluder should ever reach the delta search.
        self.assertEqual(delta_mock.call_count, 1)
        self.assertEqual(int(delta_mock.call_args.kwargs["obj_ref"]["id"]), 2)

    def test_iter_occlusion_directed_object_move_states_stops_when_candidate_budget_exhausted(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        nearer_occluder = make_object(2, "shelf", (0.0, 1.0, 2.0))
        farther_occluder = make_object(3, "cabinet", (0.0, 2.0, 2.0))
        original_visibility = {
            2: (None, "mesh_ray", "", make_l1_metrics("occluded")),
            3: (None, "mesh_ray", "", make_l1_metrics("occluded")),
        }
        counter = [0]

        with patch(
            "src.qa_generator._find_occlusion_directed_delta_for_occluder",
            return_value=None,
        ) as delta_mock:
            states = list(
                _iter_occlusion_directed_object_move_states(
                    query_obj=query_obj,
                    move_source_id=1,
                    moved_ids={1},
                    movement_scene_objects=[query_obj, nearer_occluder, farther_occluder],
                    occlusion_source_objects=[nearer_occluder, farther_occluder],
                    original_visibility=original_visibility,
                    attachment_graph={},
                    camera_pose=make_camera_pose(),
                    color_intrinsics=make_camera_intrinsics(),
                    instance_mesh_data=object(),
                    room_min=np.array([-10.0, -10.0, -10.0]),
                    room_max=np.array([10.0, 10.0, 10.0]),
                    collision_objects=None,
                    candidates_tried_counter=counter,
                    max_candidates_tried=1,
                )
            )

        self.assertEqual(states, [])
        # Only the nearer occluder (tried first) should be attempted; the
        # farther one is skipped once the shared counter hits the budget.
        self.assertEqual(delta_mock.call_count, 1)
        self.assertEqual(int(delta_mock.call_args.kwargs["obj_ref"]["id"]), 2)
        self.assertEqual(counter[0], 1)


class OcclusionDirectedIntegrationTests(unittest.TestCase):
    """Integration tests confirming Phase 1.5 is wired into
    generate_l2_object_move with the right priority and guardrails."""

    def test_generate_l2_object_move_emits_occlusion_directed_not_occluded_to_occluded_transition(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )
        directed_delta = np.array([0.37, 0.0, 0.0], dtype=np.float64)
        directed_state = SimpleNamespace(
            delta=directed_delta,
            moved_objects=objects,
            moved_ids={1, 2},
            used_changed_delta=False,
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._iter_occlusion_directed_object_move_states",
                return_value=[directed_state],
            ) as directed_mock,
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                side_effect=[
                    (
                        "not occluded",
                        "mesh_ray",
                        "resolved_visibility",
                        make_l1_metrics("not occluded"),
                        "not occluded",
                        "mesh_ray",
                        "resolved_visibility",
                        make_l1_metrics("not occluded"),
                    ),
                    (
                        "not occluded",
                        "mesh_ray",
                        "resolved_visibility",
                        make_l1_metrics("not occluded"),
                        "occluded",
                        "mesh_ray",
                        "resolved_visibility",
                        make_l1_metrics("occluded"),
                    ),
                ],
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        self.assertTrue(directed_mock.called)
        occlusion_questions = [q for q in questions if q.get("type") == "object_move_occlusion"]
        self.assertEqual(len(occlusion_questions), 1)
        self.assertEqual(occlusion_questions[0]["old_visibility"], "not occluded")
        self.assertEqual(occlusion_questions[0]["new_visibility"], "occluded")
        self.assertEqual(occlusion_questions[0]["delta"], directed_delta.tolist())
        # The directed delta is not one of the 48 canonical grid deltas.
        self.assertNotIn(
            tuple(np.round(directed_delta, 6).tolist()),
            {tuple(np.round(d, 6).tolist()) for d in MOVEMENT_CANDIDATES},
        )

    def test_generate_l2_object_move_stops_occlusion_directed_search_once_scene_budget_exhausted(self) -> None:
        # Two independent attachment pairs in the same scene, both eligible
        # for Phase 1.5. With the scene-wide changed-question budget patched
        # to 1, only the first pair should ever reach
        # _iter_occlusion_directed_object_move_states.
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        table = make_object(3, "table", (5.0, 0.0, 2.0))
        lamp = make_object(4, "lamp", (5.2, 0.0, 2.0))
        objects = [sofa, cushion, table, lamp]
        directed_state = SimpleNamespace(
            delta=np.array([0.37, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
            used_changed_delta=False,
        )

        with (
            patch("src.qa_generator._select_object_move_state", return_value=None),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch("src.qa_generator._bbox_fully_in_frame", return_value=True),
            patch("src.qa_generator._bbox_in_frame_corner_count", return_value=(8, 8)),
            # Isolate Phase 1.5 from Phase 2's blind fallback -- otherwise the
            # second pair would still get an occlusion question through the
            # blind grid (since _query_visibility_for_object_move_state below
            # is mocked to always succeed), masking whether Phase 1.5 itself
            # actually stopped once the scene budget was exhausted.
            patch("src.qa_generator._iter_additional_object_move_states", return_value=[]),
            patch(
                "src.qa_generator._iter_occlusion_directed_object_move_states",
                return_value=[directed_state],
            ) as directed_mock,
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "not occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("not occluded"),
                    "occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("occluded"),
                ),
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
            patch("src.qa_generator.OCCLUSION_DIRECTED_MAX_CHANGED_QUESTIONS_PER_SCENE", 1),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2], 3: [4]},
                attached_by={2: 1, 4: 3},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        occlusion_questions = [q for q in questions if q.get("type") == "object_move_occlusion"]
        self.assertEqual(len(occlusion_questions), 1)
        self.assertEqual(directed_mock.call_count, 1)

    def test_generate_l2_object_move_skips_occlusion_directed_search_when_original_status_not_not_occluded(self) -> None:
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        objects = [sofa, cushion]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=objects,
            moved_ids={1, 2},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                return_value=selected_state,
            ),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("occluded"), "mesh_ray"),
            ),
            patch(
                "src.qa_generator._iter_occlusion_directed_object_move_states",
            ) as directed_mock,
            patch(
                "src.qa_generator._query_visibility_for_object_move_state",
                return_value=(
                    "occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("occluded"),
                    "occluded",
                    "mesh_ray",
                    "resolved_visibility",
                    make_l1_metrics("occluded"),
                ),
            ),
            patch(
                "src.qa_generator._iter_additional_object_move_states",
                return_value=[],
            ),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "move {obj_a} {direction_with_camera_hint} by {distance}: what is the occlusion status of {obj_b}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        directed_mock.assert_not_called()

    def test_generate_l2_object_move_skips_occlusion_directed_search_when_occlusion_disabled(self) -> None:
        objects = [
            make_object(1, "sofa", (0.0, 0.0, 2.0)),
            make_object(2, "cushion", (0.2, 0.0, 2.0)),
        ]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._iter_occlusion_directed_object_move_states",
            ) as directed_mock,
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                return_value=[],
            ),
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        directed_mock.assert_not_called()
        self.assertFalse(any(q.get("type") == "object_move_occlusion" for q in questions))


if __name__ == "__main__":
    unittest.main()
