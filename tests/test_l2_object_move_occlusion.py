import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    DISTANCE_MOVE_DIRECTIONS,
    L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER,
    L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF,
    L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
    MAX_OCCLUSION_OBJECTS_AUTO,
    MOVEMENT_CANDIDATES,
    _aabb_extent_along_direction,
    _adaptive_occlusion_directed_step,
    _adaptive_scene_scaled_cap,
    _bbox_has_min_in_frame_corners,
    _bbox_fully_in_frame,
    _counterfactual_occlusion_backend,
    _find_object_move_occlusion_changes,
    _find_occlusion_directed_delta_for_occluder,
    _iter_occlusion_directed_object_move_states,
    _make_l1_occlusion_metrics,
    _match_world_xy_direction_bin,
    _match_world_xy_direction_bins,
    _occluder_blocks_translated_query_object,
    _pairwise_occlusion_relation_after_move,
    _iter_pairwise_occlusion_directed_object_move_states,
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

    def test_select_l2_object_move_occlusion_records_keeps_all_post_move_records(self) -> None:
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
            [0, 1, 2, 3, 4, 5, 6],
        )

    def test_select_l2_object_move_occlusion_records_does_not_balance_by_old_state(self) -> None:
        records = [
            {"candidate_index": 0, "relation_unchanged": True},
            {"candidate_index": 1, "relation_unchanged": True},
        ]

        selected = _select_l2_object_move_occlusion_records(records)

        self.assertEqual([record["candidate_index"] for record in selected], [0, 1])

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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    def test_match_world_xy_direction_bin_accepts_vector_within_22_5_degrees(self) -> None:
        # +X axis is DISTANCE_MOVE_DIRECTIONS[0] (see MOVEMENT_CANDIDATES[:8]).
        vector = np.array([1.0, math.tan(math.radians(20.0)), 0.0], dtype=np.float64)

        match = _match_world_xy_direction_bin(vector)

        self.assertIsNotNone(match)
        bin_idx, unit_direction = match
        self.assertEqual(bin_idx, 0)
        np.testing.assert_allclose(unit_direction, DISTANCE_MOVE_DIRECTIONS[0])

    def test_match_world_xy_direction_bins_returns_both_at_exact_boundary(self) -> None:
        angle = math.radians(22.5)
        vector = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)

        matches = _match_world_xy_direction_bins(vector)

        self.assertEqual([idx for idx, _direction in matches], [0, 4])
        np.testing.assert_allclose(matches[0][1], DISTANCE_MOVE_DIRECTIONS[0])
        np.testing.assert_allclose(matches[1][1], DISTANCE_MOVE_DIRECTIONS[4])

    def test_match_world_xy_direction_bins_returns_only_nearest_around_boundary(self) -> None:
        below = math.radians(22.5 - 1e-4)
        above = math.radians(22.5 + 1e-4)

        below_matches = _match_world_xy_direction_bins(
            np.array([math.cos(below), math.sin(below), 0.0], dtype=np.float64)
        )
        above_matches = _match_world_xy_direction_bins(
            np.array([math.cos(above), math.sin(above), 0.0], dtype=np.float64)
        )

        self.assertEqual([idx for idx, _direction in below_matches], [0])
        self.assertEqual([idx for idx, _direction in above_matches], [4])

    def test_match_world_xy_direction_bin_rejects_zero_vector(self) -> None:
        self.assertIsNone(_match_world_xy_direction_bin(np.zeros(3)))
        self.assertEqual(_match_world_xy_direction_bins(np.zeros(3)), ())

    def test_adaptive_scene_scaled_cap_scales_linearly_and_clamps(self) -> None:
        # Below min_cap: clamps up to min_cap.
        self.assertEqual(
            _adaptive_scene_scaled_cap(5, fraction=1.0, min_cap=20, max_cap=100), 20
        )
        # Within range: scales linearly (rounded).
        self.assertEqual(
            _adaptive_scene_scaled_cap(40, fraction=1.0, min_cap=20, max_cap=100), 40
        )
        self.assertEqual(
            _adaptive_scene_scaled_cap(41, fraction=0.5, min_cap=20, max_cap=100), 20
        )
        # Above max_cap: clamps down to max_cap.
        self.assertEqual(
            _adaptive_scene_scaled_cap(500, fraction=1.0, min_cap=20, max_cap=100), 100
        )

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

    def test_select_occlusion_directed_occluder_candidates_default_cap_scales_with_pool_size(self) -> None:
        # With max_candidates left unset, a pool bigger than the old fixed
        # cap of 20 should no longer be truncated to 20 -- the adaptive
        # default should scale with the pool size instead.
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        occluders = [
            make_object(idx, f"obj{idx}", (float(idx) * 0.1, 0.0, 2.0))
            for idx in range(2, 32)  # 30 candidates, all within range
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
        )

        self.assertEqual(len(candidates), 30)
        self.assertGreater(len(candidates), 20)

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
            patch("src.qa_generator._projection_rects_overlap", return_value=True),
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

        # obj_ref is 1.0m away along the matched direction; step_m/
        # reach_margin_m are passed explicitly (0.3, matching the old fixed
        # defaults) so this test's scan sequence stays deterministic and
        # independent of the adaptive, object-size-based step computation
        # (covered separately below) -- scan starts at
        # max(0.2, 1.0 - 0.3) = 0.7, so 0.7 and 1.0 must fail here to
        # actually exercise "picks the smallest PASSING magnitude" rather
        # than just the first one tried.
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
                step_m=0.3,
                reach_margin_m=0.3,
            )

        self.assertIsNotNone(selected_state)
        self.assertAlmostEqual(float(np.linalg.norm(selected_state.delta)), 1.3, places=6)
        self.assertEqual(blocks_mock.call_count, 3)
        self.assertGreater(blocks_mock.call_count, 0)

    def test_aabb_extent_along_direction_sums_projected_axis_extents(self) -> None:
        obj = {
            "bbox_min": [-1.0, -2.0, -0.5],
            "bbox_max": [1.0, 2.0, 0.5],
        }

        # Pure +X: extent is just the X-axis size (2.0).
        self.assertAlmostEqual(
            _aabb_extent_along_direction(obj, np.array([1.0, 0.0, 0.0])), 2.0, places=6
        )
        # 45-degree diagonal in XY: sums both axes' extents (2.0 + 4.0).
        diag = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        self.assertAlmostEqual(
            _aabb_extent_along_direction(obj, diag), (2.0 + 4.0) / math.sqrt(2.0), places=6
        )

    def test_aabb_extent_along_direction_handles_missing_bbox(self) -> None:
        self.assertEqual(_aabb_extent_along_direction({}, np.array([1.0, 0.0, 0.0])), 0.0)

    def test_adaptive_occlusion_directed_step_scales_with_obj_ref_size_only_and_clamps(self) -> None:
        # query_obj's own size must NOT affect the step (each scanned
        # magnitude already tests query_obj's full translated surface
        # samples) -- only obj_ref's extent matters, and the step must
        # never exceed it (kept well below it via `fraction`). Note
        # _adaptive_occlusion_directed_step doesn't even take a query_obj
        # argument, so there's nothing for it to depend on.
        unit_direction = np.array([1.0, 0.0, 0.0])
        tiny_ref = {"bbox_min": [-0.01, -0.01, -0.01], "bbox_max": [0.01, 0.01, 0.01]}
        huge_ref = {"bbox_min": [-5.0, -5.0, -5.0], "bbox_max": [5.0, 5.0, 5.0]}
        typical_ref = {"bbox_min": [-0.15, -0.15, -0.15], "bbox_max": [0.15, 0.15, 0.15]}

        tiny_step = _adaptive_occlusion_directed_step(tiny_ref, unit_direction)
        huge_step = _adaptive_occlusion_directed_step(huge_ref, unit_direction)
        typical_step = _adaptive_occlusion_directed_step(typical_ref, unit_direction)

        # Tiny occluder: clamped to the minimum, not scaled down to ~0.
        self.assertAlmostEqual(tiny_step, 0.05, places=6)
        # Huge occluder (10m extent along +X): clamped to the maximum (1.0m),
        # not scaled up to half its own size (5.0m).
        self.assertAlmostEqual(huge_step, 1.0, places=6)
        # Typical furniture-sized occluder: between the two clamps, and
        # never larger than the occluder's own extent (0.3m along +X here).
        self.assertLess(typical_step, 1.0)
        self.assertGreater(typical_step, 0.05)
        self.assertLessEqual(
            typical_step, _aabb_extent_along_direction(typical_ref, unit_direction)
        )

    def test_find_occlusion_directed_delta_for_occluder_uses_adaptive_step_by_default(self) -> None:
        # Both objects are tiny (0.02m cubes), so the adaptive step/margin
        # clamp to the minimum (0.05m) -- much finer than the old fixed
        # 0.3m, which would have skipped straight past this narrow window.
        query_obj = {
            "id": 1, "label": "pin", "center": [0.0, 0.0, 2.0],
            "bbox_min": [-0.01, -0.01, -0.01], "bbox_max": [0.01, 0.01, 0.01],
        }
        obj_ref = {
            "id": 2, "label": "pin_ref", "center": [1.0, 0.0, 2.0],
            "bbox_min": [-0.01, -0.01, -0.01], "bbox_max": [0.01, 0.01, 0.01],
        }
        moved_objects = [query_obj, obj_ref]

        # Only the narrow band [1.02, 1.07] blocks -- a 0.3m fixed step
        # starting at max(0.2, 1.0-0.3)=0.7 would jump 0.7, 1.0, 1.3 and
        # miss it entirely.
        def fake_blocks(*, target_delta, **kwargs):
            magnitude = float(np.linalg.norm(target_delta))
            return 1.02 <= magnitude <= 1.07

        with (
            patch(
                "src.qa_generator._occluder_blocks_translated_query_object",
                side_effect=fake_blocks,
            ),
            patch("src.qa_generator.apply_movement", return_value=moved_objects),
            patch("src.qa_generator.is_within_room", return_value=True),
            patch("src.qa_generator.has_terminal_bbox_collision", return_value=False),
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

        self.assertIsNotNone(selected_state)
        magnitude = float(np.linalg.norm(selected_state.delta))
        self.assertGreaterEqual(magnitude, 1.02)
        self.assertLessEqual(magnitude, 1.07)

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

    def test_iter_occlusion_directed_object_move_states_excludes_moved_ids_and_zero_vectors(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        # Directly "north" (+Y direction) of query_obj: matches a canonical bin.
        aligned_occluder = make_object(2, "shelf", (0.0, 1.0, 2.0))
        # A coincident center has no usable floor-plane direction and should
        # never reach the mocked delta search.
        zero_vector_occluder = make_object(3, "lamp", (0.0, 0.0, 2.0))
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
                    movement_scene_objects=[query_obj, aligned_occluder, zero_vector_occluder, moved_sibling],
                    occlusion_source_objects=[aligned_occluder, zero_vector_occluder, moved_sibling],
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

    def test_iter_occlusion_directed_object_move_states_tries_both_boundary_directions(self) -> None:
        query_obj = make_object(1, "cushion", (0.0, 0.0, 2.0))
        angle = math.radians(22.5)
        occluder = make_object(
            2,
            "shelf",
            (math.cos(angle), math.sin(angle), 2.0),
        )
        original_visibility = {
            2: (None, "mesh_ray", "", make_l1_metrics("occluded")),
        }
        states_by_direction = [
            SimpleNamespace(delta=np.array([0.8, 0.0, 0.0])),
            SimpleNamespace(delta=np.array([0.6, 0.6, 0.0])),
        ]
        counter = [0]

        with patch(
            "src.qa_generator._find_occlusion_directed_delta_for_occluder",
            side_effect=states_by_direction,
        ) as delta_mock:
            states = list(
                _iter_occlusion_directed_object_move_states(
                    query_obj=query_obj,
                    move_source_id=1,
                    moved_ids={1},
                    movement_scene_objects=[query_obj, occluder],
                    occlusion_source_objects=[occluder],
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

        self.assertEqual(states, states_by_direction)
        self.assertEqual(counter, [1])
        self.assertEqual(delta_mock.call_count, 2)
        attempted_directions = [
            call.kwargs["unit_direction"] for call in delta_mock.call_args_list
        ]
        np.testing.assert_allclose(attempted_directions[0], DISTANCE_MOVE_DIRECTIONS[0])
        np.testing.assert_allclose(attempted_directions[1], DISTANCE_MOVE_DIRECTIONS[4])

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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
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

    @unittest.skip("legacy single-target semantics v1")
    def test_generate_l2_object_move_occlusion_directed_budget_scales_with_scene_size(self) -> None:
        # A scene with more movement objects than the old fixed budget of 20
        # should get a correspondingly larger occluder-directed search
        # budget (max_candidates_tried) instead of being capped at 20.
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        filler = [
            make_object(idx, f"filler{idx}", (10.0 + idx, 10.0, 2.0))
            for idx in range(3, 26)
        ]
        objects = [sofa, cushion] + filler  # 25 movement objects total

        with (
            patch("src.qa_generator._select_object_move_state", return_value=None),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ),
            patch("src.qa_generator._bbox_fully_in_frame", return_value=True),
            patch("src.qa_generator._bbox_in_frame_corner_count", return_value=(8, 8)),
            patch("src.qa_generator._iter_additional_object_move_states", return_value=[]),
            patch(
                "src.qa_generator._iter_occlusion_directed_object_move_states",
                return_value=[],
            ) as directed_mock,
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

        self.assertTrue(directed_mock.called)
        _, kwargs = directed_mock.call_args
        self.assertEqual(kwargs["max_candidates_tried"], 25)

    @unittest.skip("legacy single-target semantics v1")
    def test_generate_l2_object_move_max_occlusion_objects_auto_scales_source_pool(self) -> None:
        # With max_occlusion_objects=MAX_OCCLUSION_OBJECTS_AUTO, a scene with
        # more movement objects than the old fixed default of 20 should get
        # an adaptively larger (but still capped) occlusion source pool,
        # instead of always truncating to 20.
        sofa = make_object(1, "sofa", (0.0, 0.0, 2.0))
        cushion = make_object(2, "cushion", (0.2, 0.0, 2.0))
        filler = [
            make_object(idx, f"filler{idx}", (10.0 + idx, 10.0, 2.0))
            for idx in range(3, 73)
        ]
        objects = [sofa, cushion] + filler  # 72 movement objects total

        with (
            patch("src.qa_generator._select_object_move_state", return_value=None),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_static_target",
                return_value=(make_l1_metrics("not occluded"), "mesh_ray"),
            ) as visibility_mock,
            patch("src.qa_generator._bbox_fully_in_frame", return_value=True),
            patch("src.qa_generator._bbox_in_frame_corner_count", return_value=(8, 8)),
            patch("src.qa_generator._iter_additional_object_move_states", return_value=[]),
            patch(
                "src.qa_generator._iter_occlusion_directed_object_move_states",
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
                max_occlusion_objects=MAX_OCCLUSION_OBJECTS_AUTO,
            )

        # 72 movement objects clamp to the AUTO cap's max_cap of 50 (not the
        # old fixed 20), so exactly 50 objects should have gone through the
        # per-object visibility precompute.
        self.assertEqual(visibility_mock.call_count, 50)

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


class PairwiseOcclusionV2Tests(unittest.TestCase):
    def test_pairwise_relation_discards_mutual_partial_blocking(self) -> None:
        query = make_object(1, "cushion", (0.0, 0.0, 2.0))
        reference = make_object(2, "sofa", (0.0, 0.0, 3.0))
        with (
            patch("src.qa_generator._get_instance_intersector", side_effect=lambda _mesh, obj_id: obj_id),
            patch(
                "src.qa_generator._instance_surface_samples",
                return_value=np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
            ),
            patch(
                "src.qa_generator._in_frame_surface_sample_subset",
                side_effect=lambda points, *_args, **_kwargs: (1.0, 1.0, points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._surface_probe_subset",
                side_effect=lambda points, _limit: (points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._batch_first_hit_distances_compat",
                side_effect=[np.array([1.0]), np.array([1.0])],
            ),
        ):
            relation, _, _ = _pairwise_occlusion_relation_after_move(
                query_obj=query,
                ref_obj=reference,
                query_delta=np.zeros(3),
                ref_delta=np.zeros(3),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=SimpleNamespace(),
            )

        self.assertIsNone(relation)

    def test_pairwise_relation_supports_query_ref_neither(self) -> None:
        query = make_object(1, "cushion", (0.0, 0.0, 2.0))
        reference = make_object(2, "sofa", (0.0, 0.0, 3.0))
        mesh_data = SimpleNamespace()

        with (
            patch("src.qa_generator._get_instance_intersector", side_effect=lambda _mesh, obj_id: obj_id),
            patch(
                "src.qa_generator._instance_surface_samples",
                return_value=np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
            ),
            patch(
                "src.qa_generator._in_frame_surface_sample_subset",
                side_effect=lambda points, *_args, **_kwargs: (1.0, 1.0, points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._surface_probe_subset",
                side_effect=lambda points, _limit: (points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._batch_first_hit_distances_compat",
                side_effect=[np.array([1.0]), np.array([np.inf])],
            ),
        ):
            relation, query_ratio, ref_ratio = _pairwise_occlusion_relation_after_move(
                query_obj=query,
                ref_obj=reference,
                query_delta=np.zeros(3),
                ref_delta=np.zeros(3),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=mesh_data,
            )

        self.assertEqual(relation, L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF)
        self.assertGreater(query_ratio, ref_ratio)

        with (
            patch("src.qa_generator._get_instance_intersector", side_effect=lambda _mesh, obj_id: obj_id),
            patch(
                "src.qa_generator._instance_surface_samples",
                return_value=np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
            ),
            patch(
                "src.qa_generator._in_frame_surface_sample_subset",
                side_effect=lambda points, *_args, **_kwargs: (1.0, 1.0, points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._surface_probe_subset",
                side_effect=lambda points, _limit: (points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._batch_first_hit_distances_compat",
                side_effect=[np.array([np.inf]), np.array([1.0])],
            ),
        ):
            relation, _, _ = _pairwise_occlusion_relation_after_move(
                query_obj=query,
                ref_obj=reference,
                query_delta=np.zeros(3),
                ref_delta=np.zeros(3),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=mesh_data,
            )
        self.assertEqual(relation, L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY)

        with (
            patch("src.qa_generator._get_instance_intersector", side_effect=lambda _mesh, obj_id: obj_id),
            patch(
                "src.qa_generator._instance_surface_samples",
                return_value=np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
            ),
            patch(
                "src.qa_generator._in_frame_surface_sample_subset",
                side_effect=lambda points, *_args, **_kwargs: (1.0, 1.0, points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._surface_probe_subset",
                side_effect=lambda points, _limit: (points, np.array([0]), np.zeros((len(points), 3))),
            ),
            patch(
                "src.qa_generator._batch_first_hit_distances_compat",
                side_effect=[np.array([np.inf]), np.array([np.inf])],
            ),
        ):
            relation, _, _ = _pairwise_occlusion_relation_after_move(
                query_obj=query,
                ref_obj=reference,
                query_delta=np.zeros(3),
                ref_delta=np.zeros(3),
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=mesh_data,
            )
        self.assertEqual(relation, L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER)

    def test_pairwise_directed_search_yields_both_post_move_directions(self) -> None:
        query = make_object(1, "cushion", (0.0, 0.0, 2.0))
        reference = make_object(2, "sofa", (1.0, 0.0, 2.0))
        objects = [query, reference]
        relation_results = [
            (L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY, 0.0, 0.8),
            (L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF, 0.8, 0.0),
        ]
        with (
            patch(
                "src.qa_generator._match_world_xy_direction_bins",
                return_value=((0, np.array([1.0, 0.0, 0.0])),),
            ),
            patch("src.qa_generator._adaptive_occlusion_directed_step", return_value=0.5),
            patch("src.qa_generator.is_within_room", return_value=True),
            patch("src.qa_generator.has_terminal_bbox_collision", return_value=False),
            patch("src.qa_generator._object_blocks_translated_target_object", return_value=True),
            patch("src.qa_generator._occluder_blocks_translated_query_object", return_value=True),
            patch(
                "src.qa_generator._pairwise_occlusion_relation_after_move",
                side_effect=relation_results,
            ),
        ):
            states = list(
                _iter_pairwise_occlusion_directed_object_move_states(
                    query_obj=query,
                    ref_obj=reference,
                    move_source_id=1,
                    moved_ids={1},
                    movement_scene_objects=objects,
                    attachment_graph={},
                    camera_pose=make_camera_pose(),
                    color_intrinsics=make_camera_intrinsics(),
                    instance_mesh_data=SimpleNamespace(),
                    room_min=np.array([-5.0, -5.0, -5.0]),
                    room_max=np.array([5.0, 5.0, 5.0]),
                    collision_objects=objects,
                )
            )

        self.assertEqual(
            {relation for _state, relation, _query_ratio, _ref_ratio in states},
            {
                L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF,
                L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
            },
        )

    def test_pairwise_directed_search_recomputes_step_for_both_boundary_directions(self) -> None:
        query = make_object(1, "cushion", (0.0, 0.0, 2.0))
        angle = math.radians(22.5)
        reference = make_object(
            2,
            "sofa",
            (math.cos(angle), math.sin(angle), 2.0),
        )
        objects = [query, reference]
        counter = [0]

        with (
            patch(
                "src.qa_generator._adaptive_occlusion_directed_step",
                return_value=0.5,
            ) as step_mock,
            patch("src.qa_generator.is_within_room", return_value=False),
        ):
            states = list(
                _iter_pairwise_occlusion_directed_object_move_states(
                    query_obj=query,
                    ref_obj=reference,
                    move_source_id=1,
                    moved_ids={1},
                    movement_scene_objects=objects,
                    attachment_graph={},
                    camera_pose=make_camera_pose(),
                    color_intrinsics=make_camera_intrinsics(),
                    instance_mesh_data=SimpleNamespace(),
                    room_min=np.array([-5.0, -5.0, -5.0]),
                    room_max=np.array([5.0, 5.0, 5.0]),
                    collision_objects=objects,
                    candidates_tried_counter=counter,
                    max_candidates_tried=1,
                )
            )

        self.assertEqual(states, [])
        self.assertEqual(counter, [1])
        self.assertEqual(step_mock.call_count, 2)
        attempted_directions = [call.args[1] for call in step_mock.call_args_list]
        np.testing.assert_allclose(attempted_directions[0], DISTANCE_MOVE_DIRECTIONS[0])
        np.testing.assert_allclose(attempted_directions[1], DISTANCE_MOVE_DIRECTIONS[4])

    def test_generate_l2_object_move_emits_pairwise_v2_fields(self) -> None:
        mover = make_object(1, "table", (0.0, 0.0, 2.0))
        query = make_object(2, "lamp", (0.2, 0.0, 2.0))
        reference = make_object(3, "sofa", (0.8, 0.0, 2.0))
        moved_query = dict(query)
        moved_query["center"] = [0.7, 0.0, 2.0]
        moved_query["bbox_min"] = [0.6, -0.1, 1.9]
        moved_query["bbox_max"] = [0.8, 0.1, 2.1]
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[mover, moved_query],
            moved_ids={1, 2},
        )
        movement_camera = make_camera_pose()
        occlusion_camera = CameraPose(
            image_name="last.jpg",
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([-0.2, 0.0, 0.0], dtype=np.float64),
        )
        with (
            patch("src.qa_generator._select_object_move_state", return_value=selected_state),
            patch("src.qa_generator._iter_additional_object_move_states", return_value=[]),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._pairwise_occlusion_relation_after_move",
                return_value=(
                    L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF,
                    0.8,
                    0.0,
                ),
            ) as relation_mock,
            patch(
                "src.qa_generator._bbox_in_frame_corner_count",
                return_value=(8, 8),
            ) as bbox_mock,
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=[mover, query, reference],
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=movement_camera,
                templates={
                    "L2_object_move_occlusion": [
                        "After moving {obj_move_source}, compare {obj_query} and {obj_ref} from the last main view."
                    ]
                },
                movement_objects=[mover, query, reference],
                object_map={obj["id"]: obj for obj in [mover, query, reference]},
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=SimpleNamespace(),
                attachment_referable_object_ids=[1, 2],
                attachment_query_objects=[query],
                move_source_object_ids={1},
                reference_object_ids={3},
                occlusion_camera_pose=occlusion_camera,
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        occlusion_questions = [q for q in questions if q.get("type") == "object_move_occlusion"]
        self.assertEqual(len(occlusion_questions), 1)
        question = occlusion_questions[0]
        self.assertEqual(question["occlusion_semantics_version"], 2)
        self.assertEqual(question["query_obj_id"], 2)
        self.assertEqual(question["obj_ref_id"], 3)
        self.assertEqual(question["new_pairwise_occlusion_relation"], L2_OBJECT_MOVE_OCCLUSION_RELATION_QUERY_BY_REF)
        self.assertEqual(relation_mock.call_count, 1)
        self.assertTrue(np.any(np.asarray(relation_mock.call_args.kwargs["query_delta"])))
        self.assertIs(relation_mock.call_args.kwargs["camera_pose"], occlusion_camera)
        self.assertEqual(
            [int(call.args[0]["id"]) for call in bbox_mock.call_args_list],
            [3, 2, 3],
        )
        self.assertTrue(all(
            call.args[1] is occlusion_camera
            for call in bbox_mock.call_args_list
        ))
        self.assertEqual(bbox_mock.call_args_list[1].args[0]["center"], moved_query["center"])
        self.assertNotIn("old_correct_value", question)
        self.assertNotIn("old_pairwise_occlusion_relation", question)
        self.assertNotIn("old_query_blocking_ratio", question)
        self.assertIn("last main view", question["question"].lower())

    def test_generate_l2_object_move_prefers_reverse_occlusion_over_neither_fallback(self) -> None:
        mover = make_object(1, "table", (0.0, 0.0, 2.0))
        query = make_object(2, "lamp", (0.2, 0.0, 2.0))
        reference = make_object(3, "sofa", (0.8, 0.0, 2.0))
        generic_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[mover, query],
            moved_ids={1, 2},
        )
        directed_state = SimpleNamespace(
            delta=np.array([0.8, 0.0, 0.0], dtype=np.float64),
            moved_objects=[mover, query],
            moved_ids={1, 2},
        )
        occlusion_camera = CameraPose(
            image_name="last.jpg",
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([-0.2, 0.0, 0.0], dtype=np.float64),
        )

        with (
            patch("src.qa_generator._select_object_move_state", return_value=generic_state),
            patch("src.qa_generator._iter_additional_object_move_states", return_value=[]),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._pairwise_occlusion_relation_after_move",
                return_value=(L2_OBJECT_MOVE_OCCLUSION_RELATION_NEITHER, 0.0, 0.0),
            ),
            patch(
                "src.qa_generator._iter_pairwise_occlusion_directed_object_move_states",
                return_value=[
                    (
                        directed_state,
                        L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
                        0.0,
                        0.8,
                    )
                ],
            ) as directed_mock,
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=[mover, query, reference],
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_occlusion": [
                        "After moving {obj_move_source}, compare {obj_query} and {obj_ref}."
                    ]
                },
                movement_objects=[mover, query, reference],
                object_map={obj["id"]: obj for obj in [mover, query, reference]},
                color_intrinsics=make_camera_intrinsics(),
                instance_mesh_data=SimpleNamespace(),
                attachment_referable_object_ids=[1, 2],
                attachment_query_objects=[query],
                move_source_object_ids={1},
                reference_object_ids={3},
                occlusion_camera_pose=occlusion_camera,
                enabled_l2_object_move_types={"object_move_occlusion"},
            )

        occlusion_questions = [
            question for question in questions
            if question.get("type") == "object_move_occlusion"
        ]
        self.assertEqual(len(occlusion_questions), 1)
        self.assertEqual(directed_mock.call_count, 1)
        self.assertIs(
            directed_mock.call_args.kwargs["camera_pose"],
            occlusion_camera,
        )
        self.assertEqual(
            occlusion_questions[0]["new_pairwise_occlusion_relation"],
            L2_OBJECT_MOVE_OCCLUSION_RELATION_REF_BY_QUERY,
        )


if __name__ == "__main__":
    unittest.main()
