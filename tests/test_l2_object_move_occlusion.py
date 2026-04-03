from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    _counterfactual_occlusion_backend,
    _find_object_move_occlusion_changes,
    _make_l1_occlusion_metrics,
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
    def test_counterfactual_occlusion_backend_rejects_unsupported_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy_backend"):
            _counterfactual_occlusion_backend(
                "legacy_backend",
                ray_caster=object(),
                instance_mesh_data=SimpleNamespace(),
            )

    def test_find_object_move_occlusion_changes_tracks_l1_style_changes_for_moved_targets_only(self) -> None:
        objects = [
            make_object(1, "sofa", (0.0, 0.0, 2.0)),
            make_object(2, "cushion", (0.2, 0.0, 2.0)),
            make_object(3, "television", (1.0, 0.0, 2.0)),
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
                ],
            ),
            patch(
                "src.qa_generator._compute_l1_style_visibility_metrics_for_moved_target",
                side_effect=[
                    (make_l1_metrics("occluded"), "mesh_ray"),
                    (make_l1_metrics("not visible"), "mesh_ray"),
                ],
            ),
        ):
            changes = _find_object_move_occlusion_changes(
                original_objects=objects,
                moved_objects=objects,
                moved_ids={1, 2},
                camera_pose=make_camera_pose(),
                color_intrinsics=make_camera_intrinsics(),
                occlusion_backend="mesh_ray",
                ray_caster=object(),
                instance_mesh_data=object(),
            )

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["target_obj_id"], 1)
        self.assertEqual(changes[0]["target_obj_label"], "sofa")
        self.assertEqual(changes[0]["old"]["visibility_status"], "not occluded")
        self.assertEqual(changes[0]["new"]["visibility_status"], "occluded")

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
                return_value=(make_l1_metrics("occluded"), "mesh_ray"),
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
        self.assertEqual(question["correct_value"], "occluded")
        self.assertEqual(set(question["options"]), {"not occluded", "occluded", "not visible"})
        self.assertEqual(len(question["options"]), 3)
        self.assertEqual(question["moved_obj_label"], "sofa")
        self.assertEqual(question["query_obj_label"], "cushion")
        self.assertEqual(question["target_obj_label"], "cushion")
        self.assertEqual(question["obj_b_label"], "cushion")
        self.assertNotIn("obj_c_id", question)
        self.assertNotIn("relative to", question["question"])
        self.assertNotIn("attachment", question["question"].lower())

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

    def test_generate_l2_object_move_does_not_reuse_occlusion_changes_across_different_deltas(self) -> None:
        objects = [
            make_object(1, "box", (0.0, 0.0, 2.0)),
            make_object(2, "cup", (0.2, 0.0, 2.0)),
        ]
        base_relations = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "right",
            }
        ]
        moved_relations_box = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "front-right",
            }
        ]
        moved_relations_cup = [
            {
                "obj_a_id": 1,
                "obj_b_id": 2,
                "direction_b_rel_a": "front-right",
            }
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
                attachment_graph={},
                attached_by={2: 1},
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
            )

        self.assertEqual(mocked_helper.call_count, 2)
        self.assertEqual(
            [tuple(q["delta"]) for q in questions if q.get("type") == "object_move_agent"],
            [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )
        self.assertTrue(all(args.args[2] == 1 for args in mocked_helper.call_args_list))


if __name__ == "__main__":
    unittest.main()
