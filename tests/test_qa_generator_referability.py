import os
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.qa_generator import (
    enrich_objects_with_distance_geometry,
    _cap_l3_unchanged_ratio,
    _ensure_question_mentions,
    _enforce_in_frame_mentions,
    _select_l2_object_move_occlusion_records,
    generate_all_questions,
    generate_l1_occlusion_questions,
    generate_l2_object_move,
    generate_l2_object_move_object_centric,
    generate_l2_object_move_allocentric,
    generate_l2_object_rotate_object_centric,
    generate_l3_attachment_chain,
    generate_l3_coordinate_rotation,
    generate_l3_coordinate_rotation_allocentric,
    generate_l3_coordinate_rotation_object_centric,
)
from src.utils.colmap_loader import CameraPose


def make_camera_pose() -> CameraPose:
    return CameraPose(
        image_name="000000.jpg",
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


def make_object(obj_id: int, label: str) -> dict:
    return {
        "id": obj_id,
        "label": label,
        "center": [float(obj_id), 0.0, 1.0],
        "bbox_min": [float(obj_id), 0.0, 0.5],
        "bbox_max": [float(obj_id) + 0.2, 0.2, 1.5],
    }


def make_l2_object_move_question(
    qtype: str,
    *,
    attached: bool,
    text: str,
) -> dict:
    return {
        "level": "L2",
        "type": qtype,
        "question": text,
        "options": ["A", "B", "C", "D"],
        "answer": "A",
        "attachment_remapped": attached,
    }


class QaGeneratorReferabilityTests(unittest.TestCase):
    def test_object_move_occlusion_selector_caps_priority_unchanged_record(self) -> None:
        records = [
            {
                "candidate_index": 0,
                "query_obj_id": 1,
                "relation_unchanged": True,
                "question": {"question": "normal unchanged"},
            },
            {
                "candidate_index": 1,
                "query_obj_id": 2,
                "relation_unchanged": True,
                "attachment_priority_pair": True,
                "question": {"question": "priority unchanged"},
            },
        ]

        selected = _select_l2_object_move_occlusion_records(records)

        self.assertEqual(selected, [])

    def test_object_move_occlusion_selector_counts_priority_unchanged_in_ratio(self) -> None:
        records = [
            *[
                {
                    "candidate_index": idx,
                    "query_obj_id": idx,
                    "relation_unchanged": False,
                    "question": {"question": f"changed {idx}"},
                }
                for idx in range(4)
            ],
            {
                "candidate_index": 4,
                "query_obj_id": 40,
                "relation_unchanged": True,
                "attachment_priority_pair": True,
                "question": {"question": "priority unchanged"},
            },
            {
                "candidate_index": 5,
                "query_obj_id": 50,
                "relation_unchanged": True,
                "question": {"question": "normal unchanged"},
            },
        ]

        selected = _select_l2_object_move_occlusion_records(records)

        self.assertEqual(
            [record["query_obj_id"] for record in selected],
            [0, 1, 2, 3, 40],
        )

    def test_generate_all_questions_passes_l2_move_internal_type_filter(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
        ]
        captured: dict[str, object] = {}

        def fake_generate_l2_object_move(*_args, **kwargs):
            captured["enabled_l2_object_move_types"] = kwargs.get(
                "enabled_l2_object_move_types"
            )
            return []

        with (
            patch(
                "src.qa_generator.generate_l2_object_move",
                side_effect=fake_generate_l2_object_move,
            ),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                camera_pose=make_camera_pose(),
                templates={},
                referable_object_ids=[1, 2],
                only_question_types=[
                    "L2_object_move_distance",
                    "L2_object_move_agent",
                    "L2_object_remove",
                ],
            )

        self.assertEqual(questions, [])
        self.assertEqual(
            captured["enabled_l2_object_move_types"],
            {"object_move_agent", "object_move_distance"},
        )

    def test_generate_all_questions_only_question_types_accepts_new_reference_frame_types(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "chair"),
            make_object(4, "sofa"),
        ]

        cases = [
            (
                "L2_object_rotate_object_centric",
                "generate_l2_object_rotate_object_centric",
                {
                    "level": "L2",
                    "type": "object_rotate_object_centric",
                    "moved_obj_id": 1,
                    "moved_obj_label": "table",
                    "query_obj_id": 2,
                    "query_obj_label": "box",
                    "obj_ref_id": 3,
                    "obj_ref_label": "chair",
                    "obj_face_id": 4,
                    "obj_face_label": "sofa",
                },
            ),
            (
                "L2_object_move_allocentric",
                "generate_l2_object_move_allocentric",
                {
                    "level": "L2",
                    "type": "object_move_allocentric",
                    "moved_obj_id": 1,
                    "moved_obj_label": "table",
                    "query_obj_id": 2,
                    "query_obj_label": "box",
                    "obj_ref_id": 3,
                    "obj_ref_label": "chair",
                },
            ),
            (
                "L3_attachment_chain",
                "generate_l3_attachment_chain",
                {
                    "level": "L3",
                    "type": "attachment_chain",
                    "grandparent_id": 1,
                    "grandparent_label": "table",
                    "parent_id": 2,
                    "parent_label": "box",
                    "grandchild_id": 3,
                    "grandchild_label": "chair",
                    "neighbor_id": 4,
                    "neighbor_label": "sofa",
                    "mentioned_objects": [
                        {"role": "grandparent", "label": "table", "obj_id": 1},
                        {"role": "parent", "label": "box", "obj_id": 2},
                        {"role": "grandchild", "label": "chair", "obj_id": 3},
                        {"role": "neighbor", "label": "sofa", "obj_id": 4},
                    ],
                },
            ),
            (
                "L3_coordinate_rotation_object_centric",
                "generate_l3_coordinate_rotation_object_centric",
                {
                    "level": "L3",
                    "type": "coordinate_rotation_object_centric",
                    "obj_ref_id": 1,
                    "obj_ref_label": "table",
                    "obj_face_id": 2,
                    "obj_face_label": "box",
                    "obj_target_id": 3,
                    "obj_target_label": "chair",
                },
            ),
            (
                "L3_coordinate_rotation_allocentric",
                "generate_l3_coordinate_rotation_allocentric",
                {
                    "level": "L3",
                    "type": "coordinate_rotation_allocentric",
                    "obj_a_id": 1,
                    "obj_a_label": "table",
                    "obj_b_id": 2,
                    "obj_b_label": "box",
                },
            ),
        ]

        for public_type, generator_name, question_fields in cases:
            with self.subTest(public_type=public_type):
                question = {
                    **question_fields,
                    "question": f"{public_type} question",
                    "options": ["front", "back", "left", "right"],
                    "answer": "A",
                    "correct_value": "front",
                    "relation_unchanged": False,
                }
                with patch(f"src.qa_generator.{generator_name}", return_value=[question]):
                    questions = generate_all_questions(
                        objects=objects,
                        attachment_graph={},
                        attached_by={},
                        support_chain_graph={},
                        support_chain_by={},
                        camera_pose=make_camera_pose(),
                        templates={},
                        visible_object_ids=[1, 2, 3, 4],
                        referable_object_ids=[1, 2, 3, 4],
                        only_question_types=[public_type],
                    )

                self.assertEqual([q["type"] for q in questions], [question_fields["type"]])

    def test_object_move_object_centric_generator_emits_three_role_move_question(self) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        ref = make_object(3, "chair")
        moved_parent = {**parent, "center": [2.0, 0.0, 1.0]}
        moved_child = {**child, "center": [3.0, 0.0, 1.0]}
        moved_objects = [moved_parent, moved_child, ref]
        old_parent_center = np.array(parent["center"], dtype=np.float64)
        old_child_center = np.array(child["center"], dtype=np.float64)
        moved_parent_center = np.array(moved_parent["center"], dtype=np.float64)
        moved_child_center = np.array(moved_child["center"], dtype=np.float64)
        ref_center = np.array(ref["center"], dtype=np.float64)

        def object_centric_direction(anchor_center, _face_center, target_center, **_kwargs):
            anchor = np.asarray(anchor_center, dtype=np.float64)
            target = np.asarray(target_center, dtype=np.float64)
            if np.allclose(target, ref_center):
                if np.allclose(anchor, old_parent_center) or np.allclose(anchor, old_child_center):
                    return "left", 0.1
                if np.allclose(anchor, moved_parent_center) or np.allclose(anchor, moved_child_center):
                    return "front", 0.1
            return "left", 0.9

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator._iter_valid_object_move_states",
                return_value=[
                    (np.array([1.0, 0.0, 0.0], dtype=np.float64), moved_objects, {1, 2})
                ],
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                side_effect=object_centric_direction,
            ),
        ):
            questions = generate_l2_object_move_object_centric(
                objects=[parent, child, ref],
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_object_centric": [
                        "move {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=[parent, child, ref],
                object_map={obj["id"]: obj for obj in [parent, child, ref]},
                attachment_referable_object_ids=[1, 2, 3],
                attachment_query_objects=[parent, child, ref],
            )

        self.assertTrue(questions)
        question = questions[0]
        self.assertEqual(question["type"], "object_move_object_centric")
        self.assertEqual(question["moved_obj_id"], 1)
        self.assertEqual(question["query_obj_id"], 2)
        self.assertEqual(question["obj_ref_id"], 3)
        self.assertNotIn("obj_face_id", question)
        self.assertEqual(
            [mention["role"] for mention in question["mentioned_objects"]],
            ["moved_object", "query_object", "reference_object"],
        )

    def test_object_move_object_centric_generator_allows_move_source_as_query(self) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        ref = make_object(3, "chair")
        moved_parent = {**parent, "center": [1.5, 0.0, 1.0]}
        moved_child = {**child, "center": [2.5, 0.0, 1.0]}
        moved_objects = [moved_parent, moved_child, ref]
        old_parent_center = np.array(parent["center"], dtype=np.float64)
        old_child_center = np.array(child["center"], dtype=np.float64)
        moved_parent_center = np.array(moved_parent["center"], dtype=np.float64)
        moved_child_center = np.array(moved_child["center"], dtype=np.float64)
        ref_center = np.array(ref["center"], dtype=np.float64)

        def object_centric_direction(anchor_center, _face_center, target_center, **_kwargs):
            anchor = np.asarray(anchor_center, dtype=np.float64)
            target = np.asarray(target_center, dtype=np.float64)
            if np.allclose(target, ref_center):
                if np.allclose(anchor, old_parent_center) or np.allclose(anchor, old_child_center):
                    return "left", 0.1
                if np.allclose(anchor, moved_parent_center) or np.allclose(anchor, moved_child_center):
                    return "front", 0.1
            return "left", 0.9

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator._iter_valid_object_move_states",
                return_value=[
                    (np.array([0.5, 0.0, 0.0], dtype=np.float64), moved_objects, {1, 2})
                ],
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                side_effect=object_centric_direction,
            ),
        ):
            questions = generate_l2_object_move_object_centric(
                objects=[parent, child, ref],
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_move_object_centric": [
                        "move {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=[parent, child, ref],
                object_map={obj["id"]: obj for obj in [parent, child, ref]},
                attachment_referable_object_ids=[1, 2, 3],
                attachment_query_objects=[parent, child, ref],
            )

        question = next(q for q in questions if q["query_obj_id"] == 1)
        self.assertEqual(question["type"], "object_move_object_centric")
        self.assertEqual(question["moved_obj_id"], 1)
        self.assertEqual(question["query_obj_id"], 1)
        self.assertEqual(question["obj_ref_id"], 3)

    def test_generate_l2_object_move_skips_state_search_when_pair_budget_exhausted(self) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        objects = [parent, child]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator._select_object_move_state") as select_mock,
            patch("src.qa_generator._find_object_move_occlusion_changes") as find_mock,
            patch("src.qa_generator._generate_l2_distance_questions_for_object") as distance_mock,
        ):
            questions = generate_l2_object_move(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                pair_budget_remaining=lambda canonical_type, id_a, id_b: False,
            )

        self.assertEqual(questions, [])
        select_mock.assert_not_called()
        find_mock.assert_not_called()
        distance_mock.assert_not_called()

    def test_generate_l2_object_move_object_centric_skips_state_search_when_pair_budget_exhausted(self) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        objects = [parent, child]

        with patch("src.qa_generator._iter_valid_object_move_states") as states_mock:
            questions = generate_l2_object_move_object_centric(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                pair_budget_remaining=lambda canonical_type, id_a, id_b: False,
            )

        self.assertEqual(questions, [])
        states_mock.assert_not_called()

    def test_generate_l2_object_rotate_object_centric_skips_rotation_search_when_pair_budget_exhausted(
        self,
    ) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        face = make_object(3, "lamp")
        objects = [parent, child, face]

        with patch("src.qa_generator.find_meaningful_orbit_rotation") as rotation_mock:
            questions = generate_l2_object_rotate_object_centric(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                pair_budget_remaining=lambda canonical_type, id_a, id_b: False,
            )

        self.assertEqual(questions, [])
        rotation_mock.assert_not_called()

    def test_generate_l2_object_move_allocentric_skips_state_search_when_pair_budget_exhausted(self) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        objects = [parent, child]

        with patch("src.qa_generator._select_object_move_state") as select_mock:
            questions = generate_l2_object_move_allocentric(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
                pair_budget_remaining=lambda canonical_type, id_a, id_b: False,
            )

        self.assertEqual(questions, [])
        select_mock.assert_not_called()

    def test_generate_all_questions_keeps_salvage_only_attachment_pairs_in_attachment_path(self) -> None:
        objects = [
            make_object(9, "desk"),
            make_object(31, "cup"),
            make_object(32, "bottle"),
            make_object(34, "chair"),
        ]
        trace_events: list[dict] = []
        captured: dict[str, object] = {}

        def capture_move(
            objects_arg,
            attachment_graph_arg,
            attached_by_arg,
            camera_pose_arg,
            templates_arg,
            **kwargs,
        ):
            captured["objects"] = [int(obj["id"]) for obj in objects_arg]
            captured["attachment_query_objects"] = [
                int(obj["id"]) for obj in (kwargs.get("attachment_query_objects") or [])
            ]
            captured["movement_objects"] = [
                int(obj["id"]) for obj in (kwargs.get("movement_objects") or [])
            ]
            captured["attachment_graph"] = {
                int(parent_id): [int(child_id) for child_id in child_ids]
                for parent_id, child_ids in attachment_graph_arg.items()
            }
            captured["attachment_priority_pairs"] = list(
                kwargs.get("attachment_priority_pairs") or []
            )
            return [{
                "level": "L2",
                "type": "object_move_agent",
                "question": "If the desk moves, where is the monitor relative to the chair?",
                "options": ["A", "B", "C", "D"],
                "answer": "A",
                "correct_value": "left",
                "moved_obj_id": 9,
                "moved_obj_label": "desk",
                "query_obj_id": 31,
                "query_obj_label": "monitor",
                "obj_b_id": 31,
                "obj_b_label": "monitor",
                "obj_c_id": 34,
                "obj_c_label": "chair",
                "attachment_remapped": True,
                "mentioned_objects": [
                    {"role": "moved_object", "obj_id": 9, "label": "desk"},
                    {"role": "query_object", "obj_id": 31, "label": "monitor"},
                    {"role": "reference_object", "obj_id": 34, "label": "chair"},
                ],
            }]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", side_effect=capture_move),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={9: [31, 32, 34]},
                attached_by={31: 9, 32: 9, 34: 9},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[9, 31, 32, 34],
                referable_object_ids=[34],
                attachment_referable_object_ids=[9, 31, 32, 34],
                label_statuses={
                    "desk": "unique",
                    "cup": "unique",
                    "bottle": "unique",
                    "chair": "unique",
                },
                label_to_object_ids={
                    "desk": [9],
                    "cup": [31],
                    "bottle": [32],
                    "chair": [34],
                },
                attachment_edges=[
                    {"parent_id": 9, "child_id": 31, "type": "attached_to"},
                    {"parent_id": 9, "child_id": 32, "type": "attached_to"},
                    {"parent_id": 9, "child_id": 34, "type": "attached_to"},
                ],
                attachment_priority_pairs=[(9, 31), (9, 32)],
                trace_recorder=trace_events.append,
                trace_detail="full",
            )

        self.assertEqual(captured["objects"], [34])
        self.assertEqual(captured["attachment_query_objects"], [9, 31, 32, 34])
        self.assertEqual(captured["movement_objects"], [9, 31, 32, 34])
        self.assertEqual(captured["attachment_graph"], {9: [31, 32, 34]})
        self.assertEqual(captured["attachment_priority_pairs"], [(9, 31), (9, 32)])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["query_obj_id"], 31)
        self.assertEqual(
            questions[0]["question_referability_audit"]["decision"],
            "pass",
        )
        pool_snapshot = next(
            event for event in trace_events
            if event.get("event") == "object_pool_snapshot"
        )
        rows = {int(row["id"]): row for row in pool_snapshot["rows"]}
        self.assertTrue(rows[31]["attachment_query_pool"])
        self.assertFalse(rows[31]["question_pool"])
        self.assertTrue(rows[31]["movement_pool"])
        self.assertTrue(rows[31]["attachment_referable"])

    def test_scene0025_salvage_cache_entry_survives_attachment_preprocessing(self) -> None:
        cache_path = "output/flash0-9/0-9_20260502_174155.json"
        if not os.path.exists(cache_path):
            self.skipTest("cache file not available")
            return
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_doc = json.load(f)
        cache_frame_entry = cache_doc["frames"]["scene0025_00"]["1942.jpg"]

        self.assertEqual(cache_frame_entry["referable_object_ids"], [34])
        self.assertEqual(cache_frame_entry["attachment_referable_object_ids"], [9, 34])
        frame_entry = dict(cache_frame_entry)
        frame_entry["attachment_referable_pairs"] = [[9, 31], [9, 34]]
        frame_entry["attachment_referable_object_ids"] = [9, 31, 34]

        objects = [
            make_object(9, "desk"),
            make_object(31, "cup"),
            make_object(34, "chair"),
        ]
        captured: dict[str, object] = {}

        def capture_move(
            objects_arg,
            attachment_graph_arg,
            attached_by_arg,
            camera_pose_arg,
            templates_arg,
            **kwargs,
        ):
            captured["attachment_query_objects"] = [
                int(obj["id"]) for obj in (kwargs.get("attachment_query_objects") or [])
            ]
            captured["movement_objects"] = [
                int(obj["id"]) for obj in (kwargs.get("movement_objects") or [])
            ]
            captured["attachment_graph"] = {
                int(parent_id): [int(child_id) for child_id in child_ids]
                for parent_id, child_ids in attachment_graph_arg.items()
            }
            return []

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", side_effect=capture_move),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            generate_all_questions(
                objects=objects,
                attachment_graph={9: [31, 34]},
                attached_by={31: 9, 34: 9},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[9, 31, 34],
                referable_object_ids=frame_entry["referable_object_ids"],
                attachment_referable_object_ids=frame_entry["attachment_referable_object_ids"],
                attachment_edges=[
                    {"parent_id": 9, "child_id": 31, "type": "attached_to"},
                    {"parent_id": 9, "child_id": 34, "type": "attached_to"},
                ],
            )

        self.assertEqual(captured["attachment_graph"], {9: [31, 34]})
        self.assertEqual(captured["attachment_query_objects"], [9, 31, 34])
        self.assertEqual(captured["movement_objects"], [9, 31, 34])

    def test_enrich_objects_with_distance_geometry_skips_repeat_work_for_same_mesh(self) -> None:
        objects = [make_object(1, "chair")]
        instance_mesh_data = SimpleNamespace(
            vertices=np.zeros((1, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int64),
        )
        surface_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        triangle_ids = np.array([0], dtype=np.int64)
        barycentrics = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        with (
            patch("src.qa_generator._instance_surface_samples", return_value=surface_points) as samples_mock,
            patch(
                "src.qa_generator._instance_surface_sample_metadata",
                return_value=(triangle_ids, barycentrics),
            ) as metadata_mock,
        ):
            enrich_objects_with_distance_geometry(objects, instance_mesh_data)
            enrich_objects_with_distance_geometry(objects, instance_mesh_data)

        self.assertEqual(samples_mock.call_count, 1)
        self.assertEqual(metadata_mock.call_count, 1)

    def test_generate_all_questions_applies_attachment_surface_text_overrides_only_to_attachment_questions(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
            make_object(4, "bed"),
            make_object(5, "pillow"),
            make_object(6, "book"),
            make_object(7, "chair"),
        ]
        attachment_move_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup relative to the lamp?",
            "options": ["left", "right", "front", "back"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "query_obj_id": 1,
            "query_obj_label": "cup",
            "obj_c_id": 3,
            "obj_c_label": "lamp",
            "attachment_remapped": True,
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "query_object", "obj_id": 1, "label": "cup"},
                {"role": "reference_object", "obj_id": 3, "label": "lamp"},
            ],
        }
        attachment_chain_question = {
            "level": "L3",
            "type": "attachment_chain",
            "question": "If the bed moves, which objects move with it?",
            "options": ["the pillow", "the book", "Both the pillow and the book", "the chair"],
            "answer": "C",
            "correct_value": "Both the pillow and the book",
            "grandparent_id": 4,
            "grandparent_label": "bed",
            "parent_id": 5,
            "parent_label": "pillow",
            "grandchild_id": 6,
            "grandchild_label": "book",
            "neighbor_id": 7,
            "neighbor_label": "chair",
            "mentioned_objects": [
                {"role": "grandparent", "obj_id": 4, "label": "bed"},
                {"role": "parent", "obj_id": 5, "label": "pillow"},
                {"role": "grandchild", "obj_id": 6, "label": "book"},
                {"role": "neighbor", "obj_id": 7, "label": "chair"},
            ],
        }
        non_attachment_question = {
            "level": "L1",
            "type": "distance",
            "question": "How far is the cup from the lamp?",
            "options": ["near", "far"],
            "answer": "A",
            "correct_value": "near",
            "obj_a_id": 1,
            "obj_a_label": "cup",
            "obj_b_id": 3,
            "obj_b_label": "lamp",
            "mentioned_objects": [
                {"role": "obj_a", "obj_id": 1, "label": "cup"},
                {"role": "obj_b", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction", return_value=None),
            patch("src.qa_generator.generate_l1_distance", return_value=None),
            patch(
                "src.qa_generator.generate_l2_object_move",
                return_value=[attachment_move_question],
            ),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch(
                "src.qa_generator.generate_l3_attachment_chain",
                return_value=[attachment_chain_question],
            ),
            patch(
                "src.qa_generator.generate_l3_coordinate_rotation",
                return_value=[non_attachment_question],
            ),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
            patch("src.qa_generator._enforce_in_frame_mentions", side_effect=lambda questions, *args, **kwargs: questions),
            patch("src.qa_generator._enforce_referable_mentions", side_effect=lambda questions, *args, **kwargs: questions),
            patch("src.qa_generator._enforce_stable_facing_references", side_effect=lambda questions, *args, **kwargs: questions),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1], 4: [5], 5: [6]},
                attached_by={1: 2, 5: 4, 6: 5},
                support_chain_graph={2: [1], 4: [5], 5: [6]},
                support_chain_by={1: 2, 5: 4, 6: 5},
                camera_pose=make_camera_pose(),
                referable_object_ids=[1, 2, 3, 4, 5, 6, 7],
                attachment_referable_object_ids=[1, 2, 3, 4, 5, 6, 7],
                attachment_object_surface_text_by_id={
                    1: "blue cup",
                    2: "wooden table",
                    3: "floor lamp",
                    4: "king bed",
                    5: "blue pillow",
                    6: "red book",
                },
                visible_object_ids=[1, 2, 3, 4, 5, 6, 7],
            )

        move_question = next(q for q in questions if q["type"] == "object_move_agent")
        chain_question = next(q for q in questions if q["type"] == "attachment_chain")
        distance_question = next(q for q in questions if q["type"] == "distance")

        self.assertIn("wooden table", move_question["question"])
        self.assertIn("blue cup", move_question["question"])
        self.assertIn("floor lamp", move_question["question"])
        self.assertEqual(move_question["moved_obj_label"], "wooden table")
        self.assertEqual(move_question["query_obj_label"], "blue cup")
        self.assertEqual(move_question["obj_c_label"], "floor lamp")
        self.assertEqual(
            [item["label"] for item in move_question["mentioned_objects"]],
            ["wooden table", "blue cup", "floor lamp"],
        )

        self.assertIn("king bed", chain_question["question"])
        self.assertEqual(chain_question["grandparent_label"], "king bed")
        self.assertEqual(chain_question["parent_label"], "blue pillow")
        self.assertEqual(chain_question["grandchild_label"], "red book")
        self.assertIn("Both the blue pillow and the red book", chain_question["correct_value"])
        self.assertIn("the blue pillow", chain_question["options"])
        self.assertIn("the red book", chain_question["options"])

        self.assertEqual(distance_question["question"], "How far is the cup from the lamp?")
        self.assertEqual(distance_question["obj_a_label"], "cup")
        self.assertEqual(distance_question["obj_b_label"], "lamp")

    def test_generate_all_questions_skips_trace_snapshots_without_trace_recorder(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]
        generated_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup relative to the lamp?",
            "options": ["left", "right", "front", "back"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "query_obj_id": 1,
            "query_obj_label": "cup",
            "obj_c_id": 3,
            "obj_c_label": "lamp",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "query_object", "obj_id": 1, "label": "cup"},
                {"role": "reference_object", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[generated_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
            patch(
                "src.qa_generator.json.dumps",
                side_effect=AssertionError("trace snapshot should be skipped"),
            ),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2, 3],
                attachment_edges=[],
            )

        self.assertIn("trace_question_id", generated_question)
        self.assertEqual(generated_question.get("_trace_source"), "generate_l2_object_move")

    def test_ensure_question_mentions_includes_obj_c_id(self) -> None:
        question = {
            "question": "If the table moves, where is the cup relative to the lamp?",
            "obj_a_id": 1,
            "obj_a_label": "cup",
            "obj_b_id": 2,
            "obj_b_label": "table",
            "obj_c_id": 3,
            "obj_c_label": "lamp",
            "mentioned_objects": [
                {"role": "query_object", "label": "cup", "obj_id": 1},
                {"role": "moved_object", "label": "table", "obj_id": 2},
            ],
        }

        normalized = _ensure_question_mentions(
            question,
            {
                1: make_object(1, "cup"),
                2: make_object(2, "table"),
                3: make_object(3, "lamp"),
            },
        )

        mentions = normalized["mentioned_objects"]
        self.assertTrue(
            any(
                mention.get("role") == "obj_c"
                and mention.get("obj_id") == 3
                and mention.get("label") == "lamp"
                for mention in mentions
            )
        )


    def test_l3_support_chain_only_sees_fully_referable_subgraph(self) -> None:
        captured: dict[str, object] = {}

        def capture_l3(objects, attachment_graph, attached_by, camera_pose, templates, **kwargs):
            captured["object_ids"] = [int(o["id"]) for o in objects]
            captured["attachment_graph"] = attachment_graph
            captured["attached_by"] = attached_by
            captured["ordinary_reference_ids"] = [
                int(o["id"])
                for o in kwargs.get("ordinary_reference_objects", [])
            ]
            return []

        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
        ]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", side_effect=capture_l3),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            generate_all_questions(
                objects=objects,
                attachment_graph={1: [2], 2: [3]},
                attached_by={2: 1, 3: 2},
                support_chain_graph={1: [2], 2: [3]},
                support_chain_by={2: 1, 3: 2},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 3],
                label_counts=None,
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    {"parent_id": 2, "child_id": 3, "type": "supported_by"},
                ],
            )

        self.assertEqual(captured["object_ids"], [1, 3])
        self.assertEqual(captured["ordinary_reference_ids"], [1, 3])
        self.assertEqual(captured["attachment_graph"], {})
        self.assertEqual(captured["attached_by"], {})

    def test_l3_attachment_chain_uses_ordinary_referable_neighbor(self) -> None:
        attachment_objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
        ]
        ordinary_neighbor = make_object(4, "chair")

        questions = generate_l3_attachment_chain(
            attachment_objects,
            {1: [2], 2: [3]},
            {2: 1, 3: 2},
            make_camera_pose(),
            {},
            ordinary_reference_objects=[ordinary_neighbor],
        )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["grandparent_id"], 1)
        self.assertEqual(questions[0]["parent_id"], 2)
        self.assertEqual(questions[0]["grandchild_id"], 3)
        self.assertEqual(questions[0]["neighbor_id"], 4)

    def test_generate_all_questions_separates_pair_graph_from_physics_context(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "chair"),
        ]
        captured: dict[str, object] = {}

        def capture_move(objects_arg, attachment_graph_arg, *_args, **kwargs):
            captured["ordinary_ids"] = [int(obj["id"]) for obj in objects_arg]
            captured["physics_graph"] = attachment_graph_arg
            captured["candidate_graph"] = kwargs.get("candidate_attachment_graph")
            captured["attachment_query_ids"] = [
                int(obj["id"])
                for obj in kwargs.get("attachment_query_objects", [])
            ]
            return []

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", side_effect=capture_move),
        ):
            generate_all_questions(
                objects=objects,
                attachment_graph={1: [2], 2: [3]},
                attached_by={2: 1, 3: 2},
                support_chain_graph={1: [2], 2: [3]},
                support_chain_by={2: 1, 3: 2},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[3],
                attachment_referable_object_ids=[1, 2],
                attachment_referable_pairs=[[1, 2]],
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    {"parent_id": 2, "child_id": 3, "type": "supported_by"},
                ],
                only_question_types=["L2_object_move_agent"],
            )

        self.assertEqual(captured["ordinary_ids"], [3])
        self.assertEqual(captured["attachment_query_ids"], [1, 2])
        self.assertEqual(captured["candidate_graph"], {1: [2]})
        self.assertEqual(captured["physics_graph"], {1: [2], 2: [3]})

    def test_generate_all_questions_raises_instead_of_filtering_role_pool_leak(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "chair"),
        ]
        leaked_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the box relative to the chair?",
            "options": ["left", "right", "front", "back"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 1,
            "moved_obj_label": "table",
            "query_obj_id": 2,
            "query_obj_label": "box",
            "obj_c_id": 3,
            "obj_c_label": "chair",
            "attachment_remapped": True,
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 1, "label": "table"},
                {"role": "query_object", "obj_id": 2, "label": "box"},
                {"role": "reference_object", "obj_id": 3, "label": "chair"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator.generate_l2_object_move",
                return_value=[leaked_question],
            ),
        ):
            with self.assertRaisesRegex(AssertionError, "Referability invariant"):
                generate_all_questions(
                    objects=objects,
                    attachment_graph={1: [2]},
                    attached_by={2: 1},
                    support_chain_graph={1: [2]},
                    support_chain_by={2: 1},
                    camera_pose=make_camera_pose(),
                    templates={},
                    visible_object_ids=[1, 2, 3],
                    referable_object_ids=[],
                    attachment_referable_object_ids=[1, 2, 3],
                    attachment_referable_pairs=[[1, 2]],
                    attachment_edges=[
                        {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    ],
                    only_question_types=["L2_object_move_agent"],
                )

    def test_generate_all_questions_object_remove_uses_only_ordinary_pool(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "chair"),
            make_object(4, "sofa"),
        ]
        captured: dict[str, object] = {}

        def capture_remove(objects_arg, *_args, **kwargs):
            captured["object_ids"] = [int(obj["id"]) for obj in objects_arg]
            captured["attachment_query_objects"] = kwargs.get("attachment_query_objects")
            return []

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", side_effect=capture_remove),
        ):
            generate_all_questions(
                objects=objects,
                attachment_graph={1: [2]},
                attached_by={2: 1},
                support_chain_graph={1: [2]},
                support_chain_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3, 4],
                referable_object_ids=[3, 4],
                attachment_referable_object_ids=[1, 2],
                attachment_referable_pairs=[[1, 2]],
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                ],
                only_question_types=["L2_object_remove"],
            )

        self.assertEqual(captured["object_ids"], [3, 4])
        self.assertIsNone(captured["attachment_query_objects"])

    def test_generate_all_questions_l3_attachment_move_gets_ordinary_references(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
            make_object(4, "chair"),
        ]
        captured: dict[str, object] = {}

        def capture_attachment_move(objects_arg, attachment_graph_arg, *_args, **kwargs):
            captured["ordinary_ids"] = [int(obj["id"]) for obj in objects_arg]
            captured["physics_graph"] = attachment_graph_arg
            captured["candidate_graph"] = kwargs.get("candidate_attachment_graph")
            captured["attachment_query_ids"] = [
                int(obj["id"])
                for obj in kwargs.get("attachment_query_objects", [])
            ]
            return []

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator.generate_l3_attachment_move",
                side_effect=capture_attachment_move,
            ),
        ):
            generate_all_questions(
                objects=objects,
                attachment_graph={1: [2], 2: [3]},
                attached_by={2: 1, 3: 2},
                support_chain_graph={1: [2], 2: [3]},
                support_chain_by={2: 1, 3: 2},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3, 4],
                referable_object_ids=[4],
                attachment_referable_object_ids=[1, 2, 3],
                attachment_referable_pairs=[[1, 2], [2, 3]],
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    {"parent_id": 2, "child_id": 3, "type": "supported_by"},
                ],
                only_question_types=["L3_attachment_move"],
            )

        self.assertEqual(captured["ordinary_ids"], [4])
        self.assertEqual(captured["attachment_query_ids"], [1, 2, 3])
        self.assertEqual(captured["candidate_graph"], {1: [2], 2: [3]})
        self.assertEqual(captured["physics_graph"], {1: [2], 2: [3]})

    def test_l2_move_skips_state_search_without_referable_pair(self) -> None:
        parent = make_object(1, "table")
        child = make_object(2, "box")
        reference = make_object(3, "chair")

        with patch("src.qa_generator._select_object_move_state") as state_mock:
            questions = generate_l2_object_move(
                objects=[reference],
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[parent, child, reference],
                object_map={1: parent, 2: child, 3: reference},
                attachment_referable_object_ids=[1, 2],
                attachment_query_objects=[parent, child],
                candidate_attachment_graph={},
            )

        self.assertEqual(questions, [])
        state_mock.assert_not_called()

    def test_generate_all_questions_keeps_attachment_chain_with_attachment_referable_ids(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
            make_object(4, "chair"),
        ]
        attachment_question = {
            "level": "L3",
            "type": "attachment_chain",
            "question": "If the table moves, which objects move with it?",
            "options": ["box", "cup", "Both box and cup", "chair"],
            "answer": "C",
            "correct_value": "Both box and cup",
            "mentioned_objects": [
                {"role": "grandparent", "obj_id": 1, "label": "table"},
                {"role": "parent", "obj_id": 2, "label": "box"},
                {"role": "grandchild", "obj_id": 3, "label": "cup"},
                {"role": "neighbor", "obj_id": 4, "label": "chair"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[attachment_question]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={1: [2], 2: [3]},
                attached_by={2: 1, 3: 2},
                support_chain_graph={1: [2], 2: [3]},
                support_chain_by={2: 1, 3: 2},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3, 4],
                referable_object_ids=[1],
                attachment_referable_object_ids=[1, 2, 3, 4],
                label_statuses={
                    "table": "unique",
                    "box": "unique",
                    "cup": "unique",
                    "chair": "unique",
                },
                label_to_object_ids={
                    "table": [1],
                    "box": [2],
                    "cup": [3],
                    "chair": [4],
                },
                attachment_edges=[
                    {"parent_id": 1, "child_id": 2, "type": "supported_by"},
                    {"parent_id": 2, "child_id": 3, "type": "supported_by"},
                ],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "attachment_chain")

    def test_l2_object_move_attachment_counts_are_left_unchanged_during_generation(self) -> None:
        objects = [
            make_object(1, "table"),
            make_object(2, "box"),
            make_object(3, "cup"),
        ]

        l2_move_questions = [
            make_l2_object_move_question("object_move_agent", attached=True, text="agent attached 1"),
            make_l2_object_move_question("object_move_agent", attached=True, text="agent attached 2"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 2"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 3"),
            make_l2_object_move_question("object_move_distance", attached=True, text="distance attached 1"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 1"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 2"),
        ]
        l2_object_centric_questions = [
            make_l2_object_move_question("object_rotate_object_centric", attached=True, text="oc attached 1"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="oc free 1"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="oc free 2"),
        ]
        l2_allocentric_questions = [
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 1"),
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 2"),
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 3"),
            make_l2_object_move_question("object_move_allocentric", attached=False, text="allo free 4"),
        ]
        viewpoint_questions = [
            {
                "level": "L2",
                "type": "viewpoint_move",
                "question": "viewpoint",
                "options": ["A", "B", "C", "D"],
                "answer": "A",
            },
        ]

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=l2_move_questions),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=viewpoint_questions),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=l2_object_centric_questions),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=l2_allocentric_questions),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
            patch("src.qa_generator._ensure_question_mentions", side_effect=lambda q, *_: q),
            patch("src.qa_generator._enforce_stable_facing_references", side_effect=lambda qs, *_: qs),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2, 3],
                label_counts=None,
                attachment_edges=[],
            )

        counts: dict[str, tuple[int, int]] = {}
        for q in questions:
            qtype = q.get("type")
            if not (
                str(qtype).startswith("object_move_")
                or str(qtype) == "object_rotate_object_centric"
            ):
                continue
            attached, unattached = counts.get(qtype, (0, 0))
            if q.get("attachment_remapped", False):
                attached += 1
            else:
                unattached += 1
            counts[qtype] = (attached, unattached)

        self.assertEqual(counts["object_move_agent"], (2, 3))
        self.assertEqual(counts["object_move_distance"], (1, 2))
        self.assertEqual(counts["object_rotate_object_centric"], (1, 2))
        self.assertEqual(counts.get("object_move_allocentric", (0, 0)), (0, 4))
        self.assertEqual(sum(1 for q in questions if q.get("type") == "viewpoint_move"), 1)

    def test_generate_all_questions_drops_questions_with_nonreferable_mentions(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]

        leaked_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup relative to the lamp?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "query_object", "obj_id": 1, "label": "cup"},
                {"role": "reference_object", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1]},
                attached_by={1: 2},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 3],
                attachment_edges=[
                    {"parent_id": 2, "child_id": 1, "type": "supported_by"},
                ],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_questions_via_obj_c_fallback(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]

        leaked_question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup relative to the lamp?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "query_obj_id": 1,
            "query_obj_label": "cup",
            "obj_c_id": 3,
            "obj_c_label": "lamp",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "query_object", "obj_id": 1, "label": "cup"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1]},
                attached_by={1: 2},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2],
                label_statuses={"cup": "unique", "table": "unique", "lamp": "absent"},
                label_to_object_ids={"cup": [1], "table": [2], "lamp": [3]},
                attachment_edges=[
                    {"parent_id": 2, "child_id": 1, "type": "supported_by"},
                ],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_drops_same_object_multi_role_from_fallback_fields(self) -> None:
        objects = [make_object(1, "chair")]

        leaked_question = {
            "level": "L1",
            "type": "direction_agent",
            "question": "Where is the chair relative to itself?",
            "options": ["left", "right", "front", "back"],
            "answer": "A",
            "correct_value": "left",
            "obj_a_id": 1,
            "obj_a_label": "chair",
            "obj_b_id": 1,
            "obj_b_label": "chair",
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[leaked_question]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1],
                referable_object_ids=[1],
                label_statuses={"chair": "unique"},
                label_to_object_ids={"chair": [1]},
                attachment_edges=[],
            )

        self.assertEqual(questions, [])

    def test_generate_all_questions_keeps_static_occlusion_not_visible(self) -> None:
        objects = [make_object(3, "lamp")]
        occlusion_question = {
            "level": "L1",
            "type": "occlusion",
            "question": "Is the lamp visible?",
            "options": ["fully visible", "partially occluded", "not visible"],
            "answer": "C",
            "correct_value": "not visible",
            "obj_a_id": None,
            "obj_a_label": "lamp",
            "occlusion_decision_source": "vlm_out_of_frame_label_review",
            "mentioned_objects": [
                {"role": "target", "label": "lamp", "obj_id": None},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[occlusion_question]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
            patch("src.qa_generator._ensure_question_mentions", side_effect=lambda q, *_: q),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[3],
                referable_object_ids=[],
                occlusion_eligible_object_ids=[],
                label_statuses={},
                label_to_object_ids={},
                out_of_frame_not_visible_labels=["lamp"],
                out_of_frame_label_to_object_ids={"lamp": [3]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "occlusion")
        self.assertEqual(questions[0]["correct_value"], "not visible")
        self.assertEqual(
            questions[0]["question_referability_audit"]["decision"],
            "pass",
        )
        self.assertTrue(
            any(
                bool(item.get("exempt"))
                for item in questions[0]["question_referability_audit"]["mentioned_objects"]
            )
        )

    def test_in_frame_filter_keeps_static_occlusion_not_visible(self) -> None:
        kept = _enforce_in_frame_mentions(
            [
                {
                    "level": "L1",
                    "type": "occlusion",
                    "question": "Is the lamp visible?",
                    "options": ["fully visible", "partially occluded", "not visible"],
                    "answer": "C",
                    "correct_value": "not visible",
                    "mentioned_objects": [
                        {"role": "target", "label": "lamp", "obj_id": 3},
                    ],
                }
            ],
            [],
        )

        self.assertEqual(len(kept), 1)

    def test_generate_all_questions_keeps_visible_static_occlusion_target_without_ratio_gate(self) -> None:
        objects = [make_object(3, "chair")]
        occlusion_question = {
            "level": "L1",
            "type": "occlusion",
            "question": "Is the chair occluded?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "B",
            "correct_value": "occluded",
            "obj_a_id": 3,
            "obj_a_label": "chair",
            "mentioned_objects": [
                {"role": "target", "label": "chair", "obj_id": 3},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[occlusion_question]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[3],
                referable_object_ids=[3],
                occlusion_eligible_object_ids=[],
                mention_in_frame_ratio_by_obj_id={3: 0.59},
                label_statuses={"chair": "unique"},
                label_to_object_ids={"chair": [3]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "occlusion")

    def test_in_frame_filter_keeps_object_move_allocentric_when_mentions_are_visible(self) -> None:
        kept = _enforce_in_frame_mentions(
            [
                {
                    "level": "L2",
                    "type": "object_move_allocentric",
                    "question": "If the table moves east, where is the cup from the chair?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                    "correct_value": "north",
                    "mentioned_objects": [
                        {"role": "moved_object", "label": "table", "obj_id": 2},
                        {"role": "query_object", "label": "cup", "obj_id": 1},
                        {"role": "reference_object", "label": "chair", "obj_id": 3},
                    ],
                }
            ],
            None,
            visible_object_ids=[1, 2, 3],
            mention_in_frame_ratio_by_obj_id={1: 0.49, 2: 0.20, 3: 0.05},
        )

        self.assertEqual(len(kept), 1)

    def test_in_frame_filter_keeps_coordinate_rotation_agent_when_mentions_are_visible(self) -> None:
        question = {
            "level": "L3",
            "type": "coordinate_rotation_agent",
            "question": "After room rotation, where is the cup from the table?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "mentioned_objects": [
                {"role": "obj_a", "label": "cup", "obj_id": 1},
                {"role": "obj_b", "label": "table", "obj_id": 2},
            ],
        }

        kept_below = _enforce_in_frame_mentions(
            [question],
            None,
            visible_object_ids=[1, 2],
            mention_in_frame_ratio_by_obj_id={1: 0.50, 2: 0.49},
        )
        kept_at_threshold = _enforce_in_frame_mentions(
            [question],
            None,
            visible_object_ids=[1, 2],
            mention_in_frame_ratio_by_obj_id={1: 0.50, 2: 0.50},
        )

        self.assertEqual(len(kept_below), 1)
        self.assertEqual(len(kept_at_threshold), 1)

    def test_in_frame_filter_keeps_attachment_chain_when_mentions_are_visible(self) -> None:
        kept = _enforce_in_frame_mentions(
            [
                {
                    "level": "L3",
                    "type": "attachment_chain",
                    "question": "If the table moves, which objects also move?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "C",
                    "correct_value": "Both the box and the cup",
                    "mentioned_objects": [
                        {"role": "grandparent", "label": "table", "obj_id": 1},
                        {"role": "parent", "label": "box", "obj_id": 2},
                        {"role": "grandchild", "label": "cup", "obj_id": 3},
                    ],
                }
            ],
            None,
            visible_object_ids=[1, 2, 3],
            mention_in_frame_ratio_by_obj_id={1: 0.95, 2: 0.60, 3: 0.59},
        )

        self.assertEqual(len(kept), 1)

    def test_generate_all_questions_does_not_flag_explicit_role_with_matching_legacy_alias(self) -> None:
        objects = [make_object(1, "cup")]
        question = {
            "level": "L2",
            "type": "object_move_agent",
            "question": "If the table moves, where is the cup?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "left",
            "query_obj_id": 1,
            "query_obj_label": "cup",
            "mentioned_objects": [
                {"role": "query_object", "obj_id": 1, "label": "cup"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1],
                referable_object_ids=[1],
                label_statuses={"cup": "unique"},
                label_to_object_ids={"cup": [1]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        audit = questions[0]["question_referability_audit"]
        self.assertEqual(audit["decision"], "pass")
        self.assertEqual(audit["reason_codes"], [])
        self.assertEqual(
            audit["mentioned_objects"][0]["explicit_roles"],
            ["query_object"],
        )
        self.assertEqual(
            audit["mentioned_objects"][0]["fallback_roles"],
            ["query_obj"],
        )

    def test_generate_all_questions_drops_object_move_occlusion_nonreferable_target(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
            make_object(3, "lamp"),
        ]
        leaked_question = {
            "level": "L2",
            "type": "object_move_occlusion",
            "question": "If the table moves, is the lamp visible from the camera?",
            "options": ["fully visible", "partially occluded", "not visible", "unknown"],
            "answer": "C",
            "correct_value": "not visible",
            "moved_obj_id": 2,
            "moved_obj_label": "table",
            "target_obj_id": 3,
            "target_obj_label": "lamp",
            "mentioned_objects": [
                {"role": "moved_object", "obj_id": 2, "label": "table"},
                {"role": "target_object", "obj_id": 3, "label": "lamp"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[leaked_question]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={2: [1]},
                attached_by={1: 2},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2, 3],
                referable_object_ids=[1, 2],
                label_statuses={"cup": "unique", "table": "unique", "lamp": "absent"},
                label_to_object_ids={"cup": [1], "table": [2], "lamp": [3]},
                attachment_edges=[
                    {"parent_id": 2, "child_id": 1, "type": "supported_by"},
                ],
            )

        self.assertEqual(questions, [])

    def test_in_frame_filter_keeps_object_move_occlusion_when_mentions_are_visible(self) -> None:
        kept = _enforce_in_frame_mentions(
            [
                {
                    "level": "L2",
                    "type": "object_move_occlusion",
                    "question": "If the table moves, is the lamp visible from the camera?",
                    "options": ["A", "B", "C"],
                    "answer": "C",
                    "correct_value": "not visible",
                    "mentioned_objects": [
                        {"role": "moved_object", "obj_id": 2, "label": "table"},
                        {"role": "target_object", "obj_id": 3, "label": "lamp"},
                    ],
                }
            ],
            None,
            visible_object_ids=[1, 2, 3],
            mention_in_frame_ratio_by_obj_id={1: 0.95, 2: 0.20, 3: 0.30},
        )

        self.assertEqual(len(kept), 1)

    def test_generate_all_questions_keeps_direction_agent_question_without_ratio_gate(self) -> None:
        objects = [
            make_object(1, "cup"),
            make_object(2, "table"),
        ]
        direction_question = {
            "level": "L1",
            "type": "direction_agent",
            "question": "Where is the cup relative to the table?",
            "options": ["left", "right", "front", "behind"],
            "answer": "A",
            "correct_value": "left",
            "obj_a_id": 1,
            "obj_a_label": "cup",
            "obj_b_id": 2,
            "obj_b_label": "table",
            "mentioned_objects": [
                {"role": "query_object", "obj_id": 1, "label": "cup"},
                {"role": "reference_object", "obj_id": 2, "label": "table"},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[direction_question]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2],
                referable_object_ids=[1, 2],
                occlusion_eligible_object_ids=[],
                mention_in_frame_ratio_by_obj_id={1: 0.95, 2: 0.49},
                label_statuses={"cup": "unique", "table": "unique"},
                label_to_object_ids={"cup": [1], "table": [2]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "direction_agent")

    def test_generate_all_questions_keeps_viewpoint_move_target_without_ratio_gate(self) -> None:
        objects = [make_object(1, "chair")]
        viewpoint_question = {
            "level": "L2",
            "type": "viewpoint_move",
            "question": "If the camera moves right, what is the occlusion status of the chair?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "A",
            "correct_value": "not occluded",
            "obj_a_id": 1,
            "obj_a_label": "chair",
            "mentioned_objects": [
                {"role": "target", "label": "chair", "obj_id": 1},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[viewpoint_question]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1],
                referable_object_ids=[1],
                occlusion_eligible_object_ids=[],
                mention_in_frame_ratio_by_obj_id={1: 0.59},
                label_statuses={"chair": "unique"},
                label_to_object_ids={"chair": [1]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "viewpoint_move")

    def test_generate_all_questions_keeps_object_remove_pair_without_ratio_gate(self) -> None:
        objects = [
            make_object(1, "chair"),
            make_object(2, "table"),
        ]
        remove_question = {
            "level": "L2",
            "type": "object_remove",
            "question": "If the table were removed, what would be the occlusion status of the chair?",
            "options": ["not occluded", "occluded", "not visible"],
            "answer": "A",
            "correct_value": "not occluded",
            "removed_obj_id": 2,
            "removed_obj_label": "table",
            "obj_b_id": 1,
            "obj_b_label": "chair",
            "mentioned_objects": [
                {"role": "removed_object", "label": "table", "obj_id": 2},
                {"role": "remaining_object", "label": "chair", "obj_id": 1},
            ],
        }

        with (
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator.generate_l1_occlusion_questions", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l1_direction_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move", return_value=[]),
            patch("src.qa_generator.generate_l2_viewpoint_move", return_value=[]),
            patch("src.qa_generator.generate_l2_object_remove", return_value=[remove_question]),
            patch("src.qa_generator.generate_l2_object_rotate_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l2_object_move_allocentric", return_value=[]),
            patch("src.qa_generator.generate_l3_attachment_chain", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_object_centric", return_value=[]),
            patch("src.qa_generator.generate_l3_coordinate_rotation_allocentric", return_value=[]),
        ):
            questions = generate_all_questions(
                objects=objects,
                attachment_graph={},
                attached_by={},
                support_chain_graph={},
                support_chain_by={},
                camera_pose=make_camera_pose(),
                templates={},
                visible_object_ids=[1, 2],
                referable_object_ids=[1, 2],
                occlusion_eligible_object_ids=[],
                mention_in_frame_ratio_by_obj_id={1: 0.95, 2: 0.59},
                label_statuses={"chair": "unique", "table": "unique"},
                label_to_object_ids={"chair": [1], "table": [2]},
                attachment_edges=[],
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["type"], "object_remove")

    def test_l1_occlusion_skips_multiple_status_without_unique_instance(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cup")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="depth",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            label_statuses={"cup": "multiple"},
            referable_object_ids=[1],
        )

        self.assertEqual(questions, [])

    def test_l1_occlusion_skips_unique_status_when_only_candidate_is_not_referable(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cabinet")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="depth",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            label_statuses={"cabinet": "unique"},
            referable_object_ids=[],
        )

        self.assertEqual(questions, [])

    def test_l1_occlusion_never_computes_geometry_for_nonreferable_objects(self) -> None:
        with (
            patch(
                "src.qa_generator._compute_l1_occlusion_metrics",
                return_value=object(),
            ) as metrics_mock,
            patch(
                "src.qa_generator._resolve_l1_occlusion_decision",
                return_value=(None, "geometry_from_vlm_unique", False),
            ),
        ):
            questions = generate_l1_occlusion_questions(
                objects=[make_object(1, "chair"), make_object(2, "cabinet")],
                camera_pose=make_camera_pose(),
                color_intrinsics=None,
                depth_image=None,
                depth_intrinsics=None,
                occlusion_backend="mesh_ray",
                ray_caster=None,
                instance_mesh_data=None,
                templates={},
                label_statuses={"chair": "unique", "cabinet": "unique"},
                referable_object_ids=[1],
            )

        self.assertEqual(questions, [])
        self.assertEqual(metrics_mock.call_count, 1)
        self.assertEqual(int(metrics_mock.call_args.kwargs["obj"]["id"]), 1)

    def test_l1_occlusion_generates_not_visible_from_out_of_frame_label_review(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cup")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            out_of_frame_not_visible_labels=["cup"],
            out_of_frame_label_to_object_ids={"cup": [1]},
        )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["correct_value"], "not visible")
        self.assertEqual(
            questions[0]["occlusion_decision_source"],
            "vlm_out_of_frame_label_review",
        )
        self.assertIsNone(questions[0]["obj_a_id"])

    def test_l1_occlusion_requires_out_of_frame_label_mapping_to_emit_not_visible(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cup")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            out_of_frame_not_visible_labels=["cup"],
            out_of_frame_label_to_object_ids={},
        )

        self.assertEqual(questions, [])

    def test_l1_occlusion_scans_out_of_frame_labels_until_it_finds_a_mapping(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cup")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            out_of_frame_not_visible_labels=["bad", "cup"],
            out_of_frame_label_to_object_ids={"cup": [1]},
        )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["correct_value"], "not visible")
        self.assertEqual(questions[0]["occlusion_decision_source"], "vlm_out_of_frame_label_review")

    def test_l1_occlusion_emits_at_most_one_not_visible_question_from_out_of_frame_labels(self) -> None:
        questions = generate_l1_occlusion_questions(
            objects=[make_object(1, "cup"), make_object(2, "lamp")],
            camera_pose=make_camera_pose(),
            color_intrinsics=None,
            depth_image=None,
            depth_intrinsics=None,
            occlusion_backend="mesh_ray",
            ray_caster=None,
            instance_mesh_data=None,
            templates={},
            out_of_frame_not_visible_labels=["cup", "lamp"],
            out_of_frame_label_to_object_ids={"cup": [1], "lamp": [2]},
        )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["correct_value"], "not visible")

    def test_leaf_attached_child_stays_its_own_l2_intervention_source(self) -> None:
        child = make_object(1, "cup")
        hidden_parent = make_object(2, "table")
        ref = make_object(3, "chair")
        face = make_object(4, "lamp")
        moved_child = {
            **child,
            "center": [1.5, 0.0, 1.0],
            "bbox_min": [1.5, 0.0, 0.5],
            "bbox_max": [1.7, 0.2, 1.5],
        }
        attachment_graph = {2: [1]}
        attached_by = {1: 2}
        movement_objects = [child, hidden_parent, ref, face]
        moved_scene_objects = [moved_child, hidden_parent, ref, face]
        object_map = {obj["id"]: obj for obj in movement_objects}
        moved_object_map = {obj["id"]: obj for obj in moved_scene_objects}
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=moved_scene_objects,
            moved_ids={1},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                side_effect=lambda *args, **kwargs: selected_state if args[2] == 1 else None,
            ),
            patch("src.qa_generator._iter_valid_object_move_states", return_value=[]),
            patch(
                "src.qa_generator.compute_all_relations",
                side_effect=[
                    [{
                        "obj_a_id": 1,
                        "obj_b_id": 3,
                        "direction_b_rel_a": "left",
                    }],
                    [{
                        "obj_a_id": 1,
                        "obj_b_id": 3,
                        "direction_b_rel_a": "front-left",
                    }],
                ],
            ),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            move_questions = generate_l2_object_move(
                objects=[child, ref],
                attachment_graph=attachment_graph,
                attached_by=attached_by,
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=object_map,
            )

        move_question = next(q for q in move_questions if q.get("type") == "object_move_agent")
        self.assertEqual(move_question["moved_obj_id"], 1)
        self.assertEqual(move_question["moved_obj_label"], "cup")
        self.assertFalse(move_question["attachment_remapped"])

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                side_effect=lambda objects, graph, target_id, pivot_id, **kwargs: (
                    [{
                        "angle": 90,
                        "rotation_direction": "clockwise",
                        "signed_angle": -90,
                        "objects": moved_scene_objects,
                    }]
                    if target_id == 1 and pivot_id == 4
                    else []
                ),
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                side_effect=[("left", 0.1), ("front", 0.1)],
            ),
        ):
            rotate_questions = generate_l2_object_rotate_object_centric(
                objects=[child, ref, face],
                attachment_graph=attachment_graph,
                attached_by=attached_by,
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=object_map,
            )

        rotate_question = next(q for q in rotate_questions if q.get("type") == "object_rotate_object_centric")
        self.assertEqual(rotate_question["moved_obj_id"], 1)
        self.assertFalse(rotate_question["attachment_remapped"])

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                side_effect=lambda *args, **kwargs: (
                    SimpleNamespace(
                        delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
                        moved_objects=moved_scene_objects,
                        moved_ids={1},
                    )
                    if args[2] == 1
                    else None
                ),
            ),
            patch(
                "src.qa_generator.primary_direction_allocentric",
                side_effect=[("east", 0.1), ("north", 0.1)],
            ),
        ):
            allocentric_questions = generate_l2_object_move_allocentric(
                objects=[child, ref],
                attachment_graph=attachment_graph,
                attached_by=attached_by,
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=moved_object_map,
            )

        allocentric_question = next(q for q in allocentric_questions if q.get("type") == "object_move_allocentric")
        self.assertEqual(allocentric_question["moved_obj_id"], 1)
        self.assertFalse(allocentric_question["attachment_remapped"])

    def test_attachment_referable_parent_source_is_enumerated_from_movement_objects(self) -> None:
        child = make_object(1, "cup")
        hidden_parent = make_object(2, "table")
        ref = make_object(3, "chair")
        face = make_object(4, "lamp")
        moved_child = {
            **child,
            "center": [1.5, 0.0, 1.0],
            "bbox_min": [1.5, 0.0, 0.5],
            "bbox_max": [1.7, 0.2, 1.5],
        }
        moved_parent = {
            **hidden_parent,
            "center": [2.5, 0.0, 1.0],
            "bbox_min": [2.5, 0.0, 0.5],
            "bbox_max": [2.7, 0.2, 1.5],
        }
        movement_objects = [child, hidden_parent, ref, face]
        moved_scene_objects = [moved_child, moved_parent, ref, face]
        object_map = {obj["id"]: obj for obj in movement_objects}
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=moved_scene_objects,
            moved_ids={1, 2},
        )

        def select_parent_state(*args, **_kwargs):
            return selected_state if args[2] == 2 else None

        with (
            patch("src.qa_generator._select_object_move_state", side_effect=select_parent_state),
            patch(
                "src.qa_generator.compute_all_relations",
                side_effect=[
                    [{
                        "obj_a_id": 1,
                        "obj_b_id": 3,
                        "direction_b_rel_a": "left",
                    }],
                    [{
                        "obj_a_id": 1,
                        "obj_b_id": 3,
                        "direction_b_rel_a": "front-left",
                    }],
                ],
            ),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            move_questions = generate_l2_object_move(
                objects=[child, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=object_map,
                attachment_referable_object_ids=[2],
            )

        move_question = next(q for q in move_questions if q.get("type") == "object_move_agent")
        self.assertEqual(move_question["moved_obj_id"], 2)
        self.assertEqual(move_question["query_obj_id"], 1)
        self.assertTrue(move_question["attachment_remapped"])
        self.assertEqual(move_question["trace_reason"], "attachment_agent_relation_change")

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                side_effect=lambda _objects, _graph, target_id, pivot_id, **_kwargs: (
                    [{
                        "angle": 90,
                        "rotation_direction": "clockwise",
                        "signed_angle": -90,
                        "objects": moved_scene_objects,
                    }]
                    if target_id == 2 and pivot_id == 4
                    else []
                ),
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                side_effect=[("left", 0.1), ("front", 0.1)],
            ),
        ):
            rotate_questions = generate_l2_object_rotate_object_centric(
                objects=[child, ref, face],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_rotate_object_centric": [
                        "rotate {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=movement_objects,
                object_map=object_map,
                attachment_referable_object_ids=[2],
            )

        rotate_question = next(q for q in rotate_questions if q.get("type") == "object_rotate_object_centric")
        self.assertEqual(rotate_question["moved_obj_id"], 2)
        self.assertEqual(rotate_question["query_obj_id"], 1)
        self.assertTrue(rotate_question["attachment_remapped"])
        self.assertEqual(
            rotate_question["trace_reason"],
            "attachment_object_centric_relation_change",
        )

        with (
            patch("src.qa_generator._select_object_move_state", side_effect=select_parent_state),
            patch(
                "src.qa_generator.primary_direction_allocentric",
                side_effect=[("east", 0.1), ("north", 0.1)],
            ),
        ):
            allocentric_questions = generate_l2_object_move_allocentric(
                objects=[child, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=object_map,
                attachment_referable_object_ids=[2],
            )

        allocentric_question = next(
            q for q in allocentric_questions if q.get("type") == "object_move_allocentric"
        )
        self.assertEqual(allocentric_question["moved_obj_id"], 2)
        self.assertEqual(allocentric_question["query_obj_id"], 1)
        self.assertTrue(allocentric_question["attachment_remapped"])
        self.assertEqual(
            allocentric_question["trace_reason"],
            "attachment_allocentric_relation_change",
        )

    def test_attachment_chain_questions_include_trace_reason(self) -> None:
        grandparent = make_object(1, "bed")
        parent = make_object(2, "pillow")
        grandchild = make_object(3, "book")
        neighbor = make_object(4, "chair")

        questions = generate_l3_attachment_chain(
            objects=[grandparent, parent, grandchild, neighbor],
            attachment_graph={1: [2], 2: [3]},
            attached_by={2: 1, 3: 2},
            camera_pose=make_camera_pose(),
            templates={
                "L3_attachment_chain": [
                    "If {obj_a} moves, which objects move with it?"
                ]
            },
        )

        self.assertTrue(questions)
        question = questions[0]
        self.assertEqual(question["type"], "attachment_chain")
        self.assertEqual(question["trace_reason"], "attachment_chain_two_hop_inference")

    def test_nonreferable_movement_source_is_still_filtered_out(self) -> None:
        child = make_object(1, "cup")
        hidden_parent = make_object(2, "table")
        ref = make_object(3, "chair")
        face = make_object(4, "lamp")
        moved_child = {
            **child,
            "center": [1.5, 0.0, 1.0],
            "bbox_min": [1.5, 0.0, 0.5],
            "bbox_max": [1.7, 0.2, 1.5],
        }
        moved_parent = {
            **hidden_parent,
            "center": [2.5, 0.0, 1.0],
            "bbox_min": [2.5, 0.0, 0.5],
            "bbox_max": [2.7, 0.2, 1.5],
        }
        movement_objects = [child, hidden_parent, ref, face]
        moved_scene_objects = [moved_child, moved_parent, ref, face]
        object_map = {obj["id"]: obj for obj in movement_objects}
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=moved_scene_objects,
            moved_ids={1, 2},
        )
        move_source_calls: list[int] = []
        rotate_source_calls: list[int] = []
        allocentric_source_calls: list[int] = []

        def record_move_source(*args, **_kwargs):
            move_source_calls.append(args[2])
            return selected_state

        def record_allocentric_source(*args, **_kwargs):
            allocentric_source_calls.append(args[2])
            return selected_state

        with (
            patch("src.qa_generator._select_object_move_state", side_effect=record_move_source),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            move_questions = generate_l2_object_move(
                objects=[child, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=object_map,
                attachment_referable_object_ids=[],
            )

        self.assertEqual(move_questions, [])
        self.assertEqual(move_source_calls, [])

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                side_effect=lambda _objects, _graph, target_id, _pivot_id, **_kwargs: (
                    rotate_source_calls.append(target_id) or [{
                        "angle": 90,
                        "rotation_direction": "clockwise",
                        "signed_angle": -90,
                        "objects": moved_scene_objects,
                    }]
                ),
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                return_value=("left", 0.1),
            ),
        ):
            rotate_questions = generate_l2_object_rotate_object_centric(
                objects=[child, ref, face],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_rotate_object_centric": [
                        "rotate {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=movement_objects,
                object_map=object_map,
                attachment_referable_object_ids=[],
            )

        self.assertEqual(rotate_questions, [])
        self.assertEqual(rotate_source_calls, [])

        with (
            patch("src.qa_generator._select_object_move_state", side_effect=record_allocentric_source),
            patch(
                "src.qa_generator.primary_direction_allocentric",
                return_value=("east", 0.1),
            ),
        ):
            allocentric_questions = generate_l2_object_move_allocentric(
                objects=[child, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=movement_objects,
                object_map=object_map,
                attachment_referable_object_ids=[],
            )

        self.assertEqual(allocentric_questions, [])
        self.assertEqual(allocentric_source_calls, [])

    def test_object_move_middle_node_source_only_moves_downward(self) -> None:
        root = make_object(1, "wardrobe")
        middle = make_object(2, "shelf")
        leaf = make_object(3, "box")
        ref = make_object(4, "chair")
        moved_middle = {
            **middle,
            "center": [2.5, 0.0, 1.0],
            "bbox_min": [2.5, 0.0, 0.5],
            "bbox_max": [2.7, 0.2, 1.5],
        }
        moved_leaf = {
            **leaf,
            "center": [3.5, 0.0, 1.0],
            "bbox_min": [3.5, 0.0, 0.5],
            "bbox_max": [3.7, 0.2, 1.5],
        }
        moved_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[root, moved_middle, moved_leaf, ref],
            moved_ids={2, 3},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                side_effect=lambda *args, **kwargs: moved_state if args[2] == 2 else None,
            ),
            patch("src.qa_generator._iter_valid_object_move_states", return_value=[]),
            patch(
                "src.qa_generator.compute_all_relations",
                side_effect=[
                    [{
                        "obj_a_id": 4,
                        "obj_b_id": 3,
                        "direction_b_rel_a": "left",
                    }],
                    [{
                        "obj_a_id": 4,
                        "obj_b_id": 3,
                        "direction_b_rel_a": "front-left",
                    }],
                ],
            ),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=[middle, leaf, ref],
                attachment_graph={1: [2], 2: [3]},
                attached_by={2: 1, 3: 2},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[root, middle, leaf, ref],
                object_map={obj["id"]: obj for obj in [root, middle, leaf, ref]},
            )

        question = next(q for q in questions if q.get("query_obj_id") == 3)
        self.assertEqual(question["moved_obj_id"], 2)
        self.assertEqual(question["moved_obj_label"], "shelf")
        self.assertTrue(question["attachment_remapped"])
        self.assertNotEqual(question["moved_obj_id"], 1)

    def test_object_move_parent_source_keeps_direct_attachment_pair_question(self) -> None:
        parent = make_object(1, "bed")
        child = make_object(2, "pillow")
        moved_parent = {
            **parent,
            "center": [1.5, 0.0, 1.0],
            "bbox_min": [1.5, 0.0, 0.5],
            "bbox_max": [1.7, 0.2, 1.5],
        }
        moved_child = {
            **child,
            "center": [2.5, 0.0, 1.0],
            "bbox_min": [2.5, 0.0, 0.5],
            "bbox_max": [2.7, 0.2, 1.5],
        }
        moved_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[moved_parent, moved_child],
            moved_ids={1, 2},
        )

        with (
            patch(
                "src.qa_generator._select_object_move_state",
                side_effect=lambda *args, **kwargs: moved_state if args[2] == 1 else None,
            ),
            patch("src.qa_generator._iter_valid_object_move_states", return_value=[]),
            patch(
                "src.qa_generator.compute_all_relations",
                side_effect=[
                    [{
                        "obj_a_id": 1,
                        "obj_b_id": 2,
                        "direction_b_rel_a": "right",
                    }],
                    [{
                        "obj_a_id": 1,
                        "obj_b_id": 2,
                        "direction_b_rel_a": "right",
                    }],
                ],
            ),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=[parent, child],
                attachment_graph={1: [2]},
                attached_by={2: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[parent, child],
                object_map={1: parent, 2: child},
            )

        self.assertTrue(
            any(
                q.get("moved_obj_id") == 1 and q.get("query_obj_id") == 1
                for q in questions
            )
        )
        question = next(q for q in questions if q.get("query_obj_id") == 2)
        self.assertEqual(question["moved_obj_id"], 1)
        self.assertTrue(question["attachment_remapped"])
        self.assertTrue(question["relation_unchanged"])

    def test_object_move_priority_child_overrides_attachment_pair_metadata(self) -> None:
        parent = make_object(1, "desk")
        first_child = make_object(2, "keyboard")
        priority_child = make_object(3, "bottle")
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[parent, first_child, priority_child],
            moved_ids={1, 2, 3},
        )

        def fake_distance_questions(*, query_obj, **_kwargs):
            if int(query_obj["id"]) != 3:
                return []
            return [{
                "level": "L2",
                "type": "object_move_distance",
                "question": "distance",
                "options": ["A", "B"],
                "answer": "A",
                "moved_obj_id": 1,
                "query_obj_id": 3,
                "attachment_remapped": True,
                "attachment_pair_id": "1->2",
                "attachment_parent_id": 1,
                "attachment_child_id": 2,
            }]

        with (
            patch("src.qa_generator._select_object_move_state", return_value=selected_state),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                side_effect=fake_distance_questions,
            ),
        ):
            questions = generate_l2_object_move(
                objects=[parent, first_child, priority_child],
                attachment_graph={1: [2, 3]},
                attached_by={2: 1, 3: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[parent, first_child, priority_child],
                object_map={1: parent, 2: first_child, 3: priority_child},
                attachment_priority_pairs=[(1, 3)],
                enabled_l2_object_move_types={"object_move_distance"},
            )

        question = questions[0]
        self.assertEqual(question["query_obj_id"], 3)
        self.assertTrue(question["attachment_priority_pair"])
        self.assertEqual(question["attachment_pair_id"], "1->3")
        self.assertEqual(question["attachment_child_id"], 3)

    def test_object_move_agent_marks_priority_child_query(self) -> None:
        parent = make_object(1, "desk")
        first_child = make_object(2, "keyboard")
        priority_child = make_object(3, "bottle")
        ref = make_object(4, "chair")
        moved_parent = {**parent, "center": [1.5, 0.0, 1.0]}
        moved_first_child = {**first_child, "center": [2.5, 0.0, 1.0]}
        moved_priority_child = {**priority_child, "center": [3.5, 0.0, 1.0]}
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[moved_parent, moved_first_child, moved_priority_child, ref],
            moved_ids={1, 2, 3},
        )
        relation_call_count = 0

        def fake_relations(*_args, **_kwargs):
            nonlocal relation_call_count
            relation_call_count += 1
            if relation_call_count == 1:
                return [{
                    "obj_a_id": 3,
                    "obj_b_id": 4,
                    "direction_b_rel_a": "left",
                }]
            return [{
                "obj_a_id": 3,
                "obj_b_id": 4,
                "direction_b_rel_a": "front-left",
            }]

        with (
            patch("src.qa_generator._select_object_move_state", return_value=selected_state),
            patch("src.qa_generator._iter_valid_object_move_states", return_value=[]),
            patch("src.qa_generator.compute_all_relations", side_effect=fake_relations),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=[parent, first_child, priority_child, ref],
                attachment_graph={1: [2, 3]},
                attached_by={2: 1, 3: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[parent, first_child, priority_child, ref],
                object_map={obj["id"]: obj for obj in [parent, first_child, priority_child, ref]},
                attachment_priority_pairs=[(1, 3)],
                enabled_l2_object_move_types={"object_move_agent"},
            )

        question = next(q for q in questions if q.get("type") == "object_move_agent")
        self.assertEqual(question["query_obj_id"], 3)
        self.assertTrue(question["attachment_priority_pair"])
        self.assertEqual(question["attachment_pair_id"], "1->3")
        self.assertEqual(question["attachment_child_id"], 3)

    def test_object_move_agent_allows_priority_child_unchanged_fallback(self) -> None:
        parent = make_object(1, "desk")
        priority_child = make_object(3, "bottle")
        ref = make_object(4, "chair")
        selected_state = SimpleNamespace(
            delta=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            moved_objects=[parent, priority_child, ref],
            moved_ids={1, 3},
        )

        with (
            patch("src.qa_generator._select_object_move_state", return_value=selected_state),
            patch("src.qa_generator._iter_valid_object_move_states", return_value=[]),
            patch(
                "src.qa_generator.compute_all_relations",
                return_value=[{
                    "obj_a_id": 3,
                    "obj_b_id": 4,
                    "direction_b_rel_a": "left",
                }],
            ),
            patch("src.qa_generator._generate_l2_distance_questions_for_object", return_value=[]),
        ):
            questions = generate_l2_object_move(
                objects=[parent, priority_child, ref],
                attachment_graph={1: [3]},
                attached_by={3: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[parent, priority_child, ref],
                object_map={obj["id"]: obj for obj in [parent, priority_child, ref]},
                attachment_priority_pairs=[(1, 3)],
                enabled_l2_object_move_types={"object_move_agent"},
            )

        question = next(q for q in questions if q.get("type") == "object_move_agent")
        self.assertEqual(question["query_obj_id"], 3)
        self.assertTrue(question["relation_unchanged"])
        self.assertTrue(question["attachment_priority_pair"])
        self.assertEqual(question["attachment_pair_id"], "1->3")

    def test_object_move_distance_allows_priority_child_unchanged_fallback(self) -> None:
        parent = make_object(1, "desk")
        priority_child = make_object(3, "bottle")
        captured = []

        def fake_distance_questions(**kwargs):
            captured.append((
                int(kwargs["move_source_id"]),
                int(kwargs["query_obj"]["id"]),
                kwargs.get("allow_unchanged_fallback"),
            ))
            return [{
                "level": "L2",
                "type": "object_move_distance",
                "question": "distance",
                "options": ["A", "B"],
                "answer": "A",
                "moved_obj_id": 1,
                "query_obj_id": 3,
                "attachment_remapped": True,
            }]

        with (
            patch("src.qa_generator._select_object_move_state", return_value=None),
            patch("src.qa_generator.compute_all_relations", return_value=[]),
            patch(
                "src.qa_generator._generate_l2_distance_questions_for_object",
                side_effect=fake_distance_questions,
            ),
        ):
            questions = generate_l2_object_move(
                objects=[parent, priority_child],
                attachment_graph={1: [3]},
                attached_by={3: 1},
                camera_pose=make_camera_pose(),
                templates={},
                movement_objects=[parent, priority_child],
                object_map={1: parent, 3: priority_child},
                attachment_priority_pairs=[(1, 3)],
                enabled_l2_object_move_types={"object_move_distance"},
            )

        self.assertIn((1, 3, True), captured)
        question = questions[0]
        self.assertTrue(question["attachment_priority_pair"])
        self.assertEqual(question["attachment_pair_id"], "1->3")
        self.assertEqual(question["attachment_child_id"], 3)

    def test_rotate_generator_prefers_changed_rotation_over_larger_unchanged_attachment_fallback(self) -> None:
        child = make_object(1, "cup")
        parent = make_object(2, "table")
        face = make_object(3, "lamp")
        ref = make_object(4, "chair")
        movement_objects = [child, parent, face, ref]
        rotated_180 = [
            {**child, "center": [1.0, 0.0, 1.0]},
            {**parent, "center": [2.0, 0.0, 1.0]},
            face,
            ref,
        ]
        rotated_135 = [
            {**child, "center": [0.0, 1.0, 1.0]},
            {**parent, "center": [1.0, 1.0, 1.0]},
            face,
            ref,
        ]
        old_child_center = np.array(child["center"], dtype=np.float64)
        changed_child_center = np.array(rotated_135[0]["center"], dtype=np.float64)
        ref_center = np.array(ref["center"], dtype=np.float64)

        def object_centric_direction(anchor_center, _face_center, target_center, **_kwargs):
            anchor = np.asarray(anchor_center, dtype=np.float64)
            target = np.asarray(target_center, dtype=np.float64)
            if np.allclose(target, ref_center):
                if np.allclose(anchor, changed_child_center):
                    return "front", 0.1
                if np.allclose(anchor, old_child_center):
                    return "left", 0.1
            return "left", 0.9

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                side_effect=lambda _objects, _graph, target_id, pivot_id, **_kwargs: (
                    [
                        {
                            "angle": 180,
                            "rotation_direction": "clockwise",
                            "signed_angle": -180,
                            "objects": rotated_180,
                        },
                        {
                            "angle": 135,
                            "rotation_direction": "clockwise",
                            "signed_angle": -135,
                            "objects": rotated_135,
                        },
                    ]
                    if target_id == 2 and pivot_id == 3
                    else []
                ),
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                side_effect=object_centric_direction,
            ),
        ):
            questions = generate_l2_object_rotate_object_centric(
                objects=[parent, child, face, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_rotate_object_centric": [
                        "rotate {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=movement_objects,
                object_map={obj["id"]: obj for obj in movement_objects},
                attachment_referable_object_ids=[1, 2, 3, 4],
            )

        question = next(q for q in questions if q.get("query_obj_id") == 1)
        self.assertEqual(question["rotation_angle"], 135)
        self.assertEqual(question["rotation_direction"], "clockwise")
        self.assertFalse(question["relation_unchanged"])

    def test_downward_propagated_rotate_questions_can_be_kept_when_answer_is_unchanged(self) -> None:
        child = make_object(1, "cup")
        parent = make_object(2, "table")
        face = make_object(3, "lamp")
        ref = make_object(4, "chair")
        objects = [child, parent, face, ref]
        rotated_180 = [
            {**make_object(1, "cup"), "center": [1.0, 0.0, 1.0]},
            {**make_object(2, "table"), "center": [2.0, 0.0, 1.0]},
            face,
            ref,
        ]
        rotated_135 = [
            {**make_object(1, "cup"), "center": [0.0, 1.0, 1.0]},
            {**make_object(2, "table"), "center": [1.0, 1.0, 1.0]},
            face,
            ref,
        ]
        old_child_center = np.array(child["center"], dtype=np.float64)
        ref_center = np.array(ref["center"], dtype=np.float64)

        def object_centric_direction(anchor_center, _face_center, target_center, **_kwargs):
            anchor = np.asarray(anchor_center, dtype=np.float64)
            target = np.asarray(target_center, dtype=np.float64)
            if np.allclose(anchor, old_child_center) and np.allclose(target, ref_center):
                return "left", 0.1
            return "left", 0.9

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                side_effect=lambda _objects, _graph, target_id, pivot_id, **_kwargs: (
                    [
                        {
                            "angle": 180,
                            "rotation_direction": "clockwise",
                            "signed_angle": -180,
                            "objects": rotated_180,
                        },
                        {
                            "angle": 135,
                            "rotation_direction": "clockwise",
                            "signed_angle": -135,
                            "objects": rotated_135,
                        },
                    ]
                    if target_id == 2 and pivot_id == 3
                    else []
                ),
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                side_effect=object_centric_direction,
            ),
        ):
            questions = generate_l2_object_rotate_object_centric(
                objects=[child, parent, face, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_rotate_object_centric": [
                        "rotate {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=objects,
                object_map={obj["id"]: obj for obj in objects},
            )

        self.assertTrue(questions)
        question = next(q for q in questions if q.get("attachment_remapped"))
        self.assertEqual(question["type"], "object_rotate_object_centric")
        self.assertTrue(question["attachment_remapped"])
        self.assertTrue(question["relation_unchanged"])
        self.assertEqual(question["correct_value"], "left")
        self.assertEqual(question["old_correct_value"], "left")
        self.assertEqual(question["new_correct_value"], "left")
        self.assertTrue(question["has_attachment_chain"])
        self.assertEqual(question["rotation_angle"], 180)

    def test_rotate_parent_source_can_query_attached_child_without_upward_remap(self) -> None:
        child = make_object(1, "cup")
        parent = make_object(2, "table")
        face = make_object(3, "lamp")
        ref = make_object(4, "chair")
        movement_objects = [child, parent, face, ref]
        rotated_objects = [make_object(1, "cup"), make_object(2, "table"), face, ref]

        with (
            patch("src.qa_generator._has_stable_object_centric_facing", return_value=True),
            patch(
                "src.qa_generator.find_meaningful_orbit_rotation",
                return_value=[{
                    "angle": 90,
                    "rotation_direction": "clockwise",
                    "signed_angle": -90,
                    "objects": rotated_objects,
                }],
            ),
            patch(
                "src.qa_generator.primary_direction_object_centric",
                return_value=("left", 0.1),
            ),
        ):
            questions = generate_l2_object_rotate_object_centric(
                objects=[parent, child, face, ref],
                attachment_graph={2: [1]},
                attached_by={1: 2},
                camera_pose=make_camera_pose(),
                templates={
                    "L2_object_rotate_object_centric": [
                        "rotate {obj_move_source} {direction} by {distance}: where is {obj_ref} from {obj_query}?"
                    ]
                },
                movement_objects=movement_objects,
                object_map={obj["id"]: obj for obj in movement_objects},
                attachment_referable_object_ids=[1, 2, 3, 4],
            )

        question = next(q for q in questions if q.get("query_obj_id") == 1)
        self.assertEqual(question["moved_obj_id"], 2)
        self.assertEqual(question["moved_obj_label"], "table")
        self.assertTrue(question["attachment_remapped"])
        self.assertTrue(question["relation_unchanged"])

    def test_full_quality_pipeline_balances_l2_attachment_per_scene(self) -> None:
        from src.quality_control import full_quality_pipeline

        questions = [
            make_l2_object_move_question("object_move_agent", attached=True, text="agent attached 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 1"),
            make_l2_object_move_question("object_move_agent", attached=False, text="agent free 2"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 1"),
            make_l2_object_move_question("object_move_distance", attached=False, text="distance free 2"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 1"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 2"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 3"),
            make_l2_object_move_question("object_rotate_object_centric", attached=False, text="rotate free 4"),
            {
                "level": "L2",
                "type": "viewpoint_move",
                "question": "viewpoint",
                "options": ["fully visible", "partially occluded", "not visible", "unknown"],
                "answer": "A",
                "correct_value": "fully visible",
            },
        ]
        for q in questions:
            q.setdefault("correct_value", "A")
        for idx, q in enumerate(questions):
            if str(q.get("type", "")).startswith("object_move_") or q.get("type") == "object_rotate_object_centric":
                q["moved_obj_id"] = idx + 1

        filtered = full_quality_pipeline(questions)

        counts: dict[str, tuple[int, int]] = {}
        for q in filtered:
            qtype = str(q.get("type", ""))
            if not (qtype.startswith("object_move_") or qtype == "object_rotate_object_centric"):
                continue
            attached, unattached = counts.get(qtype, (0, 0))
            if q.get("attachment_remapped", False):
                attached += 1
            else:
                unattached += 1
            counts[qtype] = (attached, unattached)

        self.assertEqual(counts.get("object_move_agent", (0, 0)), (1, 0))
        self.assertEqual(counts.get("object_move_distance", (0, 0)), (0, 0))
        self.assertEqual(counts.get("object_rotate_object_centric", (0, 0)), (0, 0))
        self.assertEqual(sum(1 for q in filtered if q.get("type") == "viewpoint_move"), 1)

    def test_full_quality_pipeline_applies_l3_ratio_cap_and_global_dedup(self) -> None:
        from src.quality_control import full_quality_pipeline

        questions = []
        for idx in range(4):
            questions.append({
                "scene_id": "s1",
                "level": "L3",
                "type": "coordinate_rotation_agent",
                "question": f"changed {idx}",
                "options": ["north", "south", "east", "west"],
                "answer": "A",
                "correct_value": "north",
                "obj_a_id": idx + 10,
                "obj_b_id": idx + 20,
                "relation_unchanged": False,
            })
        questions.extend([
            {
                "scene_id": "s1",
                "level": "L3",
                "type": "coordinate_rotation_agent",
                "question": "unchanged first",
                "options": ["north", "south", "east", "west"],
                "answer": "A",
                "correct_value": "north",
                "obj_a_id": 1,
                "obj_b_id": 2,
                "relation_unchanged": True,
            },
            {
                "scene_id": "s1",
                "level": "L3",
                "type": "coordinate_rotation_agent",
                "question": "unchanged second",
                "options": ["north", "south", "east", "west"],
                "answer": "A",
                "correct_value": "north",
                "obj_a_id": 3,
                "obj_b_id": 4,
                "relation_unchanged": True,
            },
        ])
        for answer in ("A", "B", "C"):
            questions.append({
                "scene_id": "s1",
                "level": "L3",
                "type": "attachment_chain",
                "question": "same chain",
                "options": ["north", "south", "east", "west"],
                "answer": answer,
                "correct_value": "north",
                "grandparent_id": 1,
                "parent_id": 2,
                "grandchild_id": 3,
                "neighbor_id": 4,
                "relation_unchanged": False,
            })

        result = full_quality_pipeline(questions)

        unchanged_agent = [
            q for q in result
            if q["type"] == "coordinate_rotation_agent" and q.get("relation_unchanged") is True
        ]
        chain = [q for q in result if q["type"] == "attachment_chain"]
        self.assertEqual([q["question"] for q in unchanged_agent], ["unchanged first"])
        self.assertEqual(len(chain), 1)

    # ------------------------------------------------------------------
    # balance_l2_attachment_per_scene unit tests
    # ------------------------------------------------------------------

    def _make_att_question(
        self,
        qtype,
        attached,
        unchanged=False,
        pair_id="",
        scene="s1",
        mentioned_object_ids=None,
        text=None,
    ):
        q = make_l2_object_move_question(
            qtype,
            attached=attached,
            text=text or f"{qtype} text",
        )
        q["scene_id"] = scene
        if attached:
            q["relation_unchanged"] = unchanged
            if pair_id:
                q["attachment_pair_id"] = pair_id
            if mentioned_object_ids is not None:
                q["mentioned_objects"] = [
                    {"role": f"mentioned_{i}", "obj_id": obj_id, "label": f"obj{obj_id}"}
                    for i, obj_id in enumerate(mentioned_object_ids)
                ]
        return q

    def test_balance_changed_dedup_by_pair_id_and_other_mentions(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->2",
                mentioned_object_ids=[1, 2, 3],
            ),
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->2",
                mentioned_object_ids=[2, 1, 3],
            ),
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->3",
                mentioned_object_ids=[1, 3, 2],
            ),
        ]
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 2)
        pair_ids = [q.get("attachment_pair_id") for q in result]
        self.assertEqual(pair_ids, ["1->2", "1->3"])

    def test_balance_changed_same_pair_keeps_different_other_mentions(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->2",
                mentioned_object_ids=[1, 2, 3],
            ),
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->2",
                mentioned_object_ids=[1, 2, 4],
            ),
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual(len(result), 2)
        other_ids = [
            tuple(
                mention["obj_id"]
                for mention in q["mentioned_objects"]
                if mention["obj_id"] not in (1, 2)
            )
            for q in result
        ]
        self.assertEqual(other_ids, [(3,), (4,)])

    def test_balance_changed_same_pair_falls_back_to_legacy_object_ids(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            {
                **self._make_att_question("object_move_agent", attached=True, pair_id="1->2"),
                "obj_a_id": 1,
                "obj_b_id": 2,
                "obj_c_id": 3,
            },
            {
                **self._make_att_question("object_move_agent", attached=True, pair_id="1->2"),
                "obj_a_id": 1,
                "obj_b_id": 2,
                "obj_c_id": 3,
            },
            {
                **self._make_att_question("object_move_agent", attached=True, pair_id="1->2"),
                "obj_a_id": 1,
                "obj_b_id": 2,
                "obj_c_id": 4,
            },
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual([q["obj_c_id"] for q in result], [3, 4])

    def test_balance_unattached_cap(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = []
        for i in range(8):
            questions.append(self._make_att_question("object_move_agent", attached=True, pair_id=f"1->{i}"))
        for i in range(5):
            questions.append(self._make_att_question("object_move_agent", attached=False))
        result = balance_l2_attachment_per_scene(questions)
        attached_count = sum(1 for q in result if q.get("attachment_remapped"))
        unattached_count = sum(1 for q in result if not q.get("attachment_remapped"))
        self.assertEqual(attached_count, 8)
        self.assertEqual(unattached_count, 2)

    def test_balance_unchanged_cap(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = []
        for i in range(8):
            questions.append(self._make_att_question("object_move_agent", attached=True, pair_id=f"1->{i}"))
        for i in range(5):
            questions.append(self._make_att_question("object_move_agent", attached=True, pair_id=f"2->{i}", unchanged=True))
        result = balance_l2_attachment_per_scene(questions)
        unchanged_count = sum(1 for q in result if q.get("relation_unchanged"))
        self.assertEqual(unchanged_count, 2)

    def test_balance_priority_unchanged_does_not_bypass_changed_ratio(self):
        from src.quality_control import balance_l2_attachment_per_scene

        questions = [
            {
                **self._make_att_question(
                    "object_move_agent",
                    attached=True,
                    pair_id="1->2",
                    unchanged=True,
                    text="priority unchanged",
                ),
                "attachment_priority_pair": True,
            }
        ]

        self.assertEqual(balance_l2_attachment_per_scene(questions), [])

        questions = [
            self._make_att_question("object_move_agent", attached=True, pair_id=f"1->{i}", text=f"changed {i}")
            for i in range(4)
        ] + [
            {
                **self._make_att_question(
                    "object_move_agent",
                    attached=True,
                    pair_id="2->3",
                    unchanged=True,
                    text="priority unchanged",
                ),
                "attachment_priority_pair": True,
            },
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="3->4",
                unchanged=True,
                text="normal unchanged",
            ),
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual(
            [q["question"] for q in result if q.get("relation_unchanged")],
            ["priority unchanged"],
        )

    def test_balance_zero_changed_caps_to_zero(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            self._make_att_question("object_move_agent", attached=False),
            self._make_att_question("object_move_agent", attached=False),
        ]
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 0)

    def test_balance_one_changed_caps_to_zero(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            self._make_att_question("object_move_agent", attached=True, pair_id="1->2"),
            self._make_att_question("object_move_agent", attached=False),
            self._make_att_question("object_move_agent", attached=True, unchanged=True, pair_id="1->2"),
        ]
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].get("attachment_remapped"))
        self.assertFalse(result[0].get("relation_unchanged"))

    def test_balance_three_changed_caps_to_zero(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = []
        for i in range(3):
            questions.append(self._make_att_question("object_move_agent", attached=True, pair_id=f"1->{i}"))
        for i in range(2):
            questions.append(self._make_att_question("object_move_agent", attached=False))
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 3)

    def test_balance_changed_dedup_does_not_cross_scene_boundaries(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->2",
                scene="s1",
                text="s1 changed",
                mentioned_object_ids=[1, 2, 3],
            ),
            self._make_att_question(
                "object_move_agent",
                attached=True,
                pair_id="1->2",
                scene="s2",
                text="s2 changed",
                mentioned_object_ids=[1, 2, 3],
            ),
        ]
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            [(q["scene_id"], q["question"]) for q in result],
            [("s1", "s1 changed"), ("s2", "s2 changed")],
        )

    def test_balance_different_qtypes_independent(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            self._make_att_question("object_move_agent", attached=True, pair_id="1->2"),
            self._make_att_question("object_move_agent", attached=False),
            self._make_att_question("object_move_distance", attached=True, pair_id="1->2"),
            self._make_att_question("object_move_distance", attached=False),
        ]
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 2)

    def test_balance_preserves_generation_order(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = []
        for i in range(4):
            questions.append(
                self._make_att_question(
                    "object_move_agent",
                    attached=True,
                    pair_id=f"1->{i}",
                    text=f"changed {i}",
                )
            )
            questions.append(
                self._make_att_question(
                    "object_move_agent",
                    attached=False,
                    text=f"free {i}",
                )
            )
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 5)
        self.assertEqual(
            [q["question"] for q in result],
            ["changed 0", "free 0", "changed 1", "changed 2", "changed 3"],
        )

    def test_balance_non_l2_move_passes_through(self):
        from src.quality_control import balance_l2_attachment_per_scene
        questions = [
            {"level": "L1", "type": "distance", "question": "L1 dist"},
            {"level": "L3", "type": "attachment_chain", "question": "L3 chain"},
        ]
        result = balance_l2_attachment_per_scene(questions)
        self.assertEqual(len(result), 2)

    def test_balance_unchanged_cap_is_batch_level_per_qtype(self):
        from src.quality_control import balance_l2_attachment_per_scene

        questions = [
            self._make_att_question("object_move_agent", attached=True, pair_id="1->2", scene="s1", text="s1 c1"),
            self._make_att_question("object_move_agent", attached=True, pair_id="1->3", scene="s1", text="s1 c2"),
            self._make_att_question("object_move_agent", attached=True, unchanged=True, pair_id="4->5", scene="s1", text="s1 u1"),
            self._make_att_question("object_move_agent", attached=True, pair_id="6->7", scene="s2", text="s2 c1"),
            self._make_att_question("object_move_agent", attached=True, pair_id="6->8", scene="s2", text="s2 c2"),
            self._make_att_question("object_move_agent", attached=True, unchanged=True, pair_id="9->10", scene="s2", text="s2 u1"),
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual(
            [q["question"] for q in result if q.get("relation_unchanged") is True],
            ["s1 u1"],
        )

    def test_balance_unattached_cap_is_batch_level_per_qtype(self):
        from src.quality_control import balance_l2_attachment_per_scene

        questions = [
            self._make_att_question("object_move_agent", attached=True, pair_id="1->2", scene="s1", text="s1 c1"),
            self._make_att_question("object_move_agent", attached=True, pair_id="1->3", scene="s1", text="s1 c2"),
            self._make_att_question("object_move_agent", attached=False, scene="s1", text="s1 free"),
            self._make_att_question("object_move_agent", attached=True, pair_id="4->5", scene="s2", text="s2 c1"),
            self._make_att_question("object_move_agent", attached=True, pair_id="4->6", scene="s2", text="s2 c2"),
            self._make_att_question("object_move_agent", attached=False, scene="s2", text="s2 free"),
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual(
            [q["question"] for q in result if not q.get("attachment_remapped")],
            ["s1 free"],
        )

    def test_balance_unchanged_candidates_are_capped_at_three_per_scene(self):
        from src.quality_control import balance_l2_attachment_per_scene

        questions = [
            *[
                self._make_att_question(
                    "object_move_agent",
                    attached=True,
                    pair_id=f"1->{i}",
                    scene="s1",
                    text=f"s1 c{i}",
                )
                for i in range(16)
            ],
            *[
                self._make_att_question(
                    "object_move_agent",
                    attached=True,
                    unchanged=True,
                    pair_id=f"10->{i}",
                    scene="s1",
                    text=f"s1 u{i}",
                )
                for i in range(5)
            ],
            self._make_att_question(
                "object_move_agent",
                attached=True,
                unchanged=True,
                pair_id="20->1",
                scene="s2",
                text="s2 u0",
            ),
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual(
            [q["question"] for q in result if q.get("relation_unchanged") is True],
            ["s1 u0", "s1 u1", "s1 u2", "s2 u0"],
        )

    def test_balance_unattached_candidates_are_capped_at_three_per_scene(self):
        from src.quality_control import balance_l2_attachment_per_scene

        questions = [
            *[
                self._make_att_question(
                    "object_move_agent",
                    attached=True,
                    pair_id=f"1->{i}",
                    scene="s1",
                    text=f"s1 c{i}",
                )
                for i in range(16)
            ],
            *[
                self._make_att_question(
                    "object_move_agent",
                    attached=False,
                    scene="s1",
                    text=f"s1 free{i}",
                )
                for i in range(5)
            ],
            self._make_att_question(
                "object_move_agent",
                attached=False,
                scene="s2",
                text="s2 free0",
            ),
        ]

        result = balance_l2_attachment_per_scene(questions)

        self.assertEqual(
            [q["question"] for q in result if not q.get("attachment_remapped")],
            ["s1 free0", "s1 free1", "s1 free2", "s2 free0"],
        )

    def test_coordinate_rotation_object_centric_generates_unchanged_candidates(self) -> None:
        """coordinate_rotation_object_centric includes unchanged answers."""
        objects = [
            make_object(1, "table"),
            make_object(2, "chair"),
            make_object(3, "lamp"),
        ]
        with patch("src.qa_generator._has_stable_object_centric_facing", return_value=True), \
             patch("src.qa_generator._direction_suppression_reason", return_value=None), \
             patch("src.qa_generator.primary_direction_object_centric", return_value=("north", 0.0)), \
             patch("src.qa_generator.generate_options", return_value=(["left", "right", "front", "back"], "A")):
            questions = generate_l3_coordinate_rotation_object_centric(
                objects,
                make_camera_pose(),
                templates={
                    "L3_coordinate_rotation_object_centric": [
                        "Standing at {obj_ref} facing {obj_face}, where is {obj_target}?",
                    ],
                },
            )
        self.assertGreater(len(questions), 0)
        self.assertTrue(all(q.get("relation_unchanged") is True for q in questions))

    def test_coordinate_rotation_object_centric_preserves_original_heading(self) -> None:
        objects = [
            {
                "id": 1,
                "label": "chair",
                "center": [0.0, 0.0, 1.0],
                "bbox_min": [-0.1, -0.1, 0.5],
                "bbox_max": [0.1, 0.1, 1.5],
            },
            {
                "id": 2,
                "label": "table",
                "center": [0.0, 2.0, 1.0],
                "bbox_min": [-0.1, 1.9, 0.5],
                "bbox_max": [0.1, 2.1, 1.5],
            },
            {
                "id": 3,
                "label": "lamp",
                "center": [1.0, 1.0, 1.0],
                "bbox_min": [0.9, 0.9, 0.5],
                "bbox_max": [1.1, 1.1, 1.5],
            },
        ]

        with patch("src.qa_generator._direction_suppression_reason", return_value=None), \
             patch("src.qa_generator.generate_options", side_effect=lambda correct, _pool: ([correct, "front", "back", "left"], "A")):
            questions = generate_l3_coordinate_rotation_object_centric(
                objects,
                make_camera_pose(),
                templates={
                    "L3_coordinate_rotation_object_centric": [
                        "If you were {obj_ref} at its rotated position and kept facing the same horizontal direction that originally pointed from {obj_ref} toward {obj_face}, where is {obj_target}?",
                    ],
                },
            )

        question = next(
            q for q in questions
            if q["rotation_angle"] == 90
            and q["obj_ref_label"] == "chair"
            and q["obj_face_label"] == "table"
            and q["obj_target_label"] == "lamp"
        )

        self.assertEqual(question["old_direction"], "front-right")
        self.assertEqual(question["new_direction"], "back-right")
        self.assertFalse(question["relation_unchanged"])
        self.assertEqual(question["facing_mode"], "preserve_original_heading")
        self.assertIn("kept facing the same horizontal direction", question["question"])
        self.assertNotIn("rotated position and faced toward", question["question"])

        anchor = np.asarray(question["facing_anchor_center"], dtype=float)
        facing = np.asarray(question["facing_target_center"], dtype=float)
        np.testing.assert_allclose(facing - anchor, [0.0, 2.0, 0.0], atol=1e-8)
        self.assertFalse(np.allclose(facing - anchor, [2.0, 0.0, 0.0], atol=1e-8))

        self.assertTrue(any(q["relation_unchanged"] is False for q in questions))

    # ------------------------------------------------------------------
    # L3 coordinate_rotation_agent: unchanged candidates
    # ------------------------------------------------------------------

    def test_coordinate_rotation_agent_generates_unchanged_candidates(self) -> None:
        """coordinate_rotation_agent includes relation_unchanged=True candidates."""
        objects = [
            make_object(1, "table"),
            make_object(2, "chair"),
            make_object(3, "lamp"),
        ]
        _rel = lambda a, b, d: {
            "obj_a_id": a, "obj_a_label": f"obj{a}",
            "obj_b_id": b, "obj_b_label": f"obj{b}",
            "direction_b_rel_a": d,
        }
        orig_rels = [_rel(1, 2, "east"), _rel(1, 3, "east"), _rel(2, 3, "east")]

        def mock_compute_all_relations(objs, *_args, **_kwargs):
            centers = sorted(int(o["center"][0]) for o in objs)
            if centers == [1, 2, 3]:
                return orig_rels
            return [_rel(1, 2, "east"), _rel(1, 3, "north"), _rel(2, 3, "north")]

        with patch("src.qa_generator.compute_all_relations", side_effect=mock_compute_all_relations), \
             patch("src.qa_generator.apply_coordinate_rotation", side_effect=lambda objs, _angle: [
                 {**o, "center": [float(-o["center"][0]), 0.0, 1.0]} for o in objs
             ]), \
             patch("src.qa_generator._direction_suppression_reason", return_value=None), \
             patch("src.qa_generator.generate_options", return_value=(['left', 'right', 'front', 'back'], 'A')):
            questions = generate_l3_coordinate_rotation(
                objects, make_camera_pose(), templates={
                    "L3_coordinate_rotation_agent": [
                        "After a {angle}° rotation, where is {obj_a} relative to {obj_b}?",
                    ]
                },
            )
        unchanged = [q for q in questions if q.get("relation_unchanged")]
        changed = [q for q in questions if not q.get("relation_unchanged", False)]
        self.assertGreater(len(unchanged), 0, "Should generate some unchanged candidates")
        self.assertGreater(len(changed), 0, "Should generate some changed candidates")

    # ------------------------------------------------------------------
    # L3 allocentric: still skips unchanged
    # ------------------------------------------------------------------

    def test_coordinate_rotation_allocentric_generates_unchanged_candidates(self) -> None:
        """Allocentric now generates relation_unchanged=True candidates."""
        objects = [
            make_object(1, "table"),
            make_object(2, "chair"),
        ]
        with patch("src.qa_generator._direction_suppression_reason", return_value=None), \
             patch("src.qa_generator.primary_direction_allocentric", return_value=("north", 0.0)), \
             patch("src.qa_generator.generate_options", return_value=(['north', 'south', 'east', 'west'], 'A')):
            questions = generate_l3_coordinate_rotation_allocentric(
                objects, make_camera_pose(), templates={
                    "L3_coordinate_rotation_allocentric": [
                        "After a {angle}° rotation, what cardinal direction is {obj_a} from {obj_b}?",
                    ]
                },
            )
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertIn("relation_unchanged", q)

    # ------------------------------------------------------------------
    # L3 unchanged ratio cap
    # ------------------------------------------------------------------

    def _make_l3_question(
        self,
        qtype: str,
        *,
        unchanged: bool,
        question_text: str = "Q",
        scene_id: str = "scene0000_00",
    ) -> dict:
        return {
            "scene_id": scene_id,
            "level": "L3",
            "type": qtype,
            "question": question_text,
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "correct_value": "A",
            "relation_unchanged": unchanged,
        }

    def test_cap_l3_unchanged_ratio_keeps_all_changed(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="Q1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="Q2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="Q3"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        self.assertEqual(len(result), 3)

    def test_cap_l3_unchanged_ratio_zero_changed_drops_all_unchanged(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="Q1"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="Q2"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        self.assertEqual(len(result), 0)

    def test_cap_l3_unchanged_ratio_one_changed_drops_all_unchanged(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=False, question_text="changed"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="unch1"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="unch2"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["relation_unchanged"])

    def test_cap_l3_unchanged_ratio_three_changed_drops_all_unchanged(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c3"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="u1"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        changed_count = sum(1 for q in result if not q["relation_unchanged"])
        unchanged_count = sum(1 for q in result if q["relation_unchanged"])
        self.assertEqual(changed_count, 3)
        self.assertEqual(unchanged_count, 0)

    def test_cap_l3_unchanged_ratio_four_changed_keeps_one_unchanged(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c3"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="c4"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="u1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="u2"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        changed_count = sum(1 for q in result if not q["relation_unchanged"])
        unchanged_count = sum(1 for q in result if q["relation_unchanged"])
        self.assertEqual(changed_count, 4)
        self.assertEqual(unchanged_count, 1)

    def test_cap_l3_unchanged_ratio_keeps_unchanged_in_generation_order(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=False, question_text="c1"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=False, question_text="c2"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=False, question_text="c3"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=False, question_text="c4"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="first_unch"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="second_unch"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        unchanged_kept = [q for q in result if q["relation_unchanged"]]
        self.assertEqual(len(unchanged_kept), 1)
        self.assertEqual(unchanged_kept[0]["question"], "first_unch")

    def test_cap_l3_unchanged_ratio_excludes_attachment_chain(self) -> None:
        questions = [
            self._make_l3_question("attachment_chain", unchanged=False, question_text="chain"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "attachment_chain")

    def test_cap_l3_unchanged_ratio_per_type_group(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="ra_c1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="ra_c2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="ra_c3"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="ra_c4"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="ra_u1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="ra_u2"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=False, question_text="oc_c1"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="oc_u1"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="oc_u2"),
            self._make_l3_question("coordinate_rotation_object_centric", unchanged=True, question_text="oc_u3"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        ra_unchanged = sum(
            1 for q in result
            if q["type"] == "coordinate_rotation_agent" and q["relation_unchanged"]
        )
        oc_unchanged = sum(
            1 for q in result
            if q["type"] == "coordinate_rotation_object_centric" and q["relation_unchanged"]
        )
        self.assertEqual(ra_unchanged, 1)
        self.assertEqual(oc_unchanged, 0)

    def test_cap_l3_unchanged_ratio_uses_batch_level_cap_across_scenes(self) -> None:
        questions = [
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="s1_c1", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="s1_c2", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="s1_c3", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="s1_c4", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s1_u1", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=False, question_text="s2_c1", scene_id="s2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s2_u1", scene_id="s2"),
        ]
        result = _cap_l3_unchanged_ratio(questions)
        kept_texts = {q["question"] for q in result}
        self.assertIn("s1_u1", kept_texts)
        self.assertNotIn("s2_u1", kept_texts)

    def test_cap_l3_unchanged_ratio_caps_candidates_at_three_per_scene(self) -> None:
        questions = [
            *[
                self._make_l3_question(
                    "coordinate_rotation_agent",
                    unchanged=False,
                    question_text=f"c{i}",
                    scene_id="s1",
                )
                for i in range(16)
            ],
            *[
                self._make_l3_question(
                    "coordinate_rotation_agent",
                    unchanged=True,
                    question_text=f"s1_u{i}",
                    scene_id="s1",
                )
                for i in range(5)
            ],
            self._make_l3_question(
                "coordinate_rotation_agent",
                unchanged=True,
                question_text="s2_u0",
                scene_id="s2",
            ),
        ]

        result = _cap_l3_unchanged_ratio(questions)

        self.assertEqual(
            [q["question"] for q in result if q["relation_unchanged"]],
            ["s1_u0", "s1_u1", "s1_u2", "s2_u0"],
        )

    def test_cap_l3_unchanged_ratio_preserves_generation_order_after_both_filters(self) -> None:
        questions = [
            *[
                self._make_l3_question(
                    "coordinate_rotation_agent",
                    unchanged=False,
                    question_text=f"c{i}",
                    scene_id="s1",
                )
                for i in range(20)
            ],
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s1_u1", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s2_u1", scene_id="s2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s1_u2", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s2_u2", scene_id="s2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s1_u3", scene_id="s1"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s2_u3", scene_id="s2"),
            self._make_l3_question("coordinate_rotation_agent", unchanged=True, question_text="s1_u4", scene_id="s1"),
        ]

        result = _cap_l3_unchanged_ratio(questions)

        self.assertEqual(
            [q["question"] for q in result if q["relation_unchanged"]],
            ["s1_u1", "s2_u1", "s1_u2", "s2_u2", "s1_u3"],
        )

if __name__ == "__main__":
    unittest.main()
