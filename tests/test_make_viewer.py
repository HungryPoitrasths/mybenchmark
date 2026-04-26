import json
from pathlib import Path
import sys
import unittest
from unittest import mock

from scripts.make_viewer import (
    build_simple_viewer_html,
    build_viewer_html,
    build_task_summary_v2,
    filter_viewer_questions,
    main,
    question_review_notes,
    select_viewer_source_questions,
)


def make_object_move_question(
    *,
    qtype: str,
    scene_id: str = "scene0000_00",
    image_name: str = "000.jpg",
    attached: bool,
    unchanged: bool = False,
    text: str,
) -> dict:
    return {
        "type": qtype,
        "scene_id": scene_id,
        "image_name": image_name,
        "attachment_remapped": attached,
        "relation_unchanged": unchanged,
        "question": text,
    }


def write_questions_json(root: Path, questions: list[dict]) -> Path:
    questions_path = root / "questions.json"
    questions_path.write_text(
        json.dumps({"questions": questions}, ensure_ascii=False),
        encoding="utf-8",
    )
    return questions_path


def run_make_viewer_cli(argv: list[str]) -> None:
    with mock.patch.object(sys, "argv", argv), mock.patch("scripts.make_viewer.Image", object()):
        main()


def workspace_case_dir(name: str) -> Path:
    root = Path("tests") / "_tmp" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


class MakeViewerTests(unittest.TestCase):
    def test_attachment_only_keeps_attachment_chain_and_attached_object_moves(self) -> None:
        questions = [
            {"type": "attachment_chain"},
            {"type": "object_move_agent", "attachment_remapped": True},
            {"type": "object_move_distance", "attachment_remapped": False},
            {"type": "object_move_object_centric", "attachment_remapped": True},
            {"type": "object_move_allocentric", "attachment_remapped": True},
            {"type": "viewpoint_move"},
            {"type": "attachment_type"},
        ]

        filtered = select_viewer_source_questions(questions, attachment_only=True)

        self.assertEqual(
            [q["type"] for q in filtered],
            [
                "attachment_chain",
                "object_move_agent",
                "object_move_object_centric",
                "object_move_allocentric",
            ],
        )

    def test_qtypes_filter_still_works_without_attachment_only(self) -> None:
        questions = [
            {"type": "object_move_agent", "attachment_remapped": True},
            {"type": "viewpoint_move"},
            {"type": "attachment_type"},
        ]

        filtered = select_viewer_source_questions(
            questions,
            requested_qtypes={"viewpoint_move"},
        )

        self.assertEqual(filtered, [{"type": "viewpoint_move"}])

    def test_qtypes_filter_accepts_canonical_object_rotate_label_for_legacy_input(self) -> None:
        questions = [
            {"type": "object_move_object_centric", "attachment_remapped": True},
            {"type": "object_move_agent", "attachment_remapped": True},
        ]

        filtered = select_viewer_source_questions(
            questions,
            requested_qtypes={"object_rotate_object_centric"},
        )

        self.assertEqual(filtered, [{"type": "object_move_object_centric", "attachment_remapped": True}])

    def test_viewer_attachment_filter_drops_unattached_when_no_attached_globally(self) -> None:
        questions = [
            make_object_move_question(qtype="object_move_distance", attached=False, text="keep 1"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 2"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 3"),
            make_object_move_question(
                qtype="object_move_distance",
                scene_id="scene0000_00",
                image_name="001.jpg",
                attached=False,
                text="keep other frame",
            ),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual(filtered, [])

    def test_viewer_attachment_filter_keeps_two_to_one_ratio_per_frame(self) -> None:
        questions = [
            make_object_move_question(qtype="object_rotate_object_centric", attached=True, text="attached"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="free 1"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="free 2"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="drop 3"),
            make_object_move_question(qtype="object_rotate_object_centric", attached=False, text="drop 4"),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual(
            [q["question"] for q in filtered],
            ["attached", "free 1", "free 2"],
        )

    def test_viewer_attachment_filter_caps_unattached_globally_per_qtype(self) -> None:
        questions = [
            make_object_move_question(qtype="object_move_agent", attached=True, text="attached"),
            make_object_move_question(qtype="object_move_agent", attached=False, text="free 1"),
            make_object_move_question(qtype="object_move_agent", attached=False, text="free 2"),
            make_object_move_question(
                qtype="object_move_agent",
                image_name="001.jpg",
                attached=False,
                text="drop 3",
            ),
            make_object_move_question(
                qtype="object_move_agent",
                image_name="002.jpg",
                attached=False,
                text="drop 4",
            ),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual(
            [q["question"] for q in filtered],
            ["attached", "free 1", "free 2"],
        )

    def test_viewer_shows_attachment_unchanged_questions_by_default(self) -> None:
        questions = [
            make_object_move_question(
                qtype="object_rotate_object_centric",
                attached=True,
                unchanged=True,
                text="attached unchanged",
            ),
        ]

        filtered = filter_viewer_questions(questions)

        self.assertEqual([q["question"] for q in filtered], ["attached unchanged"])

    def test_viewer_can_still_hide_attachment_unchanged_questions(self) -> None:
        questions = [
            make_object_move_question(
                qtype="object_rotate_object_centric",
                attached=True,
                unchanged=True,
                text="attached unchanged",
            ),
        ]

        filtered = filter_viewer_questions(
            questions,
            include_attachment_unchanged=False,
        )

        self.assertEqual(filtered, [])

    def test_build_viewer_html_keeps_unattached_object_moves_by_default(self) -> None:
        questions = [
            make_object_move_question(qtype="object_move_distance", attached=False, text="keep 1"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 2"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 3"),
            make_object_move_question(
                qtype="object_move_distance",
                scene_id="scene0000_00",
                image_name="001.jpg",
                attached=False,
                text="keep other frame",
            ),
        ]

        html_text = build_viewer_html(questions, Path("."))

        self.assertIn("L2_object_move_distance: with_attachment=0, without_attachment=4", html_text)
        self.assertIn("keep 1", html_text)
        self.assertIn("drop 2", html_text)
        self.assertIn("drop 3", html_text)
        self.assertIn("keep other frame", html_text)

    def test_build_viewer_html_can_opt_into_legacy_auto_filters(self) -> None:
        questions = [
            make_object_move_question(qtype="object_move_distance", attached=False, text="keep 1"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 2"),
            make_object_move_question(qtype="object_move_distance", attached=False, text="drop 3"),
            make_object_move_question(
                qtype="object_move_distance",
                scene_id="scene0000_00",
                image_name="001.jpg",
                attached=False,
                text="keep other frame",
            ),
        ]

        html_text = build_viewer_html(questions, Path("."), apply_filters=True)

        self.assertIn("L2_object_move_distance: with_attachment=0, without_attachment=0", html_text)
        self.assertNotIn("keep 1", html_text)
        self.assertNotIn("drop 2", html_text)
        self.assertNotIn("drop 3", html_text)
        self.assertNotIn("keep other frame", html_text)

    def test_build_viewer_html_still_hides_attachment_unchanged_without_auto_filters(self) -> None:
        questions = [
            make_object_move_question(
                qtype="object_rotate_object_centric",
                attached=True,
                unchanged=True,
                text="attached unchanged",
            ),
            make_object_move_question(
                qtype="object_rotate_object_centric",
                attached=True,
                unchanged=False,
                text="attached changed",
            ),
        ]

        html_text = build_viewer_html(
            questions,
            Path("."),
            include_attachment_unchanged=False,
        )

        self.assertNotIn("attached unchanged", html_text)
        self.assertIn("attached changed", html_text)
        self.assertIn(
            "L2_object_rotate_object_centric: with_attachment=1, without_attachment=0",
            html_text,
        )

    def test_build_viewer_html_preserves_chinese_audit_text_with_utf8(self) -> None:
        questions = [
            {
                "type": "distance",
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "question": "Where is the tv stand?",
                "question_post_generation_review": {
                    "decision": "manual_review",
                    "mesh_object_reviews": [
                        {
                            "label": "tv stand",
                            "obj_id": 30,
                            "decision": "manual_review",
                            "mesh_mask_reason_codes": ["low_iou", "high_over_coverage"],
                            "reason_codes": ["mesh_low_iou", "mesh_high_over_coverage"],
                        }
                    ],
                    "dinox_label_reviews": [
                        {
                            "label": "tv stand",
                            "decision": "pass",
                            "reason_codes": [],
                        }
                    ],
                },
            }
        ]

        html_text = build_viewer_html(questions, Path("."))

        self.assertIn('<meta charset="UTF-8">', html_text)
        self.assertIn("生成后审核", html_text)
        self.assertIn("审核结论：需要人工复核", html_text)
        self.assertIn("问题概述：1 个对象需要复核。", html_text)
        self.assertIn("tv stand#30：2D 检测已通过，但 3D 网格投影与该物体不够匹配", html_text)

        raw_bytes = html_text.encode("utf-8")
        roundtrip_text = raw_bytes.decode("utf-8")
        self.assertIn("生成后审核", roundtrip_text)
        self.assertIn("审核结论：需要人工复核", roundtrip_text)
        self.assertIn("生成后审核".encode("utf-8"), raw_bytes)

    def test_task_summary_v2_canonicalizes_legacy_object_centric_type(self) -> None:
        questions = [
            {"type": "object_move_object_centric", "attachment_remapped": True},
        ]

        summary = build_task_summary_v2(questions)

        self.assertIn(
            "L2_object_rotate_object_centric: with_attachment=1, without_attachment=0",
            summary,
        )
        self.assertIn(
            "L2_object_move_all: with_attachment=1, without_attachment=0",
            summary,
        )
        self.assertNotIn("Viewer Slice", summary)
        self.assertNotIn("raw=", summary)
        self.assertNotIn("shown=", summary)
        self.assertNotIn("hidden=", summary)
        self.assertNotIn("total=", summary)
        self.assertNotIn("with_attachment_changed=", summary)
        self.assertNotIn("with_attachment_unchanged=", summary)
        self.assertNotIn("object_move_object_centric", summary)

    def test_task_summary_v2_does_not_list_object_rotate_in_other_types(self) -> None:
        questions = [
            {"type": "object_rotate_object_centric", "attachment_remapped": True},
            {"type": "custom_unknown"},
            {"type": "custom_unknown"},
        ]

        summary = build_task_summary_v2(questions)

        self.assertIn(
            "L2_object_rotate_object_centric: with_attachment=1, without_attachment=0",
            summary,
        )
        self.assertIn("Other Types:</strong> custom_unknown=2", summary)
        self.assertNotIn("Other Types:</strong> object_rotate_object_centric=1", summary)

    def test_task_summary_v2_counts_attachment_unchanged_in_with_attachment_bucket(self) -> None:
        questions = [
            {
                "type": "object_rotate_object_centric",
                "attachment_remapped": True,
                "relation_unchanged": True,
            },
            {
                "type": "object_rotate_object_centric",
                "attachment_remapped": False,
                "relation_unchanged": False,
            },
        ]

        summary = build_task_summary_v2(questions)

        self.assertIn(
            "L2_object_rotate_object_centric: with_attachment=1, without_attachment=1",
            summary,
        )

    def test_task_summary_v2_counts_only_downward_attachment_propagation(self) -> None:
        questions = [
            {
                "type": "object_move_agent",
                "attachment_remapped": True,
                "moved_obj_id": 1,
                "query_obj_id": 2,
            },
            {
                "type": "object_move_agent",
                "attachment_remapped": False,
                "moved_obj_id": 2,
                "query_obj_id": 2,
            },
        ]

        summary = build_task_summary_v2(questions)

        self.assertIn(
            "L2_object_move_agent: with_attachment=1, without_attachment=1",
            summary,
        )

    def test_question_review_notes_renders_referability_audit(self) -> None:
        notes = question_review_notes(
            {
                "question_referability_audit": {
                    "decision": "drop",
                    "reason_codes": [
                        "mentioned_label_not_unique",
                        "mentioned_label_not_resolved",
                    ],
                    "frame_referable_object_ids": [5],
                    "mentioned_objects": [
                        {
                            "role": "reference",
                            "label": "curtain",
                            "obj_id": None,
                            "label_status": "multiple",
                            "candidate_object_ids": [2, 3],
                            "referable_object_ids": [],
                            "passes_referability_check": False,
                            "reason_codes": [
                                "mentioned_label_not_unique",
                                "mentioned_label_not_resolved",
                            ],
                        }
                    ],
                }
            }
        )

        self.assertIn("Referability Audit", notes)
        self.assertIn("decision: drop", notes)
        self.assertIn(
            "reason codes: mentioned_label_not_unique, mentioned_label_not_resolved",
            notes,
        )
        self.assertIn("frame referable ids: 5", notes)
        self.assertIn(
            "reference: label=curtain, obj_id=-, label_status=multiple, candidates=2, 3, referable=-, result=drop, reasons=mentioned_label_not_unique, mentioned_label_not_resolved",
            notes,
        )

    def test_question_review_notes_renders_instance_presence_review(self) -> None:
        notes = question_review_notes(
            {
                "manual_review_reason": "VLM flagged mentioned objects: cabinet#42=unsure",
                "question_presence_review": {
                    "decision": "manual_review",
                    "flagged_labels": ["cabinet"],
                    "flagged_object_ids": [42],
                    "object_reviews": [
                        {
                            "label": "cabinet",
                            "obj_id": 42,
                            "roles": ["reference", "target"],
                            "status": "unsure",
                            "reason": "invalid_crop",
                        }
                    ],
                },
            }
        )

        self.assertIn("flagged object ids: 42", notes)
        self.assertIn("cabinet#42 [reference, target]: unsure (invalid_crop)", notes)

    def test_question_review_notes_renders_post_generation_audit(self) -> None:
        notes = question_review_notes(
            {
                "question_post_generation_review": {
                    "decision": "manual_review",
                    "reason_codes": ["dinox_multiple_strong_detections:chair", "mesh_low_iou:chair#1"],
                    "flagged_labels": ["chair"],
                    "flagged_object_ids": [1],
                    "dinox_label_reviews": [
                        {
                            "label": "chair",
                            "decision": "manual_review",
                            "strong_detection_count": 2,
                            "matched_object_ids": [1],
                            "reason_codes": ["dinox_multiple_strong_detections"],
                        }
                    ],
                    "mesh_object_reviews": [
                        {
                            "label": "chair",
                            "obj_id": 1,
                            "decision": "manual_review",
                            "topology_status": "pass",
                            "mesh_mask_status": "fail",
                            "reason_codes": ["mesh_low_iou"],
                        }
                    ],
                }
            }
        )

        self.assertIn("生成后审核", notes)
        self.assertIn("问题概述：1 个对象需要复核。", notes)
        self.assertIn(
            "chair#1：2D 检测和 3D 网格检查都需要复核。2D 问题：找到了多个较强的 2D 检测结果。3D 问题：3D 投影与检测结果的重叠度过低。",
            notes,
        )

    def test_question_review_notes_renders_post_generation_audit_ok_objects(self) -> None:
        notes = question_review_notes(
            {
                "question_post_generation_review": {
                    "decision": "manual_review",
                    "mesh_object_reviews": [
                        {
                            "label": "tv stand",
                            "obj_id": 30,
                            "decision": "manual_review",
                            "topology_status": "warn",
                            "mesh_mask_status": "fail",
                            "mesh_mask_reason_codes": [
                                "low_iou",
                                "high_over_coverage",
                                "bad_area_ratio",
                            ],
                            "reason_codes": [
                                "mesh_low_iou",
                                "mesh_high_over_coverage",
                                "mesh_bad_area_ratio",
                            ],
                        },
                        {
                            "label": "pillow",
                            "obj_id": 61,
                            "decision": "pass",
                            "topology_status": "warn",
                            "mesh_mask_status": "pass",
                            "mesh_mask_reason_codes": [],
                            "reason_codes": [],
                        },
                    ],
                    "dinox_label_reviews": [
                        {
                            "label": "tv stand",
                            "decision": "pass",
                            "strong_detection_count": 1,
                            "matched_object_ids": [30],
                            "reason_codes": [],
                        },
                        {
                            "label": "pillow",
                            "decision": "pass",
                            "strong_detection_count": 1,
                            "matched_object_ids": [61],
                            "reason_codes": [],
                        },
                    ],
                }
            }
        )

        self.assertIn("问题概述：1 个对象需要复核。", notes)
        self.assertIn(
            "tv stand#30：2D 检测已通过，但 3D 网格投影与该物体不够匹配，原因是3D 投影与检测结果的重叠度过低、3D 投影覆盖了过多额外区域和3D 投影面积和检测结果不一致。",
            notes,
        )
        self.assertIn("pillow#61：正常。2D 检测和 3D 网格检查都已通过。", notes)


    def test_main_without_simple_output_matches_build_viewer_html(self) -> None:
        questions = [
            {
                "type": "direction_agent",
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "question": "Where is the lamp relative to the desk?",
                "options": ["left", "right", "front"],
                "answer": "A",
                "correct_value": "left",
                "obj_a_label": "desk",
                "obj_b_label": "lamp",
            }
        ]

        root = workspace_case_dir("make_viewer_main_only")
        questions_path = write_questions_json(root, questions)
        image_root = root / "images"
        image_root.mkdir(exist_ok=True)
        output_path = root / "viewer.html"

        expected_html = build_viewer_html(questions, image_root)
        run_make_viewer_cli(
            [
                "make_viewer.py",
                "--questions",
                str(questions_path),
                "--image_root",
                str(image_root),
                "--output",
                str(output_path),
            ]
        )

        self.assertEqual(output_path.read_text(encoding="utf-8"), expected_html)

    def test_main_with_simple_output_writes_both_html_files(self) -> None:
        questions = [
            {
                "type": "direction_agent",
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "question": "Where is the lamp relative to the desk?",
                "options": ["left", "right", "front"],
                "answer": "A",
                "correct_value": "left",
                "obj_a_label": "desk",
                "obj_b_label": "lamp",
            }
        ]

        root = workspace_case_dir("make_viewer_main_dual")
        questions_path = write_questions_json(root, questions)
        image_root = root / "images"
        image_root.mkdir(exist_ok=True)
        output_path = root / "viewer.html"
        simple_output_path = root / "viewer_simple.html"

        run_make_viewer_cli(
            [
                "make_viewer.py",
                "--questions",
                str(questions_path),
                "--image_root",
                str(image_root),
                "--output",
                str(output_path),
                "--simple_output",
                str(simple_output_path),
            ]
        )

        full_html = output_path.read_text(encoding="utf-8")
        simple_html = simple_output_path.read_text(encoding="utf-8")
        self.assertTrue(output_path.exists())
        self.assertTrue(simple_output_path.exists())
        self.assertIn("Where is the lamp relative to the desk?", full_html)
        self.assertIn("A.&nbsp; left", full_html)
        self.assertIn("image not found", full_html)
        self.assertIn("image not found", simple_html)
        self.assertNotIn("Where is the lamp relative to the desk?", simple_html)
        self.assertNotIn("A.&nbsp; left", simple_html)
        self.assertIn("Objects", simple_html)
        self.assertIn("Relations", simple_html)

    def test_build_simple_viewer_html_direction_agent_uses_query_then_reference(self) -> None:
        html_text = build_simple_viewer_html(
            [
                {
                    "type": "direction_agent",
                    "scene_id": "scene0000_00",
                    "image_name": "000.jpg",
                    "question": "Full text should stay hidden",
                    "options": ["front", "left", "right"],
                    "answer": "B",
                    "correct_value": "left",
                    "obj_a_label": "desk",
                    "obj_b_label": "lamp",
                }
            ],
            Path("."),
        )

        self.assertNotIn("Full text should stay hidden", html_text)
        self.assertNotIn("A.&nbsp; front", html_text)
        self.assertIn("direction=left", html_text)
        self.assertLess(html_text.index("query=lamp"), html_text.index("reference=desk"))

    def test_build_simple_viewer_html_renders_representative_qtypes(self) -> None:
        questions = [
            {
                "type": "occlusion",
                "scene_id": "scene0000_00",
                "image_name": "000.jpg",
                "correct_value": "occluded",
                "obj_a_label": "chair",
            },
            {
                "type": "distance",
                "scene_id": "scene0000_00",
                "image_name": "001.jpg",
                "correct_value": "close (1.5-3m)",
                "distance_m": 2.3,
                "obj_a_label": "desk",
                "obj_b_label": "lamp",
            },
            {
                "type": "object_move_occlusion",
                "scene_id": "scene0000_00",
                "image_name": "002.jpg",
                "moved_obj_label": "cart",
                "query_obj_label": "box",
                "old_visibility": "visible",
                "new_visibility": "occluded",
            },
            {
                "type": "object_move_allocentric",
                "scene_id": "scene0000_00",
                "image_name": "003.jpg",
                "moved_obj_label": "table",
                "query_obj_label": "bag",
                "obj_ref_label": "door",
                "camera_cardinal": "west",
                "old_correct_value": "left",
                "new_correct_value": "right",
            },
            {
                "type": "object_rotate_object_centric",
                "scene_id": "scene0000_00",
                "image_name": "004.jpg",
                "moved_obj_label": "desk",
                "query_obj_label": "chair",
                "obj_face_label": "monitor",
                "obj_ref_label": "window",
                "rotation_angle": 90,
                "rotation_direction": "clockwise",
                "old_correct_value": "front",
                "new_correct_value": "left",
            },
            {
                "type": "attachment_chain",
                "scene_id": "scene0000_00",
                "image_name": "005.jpg",
                "grandparent_label": "desk",
                "parent_label": "monitor",
                "grandchild_label": "webcam",
                "neighbor_label": "chair",
                "chain_depth": 2,
                "correct_value": "yes",
            },
        ]

        html_text = build_simple_viewer_html(questions, Path("."))

        self.assertIn("target=chair", html_text)
        self.assertIn("visibility=occluded", html_text)
        self.assertIn("distance_bin=close (1.5-3m)", html_text)
        self.assertIn("distance_m=2.3", html_text)
        self.assertIn("old_visibility=visible", html_text)
        self.assertIn("new_visibility=occluded", html_text)
        self.assertIn("camera=west", html_text)
        self.assertIn("old_direction=left", html_text)
        self.assertIn("new_direction=right", html_text)
        self.assertIn("rotation_angle=90", html_text)
        self.assertIn("rotation_direction=clockwise", html_text)
        self.assertIn("chain_depth=2", html_text)
        self.assertIn("displaced=yes", html_text)

    def test_build_simple_viewer_html_unknown_qtype_falls_back_to_mentioned_objects(self) -> None:
        html_text = build_simple_viewer_html(
            [
                {
                    "type": "custom_unknown",
                    "scene_id": "scene0000_00",
                    "image_name": "000.jpg",
                    "correct_value": "around",
                    "mentioned_objects": [
                        {"role": "anchor", "label": "lamp"},
                        {"role": "target", "label": "chair"},
                    ],
                }
            ],
            Path("."),
        )

        self.assertIn("anchor=lamp", html_text)
        self.assertIn("target=chair", html_text)
        self.assertIn("correct_value=around", html_text)

    def test_build_simple_viewer_html_missing_images_do_not_break_output(self) -> None:
        html_text = build_simple_viewer_html(
            [
                {
                    "type": "object_move_agent",
                    "scene_id": "scene0000_00",
                    "image_name": "missing.jpg",
                    "moved_obj_label": "cart",
                    "obj_b_label": "box",
                    "obj_c_label": "wall",
                    "old_correct_value": "left",
                    "new_correct_value": "right",
                }
            ],
            Path("."),
        )

        self.assertIn("image not found", html_text)
        self.assertIn("moved=cart", html_text)
        self.assertIn("query=box", html_text)
        self.assertIn("reference=wall", html_text)


if __name__ == "__main__":
    unittest.main()
