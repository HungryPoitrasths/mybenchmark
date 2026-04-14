from pathlib import Path
import unittest

from scripts.make_viewer import (
    build_viewer_html,
    build_task_summary_v2,
    filter_viewer_questions,
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

        self.assertIn("Post-Generation Audit", notes)
        self.assertIn("reason codes: dinox_multiple_strong_detections:chair, mesh_low_iou:chair#1", notes)
        self.assertIn("DINO-X chair: decision=manual_review, strong=2, matched=1", notes)
        self.assertIn("Mesh chair#1: decision=manual_review, topology=pass, mesh=fail, reasons=mesh_low_iou", notes)


if __name__ == "__main__":
    unittest.main()
