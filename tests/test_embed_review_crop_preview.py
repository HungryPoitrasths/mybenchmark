import unittest

from scripts.embed_review_crop_preview import (
    CROP_PREVIEW_END,
    CROP_PREVIEW_START,
    ReviewTarget,
    _find_matching_card,
    build_crop_preview_html,
    ensure_preview_style,
    inject_crop_preview,
    parse_card_footer,
    parse_card_targets,
    parse_review_targets,
)


SAMPLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
body{font-family:Arial}
</style>
</head>
<body>
<div class="card">
  <div class="body">
    <p class="qtext">Question A</p>
    <div class="review-notes"><div class="review-block"><div class="review-title">Manual Review</div><div class="review-line">VLM flagged mentioned objects: lamp#3=absent</div></div><div class="review-block"><div class="review-title">VLM Review</div><div class="review-line">decision: manual_review</div><div class="review-line">flagged labels: lamp</div><div class="review-line">flagged object ids: 3</div><div class="review-line">lamp#3 [reference]: absent (not visible in crop)</div><div class="review-line">table#10 [target]: present (table surface is visible)</div></div><div class="review-block"><div class="review-title">Referability Audit</div><div class="review-line">decision: pass</div></div></div>
    <div class="footer">scene9999_00 &nbsp;/&nbsp; 0100.jpg</div>
  </div>
</div>
<div class="card">
  <div class="body">
    <p class="qtext">Question B</p>
    <div class="review-notes"><div class="review-block"><div class="review-title">Manual Review</div><div class="review-line">VLM flagged mentioned objects: sofa#26=unsure</div></div><div class="review-block"><div class="review-title">VLM Review</div><div class="review-line">decision: manual_review</div><div class="review-line">flagged labels: sofa</div><div class="review-line">flagged object ids: 26</div><div class="review-line">sofa#26 [reference]: unsure (missing_obj_id_in_vlm_response)</div><div class="review-line">curtain#34 [target]: present (The crop clearly shows the curtain.)</div></div><div class="review-block"><div class="review-title">Referability Audit</div><div class="review-line">decision: pass</div><div class="review-line">reason codes: -</div></div></div>
    <div class="footer">scene0000_01 &nbsp;/&nbsp; 2520.jpg</div>
  </div>
</div>
</body>
</html>
"""

AUDIT_ONLY_CARD = """
<div class="card">
  <div class="body">
    <p class="qtext">Question C</p>
    <div class="review-notes"><div class="review-block"><div class="review-title">Referability Audit</div><div class="review-line">decision: pass</div><div class="review-line">reason codes: -</div><div class="review-line">frame referable ids: 3</div><div class="review-line">target: label=bed, obj_id=3, label_status=unique, candidates=3, referable=3, result=pass, reasons=-</div></div></div>
    <div class="footer">scene0006_01 &nbsp;/&nbsp; 1890.jpg</div>
  </div>
</div>
"""


class EmbedReviewCropPreviewTests(unittest.TestCase):
    def test_find_matching_card_uses_footer_and_match_text(self) -> None:
        start, end, card_html = _find_matching_card(
            SAMPLE_HTML,
            scene_id="scene0000_01",
            image_name="2520.jpg",
            match_text="sofa#26=unsure",
        )

        self.assertGreater(end, start)
        self.assertIn("Question B", card_html)
        self.assertNotIn("Question A", card_html)

    def test_parse_review_targets_extracts_object_lines_only(self) -> None:
        _, _, card_html = _find_matching_card(
            SAMPLE_HTML,
            scene_id="scene0000_01",
            image_name="2520.jpg",
            match_text="sofa#26=unsure",
        )

        targets = parse_review_targets(card_html)

        self.assertEqual(
            targets,
            [
                ReviewTarget(
                    label="sofa",
                    obj_id=26,
                    roles=("reference",),
                    status="unsure",
                    reason="missing_obj_id_in_vlm_response",
                ),
                ReviewTarget(
                    label="curtain",
                    obj_id=34,
                    roles=("target",),
                    status="present",
                    reason="The crop clearly shows the curtain.",
                ),
            ],
        )

    def test_parse_card_targets_merges_vlm_review_and_audit_roles(self) -> None:
        _, _, card_html = _find_matching_card(
            SAMPLE_HTML,
            scene_id="scene0000_01",
            image_name="2520.jpg",
            match_text="sofa#26=unsure",
        )

        targets = parse_card_targets(card_html)

        self.assertEqual(
            targets,
            [
                ReviewTarget(
                    label="sofa",
                    obj_id=26,
                    roles=("reference",),
                    status="unsure",
                    reason="missing_obj_id_in_vlm_response",
                ),
                ReviewTarget(
                    label="curtain",
                    obj_id=34,
                    roles=("target",),
                    status="present",
                    reason="The crop clearly shows the curtain.",
                ),
            ],
        )

    def test_parse_card_targets_falls_back_to_referability_audit(self) -> None:
        targets = parse_card_targets(AUDIT_ONLY_CARD)

        self.assertEqual(
            targets,
            [
                ReviewTarget(
                    label="bed",
                    obj_id=3,
                    roles=("target",),
                    status="pass",
                    reason="",
                ),
            ],
        )

    def test_parse_card_footer_reads_scene_and_image(self) -> None:
        scene_id, image_name = parse_card_footer(AUDIT_ONLY_CARD)

        self.assertEqual(scene_id, "scene0006_01")
        self.assertEqual(image_name, "1890.jpg")

    def test_build_crop_preview_html_renders_images_and_missing_state(self) -> None:
        preview_html = build_crop_preview_html(
            [
                ReviewTarget(
                    label="sofa",
                    obj_id=26,
                    roles=("reference",),
                    status="unsure",
                    reason="missing_obj_id_in_vlm_response",
                ),
                ReviewTarget(
                    label="curtain",
                    obj_id=34,
                    roles=("target",),
                    status="present",
                    reason="",
                ),
            ],
            {
                26: {
                    "reason": "invalid_crop",
                    "roi_bounds_px": [10, 30, 40, 60],
                    "crop_bounds_px": [0, 40, 30, 70],
                },
                34: {
                    "image_b64": "ZmFrZQ==",
                    "mime": "image/jpeg",
                    "roi_bounds_px": [1, 2, 3, 4],
                    "crop_bounds_px": [0, 5, 0, 6],
                },
            },
            image_name="2520.jpg",
        )

        self.assertIn(CROP_PREVIEW_START, preview_html)
        self.assertIn(CROP_PREVIEW_END, preview_html)
        self.assertIn("Exact per-object crops used by question presence review for 2520.jpg.", preview_html)
        self.assertIn("Crop unavailable: invalid_crop", preview_html)
        self.assertIn('src="data:image/jpeg;base64,ZmFrZQ=="', preview_html)
        self.assertIn("roi_bounds_px: [10, 30, 40, 60]", preview_html)
        self.assertIn("crop_bounds_px: [0, 5, 0, 6]", preview_html)

    def test_inject_crop_preview_inserts_before_review_notes_and_replaces_existing(self) -> None:
        _, _, card_html = _find_matching_card(
            SAMPLE_HTML,
            scene_id="scene0000_01",
            image_name="2520.jpg",
            match_text="sofa#26=unsure",
        )
        first_preview = "<!-- crop-preview:start --><div>first</div><!-- crop-preview:end -->"
        second_preview = "<!-- crop-preview:start --><div>second</div><!-- crop-preview:end -->"

        injected = inject_crop_preview(card_html, first_preview)
        replaced = inject_crop_preview(injected, second_preview)

        self.assertLess(replaced.index("second"), replaced.index('<div class="review-notes">'))
        self.assertEqual(replaced.count(CROP_PREVIEW_START), 1)
        self.assertEqual(replaced.count(CROP_PREVIEW_END), 1)
        self.assertNotIn("first", replaced)
        self.assertIn("second", replaced)

    def test_ensure_preview_style_is_idempotent(self) -> None:
        once = ensure_preview_style(SAMPLE_HTML)
        twice = ensure_preview_style(once)

        self.assertEqual(once.count(".crop-preview{"), 1)
        self.assertEqual(twice.count(".crop-preview{"), 1)
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
