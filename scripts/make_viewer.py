#!/usr/bin/env python3
"""Generate a self-contained HTML viewer for QA validation.

Each question is shown next to its source image.
The output is a single HTML file with base64-embedded images - no server
required, just open it in any browser.

Dataset is inferred per-question from the ``_dataset`` / ``dataset`` field,
so a single viewer can display mixed ScanNet + ScanNet++ questions.

Usage:
    # ScanNet only (backward-compatible)
    python scripts/make_viewer.py \\
        --questions output/pilot/human_validation_sample.json \\
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \\
        --output output/pilot/viewer.html

    # ScanNet++ only
    python scripts/make_viewer.py \\
        --questions output/scannetpp/benchmark.json \\
        --scannetpp_image_root output/scannetpp_iphone_frames \\
        --scannetpp_sensor iphone \\
        --output output/scannetpp/viewer.html \\
        --simple_output output/scannetpp/viewer_simple.html

    # Mixed ScanNet + ScanNet++ (dual roots)
    python scripts/make_viewer.py \\
        --questions output/fixed_type_eval_50scene.json \\
        --scannet_image_root /home/lihongxing/datasets/ScanNet/data/scans \\
        --scannetpp_image_root output/scannetpp_iphone_frames \\
        --scannetpp_sensor iphone \\
        --output output/bench.html \\
        --simple_output output/bench_simple.html

    # Attachment-only viewers
    python scripts/make_viewer.py \\
        --questions output/pilot_meshray/benchmark.json \\
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \\
        --output output/pilot_meshray/attachment_viewer.html \\
        --attachment_only

"""

from __future__ import annotations

import argparse
import base64
import html
import json
import random
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


SUMMARY_GROUPS = [
    (
        "L1 静态感知",
        [
            ("direction_agent", "L1_direction_agent"),
            ("occlusion", "L1_occlusion"),
            ("distance", "L1_distance"),
            ("direction_object_centric", "L1_direction_object_centric"),
            ("direction_allocentric", "L1_direction_allocentric"),
        ],
    ),
    (
        "L2 干预题",
        [
            ("object_move_agent", "L2_object_move_agent"),
            ("object_move_distance", "L2_object_move_distance"),
            ("object_move_occlusion", "L2_object_move_occlusion"),
            ("object_move_object_centric", "L2_object_move_object_centric"),
            ("object_rotate_object_centric", "L2_object_rotate_object_centric"),
            ("object_move_allocentric", "L2_object_move_allocentric"),
            ("object_remove", "L2_object_remove"),
        ],
    ),
    (
        "L3 多跳 / 反事实",
        [
            ("attachment_chain", "L3_attachment_chain"),
            ("attachment_move", "L3_attachment_move"),
            ("coordinate_rotation_agent", "L3_coordinate_rotation_agent"),
            (
                "coordinate_rotation_object_centric",
                "L3_coordinate_rotation_object_centric",
            ),
            ("coordinate_rotation_allocentric", "L3_coordinate_rotation_allocentric"),
        ],
    ),
]


QTYPE_DISPLAY: dict[str, str] = {
    raw: label
    for _, items in SUMMARY_GROUPS
    for raw, label in items
}
QTYPE_LEVEL = {
    raw: section_title.split()[0]
    for section_title, items in SUMMARY_GROUPS
    for raw, _ in items
}
LEVEL_DISPLAY_ORDER = ["L1", "L2", "L3"]
QUESTION_TYPE_ALIASES: dict[str, str] = {
    "L2_object_move_object_centric": "object_move_object_centric",
}

OBJECT_MOVE_TYPES = {
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
}
REMOVED_TYPES = {"attachment_type", "support_move_consequence"}
VIEWER_QTYPE_ORDER = [
    "direction_agent",
    "occlusion",
    "distance",
    "direction_object_centric",
    "direction_allocentric",
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "object_remove",
    "attachment_chain",
    "attachment_move",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
]
SIMPLE_VIEWER_FIELD_SPECS: dict[
    str,
    dict[str, list[tuple[str, tuple[str, ...]]]],
] = {
    "direction_agent": {
        "objects": [
            ("query", ("obj_b_label",)),
            ("reference", ("obj_a_label",)),
        ],
        "relations": [("direction", ("correct_value",))],
    },
    "occlusion": {
        "objects": [("target", ("obj_a_label",))],
        "relations": [("visibility", ("correct_value",))],
    },
    "distance": {
        "objects": [
            ("obj_a", ("obj_a_label",)),
            ("obj_b", ("obj_b_label",)),
        ],
        "relations": [
            ("distance_bin", ("correct_value",)),
            ("distance_m", ("distance_m",)),
        ],
    },
    "direction_object_centric": {
        "objects": [
            ("query", ("obj_target_label",)),
            ("reference", ("obj_ref_label",)),
            ("facing", ("obj_face_label",)),
        ],
        "relations": [("direction", ("correct_value",))],
    },
    "direction_allocentric": {
        "objects": [
            ("query", ("obj_a_label",)),
            ("reference", ("obj_b_label",)),
        ],
        "relations": [
            ("camera", ("camera_cardinal",)),
            ("direction", ("correct_value",)),
        ],
    },
    "object_move_agent": {
        "objects": [
            ("moved", ("moved_obj_label",)),
            ("query", ("obj_b_label", "query_obj_label")),
            ("reference", ("obj_c_label",)),
        ],
        "relations": [
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
    "object_move_distance": {
        "objects": [
            ("moved", ("moved_obj_label",)),
            ("obj_b", ("obj_b_label", "query_obj_label")),
            ("obj_c", ("obj_c_label",)),
        ],
        "relations": [
            ("old_distance_bin", ("old_distance_bin", "old_correct_value")),
            ("new_distance_bin", ("new_distance_bin", "new_correct_value", "correct_value")),
            ("old_distance_m", ("old_distance_m",)),
            ("new_distance_m", ("new_distance_m",)),
        ],
    },
    "object_move_occlusion": {
        "objects": [
            ("moved", ("moved_obj_label",)),
            ("target", ("query_obj_label", "target_obj_label")),
        ],
        "relations": [
            ("old_visibility", ("old_visibility", "old_correct_value")),
            ("new_visibility", ("new_visibility", "new_correct_value", "correct_value")),
        ],
    },
    "object_move_object_centric": {
        "objects": [
            ("moved", ("moved_obj_label",)),
            ("query", ("query_obj_label",)),
            ("reference", ("obj_ref_label",)),
        ],
        "relations": [
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
    "object_remove": {
        "objects": [
            ("removed", ("removed_obj_label",)),
            ("target", ("obj_b_label",)),
        ],
        "relations": [
            ("old_visibility", ("old_visibility", "old_correct_value")),
            ("new_visibility", ("new_visibility", "new_correct_value", "correct_value")),
        ],
    },
    "object_rotate_object_centric": {
        "objects": [
            ("moved", ("moved_obj_label",)),
            ("query", ("query_obj_label",)),
            ("facing", ("obj_face_label",)),
            ("reference", ("obj_ref_label",)),
        ],
        "relations": [
            ("rotation_angle", ("rotation_angle",)),
            ("rotation_direction", ("rotation_direction",)),
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
    "object_move_allocentric": {
        "objects": [
            ("moved", ("moved_obj_label",)),
            ("query", ("query_obj_label",)),
            ("reference", ("obj_ref_label",)),
        ],
        "relations": [
            ("camera", ("camera_cardinal",)),
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
    "attachment_chain": {
        "objects": [
            ("moved", ("grandparent_label",)),
            ("child", ("parent_label",)),
            ("grandchild", ("grandchild_label",)),
            ("contrast", ("neighbor_label",)),
        ],
        "relations": [
            ("chain_depth", ("chain_depth",)),
            ("displaced", ("correct_value",)),
        ],
    },
    "attachment_move": {
        "objects": [
            ("root", ("root_label",)),
            ("parent", ("parent_label",)),
            ("grandchild", ("grandchild_label",)),
            ("query", ("query_obj_label",)),
            ("reference", ("obj_ref_label",)),
        ],
        "relations": [
            ("reference_frame", ("reference_frame",)),
            ("query_role", ("query_role",)),
            ("old_direction", ("old_correct_value",)),
            ("new_direction", ("new_correct_value", "correct_value")),
        ],
    },
    "coordinate_rotation_agent": {
        "objects": [
            ("query", ("obj_a_label",)),
            ("reference", ("obj_b_label",)),
        ],
        "relations": [
            ("rotation_angle", ("rotation_angle",)),
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
    "coordinate_rotation_object_centric": {
        "objects": [
            ("query", ("obj_target_label",)),
            ("reference", ("obj_ref_label",)),
            ("facing", ("obj_face_label",)),
        ],
        "relations": [
            ("rotation_angle", ("rotation_angle",)),
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
    "coordinate_rotation_allocentric": {
        "objects": [
            ("query", ("obj_a_label",)),
            ("reference", ("obj_b_label",)),
        ],
        "relations": [
            ("camera", ("camera_cardinal",)),
            ("rotation_angle", ("rotation_angle",)),
            ("old_direction", ("old_direction", "old_correct_value")),
            ("new_direction", ("new_direction", "new_correct_value", "correct_value")),
        ],
    },
}


def _canonical_qtype(qtype: str) -> str:
    canonical = str(qtype or "").strip()
    return QUESTION_TYPE_ALIASES.get(canonical, canonical)


def is_attachment_unchanged_object_move(question: dict) -> bool:
    qtype = _canonical_qtype(str(question.get("type", "")).strip())
    return (
        qtype in OBJECT_MOVE_TYPES
        and bool(question.get("attachment_remapped", False))
        and bool(question.get("relation_unchanged", False))
    )


def _canonical_type_counter(questions: list[dict]) -> Counter:
    counter: Counter = Counter()
    for question in questions:
        qtype = _canonical_qtype(str(question.get("type", "")).strip()) or "unknown"
        counter[qtype] += 1
    return counter


def _summarize_object_move_questions(
    questions: list[dict],
) -> tuple[Counter, Counter]:
    attached_counter: Counter = Counter()
    unattached_counter: Counter = Counter()
    for question in questions:
        qtype = _canonical_qtype(str(question.get("type", "")).strip())
        if qtype not in OBJECT_MOVE_TYPES:
            continue
        if bool(question.get("attachment_remapped", False)):
            attached_counter[qtype] += 1
        else:
            unattached_counter[qtype] += 1
    return (
        attached_counter,
        unattached_counter,
    )


def select_viewer_source_questions(
    questions: list[dict],
    *,
    requested_qtypes: set[str] | None = None,
    attachment_only: bool = False,
) -> list[dict]:
    filtered = [
        q for q in questions
        if _canonical_qtype(str(q.get("type", "")).strip()) not in REMOVED_TYPES
    ]
    if attachment_only:
        return [q for q in filtered if is_attachment_viewer_question(q)]
    if requested_qtypes:
        canonical_requested_qtypes = {
            _canonical_qtype(str(qtype).strip())
            for qtype in requested_qtypes
            if str(qtype).strip()
        }
        return [
            q for q in filtered
            if _canonical_qtype(str(q.get("type", "")).strip()) in canonical_requested_qtypes
        ]
    return filtered


def img_to_b64(path: Path, max_width: int = 480) -> str | None:
    if Image is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_width:
            img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def build_task_summary(type_counter: Counter) -> str:
    known_keys = {
        key
        for _, items in SUMMARY_GROUPS
        for key, _ in items
    }
    parts: list[str] = []

    for section_title, items in SUMMARY_GROUPS:
        lines = [f'<div class="summary-line"><strong>{section_title}：</strong></div>']
        for key, label in items:
            lines.append(
                f'<div class="summary-line">{label}：{type_counter.get(key, 0)}</div>'
            )
        parts.append(f'<div class="summary-section">{"".join(lines)}</div>')

    other_items = [
        (qtype, count)
        for qtype, count in sorted(type_counter.items())
        if qtype not in known_keys
    ]
    if other_items:
        other_text = "；".join(f"{qtype}={count}" for qtype, count in other_items)
        parts.append(
            f'<div class="summary-other"><strong>其他类型：</strong>{other_text}</div>'
        )

    return "".join(parts)


def build_task_summary_v2(displayed_questions: list[dict]) -> str:
    known_keys = {
        key
        for _, items in SUMMARY_GROUPS
        for key, _ in items
    }
    displayed_type_counter = _canonical_type_counter(displayed_questions)
    displayed_attached_counter, displayed_unattached_counter = (
        _summarize_object_move_questions(displayed_questions)
    )
    parts: list[str] = []

    for section_title, items in SUMMARY_GROUPS:
        lines = [f'<div class="summary-line"><strong>{section_title}</strong></div>']
        object_move_keys = [key for key, _ in items if key in OBJECT_MOVE_TYPES]
        if object_move_keys:
            attached = sum(displayed_attached_counter.get(key, 0) for key in object_move_keys)
            unattached = sum(displayed_unattached_counter.get(key, 0) for key in object_move_keys)
            lines.append(
                f'<div class="summary-line">L2_object_move_all: with_attachment={attached}, '
                f'without_attachment={unattached}</div>'
            )
        for key, label in items:
            if key in OBJECT_MOVE_TYPES:
                attached = displayed_attached_counter.get(key, 0)
                unattached = displayed_unattached_counter.get(key, 0)
                lines.append(
                    f'<div class="summary-line">{label}: with_attachment={attached}, '
                    f'without_attachment={unattached}</div>'
                )
            else:
                lines.append(
                f'<div class="summary-line">{label}: {displayed_type_counter.get(key, 0)}</div>'
                )
        parts.append(f'<div class="summary-section">{"".join(lines)}</div>')

    other_items = [
        (qtype, count)
        for qtype, count in sorted(displayed_type_counter.items())
        if qtype not in known_keys
    ]
    if other_items:
        other_text = "; ".join(f"{qtype}={count}" for qtype, count in other_items)
        parts.append(
            f'<div class="summary-other"><strong>Other Types:</strong> {other_text}</div>'
        )

    return "".join(parts)


def order_questions_for_viewer(questions: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    order_index = {qtype: idx for idx, qtype in enumerate(VIEWER_QTYPE_ORDER)}
    grouped: dict[str, list[dict]] = {}
    for q in questions:
        qtype = _canonical_qtype(str(q.get("type", "")).strip()) or "unknown"
        grouped.setdefault(qtype, []).append(q)

    for group in grouped.values():
        rng.shuffle(group)

    ordered_qtypes = sorted(
        grouped,
        key=lambda qtype: (order_index.get(qtype, len(VIEWER_QTYPE_ORDER)), qtype),
    )
    ordered_questions: list[dict] = []
    for qtype in ordered_qtypes:
        ordered_questions.extend(grouped[qtype])
    return ordered_questions


def _apply_attachment_viewer_filter(questions: list[dict]) -> list[dict]:
    """Apply viewer-only attachment balancing per (scene, frame, canonical qtype)."""
    keep_mask = [True] * len(questions)
    grouped_indices: dict[tuple[str, str, str], list[int]] = {}
    for idx, question in enumerate(questions):
        qtype = _canonical_qtype(str(question.get("type", "")).strip())
        if qtype not in OBJECT_MOVE_TYPES:
            continue
        key = (
            str(question.get("scene_id", "")).strip(),
            str(question.get("image_name", "")).strip(),
            qtype,
        )
        grouped_indices.setdefault(key, []).append(idx)

    for indices in grouped_indices.values():
        attached = [
            idx for idx in indices
            if bool(questions[idx].get("attachment_remapped", False))
        ]
        unattached = [
            idx for idx in indices
            if not bool(questions[idx].get("attachment_remapped", False))
        ]
        if not unattached:
            continue

        allowed_unattached = (2 * len(attached)) if attached else 1
        if len(unattached) <= allowed_unattached:
            continue

        kept_unattached = set(unattached[:allowed_unattached])
        for idx in unattached:
            if idx not in kept_unattached:
                keep_mask[idx] = False

    return [q for idx, q in enumerate(questions) if keep_mask[idx]]


def _apply_global_object_move_ratio_filter(questions: list[dict]) -> list[dict]:
    """Cap unattached object-move questions to a 2:1 ratio per canonical qtype."""
    keep_mask = [True] * len(questions)
    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        qtype = _canonical_qtype(str(question.get("type", "")).strip())
        if qtype not in OBJECT_MOVE_TYPES:
            continue
        grouped_indices.setdefault(qtype, []).append(idx)

    for indices in grouped_indices.values():
        attached = [
            idx for idx in indices
            if bool(questions[idx].get("attachment_remapped", False))
        ]
        unattached = [
            idx for idx in indices
            if not bool(questions[idx].get("attachment_remapped", False))
        ]
        allowed_unattached = 2 * len(attached)
        kept_unattached = set(unattached[:allowed_unattached])
        for idx in unattached:
            if idx not in kept_unattached:
                keep_mask[idx] = False

    return [q for idx, q in enumerate(questions) if keep_mask[idx]]


def is_attachment_viewer_question(question: dict) -> bool:
    qtype = _canonical_qtype(str(question.get("type", "")).strip())
    if qtype in {"attachment_chain", "attachment_move"}:
        return True
    return qtype in OBJECT_MOVE_TYPES and bool(question.get("attachment_remapped", False))


def apply_explicit_viewer_filters(
    source_questions: list[dict],
    *,
    include_attachment_unchanged: bool = True,
) -> list[dict]:
    filtered = list(source_questions)
    if not include_attachment_unchanged:
        filtered = [
            question for question in filtered
            if not is_attachment_unchanged_object_move(question)
        ]
    return filtered


def apply_auto_viewer_filters(questions: list[dict]) -> list[dict]:
    filtered = _apply_attachment_viewer_filter(questions)
    return _apply_global_object_move_ratio_filter(filtered)


def filter_viewer_questions(
    source_questions: list[dict],
    *,
    include_attachment_unchanged: bool = True,
) -> list[dict]:
    filtered = apply_explicit_viewer_filters(
        source_questions,
        include_attachment_unchanged=include_attachment_unchanged,
    )
    return apply_auto_viewer_filters(filtered)


PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f0f2f5;margin:0;padding:20px}}
h1{{text-align:center;color:#333;margin-bottom:4px}}
.stats{{text-align:center;color:#666;font-size:14px;margin-bottom:24px}}
.viewer-actions{{max-width:1100px;margin:0 auto 14px;display:flex;justify-content:flex-end;gap:10px}}
.export-edited{{border:1px solid #1d4ed8;background:#2563eb;color:#fff;border-radius:6px;
                padding:8px 12px;font-size:13px;font-weight:700;cursor:pointer}}
.export-edited:hover{{background:#1d4ed8}}
.summary{{max-width:1100px;margin:0 auto 24px;background:#fff;border-radius:10px;
          box-shadow:0 2px 6px rgba(0,0,0,.12);padding:18px 20px}}
.summary h2{{margin:0 0 12px;color:#111;font-size:18px}}
.summary-block{{color:#374151;font-size:14px;line-height:1.7}}
.summary-line{{margin:2px 0}}
.summary-section{{margin-top:10px}}
.summary-section:first-child{{margin-top:0}}
.summary-other{{margin-top:12px;color:#6b7280}}
.card{{display:flex;background:#fff;border-radius:10px;
       box-shadow:0 2px 6px rgba(0,0,0,.12);margin-bottom:18px;overflow:hidden}}
.card[data-deleted="true"]{{background:#f3f4f6}}
.card[data-deleted="true"] .img-wrap,
.card[data-deleted="true"] .body > :not(.card-actions){{filter:grayscale(1);opacity:.42}}
.img-wrap{{flex:0 0 auto;width:480px;background:#222;display:flex;
           align-items:center;justify-content:center}}
.img-wrap img{{width:480px;display:block}}
.dual-img{{width:100%}}
.img-label{{font-size:11px;color:#999;text-align:center;padding:2px 0;background:#1a1a1a}}
.no-img{{width:480px;height:200px;display:flex;align-items:center;
         justify-content:center;color:#999;font-size:13px}}
.body{{padding:18px 20px 56px;flex:1;min-width:0;position:relative}}
.meta{{font-size:12px;color:#888;margin-bottom:10px}}
.badge{{display:inline-block;padding:2px 9px;border-radius:12px;
        font-weight:bold;font-size:11px;margin-right:6px}}
.L1{{background:#dbeafe;color:#1d4ed8}}
.L2{{background:#fce7f3;color:#9d174d}}
.L3{{background:#ede9fe;color:#5b21b6}}
.extra{{background:#f3f4f6;color:#374151}}
.attachment{{background:#d1fae5;color:#065f46}}
.unchanged{{background:#fef3c7;color:#92400e}}
.review{{background:#fee2e2;color:#991b1b}}
.qtext{{font-size:15px;font-weight:600;color:#111;margin:0 0 14px}}
.opt{{padding:7px 12px;margin:4px 0;border-radius:6px;font-size:14px;
      background:#f8f9fa;border:1px solid #e5e7eb}}
.opt.correct{{background:#dcfce7;border-color:#86efac;font-weight:700}}
.review-notes{{margin-top:14px;padding:12px 14px;border-radius:8px;background:#fff7ed;
              border:1px solid #fed7aa}}
.review-block + .review-block{{margin-top:10px}}
.review-title{{font-size:12px;font-weight:700;color:#9a3412;margin-bottom:6px;
              text-transform:uppercase;letter-spacing:.04em}}
.review-line{{font-size:13px;color:#7c2d12;line-height:1.5}}
.simple-panel{{margin-top:14px}}
.simple-section + .simple-section{{margin-top:14px}}
.simple-section-title{{font-size:12px;font-weight:700;color:#6b7280;margin-bottom:6px;
                       text-transform:uppercase;letter-spacing:.04em}}
.simple-list{{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}}
.simple-item{{padding:8px 10px;border-radius:8px;background:#f8fafc;
              border:1px solid #e5e7eb;font-size:13px;color:#374151;line-height:1.45}}
.simple-key{{font-size:11px;font-weight:700;color:#6b7280;letter-spacing:.04em;
             text-transform:uppercase}}
.simple-value{{margin-top:4px;font-size:14px;color:#111;word-break:break-word}}
.simple-empty{{font-size:13px;color:#9ca3af}}
.footer{{margin-top:14px;font-size:11px;color:#aaa}}
.idx{{float:right;color:#ccc;font-size:12px}}
.card-actions{{position:absolute;right:18px;bottom:16px;display:flex;justify-content:flex-end}}
.delete-card-toggle{{border:1px solid #d1d5db;background:#fff;color:#374151;border-radius:6px;
                     padding:7px 11px;font-size:12px;font-weight:700;cursor:pointer}}
.delete-card-toggle:hover{{background:#f9fafb;border-color:#9ca3af}}
.card[data-deleted="true"] .delete-card-toggle{{border-color:#2563eb;color:#1d4ed8}}
</style>
</head>
<body>
<h1>{title}</h1>
{stats}
<div class="viewer-actions">
  <button type="button" class="export-edited" id="export-edited-html">Export Edited HTML</button>
</div>
{summary}
{cards}
<script>
(() => {{
  const editedHtmlFilename = {edited_html_filename_json};

  function setCardDeleted(card, deleted) {{
    card.dataset.deleted = deleted ? "true" : "false";
    const button = card.querySelector(".delete-card-toggle");
    if (button) {{
      button.textContent = deleted ? "Restore Card" : "Delete Card";
      button.setAttribute("aria-pressed", deleted ? "true" : "false");
    }}
  }}

  document.querySelectorAll(".card").forEach((card) => {{
    const deleted = card.dataset.deleted === "true";
    if (!card.dataset.deleted) {{
      card.dataset.deleted = "false";
    }}
    setCardDeleted(card, deleted);
  }});

  document.addEventListener("click", (event) => {{
    const button = event.target.closest(".delete-card-toggle");
    if (!button) {{
      return;
    }}
    const card = button.closest(".card");
    if (!card) {{
      return;
    }}
    setCardDeleted(card, card.dataset.deleted !== "true");
  }});

  const exportButton = document.getElementById("export-edited-html");
  if (exportButton) {{
    exportButton.addEventListener("click", () => {{
      const htmlText = "<!DOCTYPE html>\\n" + document.documentElement.outerHTML;
      const blob = new Blob([htmlText], {{type: "text/html;charset=utf-8"}});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = editedHtmlFilename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }});
  }}
}})();
</script>
</body>
</html>
"""

CARD = """\
<div class="card" data-deleted="false">
  <div class="img-wrap">{img}</div>
  <div class="body">
    {meta}
    <p class="qtext">{question}</p>
    {options}
    {review_notes}
    {footer}
    <div class="card-actions"><button type="button" class="delete-card-toggle" aria-pressed="false">Delete Card</button></div>
  </div>
</div>"""

SIMPLE_CARD = """\
<div class="card" data-deleted="false">
  <div class="img-wrap">{img}</div>
  <div class="body">
    {meta}
    {question}
    {options}
    <div class="simple-panel">
      {objects}
      {relations}
    </div>
    {footer}
    <div class="card-actions"><button type="button" class="delete-card-toggle" aria-pressed="false">Delete Card</button></div>
  </div>
</div>"""


def build_stats_bar(displayed_questions: list[dict]) -> str:
    level_counter: Counter = Counter()
    for question in displayed_questions:
        qtype = _canonical_qtype(str(question.get("type", "")).strip())
        level = QTYPE_LEVEL.get(qtype)
        if level:
            level_counter[level] += 1

    parts = [f"{len(displayed_questions)} questions"]
    for level in LEVEL_DISPLAY_ORDER:
        parts.append(f"{level}: {level_counter.get(level, 0)}")
    return '<div class="stats">' + " &nbsp;&middot;&nbsp; ".join(
        html.escape(part) for part in parts
    ) + "</div>"


def _default_edited_html_filename(output_path: str | Path) -> str:
    path = Path(output_path)
    return f"{path.stem}_edited{path.suffix or '.html'}"


def prepare_viewer_questions(
    questions: list[dict],
    *,
    requested_qtypes: set[str] | None = None,
    attachment_only: bool = False,
    include_attachment_unchanged: bool = True,
    apply_filters: bool = False,
    shuffle_seed: int = 42,
) -> list[dict]:
    displayed_questions = select_viewer_source_questions(
        list(questions),
        requested_qtypes=requested_qtypes,
        attachment_only=attachment_only,
    )
    displayed_questions = apply_explicit_viewer_filters(
        displayed_questions,
        include_attachment_unchanged=include_attachment_unchanged,
    )
    if apply_filters:
        displayed_questions = apply_auto_viewer_filters(displayed_questions)
    return order_questions_for_viewer(displayed_questions, seed=shuffle_seed)


def _stringify_review_value(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            rendered = _stringify_review_value(item)
            if rendered != "-":
                parts.append(rendered)
        return ", ".join(parts) if parts else "-"
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            rendered = _stringify_review_value(item)
            if rendered != "-":
                parts.append(f"{key}={rendered}")
        return ", ".join(parts) if parts else "-"
    return str(value)


def _render_review_block(title: str, lines: list[str]) -> str:
    rendered_lines = "".join(
        f'<div class="review-line">{html.escape(line)}</div>'
        for line in lines
        if line.strip()
    )
    if not rendered_lines:
        return ""
    return (
        '<div class="review-block">'
        f'<div class="review-title">{html.escape(title)}</div>'
        f"{rendered_lines}"
        "</div>"
    )


def _format_presence_object_review(item: dict[str, object]) -> str:
    label = str(item.get("label", "")).strip() or "object"
    obj_id = item.get("obj_id")
    roles = [
        str(role).strip()
        for role in item.get("roles", [])
        if str(role).strip()
    ] if isinstance(item.get("roles"), list) else []
    status = str(item.get("status", "")).strip() or "-"
    reason = str(item.get("reason", "")).strip()

    name = f"{label}#{obj_id}" if obj_id not in (None, "") else label
    if roles:
        name += f" [{', '.join(roles)}]"
    if reason:
        return f"{name}: {status} ({reason})"
    return f"{name}: {status}"


def _normalize_reason_codes(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip().lower()
        if text:
            normalized.append(text)
    return normalized


def _join_phrases(parts: list[str]) -> str:
    filtered = [part.strip() for part in parts if str(part).strip()]
    if not filtered:
        return "需要进一步复核"
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]}和{filtered[1]}"
    return "、".join(filtered[:-1]) + f"和{filtered[-1]}"


def _format_audit_object_name(label: object, obj_id: object) -> str:
    label_text = str(label or "").strip() or "物体"
    obj_id_text = _stringify_review_value(obj_id)
    return f"{label_text}#{obj_id_text}" if obj_id_text != "-" else label_text


def _humanize_dinox_reason(code: str) -> str:
    mapping = {
        "dinox_error": "2D 检测器执行失败",
        "dinox_no_strong_detection": "没有找到足够可靠的 2D 检测结果",
        "dinox_multiple_strong_detections": "找到了多个较强的 2D 检测结果",
        "dinox_detection_misses_target": "2D 检测结果和提到的物体对不上",
    }
    text = str(code or "").strip().lower()
    return mapping.get(text, text.replace("_", " "))


def _humanize_mesh_reason(code: str) -> str:
    mapping = {
        "mesh_low_iou": "3D 投影与检测结果的重叠度过低",
        "mesh_high_under_coverage": "3D 投影漏掉了过多目标区域",
        "mesh_high_over_coverage": "3D 投影覆盖了过多额外区域",
        "mesh_bad_area_ratio": "3D 投影面积和检测结果不一致",
        "mesh_high_depth_bad_ratio": "3D 投影深度与图像深度不一致",
        "mesh_projects_out_of_frame": "3D 投影超出了画面范围",
        "topology_fail": "3D 网格拓扑不可靠",
        "missing_projection_context": "缺少投影所需的上下文信息",
        "missing_instance_mesh_data": "缺少该实例的网格数据",
        "dinox_error": "用于辅助的 2D 检测器执行失败",
        "no_detection_overlap": "没有和该物体重叠的 2D 检测结果",
        "no_detection_mask": "没有可用的 2D 检测掩码",
        "invalid_crop": "该物体对应的裁剪区域无效",
    }
    text = str(code or "").strip().lower()
    return mapping.get(text, text.replace("_", " "))


def _summarize_dinox_review(item: dict[str, object]) -> str:
    reasons = _normalize_reason_codes(item.get("reason_codes"))
    if not reasons:
        return "2D 检测需要进一步复核"
    return _join_phrases([_humanize_dinox_reason(reason) for reason in reasons])


def _summarize_mesh_review(item: dict[str, object]) -> str:
    mesh_reasons = _normalize_reason_codes(item.get("mesh_mask_reason_codes"))
    if mesh_reasons:
        return _join_phrases([_humanize_mesh_reason(f"mesh_{reason}") for reason in mesh_reasons])
    reasons = _normalize_reason_codes(item.get("reason_codes"))
    if not reasons:
        return "3D 网格检查需要进一步复核"
    return _join_phrases([_humanize_mesh_reason(reason) for reason in reasons])


def _humanize_audit_decision(value: object) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "pass": "通过",
        "manual_review": "需要人工复核",
        "drop": "丢弃",
        "skipped": "跳过",
    }
    if not text:
        return "-"
    return mapping.get(text, text)


def _build_post_generation_audit_lines(post_generation_review: dict[str, object]) -> list[str]:
    decision = str(post_generation_review.get("decision", "-")).strip() or "-"
    mesh_reviews = [
        item for item in post_generation_review.get("mesh_object_reviews", [])
        if isinstance(item, dict)
    ]
    dinox_reviews = [
        item for item in post_generation_review.get("dinox_label_reviews", [])
        if isinstance(item, dict)
    ]
    dinox_by_label = {
        str(item.get("label", "")).strip().lower(): item
        for item in dinox_reviews
        if str(item.get("label", "")).strip()
    }

    issue_lines: list[str] = []
    ok_lines: list[str] = []
    covered_labels: set[str] = set()

    for item in mesh_reviews:
        label = str(item.get("label", "")).strip().lower()
        covered_labels.add(label)
        object_name = _format_audit_object_name(item.get("label"), item.get("obj_id"))
        mesh_ok = str(item.get("decision", "")).strip().lower() == "pass"
        dinox_review = dinox_by_label.get(label)
        dinox_ok = (
            dinox_review is None
            or str(dinox_review.get("decision", "")).strip().lower() in {"", "pass", "skipped"}
        )
        if mesh_ok and dinox_ok:
            ok_lines.append(f"{object_name}：正常。2D 检测和 3D 网格检查都已通过。")
            continue
        if mesh_ok and not dinox_ok:
            issue_lines.append(
                f"{object_name}：3D 网格检查已通过，但 2D 检测仍需复核，原因是"
                f"{_summarize_dinox_review(dinox_review)}。"
            )
            continue
        if not mesh_ok and dinox_ok:
            issue_lines.append(
                f"{object_name}：2D 检测已通过，但 3D 网格投影与该物体不够匹配，原因是"
                f"{_summarize_mesh_review(item)}。"
            )
            continue
        issue_lines.append(
            f"{object_name}：2D 检测和 3D 网格检查都需要复核。"
            f"2D 问题：{_summarize_dinox_review(dinox_review)}。"
            f"3D 问题：{_summarize_mesh_review(item)}。"
        )

    for item in dinox_reviews:
        label = str(item.get("label", "")).strip().lower()
        if label in covered_labels:
            continue
        label_name = str(item.get("label", "")).strip() or "物体"
        if str(item.get("decision", "")).strip().lower() == "pass":
            ok_lines.append(f"{label_name}：正常。2D 检测已通过。")
        elif str(item.get("decision", "")).strip().lower() != "skipped":
            issue_lines.append(
                f"{label_name}：2D 检测需要复核，原因是{_summarize_dinox_review(item)}。"
            )

    lines = [f"审核结论：{_humanize_audit_decision(decision)}"]
    if issue_lines:
        lines.append(f"问题概述：{len(issue_lines)} 个对象需要复核。")
        lines.extend(issue_lines)
        lines.extend(ok_lines)
    else:
        lines.append("问题概述：未发现审核问题。")
        lines.extend(ok_lines)
    return lines


def _format_referability_mention(item: dict[str, object]) -> str:
    role = str(item.get("role", "mentioned")).strip() or "mentioned"
    label = str(item.get("label", "")).strip() or "-"
    obj_id = _stringify_review_value(item.get("obj_id"))
    label_status = _stringify_review_value(item.get("label_status"))
    candidate_ids = _stringify_review_value(item.get("candidate_object_ids"))
    referable_ids = _stringify_review_value(item.get("referable_object_ids"))
    reasons = _stringify_review_value(item.get("reason_codes"))
    result = "pass" if bool(item.get("passes_referability_check", False)) else "drop"
    return (
        f"{role}: label={label}, obj_id={obj_id}, label_status={label_status}, "
        f"candidates={candidate_ids}, referable={referable_ids}, result={result}, "
        f"reasons={reasons}"
    )


def _is_manual_review_question(question: dict) -> bool:
    if str(question.get("manual_review_reason", "")).strip():
        return True
    post_review = question.get("question_post_generation_review")
    if isinstance(post_review, dict) and post_review.get("decision") == "manual_review":
        return True
    presence_review = question.get("question_presence_review")
    if isinstance(presence_review, dict) and presence_review.get("decision") == "manual_review":
        return True
    answer_review = question.get("question_answer_review")
    return (
        isinstance(answer_review, dict)
        and answer_review.get("decision") == "manual_review"
    )


def _build_meta_html(question: dict, idx: int) -> str:
    qtype = _canonical_qtype(str(question.get("type", "")).strip())
    level = QTYPE_LEVEL.get(qtype, "")
    qtype_label = QTYPE_DISPLAY.get(qtype, qtype or "unknown")
    badges: list[str] = []

    if level:
        badges.append(f'<span class="badge {html.escape(level)}">{html.escape(level)}</span>')
    badges.append(
        '<span class="badge extra">'
        f"{html.escape(qtype_label)}"
        "</span>"
    )
    if bool(question.get("attachment_remapped", False)):
        badges.append('<span class="badge attachment">with-attachment</span>')
    elif qtype in OBJECT_MOVE_TYPES:
        badges.append('<span class="badge extra">without-attachment</span>')
    if is_attachment_unchanged_object_move(question):
        badges.append('<span class="badge unchanged">answer-unchanged</span>')
    if _is_manual_review_question(question):
        badges.append('<span class="badge review">manual-review</span>')

    return (
        '<div class="meta">'
        + "".join(badges)
        + f'<span class="idx">#{idx}</span>'
        + "</div>"
    )


def _build_review_notes_html(
    question: dict,
    *,
    include_referability_audit: bool = False,
) -> str:
    blocks: list[str] = []

    manual_review_reason = str(question.get("manual_review_reason", "")).strip()
    if manual_review_reason:
        blocks.append(_render_review_block("Manual Review", [manual_review_reason]))

    presence_review = question.get("question_presence_review")
    if isinstance(presence_review, dict):
        lines = [
            f"decision: {str(presence_review.get('decision', '-')).strip() or '-'}",
        ]
        flagged_labels = _stringify_review_value(presence_review.get("flagged_labels"))
        if flagged_labels != "-":
            lines.append(f"flagged labels: {flagged_labels}")
        flagged_object_ids = _stringify_review_value(presence_review.get("flagged_object_ids"))
        if flagged_object_ids != "-":
            lines.append(f"flagged object ids: {flagged_object_ids}")
        object_reviews = presence_review.get("object_reviews", [])
        if isinstance(object_reviews, list):
            for item in object_reviews:
                if isinstance(item, dict):
                    lines.append(_format_presence_object_review(item))
        blocks.append(_render_review_block("VLM Review", lines))

    answer_review = question.get("question_answer_review")
    if isinstance(answer_review, dict) and answer_review.get("decision") != "skipped":
        lines = [
            f"decision: {str(answer_review.get('decision', '-')).strip() or '-'}",
        ]
        for label, key in (
            ("predicted answer", "predicted_answer"),
            ("gold answer", "gold_answer"),
            ("predicted option", "predicted_option"),
            ("gold option", "gold_option"),
        ):
            rendered = _stringify_review_value(answer_review.get(key))
            if rendered != "-":
                lines.append(f"{label}: {rendered}")
        reason = str(answer_review.get("reason", "")).strip()
        if reason:
            lines.append(f"reason: {reason}")
        blocks.append(_render_review_block("Answer Review", lines))

    post_generation_review = question.get("question_post_generation_review")
    if isinstance(post_generation_review, dict):
        lines = _build_post_generation_audit_lines(post_generation_review)
        blocks.append(_render_review_block("生成后审核", lines))

    referability_audit = question.get("question_referability_audit")
    if include_referability_audit and isinstance(referability_audit, dict):
        lines = [
            f"decision: {str(referability_audit.get('decision', '-')).strip() or '-'}",
        ]
        reason_codes = _stringify_review_value(referability_audit.get("reason_codes"))
        lines.append(f"reason codes: {reason_codes}")
        frame_referable_ids = _stringify_review_value(
            referability_audit.get("frame_referable_object_ids")
        )
        lines.append(f"frame referable ids: {frame_referable_ids}")
        mentioned_objects = referability_audit.get("mentioned_objects", [])
        if isinstance(mentioned_objects, list):
            for item in mentioned_objects:
                if isinstance(item, dict):
                    lines.append(_format_referability_mention(item))
        blocks.append(_render_review_block("Referability Audit", lines))

    if not blocks:
        return ""
    return '<div class="review-notes">' + "".join(blocks) + "</div>"


def question_review_notes(question: dict) -> str:
    return _build_review_notes_html(question, include_referability_audit=True)


def _build_footer_html(question: dict) -> str:
    scene = str(question.get("scene_id", "")).strip()
    frame = str(question.get("image_name", "")).strip()
    if not scene or not frame:
        scene = scene or "scene unavailable"
        frame = frame or "frame unavailable"
    return (
        '<div class="footer">'
        + " &nbsp;/&nbsp; ".join(html.escape(part) for part in (scene, frame))
        + "</div>"
    )


def _build_summary_html(displayed_questions: list[dict]) -> str:
    return (
        '<div class="summary">\n'
        '  <h2>Task Summary</h2>\n'
        f'  <div class="summary-block">{build_task_summary_v2(displayed_questions)}</div>\n'
        '</div>'
    )


def _infer_dataset(question: dict) -> str:
    """Infer dataset from question metadata, falling back to scene_id pattern.

    Priority:
    1. ``_dataset`` / ``dataset`` field (set during pipeline generation)
    2. scene_id pattern: ``sceneXXXX_XX`` → scannet, hex hash → scannetpp
    3. default: scannet
    """
    explicit = str(question.get("_dataset") or question.get("dataset") or "").strip().lower()
    if explicit in {"scannet", "scannetpp"}:
        return explicit

    source_text = str(question.get("_source_benchmark") or "").strip().lower()
    if "scannetpp" in source_text:
        return "scannetpp"
    if "pilot" in source_text:
        return "scannet"

    scene_id = str(question.get("scene_id", ""))
    # ScanNet scenes: scene0670_01, scene0046_02, ...
    if scene_id.startswith("scene") and "_" in scene_id:
        return "scannet"
    # ScanNet++ scenes: hex hashes like f3d64c30f8, 0d2ee665be, ...
    if scene_id and len(scene_id) >= 8:
        return "scannetpp"

    return "scannet"


def _resolve_image_path(
    question: dict,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str = "iphone",
) -> Path:
    """Resolve an image path for a question, trying multiple roots and path layouts.

    Returns the first existing path found, or the first candidate as a fallback
    (the caller handles missing files via img_to_b64).
    """
    dataset = _infer_dataset(question)
    scene = str(question.get("scene_id", ""))
    frame = str(question.get("image_name", ""))
    roots = scannetpp_roots if dataset == "scannetpp" else scannet_roots

    candidates: list[Path] = []
    for root in roots:
        if dataset == "scannetpp":
            if scannetpp_sensor == "iphone":
                candidates.append(root / scene / frame)
            elif scannetpp_sensor == "dslr":
                candidates.append(root / scene / "dslr" / "resized_images" / frame)
            else:
                raise ValueError(
                    f"scannetpp_sensor must be 'iphone' or 'dslr', got {scannetpp_sensor!r}"
                )
            # Also try alternative layouts as fallbacks
            candidates.extend([
                root / scene / frame,
                root / "scans" / scene / frame,
                root / scene / "dslr" / "resized_images" / frame,
                root / scene / "iphone" / "rgb" / frame,
                root / "scans" / scene / "dslr" / "resized_images" / frame,
                root / "scans" / scene / "iphone" / "rgb" / frame,
            ])
        else:
            candidates.extend([
                root / scene / "color" / frame,
                root / scene / frame,
            ])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    for candidate in unique:
        if candidate.exists():
            return candidate
    return unique[0] if unique else Path(".")


def _new_image_stats() -> dict[str, object]:
    return {
        "total": 0,
        "embedded": 0,
        "missing": 0,
        "missing_paths": [],
    }


def _record_image_result(
    image_stats: dict[str, object] | None,
    img_path: Path,
    *,
    embedded: bool,
) -> None:
    if image_stats is None:
        return
    image_stats["total"] = int(image_stats.get("total", 0)) + 1
    if embedded:
        image_stats["embedded"] = int(image_stats.get("embedded", 0)) + 1
        return

    image_stats["missing"] = int(image_stats.get("missing", 0)) + 1
    missing_paths = image_stats.setdefault("missing_paths", [])
    if isinstance(missing_paths, list) and len(missing_paths) < 5:
        missing_paths.append(str(img_path))


def _build_one_image_html(
    question: dict,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    max_width: int,
    scannetpp_sensor: str,
    image_stats: dict[str, object] | None,
    missing_label: str = "image not found",
) -> str:
    img_path = _resolve_image_path(question, scannet_roots, scannetpp_roots, scannetpp_sensor)
    b64 = img_to_b64(img_path, max_width)
    if b64:
        _record_image_result(image_stats, img_path, embedded=True)
        return f'<img src="data:image/jpeg;base64,{b64}">'
    _record_image_result(image_stats, img_path, embedded=False)
    return f'<div class="no-img">{html.escape(missing_label)}</div>'


def _collect_aux_image_names(question: dict) -> list[str]:
    names: list[str] = []
    raw_list = question.get("auxiliary_image_names")
    if isinstance(raw_list, list):
        names.extend(str(name).strip() for name in raw_list if str(name).strip())
    if not names:
        single = str(question.get("aux_image_name") or question.get("image_name_2") or "").strip()
        if single:
            names.append(single)
    return names


def _build_image_html(
    question: dict,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    max_width: int,
    scannetpp_sensor: str = "iphone",
    image_stats: dict[str, object] | None = None,
) -> str:
    primary = _build_one_image_html(
        question, scannet_roots, scannetpp_roots, max_width, scannetpp_sensor, image_stats,
    )
    aux_names = _collect_aux_image_names(question)
    if not aux_names:
        return primary

    blocks = [f'<div class="img-label">original</div>{primary}']
    total = len(aux_names)
    for i, aux_name in enumerate(aux_names, start=1):
        aux_q = {**question, "image_name": aux_name}
        label = "auxiliary" if total == 1 else f"auxiliary {i}/{total}"
        aux_html = _build_one_image_html(
            aux_q, scannet_roots, scannetpp_roots, max_width, scannetpp_sensor, image_stats,
            missing_label=f"{label} not found",
        )
        blocks.append(f'<div class="img-label">{html.escape(label)}</div>{aux_html}')

    return '<div class="dual-img">' + "".join(blocks) + '</div>'


def _warn_if_scannetpp_iphone_root_looks_raw(roots: list[Path]) -> None:
    """Warn when an iPhone viewer root looks like raw ScanNet++ data."""
    for root in roots:
        if (
            (root / "iphone").is_dir()
            and (root / "scans").is_dir()
            and not any(root.glob("frame_*.jpg"))
        ):
            print(
                "Warning: --scannetpp_image_root looks like a raw ScanNet++ dataset root. "
                "For scannetpp iPhone images, pass the extracted frame root, e.g. "
                "output/scannetpp_iphone_frames."
            )


def _print_image_stats(
    image_stats: dict[str, object],
    *,
    label: str,
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
) -> None:
    total = int(image_stats.get("total", 0))
    embedded = int(image_stats.get("embedded", 0))
    missing = int(image_stats.get("missing", 0))
    print(f"{label} images: embedded {embedded}/{total}; missing {missing}")
    if missing <= 0:
        return

    if scannetpp_sensor == "iphone":
        _warn_if_scannetpp_iphone_root_looks_raw(scannetpp_roots)
        print(
            "ScanNet++ iPhone image path format: "
            "<scannetpp_image_root>/<scene_id>/<image_name> "
            "(scannetpp_image_root should be the extracted frame root, not the raw dataset root)"
        )

    missing_paths = image_stats.get("missing_paths", [])
    if isinstance(missing_paths, list) and missing_paths:
        print("First missing image paths:")
        for path in missing_paths:
            print(f"  {path}")


def _build_options_html(question: dict) -> str:
    answer = str(question.get("answer", ""))
    options = question.get("options", [])
    opt_html = ""
    if isinstance(options, list):
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            cls = "opt correct" if letter == answer else "opt"
            opt_html += (
                f'<div class="{cls}">{letter}.&nbsp; {html.escape(str(opt))}</div>\n    '
            )
    return opt_html


def _simple_value_from_keys(question: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        rendered = _stringify_review_value(question.get(key))
        if rendered != "-":
            return rendered
    return "-"


def _build_simple_items(
    question: dict,
    specs: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for label, keys in specs:
        rendered = _simple_value_from_keys(question, keys)
        if rendered != "-":
            items.append((label, rendered))
    return items


def _build_simple_fallback_objects(question: dict) -> list[tuple[str, str]]:
    mentioned_objects = question.get("mentioned_objects", [])
    items: list[tuple[str, str]] = []
    if not isinstance(mentioned_objects, list):
        return items
    for item in mentioned_objects:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "mentioned")).strip() or "mentioned"
        label = str(item.get("label", "")).strip()
        if not label:
            label = _stringify_review_value(item.get("obj_id"))
        if label and label != "-":
            items.append((role, label))
    return items


def _build_simple_fallback_relations(question: dict) -> list[tuple[str, str]]:
    items = [("correct_value", _stringify_review_value(question.get("correct_value")))]
    for label, key in (
        ("old_correct_value", "old_correct_value"),
        ("new_correct_value", "new_correct_value"),
    ):
        rendered = _stringify_review_value(question.get(key))
        if rendered != "-":
            items.append((label, rendered))
    return items


def _build_simple_sections(question: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    qtype = _canonical_qtype(str(question.get("type", "")).strip())
    specs = SIMPLE_VIEWER_FIELD_SPECS.get(qtype)
    if specs is None:
        return (
            _build_simple_fallback_objects(question),
            _build_simple_fallback_relations(question),
        )
    return (
        _build_simple_items(question, specs["objects"]),
        _build_simple_items(question, specs["relations"]),
    )


def _render_simple_section(title: str, items: list[tuple[str, str]]) -> str:
    if items:
        body = "".join(
            '<div class="simple-item">'
            f'<div class="simple-key">{html.escape(label)}</div>'
            f'<div class="simple-value">{html.escape(value)}</div>'
            "</div>"
            for label, value in items
        )
        return (
            '<div class="simple-section">'
            f'<div class="simple-section-title">{html.escape(title)}</div>'
            f'<div class="simple-list">{body}</div>'
            "</div>"
        )
    return (
        '<div class="simple-section">'
        f'<div class="simple-section-title">{html.escape(title)}</div>'
        '<div class="simple-empty">-</div>'
        "</div>"
    )


def _build_full_viewer_html_from_displayed_questions(
    displayed_questions: list[dict],
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    *,
    max_width: int = 480,
    title: str = "predictive spatial reasoning benchmark",
    include_referability_audit: bool = False,
    edited_html_filename: str = "viewer_edited.html",
    scannetpp_sensor: str = "iphone",
    image_stats: dict[str, object] | None = None,
) -> str:
    summary_html = _build_summary_html(displayed_questions)
    stats_html = build_stats_bar(displayed_questions)
    cards: list[str] = []

    for idx, question in enumerate(displayed_questions, start=1):
        cards.append(
            CARD.format(
                img=_build_image_html(
                    question,
                    scannet_roots,
                    scannetpp_roots,
                    max_width,
                    scannetpp_sensor=scannetpp_sensor,
                    image_stats=image_stats,
                ),
                meta=_build_meta_html(question, idx),
                question=html.escape(str(question.get("question", ""))),
                options=_build_options_html(question),
                review_notes=_build_review_notes_html(
                    question,
                    include_referability_audit=include_referability_audit,
                ),
                footer=_build_footer_html(question),
            )
        )

    return PAGE.format(
        title=html.escape(title),
        stats=stats_html,
        summary=summary_html,
        cards="\n".join(cards),
        edited_html_filename_json=json.dumps(edited_html_filename),
    )


def _build_simple_viewer_html_from_displayed_questions(
    displayed_questions: list[dict],
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    *,
    max_width: int = 480,
    title: str = "predictive spatial reasoning benchmark (simple review)",
    edited_html_filename: str = "viewer_edited.html",
    scannetpp_sensor: str = "iphone",
    image_stats: dict[str, object] | None = None,
) -> str:
    summary_html = _build_summary_html(displayed_questions)
    stats_html = build_stats_bar(displayed_questions)
    cards: list[str] = []

    for idx, question in enumerate(displayed_questions, start=1):
        objects, relations = _build_simple_sections(question)
        cards.append(
            SIMPLE_CARD.format(
                img=_build_image_html(
                    question,
                    scannet_roots,
                    scannetpp_roots,
                    max_width,
                    scannetpp_sensor=scannetpp_sensor,
                    image_stats=image_stats,
                ),
                meta=_build_meta_html(question, idx),
                question=(
                    f'<p class="qtext">{html.escape(str(question.get("question", "")))}</p>'
                    if str(question.get("question", "")).strip()
                    else ""
                ),
                options=_build_options_html(question),
                objects=_render_simple_section("Objects", objects),
                relations=_render_simple_section("Relations", relations),
                footer=_build_footer_html(question),
            )
        )

    return PAGE.format(
        title=html.escape(title),
        stats=stats_html,
        summary=summary_html,
        cards="\n".join(cards),
        edited_html_filename_json=json.dumps(edited_html_filename),
    )


def build_viewer_html(
    questions: list[dict],
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    *,
    max_width: int = 480,
    shuffle_seed: int = 42,
    title: str = "predictive spatial reasoning benchmark",
    requested_qtypes: set[str] | None = None,
    attachment_only: bool = False,
    include_attachment_unchanged: bool = True,
    include_referability_audit: bool = False,
    apply_filters: bool = False,
    edited_html_filename: str = "viewer_edited.html",
    scannetpp_sensor: str = "iphone",
) -> str:
    displayed_questions = prepare_viewer_questions(
        questions,
        requested_qtypes=requested_qtypes,
        attachment_only=attachment_only,
        include_attachment_unchanged=include_attachment_unchanged,
        apply_filters=apply_filters,
        shuffle_seed=shuffle_seed,
    )
    return _build_full_viewer_html_from_displayed_questions(
        displayed_questions,
        scannet_roots,
        scannetpp_roots,
        max_width=max_width,
        title=title,
        include_referability_audit=include_referability_audit,
        edited_html_filename=edited_html_filename,
        scannetpp_sensor=scannetpp_sensor,
    )


def build_simple_viewer_html(
    questions: list[dict],
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    *,
    max_width: int = 480,
    shuffle_seed: int = 42,
    title: str = "predictive spatial reasoning benchmark (simple review)",
    requested_qtypes: set[str] | None = None,
    attachment_only: bool = False,
    include_attachment_unchanged: bool = True,
    apply_filters: bool = False,
    edited_html_filename: str = "viewer_edited.html",
    scannetpp_sensor: str = "iphone",
) -> str:
    displayed_questions = prepare_viewer_questions(
        questions,
        requested_qtypes=requested_qtypes,
        attachment_only=attachment_only,
        include_attachment_unchanged=include_attachment_unchanged,
        apply_filters=apply_filters,
        shuffle_seed=shuffle_seed,
    )
    return _build_simple_viewer_html_from_displayed_questions(
        displayed_questions,
        scannet_roots,
        scannetpp_roots,
        max_width=max_width,
        title=title,
        edited_html_filename=edited_html_filename,
        scannetpp_sensor=scannetpp_sensor,
    )


def main():
    parser = argparse.ArgumentParser(description="Build HTML QA viewer")
    parser.set_defaults(
        include_attachment_unchanged=True,
        include_referability_audit=False,
        apply_auto_filters=False,
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON (e.g. human_validation_sample.json)",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help=(
            "Deprecated single image root. Use --scannet_image_root / --scannetpp_image_root "
            "instead. When neither new flag is provided, this is used as the ScanNet root."
        ),
    )
    parser.add_argument(
        "--scannet_image_root",
        action="append",
        default=None,
        help="ScanNet image root (scans directory containing scene/color). May be repeated.",
    )
    parser.add_argument(
        "--scannetpp_image_root",
        action="append",
        default=None,
        help=(
            "ScanNet++ image root. May be repeated. "
            "For iPhone, pass the extracted frame root, e.g. "
            "output/scannetpp_iphone_frames, with layout "
            "<root>/<scene_id>/<image_name>. "
            "For DSLR: <root>/<scene_id>/dslr/resized_images/<image_name>."
        ),
    )
    parser.add_argument("--output", default="viewer.html")
    parser.add_argument(
        "--simple_output",
        default=None,
        help="Optional second output path for a simple structured-review HTML page",
    )
    parser.add_argument(
        "--edited_output",
        default=None,
        help="Optional filename for the edited HTML exported from the page; defaults to <output>_edited.html",
    )
    parser.add_argument(
        "--qtypes",
        default="",
        help="Comma-separated question types to keep",
    )
    parser.add_argument(
        "--attachment_only",
        action="store_true",
        help="Keep only attachment-related questions: attachment_chain, attachment_move, and attached object_move_* items",
    )
    parser.add_argument(
        "--include_attachment_unchanged",
        dest="include_attachment_unchanged",
        action="store_true",
        help="Show attachment-mediated object_move_* questions whose answer is unchanged after the move (default)",
    )
    parser.add_argument(
        "--hide_attachment_unchanged",
        dest="include_attachment_unchanged",
        action="store_false",
        help="Hide attachment-mediated object_move_* questions whose answer is unchanged after the move",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=480,
        help="Max image width in pixels (default 480)",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed used to shuffle questions within the same type",
    )
    parser.add_argument(
        "--include_referability_audit",
        dest="include_referability_audit",
        action="store_true",
        help="Render Referability Audit blocks in review cards (hidden by default)",
    )
    parser.add_argument(
        "--apply_auto_filters",
        action="store_true",
        help="Apply legacy attachment-based auto-trimming after explicit viewer filters",
    )
    parser.add_argument(
        "--dataset", type=str, choices=("scannet", "scannetpp"), default=None,
        help="Deprecated. Dataset is now inferred per-question from the _dataset / dataset field.",
    )
    parser.add_argument(
        "--scannetpp_sensor", type=str, choices=("iphone", "dslr"), default="iphone",
        help="Sensor to use when resolving scannetpp image paths",
    )
    args = parser.parse_args()

    if Image is None:
        sys.exit("Pillow is required: pip install Pillow")

    # Resolve image roots. New --scannet_image_root / --scannetpp_image_root take
    # precedence. Fall back to --image_root as scannet root for backward compat.
    scannet_roots: list[Path] = (
        [Path(p) for p in args.scannet_image_root]
        if args.scannet_image_root
        else [Path(args.image_root)] if args.image_root else []
    )
    scannetpp_roots: list[Path] = (
        [Path(p) for p in args.scannetpp_image_root]
        if args.scannetpp_image_root
        else []
    )

    if not scannet_roots and not scannetpp_roots:
        parser.error(
            "At least one of --scannet_image_root, --scannetpp_image_root, "
            "or --image_root is required."
        )

    if args.dataset is not None:
        print(
            "Note: --dataset is deprecated and ignored. "
            "Dataset is inferred per-question from the _dataset / dataset field."
        )

    with open(args.questions, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) and "questions" in data else data

    if args.attachment_only and args.qtypes:
        parser.error("--attachment_only cannot be combined with --qtypes")

    requested_qtypes: set[str] = set()
    if args.qtypes:
        requested_qtypes.update(
            qtype.strip() for qtype in args.qtypes.split(",") if qtype.strip()
        )
    displayed_questions = prepare_viewer_questions(
        questions,
        requested_qtypes=requested_qtypes,
        attachment_only=args.attachment_only,
        include_attachment_unchanged=args.include_attachment_unchanged,
        apply_filters=args.apply_auto_filters,
        shuffle_seed=args.shuffle_seed,
    )
    full_image_stats = _new_image_stats()
    html_text = _build_full_viewer_html_from_displayed_questions(
        displayed_questions,
        scannet_roots,
        scannetpp_roots,
        max_width=args.max_width,
        include_referability_audit=args.include_referability_audit,
        edited_html_filename=(
            args.edited_output
            if args.edited_output
            else _default_edited_html_filename(args.output)
        ),
        scannetpp_sensor=args.scannetpp_sensor,
        image_stats=full_image_stats,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"Saved: {out}  ({size_kb} KB)")
    _print_image_stats(
        full_image_stats,
        label="Full viewer",
        scannetpp_roots=scannetpp_roots,
        scannetpp_sensor=args.scannetpp_sensor,
    )

    if args.simple_output:
        simple_image_stats = _new_image_stats()
        simple_html = _build_simple_viewer_html_from_displayed_questions(
            displayed_questions,
            scannet_roots,
            scannetpp_roots,
            max_width=args.max_width,
            edited_html_filename=(
                args.edited_output
                if args.edited_output
                else _default_edited_html_filename(
                    args.simple_output if args.simple_output else args.output
                )
            ),
            scannetpp_sensor=args.scannetpp_sensor,
            image_stats=simple_image_stats,
        )
        simple_out = Path(args.simple_output)
        simple_out.parent.mkdir(parents=True, exist_ok=True)
        simple_out.write_text(simple_html, encoding="utf-8")
        simple_size_kb = simple_out.stat().st_size // 1024
        print(f"Saved: {simple_out}  ({simple_size_kb} KB)")
        _print_image_stats(
            simple_image_stats,
            label="Simple viewer",
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=args.scannetpp_sensor,
        )


if __name__ == "__main__":
    main()
