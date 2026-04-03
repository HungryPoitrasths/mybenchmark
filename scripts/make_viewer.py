#!/usr/bin/env python3
"""Generate a self-contained HTML viewer for QA validation.

Each question is shown next to its source image.
The output is a single HTML file with base64-embedded images - no server
required, just open it in any browser.

Usage:
    python scripts/make_viewer.py \
        --questions output/pilot/human_validation_sample.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --output output/pilot/viewer.html

    python scripts/make_viewer.py \
        --questions output/pilot_depth/benchmark.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --output output/pilot_depth/attachment_viewer.html \
        --attachment_only

    python scripts/make_viewer.py \
        --questions output/pilot_meshray/benchmark.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --output output/pilot_meshray/attachment_viewer.html \
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
            ("object_rotate_object_centric", "L2_object_rotate_object_centric"),
            ("object_move_allocentric", "L2_object_move_allocentric"),
            ("viewpoint_move", "L2_viewpoint_move"),
            ("object_remove", "L2_object_remove"),
        ],
    ),
    (
        "L3 多跳 / 反事实",
        [
            ("attachment_chain", "L3_attachment_chain"),
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
QUESTION_TYPE_ALIASES = {
    "object_move_object_centric": "object_rotate_object_centric",
}

OBJECT_MOVE_TYPES = {
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
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
    "object_rotate_object_centric",
    "object_move_allocentric",
    "viewpoint_move",
    "object_remove",
    "attachment_chain",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
]


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
    if qtype == "attachment_chain":
        return True
    return qtype in OBJECT_MOVE_TYPES and bool(question.get("attachment_remapped", False))


def filter_viewer_questions(
    source_questions: list[dict],
    *,
    include_attachment_unchanged: bool = False,
) -> list[dict]:
    filtered = list(source_questions)
    if not include_attachment_unchanged:
        filtered = [
            question for question in filtered
            if not is_attachment_unchanged_object_move(question)
        ]
    filtered = _apply_attachment_viewer_filter(filtered)
    return _apply_global_object_move_ratio_filter(filtered)


def question_badges(question: dict) -> str:
    badges: list[str] = []
    if bool(question.get("attachment_remapped", False)):
        badges.append('<span class="badge extra attachment">attachment</span>')
    if bool(question.get("relation_unchanged", False)):
        badges.append('<span class="badge extra unchanged">unchanged</span>')
    if bool(question.get("manual_review_reason")):
        badges.append('<span class="badge extra review">manual-review</span>')
    return "".join(badges)


def question_review_notes(question: dict) -> str:
    parts: list[str] = []

    manual_review_reason = str(question.get("manual_review_reason", "")).strip()
    if manual_review_reason:
        parts.append(
            '<div class="review-block">'
            '<div class="review-title">Manual Review</div>'
            f'<div class="review-line">{html.escape(manual_review_reason)}</div>'
            '</div>'
        )

    review = question.get("question_presence_review")
    if isinstance(review, dict):
        lines: list[str] = []
        decision = str(review.get("decision", "")).strip()
        if decision:
            lines.append(f"decision: {decision}")

        flagged_labels = review.get("flagged_labels", [])
        if isinstance(flagged_labels, list) and flagged_labels:
            labels_text = ", ".join(str(label) for label in flagged_labels if str(label).strip())
            if labels_text:
                lines.append(f"flagged labels: {labels_text}")

        object_reviews = review.get("object_reviews", [])
        if isinstance(object_reviews, list):
            for obj_review in object_reviews:
                if not isinstance(obj_review, dict):
                    continue
                label = str(obj_review.get("label", "")).strip()
                status = str(obj_review.get("status", "")).strip()
                reason = str(obj_review.get("reason", "")).strip()
                if not label and not status:
                    continue
                line = f"{label}: {status}" if label else status
                if reason:
                    line = f"{line} ({reason})"
                lines.append(line)

        if lines:
            review_lines = "".join(
                f'<div class="review-line">{html.escape(line)}</div>'
                for line in lines
            )
            parts.append(
                '<div class="review-block">'
                '<div class="review-title">VLM Review</div>'
                f"{review_lines}"
                "</div>"
            )

    if not parts:
        return ""
    return f'<div class="review-notes">{"".join(parts)}</div>'


PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f0f2f5;margin:0;padding:20px}}
h1{{text-align:center;color:#333;margin-bottom:4px}}
.stats{{text-align:center;color:#666;font-size:14px;margin-bottom:24px}}
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
.img-wrap{{flex:0 0 auto;width:480px;background:#222;display:flex;
           align-items:center;justify-content:center}}
.img-wrap img{{width:480px;display:block}}
.no-img{{width:480px;height:200px;display:flex;align-items:center;
         justify-content:center;color:#999;font-size:13px}}
.body{{padding:18px 20px;flex:1;min-width:0}}
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
.footer{{margin-top:14px;font-size:11px;color:#aaa}}
.idx{{float:right;color:#ccc;font-size:12px}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stats">{n} questions &nbsp;&middot;&nbsp; {levels}</div>
<div class="summary">
  <h2>Task Summary</h2>
  <div class="summary-block">{task_summary}</div>
</div>
{cards}
</body>
</html>
"""

CARD = """\
<div class="card">
  <div class="img-wrap">{img}</div>
  <div class="body">
    <div class="meta">
      <span class="badge {level}">{level}</span>
      <span class="badge" style="background:#f3f4f6;color:#374151">{qtype}</span>
      {extra_badges}
      <span class="idx">#{idx}</span>
    </div>
    <p class="qtext">{question}</p>
    {options}
    {notes}
    <div class="footer">{scene_id} &nbsp;/&nbsp; {image_name}</div>
  </div>
</div>"""


def build_viewer_html(
    questions: list[dict],
    image_root: Path,
    *,
    max_width: int = 480,
    shuffle_seed: int = 42,
    title: str = "predictive spatial reasoning benchmark",
    requested_qtypes: set[str] | None = None,
    attachment_only: bool = False,
    include_attachment_unchanged: bool = False,
    apply_filters: bool = True,
) -> str:
    displayed_questions = list(questions)
    if apply_filters:
        source_questions = select_viewer_source_questions(
            displayed_questions,
            requested_qtypes=requested_qtypes,
            attachment_only=attachment_only,
        )
        displayed_questions = filter_viewer_questions(
            source_questions,
            include_attachment_unchanged=include_attachment_unchanged,
        )
    displayed_questions = order_questions_for_viewer(displayed_questions, seed=shuffle_seed)

    level_counter: Counter = Counter()
    cards: list[str] = []

    for idx, q in enumerate(displayed_questions, 1):
        level = q.get("level", "?")
        qtype = _canonical_qtype(str(q.get("type", "")).strip()) or "unknown"
        scene = str(q.get("scene_id", ""))
        frame = str(q.get("image_name", ""))
        answer = str(q.get("answer", ""))
        opts = q.get("options", [])

        level_counter[level] += 1

        img_path = image_root / scene / "color" / frame
        b64 = img_to_b64(img_path, max_width)
        img_html = (
            f'<img src="data:image/jpeg;base64,{b64}">'
            if b64
            else '<div class="no-img">image not found</div>'
        )

        opt_html = ""
        if isinstance(opts, list):
            for i, opt in enumerate(opts):
                letter = chr(65 + i)
                cls = "opt correct" if letter == answer else "opt"
                opt_html += (
                    f'<div class="{cls}">{letter}.&nbsp; {html.escape(str(opt))}</div>\n    '
                )

        cards.append(
            CARD.format(
                img=img_html,
                level=html.escape(str(level)),
                qtype=html.escape(QTYPE_DISPLAY.get(qtype, qtype)),
                extra_badges=question_badges(q),
                idx=idx,
                question=html.escape(str(q.get("question", ""))),
                options=opt_html,
                notes=question_review_notes(q),
                scene_id=html.escape(scene),
                image_name=html.escape(frame),
            )
        )

    levels_str = " &nbsp;&middot;&nbsp; ".join(
        f"{html.escape(str(k))}: {v}" for k, v in sorted(level_counter.items())
    )
    task_summary = build_task_summary_v2(displayed_questions)
    return PAGE.format(
        title=html.escape(title),
        n=len(displayed_questions),
        levels=levels_str,
        task_summary=task_summary,
        cards="\n".join(cards),
    )


def main():
    parser = argparse.ArgumentParser(description="Build HTML QA viewer")
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON (e.g. human_validation_sample.json)",
    )
    parser.add_argument(
        "--image_root",
        required=True,
        help="Root of ScanNet scans (parent of scene dirs)",
    )
    parser.add_argument("--output", default="viewer.html")
    parser.add_argument(
        "--qtypes",
        default="",
        help="Comma-separated question types to keep",
    )
    parser.add_argument(
        "--attachment_only",
        action="store_true",
        help="Keep only attachment-related questions: attachment_chain and attached object_move_* items",
    )
    parser.add_argument(
        "--include_attachment_unchanged",
        action="store_true",
        help="Include attachment-mediated object_move_* questions whose answer is unchanged after the move",
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
    args = parser.parse_args()

    if Image is None:
        sys.exit("Pillow is required: pip install Pillow")

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
    image_root = Path(args.image_root)
    html_text = build_viewer_html(
        questions,
        image_root,
        max_width=args.max_width,
        shuffle_seed=args.shuffle_seed,
        requested_qtypes=requested_qtypes,
        attachment_only=args.attachment_only,
        include_attachment_unchanged=args.include_attachment_unchanged,
        apply_filters=True,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"Saved: {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
