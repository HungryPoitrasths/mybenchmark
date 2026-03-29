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

    python scripts/make_viewer.py \
        --questions output/pilot_cascade/benchmark.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --output output/pilot_cascade/attachment_viewer.html \
        --attachment_only
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")


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
            ("object_move_object_centric", "L2_object_rotate_object_centric"),
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


OBJECT_MOVE_TYPES = {
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
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
    "object_move_allocentric",
    "viewpoint_move",
    "object_remove",
    "attachment_chain",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
]


def img_to_b64(path: Path, max_width: int = 480) -> str | None:
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


def build_task_summary_v2(questions: list[dict], type_counter: Counter) -> str:
    known_keys = {
        key
        for _, items in SUMMARY_GROUPS
        for key, _ in items
    }
    total_counter: Counter = Counter()
    attached_counter: Counter = Counter()
    for q in questions:
        qtype = str(q.get("type", "")).strip()
        if qtype not in OBJECT_MOVE_TYPES:
            continue
        total_counter[qtype] += 1
        if bool(q.get("attachment_remapped", False)):
            attached_counter[qtype] += 1

    parts: list[str] = []
    for section_title, items in SUMMARY_GROUPS:
        lines = [f'<div class="summary-line"><strong>{section_title}</strong></div>']
        object_move_keys = [key for key, _ in items if key in OBJECT_MOVE_TYPES]
        if object_move_keys:
            total = sum(total_counter.get(key, 0) for key in object_move_keys)
            attached = sum(attached_counter.get(key, 0) for key in object_move_keys)
            unattached = total - attached
            lines.append(
                f'<div class="summary-line">L2_object_move_all: total={total}, '
                f'with_attachment={attached}, without_attachment={unattached}</div>'
            )
        for key, label in items:
            if key in OBJECT_MOVE_TYPES:
                total = total_counter.get(key, 0)
                attached = attached_counter.get(key, 0)
                unattached = total - attached
                lines.append(
                    f'<div class="summary-line">{label}: total={total}, '
                    f'with_attachment={attached}, without_attachment={unattached}</div>'
                )
            else:
                lines.append(
                f'<div class="summary-line">{label}: {type_counter.get(key, 0)}</div>'
                )
        parts.append(f'<div class="summary-section">{"".join(lines)}</div>')

    other_items = [
        (qtype, count)
        for qtype, count in sorted(type_counter.items())
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
        qtype = str(q.get("type", "")).strip() or "unknown"
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


def is_attachment_viewer_question(question: dict) -> bool:
    qtype = str(question.get("type", "")).strip()
    if qtype == "attachment_chain":
        return True
    return qtype in OBJECT_MOVE_TYPES and bool(question.get("attachment_remapped", False))


def filter_viewer_questions(
    questions: list[dict],
    *,
    requested_qtypes: set[str] | None = None,
    attachment_only: bool = False,
) -> list[dict]:
    filtered = [
        q for q in questions
        if str(q.get("type", "")).strip() not in REMOVED_TYPES
    ]
    if attachment_only:
        return [q for q in filtered if is_attachment_viewer_question(q)]
    if requested_qtypes:
        return [
            q for q in filtered
            if str(q.get("type", "")).strip() in requested_qtypes
        ]
    return filtered


PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>predictive spatial reasoning benchmark</title>
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
.qtext{{font-size:15px;font-weight:600;color:#111;margin:0 0 14px}}
.opt{{padding:7px 12px;margin:4px 0;border-radius:6px;font-size:14px;
      background:#f8f9fa;border:1px solid #e5e7eb}}
.opt.correct{{background:#dcfce7;border-color:#86efac;font-weight:700}}
.footer{{margin-top:14px;font-size:11px;color:#aaa}}
.idx{{float:right;color:#ccc;font-size:12px}}
</style>
</head>
<body>
<h1>predictive spatial reasoning benchmark</h1>
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
      <span class="idx">#{idx}</span>
    </div>
    <p class="qtext">{question}</p>
    {options}
    <div class="footer">{scene_id} &nbsp;/&nbsp; {image_name}</div>
  </div>
</div>"""


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
    questions = filter_viewer_questions(
        questions,
        requested_qtypes=requested_qtypes,
        attachment_only=args.attachment_only,
    )
    questions = order_questions_for_viewer(questions, seed=args.shuffle_seed)

    image_root = Path(args.image_root)
    level_counter: Counter = Counter()
    type_counter: Counter = Counter()
    cards: list[str] = []

    for idx, q in enumerate(questions, 1):
        level = q.get("level", "?")
        qtype = q.get("type", "")
        scene = q.get("scene_id", "")
        frame = q.get("image_name", "")
        answer = q.get("answer", "")
        opts = q.get("options", [])

        level_counter[level] += 1
        type_counter[qtype or "unknown"] += 1

        img_path = image_root / scene / "color" / frame
        b64 = img_to_b64(img_path, args.max_width)
        img_html = (
            f'<img src="data:image/jpeg;base64,{b64}">'
            if b64
            else '<div class="no-img">image not found</div>'
        )

        opt_html = ""
        for i, opt in enumerate(opts):
            letter = chr(65 + i)
            cls = "opt correct" if letter == answer else "opt"
            opt_html += f'<div class="{cls}">{letter}.&nbsp; {opt}</div>\n    '

        cards.append(
            CARD.format(
                img=img_html,
                level=level,
                qtype=qtype,
                idx=idx,
                question=q.get("question", ""),
                options=opt_html,
                scene_id=scene,
                image_name=frame,
            )
        )

        if idx % 20 == 0:
            print(f"  {idx}/{len(questions)} processed...", flush=True)

    levels_str = " &nbsp;&middot;&nbsp; ".join(
        f"{k}: {v}" for k, v in sorted(level_counter.items())
    )
    task_summary = build_task_summary_v2(questions, type_counter)
    html = PAGE.format(
        n=len(questions),
        levels=levels_str,
        task_summary=task_summary,
        cards="\n".join(cards),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"Saved: {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
