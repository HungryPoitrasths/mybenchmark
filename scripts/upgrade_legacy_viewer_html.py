#!/usr/bin/env python3
"""Upgrade legacy self-contained QA viewer HTML files.

This script preserves the original page title, rebuilds page stats and task
summary from the displayed cards, upgrades legacy cards to the richer card
layout, restores scene/frame footers from local source artifacts, and
optionally injects review blocks for manual-review pages.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_viewer import (
    QTYPE_DISPLAY,
    QTYPE_LEVEL,
    _canonical_qtype,
    build_task_summary_v2,
)
from scripts.review_viewer_html import _iter_div_ranges

QTYPE_DISPLAY_TO_RAW = {label: raw for raw, label in QTYPE_DISPLAY.items()}

CARD_START = '<div class="card">'
SUMMARY_START = '<div class="summary">'
REVIEW_NOTES_START = '<div class="review-notes">'
REVIEW_BLOCK_START = '<div class="review-block">'

TITLE_RE = re.compile(r"<title>(?P<text>.*?)</title>", flags=re.IGNORECASE | re.DOTALL)
H1_RE = re.compile(r"<h1>(?P<text>.*?)</h1>", flags=re.IGNORECASE | re.DOTALL)
STATS_RE = re.compile(r'<div class="stats">(?P<html>.*?)</div>', flags=re.IGNORECASE | re.DOTALL)
QTEXT_RE = re.compile(r'<p class="qtext">(?P<text>.*?)</p>', flags=re.IGNORECASE | re.DOTALL)
OPTION_RE = re.compile(
    r'<div class="opt(?P<correct>\s+correct)?">(?P<letter>[A-D])\.&nbsp;\s*(?P<text>.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
OPTION_ALT_RE = re.compile(
    r"<div class='opt'><span class='letter'>(?P<letter>[A-D])</span><span class='text'>(?P<text>.*?)</span></div>",
    flags=re.IGNORECASE | re.DOTALL,
)
DATA_IMAGE_RE = re.compile(
    r"<img src=[\"']data:(?P<mime>[^;]+);base64,(?P<b64>[^\"']+)[\"']",
    flags=re.IGNORECASE,
)
FILE_IMAGE_RE = re.compile(r"<img src=['\"](?P<src>[^'\"]+)['\"]", flags=re.IGNORECASE)
FOOTER_RE = re.compile(
    r'<div class="footer">\s*(?P<scene_id>[^<]+?)\s*&nbsp;/&nbsp;\s*(?P<image_name>[^<]+?)\s*</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
META_FOOTER_RE = re.compile(
    r"(?:#\d+\s*&nbsp;\s*)?(?P<scene_id>scene[^<\s]+)\s*/\s*(?P<image_name>[^<\s]+)",
    flags=re.IGNORECASE,
)
META_BADGE_RE = re.compile(
    r"<span class=\"badge(?: [^\"]*)?\"[^>]*>(?P<text>.*?)</span>",
    flags=re.IGNORECASE | re.DOTALL,
)
MANUAL_BADGE_RE = re.compile(r"<div class='badge'>(?P<text>.*?)</div>", flags=re.IGNORECASE | re.DOTALL)
MANUAL_NOTE_RE = re.compile(
    r"<div class='note'><strong>Manual note:</strong>\s*(?P<text>.*?)</div>",
    flags=re.IGNORECASE | re.DOTALL,
)
SUMMARY_NOTE_RE = re.compile(
    r'<div class="summary-note">.*?</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
REVIEW_TITLE_RE = re.compile(
    r'<div class="review-title">\s*(?P<text>.*?)\s*</div>',
    flags=re.IGNORECASE | re.DOTALL,
)

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
.summary-note{{margin-top:12px;padding:10px 12px;border-radius:8px;background:#fff7ed;
              border:1px solid #fed7aa;color:#9a3412;font-size:13px}}
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
.review{{background:#fee2e2;color:#991b1b}}
.manual{{background:#fff7ed;color:#9a3412}}
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
<h1>{heading}</h1>
{stats}
{summary}
{cards}
</body>
</html>
"""


def clean_html_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


def html_escape(value: str) -> str:
    return html.escape(value, quote=True)


def compute_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def strip_occlusion_definition(question: str) -> str:
    return re.sub(
        r" Here, 'occluded' means blocked by another object; being partly outside the image frame does not count as occlusion\.$",
        "",
        question,
    )


def normalize_object_name(value: str) -> str:
    return " ".join(value.lower().split())


def canonical_question(question: str) -> str:
    question = strip_occlusion_definition(clean_html_text(question))

    patterns = [
        (
            "direction_agent",
            [
                r"^From the camera's viewpoint, the (.+?) is in which direction relative to the (.+?)\?$",
                r"^From the current camera perspective, what is the spatial relationship of the (.+?) to the (.+?)\?$",
                r"^Looking at the scene from the camera's perspective, where is the (.+?) positioned relative to the (.+?)\?$",
            ],
            lambda m: f"direction_agent|{normalize_object_name(m.group(1))}|{normalize_object_name(m.group(2))}",
        ),
        (
            "occlusion",
            [
                r"^What is the occlusion status of the (.+?) in the current view\?$",
                r"^In the current image, is the (.+?) unoccluded, occluded by another object, or not visible\?$",
                r"^From the current viewpoint, which best describes the (.+?): not occluded, occluded, or not visible\?$",
            ],
            lambda m: f"occlusion|{normalize_object_name(m.group(1))}",
        ),
        (
            "distance",
            [
                r"^(?:Estimate )?the approximate shortest distance between the (.+?) and the (.+?), measured from their closest points\.$",
                r"^What is the approximate shortest distance between the (.+?) and the (.+?), measured from their closest points\?$",
                r"^Measured from the closest points of each object, what is the approximate shortest distance between the (.+?) and the (.+?)\?$",
                r"^Approximately how far apart are the (.+?) and the (.+?)\?$",
            ],
            lambda m: "distance|" + "|".join(
                sorted(
                    [normalize_object_name(m.group(1)), normalize_object_name(m.group(2))]
                )
            ),
        ),
    ]
    for _, regexes, builder in patterns:
        for regex in regexes:
            match = re.match(regex, question)
            if match:
                return builder(match)

    match = re.match(
        r"^Imagine you are the (.+?) and facing toward the (.+?)\. From your perspective, in which direction is the (.+?)\?$",
        question,
    )
    if match:
        return (
            "direction_object_centric|"
            f"{normalize_object_name(match.group(1))}|"
            f"{normalize_object_name(match.group(2))}|"
            f"{normalize_object_name(match.group(3))}"
        )

    match = re.match(
        r"^If you were the (.+?), looking toward the (.+?), where would the (.+?) be\?$",
        question,
    )
    if match:
        return (
            "direction_object_centric|"
            f"{normalize_object_name(match.group(1))}|"
            f"{normalize_object_name(match.group(2))}|"
            f"{normalize_object_name(match.group(3))}"
        )

    if question.startswith("Imagine all furniture is rotated"):
        if "On the floor plan" in question or "Viewed from above on the room's layout" in question:
            match = re.search(
                r"(?:On the floor plan|Viewed from above on the room's layout),? .*?(?:the )?(.+?) (?:is in which cardinal direction relative to|from the) (?:the )?(.+?)\?$",
                question,
            )
            if match:
                return (
                    "coordinate_rotation_allocentric|"
                    f"{normalize_object_name(match.group(1))}|"
                    f"{normalize_object_name(match.group(2))}"
                )
        if "From your perspective" in question or question.startswith("If you were"):
            return "coordinate_rotation_object_centric|" + question.lower()
        return "coordinate_rotation_agent|" + question.lower()

    match = re.search(
        r"(?:On the room's floor plan|Viewed from above on the room's layout),? .*?(?:the )?(.+?) (?:is in which cardinal direction relative to|from the) (?:the )?(.+?)\?$",
        question,
    )
    if match:
        return (
            "direction_allocentric|"
            f"{normalize_object_name(match.group(1))}|"
            f"{normalize_object_name(match.group(2))}"
        )

    lowered = question.lower()
    if "if the camera" in lowered or "after the camera" in lowered:
        return "viewpoint_move|" + lowered
    if " if we move " in " " + lowered or " imagine moving " in " " + lowered or lowered.startswith("from the camera's perspective, if we move"):
        if "occlusion status" in lowered or "unoccluded" in lowered or "not occluded" in lowered:
            return "object_move_occlusion|" + lowered
        if "distance" in lowered or "how far" in lowered:
            return "object_move_distance|" + lowered
        if "cardinal direction" in lowered or "floor plan" in lowered or "layout" in lowered:
            return "object_move_allocentric|" + lowered
        return "object_move_agent|" + lowered
    if lowered.startswith("if the ") and " removed" in lowered:
        return "object_remove|" + lowered
    return lowered


def infer_qtype(question: str, options: list[str]) -> str:
    question_no_occ = strip_occlusion_definition(clean_html_text(question))
    lowered = question_no_occ.lower()
    option_blob = " | ".join(opt.lower() for opt in options)
    if lowered.startswith("imagine all furniture is rotated"):
        if "from your perspective" in lowered or lowered.startswith("if you were"):
            return "coordinate_rotation_object_centric"
        if "cardinal direction" in lowered or "floor plan" in lowered or "layout" in lowered:
            return "coordinate_rotation_allocentric"
        return "coordinate_rotation_agent"
    if lowered.startswith("if the ") and " removed" in lowered:
        return "object_remove"
    if "if the camera" in lowered or "after the camera" in lowered:
        return "viewpoint_move"
    if " if we move " in " " + lowered or " imagine moving " in " " + lowered or lowered.startswith("from the camera's perspective, if we move"):
        if "occlusion status" in lowered or "unoccluded" in lowered or "not occluded" in lowered:
            return "object_move_occlusion"
        if "distance" in lowered or "how far" in lowered:
            return "object_move_distance"
        if "cardinal direction" in lowered or "floor plan" in lowered or "layout" in lowered:
            return "object_move_allocentric"
        return "object_move_agent"
    if "occlusion status" in lowered or "unoccluded" in lowered or "not occluded" in lowered:
        return "occlusion"
    if "distance" in lowered or "how far" in lowered:
        return "distance"
    if "cardinal direction" in lowered or "floor plan" in lowered or "layout" in lowered:
        return "direction_allocentric"
    if "from your perspective" in lowered or lowered.startswith("if you were") or lowered.startswith("imagine you are"):
        return "direction_object_centric"
    if "rotate" in lowered and "attachment" in lowered:
        return "attachment_chain"
    if any(token in option_blob for token in ("north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest")):
        return "direction_allocentric"
    return "direction_agent"


def extract_first_div(text: str, start_marker: str) -> str:
    ranges = _iter_div_ranges(text, start_marker)
    if not ranges:
        return ""
    start, end = ranges[0]
    return text[start:end]


def extract_review_notes(card_html: str) -> str:
    ranges = _iter_div_ranges(card_html, REVIEW_NOTES_START)
    if not ranges:
        return ""
    start, end = ranges[0]
    return card_html[start:end]


def parse_image_hash_from_card(card_html: str, base_dir: Path) -> tuple[str | None, str]:
    data_match = DATA_IMAGE_RE.search(card_html)
    if data_match is not None:
        image_bytes = base64.b64decode(data_match.group("b64"))
        image_html = f'<img src="data:{data_match.group("mime")};base64,{data_match.group("b64")}">'
        return compute_image_hash(image_bytes), image_html
    file_match = FILE_IMAGE_RE.search(card_html)
    if file_match is not None:
        src = file_match.group("src")
        image_path = (base_dir / src).resolve()
        if image_path.exists():
            image_hash = compute_image_hash(image_path.read_bytes())
        else:
            image_hash = None
        image_html = f'<img src="{html_escape(src)}">'
        return image_hash, image_html
    return None, '<div class="no-img">image not found</div>'


def parse_options(card_html: str) -> tuple[list[str], str | None]:
    options: list[str] = []
    answer: str | None = None
    matches = list(OPTION_RE.finditer(card_html))
    if not matches:
        matches = list(OPTION_ALT_RE.finditer(card_html))
    for match in matches:
        options.append(clean_html_text(match.group("text")))
        if match.re is OPTION_RE and match.group("correct"):
            answer = match.group("letter").upper()
    return options, answer


def parse_legacy_page(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    title_match = TITLE_RE.search(text)
    heading_match = H1_RE.search(text)
    stats_match = STATS_RE.search(text)
    summary_html = extract_first_div(text, SUMMARY_START)
    cards: list[dict] = []

    for start, end in _iter_div_ranges(text, CARD_START):
        card_html = text[start:end]
        image_hash, image_html = parse_image_hash_from_card(card_html, path.parent)
        question_match = QTEXT_RE.search(card_html)
        options, answer = parse_options(card_html)
        if question_match is None:
            continue
        question = clean_html_text(question_match.group("text"))
        cards.append(
            {
                "image_hash": image_hash,
                "image_html": image_html,
                "question": question,
                "canonical_question": canonical_question(question),
                "options": options,
                "answer": answer,
            }
        )

    return {
        "title": clean_html_text(title_match.group("text")) if title_match else path.stem,
        "heading": clean_html_text(heading_match.group("text")) if heading_match else path.stem,
        "stats_html": stats_match.group(0) if stats_match else "",
        "summary_html": summary_html,
        "cards": cards,
    }


def parse_rich_html_source(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    footer_by_hash: dict[str, tuple[str, str]] = {}
    badge_by_exact: dict[tuple[str, str], list[str]] = {}
    badge_by_canon: dict[tuple[str, str], list[str]] = {}
    review_by_exact: dict[tuple[str, str], str] = {}
    review_by_canon: dict[tuple[str, str], str] = {}

    for start, end in _iter_div_ranges(text, CARD_START):
        card_html = text[start:end]
        image_hash, _ = parse_image_hash_from_card(card_html, path.parent)
        question_match = QTEXT_RE.search(card_html)
        if image_hash is None or question_match is None:
            continue

        question = clean_html_text(question_match.group("text"))
        exact_key = (image_hash, question)
        canon_key = (image_hash, canonical_question(question))

        footer_match = FOOTER_RE.search(card_html)
        meta_html = extract_first_div(card_html, '<div class="meta">')
        scene = ""
        frame = ""
        if footer_match is not None:
            scene = clean_html_text(footer_match.group("scene_id"))
            frame = clean_html_text(footer_match.group("image_name"))
        elif meta_html:
            meta_match = META_FOOTER_RE.search(meta_html)
            if meta_match is not None:
                scene = clean_html_text(meta_match.group("scene_id"))
                frame = clean_html_text(meta_match.group("image_name"))
        if scene and frame:
            footer_by_hash.setdefault(image_hash, (scene, frame))

        badges = [clean_html_text(match.group("text")) for match in META_BADGE_RE.finditer(meta_html)]
        if badges:
            badge_by_exact.setdefault(exact_key, badges)
            badge_by_canon.setdefault(canon_key, badges)

        review_html = extract_review_notes(card_html)
        if review_html:
            if exact_key not in review_by_exact or len(review_html) > len(review_by_exact[exact_key]):
                review_by_exact[exact_key] = review_html
            if canon_key not in review_by_canon or len(review_html) > len(review_by_canon[canon_key]):
                review_by_canon[canon_key] = review_html

    return {
        "footer_by_hash": footer_by_hash,
        "badge_by_exact": badge_by_exact,
        "badge_by_canon": badge_by_canon,
        "review_by_exact": review_by_exact,
        "review_by_canon": review_by_canon,
    }


def parse_manifest_source(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    footer_by_hash: dict[str, tuple[str, str]] = {}
    badge_by_exact: dict[tuple[str, str], list[str]] = {}
    badge_by_canon: dict[tuple[str, str], list[str]] = {}
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return {
            "footer_by_hash": footer_by_hash,
            "badge_by_exact": badge_by_exact,
            "badge_by_canon": badge_by_canon,
        }

    for item in questions:
        if not isinstance(item, dict):
            continue
        image_path = Path(str(item.get("image_path", "")))
        if not image_path.exists():
            continue
        image_hash = compute_image_hash(image_path.read_bytes())
        question = clean_html_text(str(item.get("question", "")))
        exact_key = (image_hash, question)
        canon_key = (image_hash, canonical_question(question))
        scene = clean_html_text(str(item.get("scene_id", "")))
        frame = clean_html_text(str(item.get("image_name", "")))
        if scene and frame:
            footer_by_hash.setdefault(image_hash, (scene, frame))
        badges = [
            clean_html_text(str(badge))
            for badge in item.get("badges", [])
            if clean_html_text(str(badge))
        ]
        if badges:
            badge_by_exact.setdefault(exact_key, badges)
            badge_by_canon.setdefault(canon_key, badges)

    return {
        "footer_by_hash": footer_by_hash,
        "badge_by_exact": badge_by_exact,
        "badge_by_canon": badge_by_canon,
    }


def render_review_block(title: str, lines: list[str]) -> str:
    body = "".join(
        f'<div class="review-line">{html_escape(line)}</div>'
        for line in lines
        if line.strip()
    )
    if not body:
        return ""
    return (
        '<div class="review-block">'
        f'<div class="review-title">{html_escape(title)}</div>'
        f"{body}"
        "</div>"
    )


def parse_manual_review_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    review_by_exact: dict[tuple[str, str], str] = {}
    review_by_canon: dict[tuple[str, str], str] = {}
    footer_by_hash: dict[str, tuple[str, str]] = {}
    badge_by_exact: dict[tuple[str, str], list[str]] = {}
    badge_by_canon: dict[tuple[str, str], list[str]] = {}

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return {
            "footer_by_hash": footer_by_hash,
            "badge_by_exact": badge_by_exact,
            "badge_by_canon": badge_by_canon,
            "review_by_exact": review_by_exact,
            "review_by_canon": review_by_canon,
        }

    for item in questions:
        if not isinstance(item, dict):
            continue
        image_path = Path(str(item.get("image_path", "")))
        if not image_path.exists():
            continue
        image_hash = compute_image_hash(image_path.read_bytes())
        question = clean_html_text(str(item.get("question", "")))
        exact_key = (image_hash, question)
        canon_key = (image_hash, canonical_question(question))
        scene = clean_html_text(str(item.get("scene_id", "")))
        frame = clean_html_text(str(item.get("image_name", "")))
        if scene and frame:
            footer_by_hash.setdefault(image_hash, (scene, frame))
        badges = ["manual-review"]
        badge_by_exact.setdefault(exact_key, badges)
        badge_by_canon.setdefault(canon_key, badges)

        issue_type = clean_html_text(str(item.get("issue_type", "")))
        manual_reason = clean_html_text(str(item.get("manual_reason", "")))
        review_html = '<div class="review-notes">'
        review_html += render_review_block(
            "Manual Review",
            [
                f"issue type: {issue_type or '-'}",
                f"reason: {manual_reason or '-'}",
            ],
        )
        review_html += "</div>"
        review_by_exact.setdefault(exact_key, review_html)
        review_by_canon.setdefault(canon_key, review_html)

    return {
        "footer_by_hash": footer_by_hash,
        "badge_by_exact": badge_by_exact,
        "badge_by_canon": badge_by_canon,
        "review_by_exact": review_by_exact,
        "review_by_canon": review_by_canon,
    }


def merge_index(target: dict, source: dict) -> None:
    for key, value in source.get("footer_by_hash", {}).items():
        target["footer_by_hash"].setdefault(key, value)
    for index_name in ("badge_by_exact", "badge_by_canon", "review_by_exact", "review_by_canon"):
        for key, value in source.get(index_name, {}).items():
            if key not in target[index_name] or len(str(value)) > len(str(target[index_name][key])):
                target[index_name][key] = value


def build_source_index(project_root: Path) -> dict:
    index = {
        "footer_by_hash": {},
        "badge_by_exact": {},
        "badge_by_canon": {},
        "review_by_exact": {},
        "review_by_canon": {},
    }

    manifest_paths = [
        project_root / "output/html_extract_meshray1010_2/manifest.json",
        project_root / "output/html_extract_meshray1010_3/manifest.json",
        project_root / "output/html_extract_question_presence_review_flagged1010_2/manifest.json",
    ]
    for path in manifest_paths:
        if path.exists():
            merge_index(index, parse_manifest_source(path))

    rich_html_paths = [
        project_root / "meshray1010(2).html",
        project_root / "meshray1010(3).html",
        project_root / "meshray1010(2)_with_crops.html",
        project_root / "meshray1010.html",
        project_root / "meshray2.html",
        project_root / "meshray1.html",
        project_root / "depthviewer1.html",
        project_root / "question_presence_review_flagged.html",
        project_root / "question_presence_review_flagged2.html",
        project_root / "question_presence_review_flagged1010.html",
        project_root / "question_presence_review_flagged1010(2).html",
        project_root / "output/html_extract_meshray1010_3/manual_review_flagged.html",
        project_root / "output/html_extract_meshray1010_3/manual_compare_flagged.html",
    ]
    for path in rich_html_paths:
        if path.exists():
            merge_index(index, parse_rich_html_source(path))

    manual_review_json = project_root / "output/html_extract_meshray1010_3/manual_review_flagged.json"
    if manual_review_json.exists():
        merge_index(index, parse_manual_review_json(manual_review_json))
    return index


def strip_referability_audit(review_html: str) -> str:
    if "Referability Audit" not in review_html:
        return review_html
    kept_blocks: list[str] = []
    for start, end in _iter_div_ranges(review_html, REVIEW_BLOCK_START):
        block = review_html[start:end]
        title_match = REVIEW_TITLE_RE.search(block)
        title = clean_html_text(title_match.group("text")) if title_match else ""
        if title.lower() == "referability audit":
            continue
        kept_blocks.append(block)
    if not kept_blocks:
        return ""
    return '<div class="review-notes">' + "".join(kept_blocks) + "</div>"


def build_footer_html(scene: str | None, frame: str | None) -> str:
    if not scene or not frame:
        scene = scene or "scene unavailable"
        frame = frame or "frame unavailable"
    return f'<div class="footer">{html_escape(scene)} &nbsp;/&nbsp; {html_escape(frame)}</div>'


def resolve_source_badges(card: dict, source_index: dict) -> list[str]:
    exact_key = (card["image_hash"], card["question"])
    canon_key = (card["image_hash"], card["canonical_question"])
    return source_index["badge_by_exact"].get(exact_key) or source_index["badge_by_canon"].get(canon_key) or []


def resolve_qtype_raw(card: dict, source_index: dict) -> str:
    source_badges = resolve_source_badges(card, source_index)

    qtype_raw = ""
    for badge in source_badges:
        if badge in {"L1", "L2", "L3", "manual-review", "with-attachment", "without-attachment"}:
            continue
        if badge in QTYPE_DISPLAY:
            qtype_raw = _canonical_qtype(badge)
            break
        if badge in QTYPE_DISPLAY_TO_RAW:
            qtype_raw = _canonical_qtype(QTYPE_DISPLAY_TO_RAW[badge])
            break
    if not qtype_raw:
        qtype_raw = _canonical_qtype(infer_qtype(card["question"], card["options"]))
    return qtype_raw


def build_stats_html(cards: list[dict], source_index: dict) -> str:
    counts = {"L1": 0, "L2": 0, "L3": 0}
    for card in cards:
        qtype_raw = resolve_qtype_raw(card, source_index)
        level = QTYPE_LEVEL.get(qtype_raw)
        if level in counts:
            counts[level] += 1
    return (
        '<div class="stats">'
        f'{len(cards)} questions &nbsp;&middot;&nbsp; '
        f'L1: {counts["L1"]} &nbsp;&middot;&nbsp; '
        f'L2: {counts["L2"]} &nbsp;&middot;&nbsp; '
        f'L3: {counts["L3"]}'
        "</div>"
    )


def build_summary_html(cards: list[dict], source_index: dict) -> str:
    displayed_questions = []
    for card in cards:
        source_badges = resolve_source_badges(card, source_index)
        displayed_questions.append(
            {
                "type": resolve_qtype_raw(card, source_index),
                "attachment_remapped": "with-attachment" in source_badges,
            }
        )
    return (
        '<div class="summary">\n'
        '  <h2>Task Summary</h2>\n'
        f'  <div class="summary-block">{build_task_summary_v2(displayed_questions)}</div>\n'
        "</div>"
    )


def build_meta_html(
    card: dict,
    idx: int,
    page_mode: str,
    source_index: dict,
) -> str:
    qtype_raw = resolve_qtype_raw(card, source_index)
    qtype = QTYPE_DISPLAY.get(qtype_raw, qtype_raw or "unknown")
    level = QTYPE_LEVEL.get(qtype_raw, "")

    badges: list[str] = []
    if level:
        badges.append(f'<span class="badge {html_escape(level)}">{html_escape(level)}</span>')
    badges.append(f'<span class="badge extra">{html_escape(qtype)}</span>')
    if page_mode == "review":
        badges.append('<span class="badge review">manual-review</span>')
    badges.append(f'<span class="idx">#{idx}</span>')
    return '<div class="meta">' + "".join(badges) + "</div>"


def build_generic_review_html() -> str:
    return (
        '<div class="review-notes">'
        + render_review_block(
            "VLM Review",
            [
                "decision: manual_review",
                "details: source review fields were not embedded in this legacy export",
            ],
        )
        + "</div>"
    )


def build_review_html(card: dict, source_index: dict, *, omit_referability: bool) -> str:
    exact_key = (card["image_hash"], card["question"])
    canon_key = (card["image_hash"], card["canonical_question"])
    review_html = source_index["review_by_exact"].get(exact_key) or source_index["review_by_canon"].get(canon_key) or ""
    if omit_referability:
        review_html = strip_referability_audit(review_html)
    return review_html or build_generic_review_html()


def render_options(options: list[str], answer: str | None) -> str:
    rendered: list[str] = []
    for idx, option in enumerate(options):
        letter = chr(65 + idx)
        cls = "opt correct" if answer == letter else "opt"
        rendered.append(
            f'<div class="{cls}">{letter}.&nbsp; {html_escape(option)}</div>'
        )
    return "\n    ".join(rendered)


def render_card(card: dict, idx: int, page_mode: str, source_index: dict, *, omit_referability: bool) -> str:
    footer = source_index["footer_by_hash"].get(card["image_hash"])
    scene = footer[0] if footer else None
    frame = footer[1] if footer else None
    review_html = ""
    if page_mode == "review":
        review_html = build_review_html(card, source_index, omit_referability=omit_referability)
    return (
        '<div class="card">\n'
        f'  <div class="img-wrap">{card["image_html"]}</div>\n'
        '  <div class="body">\n'
        f'    {build_meta_html(card, idx, page_mode, source_index)}\n'
        f'    <p class="qtext">{html_escape(card["question"])}</p>\n'
        f'    {render_options(card["options"], card["answer"])}\n'
        f'    {review_html}\n'
        f'    {build_footer_html(scene, frame)}\n'
        '  </div>\n'
        '</div>'
    )


def upgrade_html(target: Path, output: Path, page_mode: str, *, omit_referability: bool) -> None:
    source_index = build_source_index(target.parent)
    page = parse_legacy_page(target)
    stats_html = build_stats_html(page["cards"], source_index)
    summary_html = build_summary_html(page["cards"], source_index)
    cards_html = "\n".join(
        render_card(card, idx, page_mode, source_index, omit_referability=omit_referability)
        for idx, card in enumerate(page["cards"], start=1)
    )
    html_text = PAGE.format(
        title=html_escape(page["title"]),
        heading=html_escape(page["heading"]),
        stats=stats_html,
        summary=summary_html,
        cards=cards_html,
    )
    output.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade a legacy viewer HTML page to the richer card layout")
    parser.add_argument("--target", type=Path, required=True, help="Legacy HTML to upgrade in-place or to --output")
    parser.add_argument("--output", type=Path, default=None, help="Optional output HTML path")
    parser.add_argument(
        "--page_mode",
        choices=("benchmark", "review"),
        required=True,
        help="Whether the target page is a benchmark page or a manual-review page",
    )
    parser.add_argument(
        "--omit_referability",
        action="store_true",
        help="Remove Referability Audit blocks from recovered review HTML",
    )
    args = parser.parse_args()

    target = args.target.resolve()
    output = (args.output or target).resolve()
    upgrade_html(target, output, args.page_mode, omit_referability=args.omit_referability)
    print(f"Upgraded {target} -> {output}")


if __name__ == "__main__":
    main()
