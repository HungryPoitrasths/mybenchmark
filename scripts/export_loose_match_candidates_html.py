#!/usr/bin/env python3
"""Export ambiguous loose-match candidates for edited merged HTML cards."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from find_strong_duplicate_questions import _card_index, _iter_card_ranges, _simple_fields
from review_viewer_html import parse_viewer_html
from rewrite_subset_from_merged_html import (
    DEFAULT_FULL,
    DEFAULT_HTML,
    ROLE_FIELD_MAP,
    _build_source_indexes,
    _choose_loose_match,
    _html_exact_key,
    _level_from_badges,
    _load_payload,
    _loose_key_from_card,
    _qtype_from_badges,
)
from make_viewer import _resolve_image_path, img_to_b64


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "edited" / "loose_match_candidate_groups.html"
DEFAULT_REPORT = PROJECT_ROOT / "output" / "edited" / "loose_match_candidate_groups_report.json"

RELATION_LABELS = {
    "direction",
    "old_direction",
    "new_direction",
    "visibility",
    "old_visibility",
    "new_visibility",
    "distance_bin",
    "old_distance_bin",
    "new_distance_bin",
    "camera",
    "rotation_angle",
    "rotation_direction",
    "chain_depth",
    "displaced",
    "reference_frame",
    "query_role",
    "old_distance_m",
    "new_distance_m",
    "distance_m",
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(value)))


def _label_match_reason(source: Any, edited: Any) -> str | None:
    source_norm = _norm(source)
    edited_norm = _norm(edited)
    if not source_norm or not edited_norm:
        return None
    if source_norm == edited_norm:
        return "object_exact"
    if source_norm in edited_norm or edited_norm in source_norm:
        return "object_overlap"
    source_tokens = _tokens(source_norm)
    edited_tokens = _tokens(edited_norm)
    if source_tokens and edited_tokens and source_tokens & edited_tokens:
        return "object_overlap"
    return None


def _cards_simple_fields(html_text: str) -> dict[int, dict[str, str]]:
    fields: dict[int, dict[str, str]] = {}
    for start, end in _iter_card_ranges(html_text):
        card_html = html_text[start:end]
        fields[_card_index(card_html)] = _simple_fields(card_html)
    return fields


def _candidate_reasons(question: dict[str, Any], card_fields: dict[str, str]) -> list[str]:
    qtype = str(question.get("type", ""))
    mapping = ROLE_FIELD_MAP.get(qtype, {})
    reasons: list[str] = []
    for label, field_value in card_fields.items():
        for key in mapping.get(label, ()):
            source_value = question.get(key, "")
            if label in RELATION_LABELS:
                if str(source_value) == str(field_value):
                    reasons.append(f"relation_exact:{label}")
                continue
            reason = _label_match_reason(source_value, field_value)
            if reason:
                reasons.append(f"{reason}:{label}")
    return sorted(set(reasons))


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    reasons = item["reasons"]
    relation = sum(reason.startswith("relation_exact:") for reason in reasons)
    exact = sum(reason.startswith("object_exact:") for reason in reasons)
    overlap = sum(reason.startswith("object_overlap:") for reason in reasons)
    question = str(item["question"].get("question", ""))
    return relation, exact, overlap, question


def _format_options(options: list[Any]) -> str:
    if not options:
        return ""
    rows = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        rows.append(f"<div><b>{letter}.</b> {html.escape(str(option))}</div>")
    return "\n".join(rows)


def _answer_text(answer: Any, options: list[Any]) -> str:
    if isinstance(answer, list):
        return ", ".join(_answer_text(item, options) for item in answer)
    letter = str(answer).upper()
    if letter in {"A", "B", "C", "D"}:
        index = ord(letter) - ord("A")
        if 0 <= index < len(options):
            return f"{letter}: {options[index]}"
    return str(answer)


def _field_table(fields: dict[str, Any]) -> str:
    if not fields:
        return "<div class=\"muted\">No simple fields</div>"
    rows = []
    for key, value in fields.items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )
    return "<table class=\"fields\">" + "\n".join(rows) + "</table>"


def _question_field_table(question: dict[str, Any]) -> str:
    qtype = str(question.get("type", ""))
    mapping = ROLE_FIELD_MAP.get(qtype, {})
    rows = []
    for label, keys in mapping.items():
        values = []
        for key in keys:
            if key in question:
                values.append(f"{key}={question.get(key)}")
        if values:
            rows.append(
                "<tr>"
                f"<th>{html.escape(label)}</th>"
                f"<td>{html.escape('; '.join(str(v) for v in values))}</td>"
                "</tr>"
            )
    return "<table class=\"fields\">" + "\n".join(rows) + "</table>" if rows else "<div class=\"muted\">No mapped fields</div>"


def _image_html(
    card: dict[str, Any],
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
    max_image_width: int,
    image_stats: dict[str, Any] | None,
) -> str:
    if not scannet_roots and not scannetpp_roots:
        return ""
    question = {
        "scene_id": str(card.get("scene_id", "")),
        "image_name": str(card.get("image_name", "")),
        "_dataset": "scannet" if str(card.get("scene_id", "")).startswith("scene") else "scannetpp",
    }
    image_path = _resolve_image_path(question, scannet_roots, scannetpp_roots, scannetpp_sensor)
    b64 = img_to_b64(image_path, max_image_width)
    if image_stats is not None:
        image_stats["total"] = int(image_stats.get("total", 0)) + 1
        if b64:
            image_stats["embedded"] = int(image_stats.get("embedded", 0)) + 1
        else:
            image_stats["missing"] = int(image_stats.get("missing", 0)) + 1
            missing_paths = image_stats.setdefault("missing_paths", [])
            if isinstance(missing_paths, list) and len(missing_paths) < 20:
                missing_paths.append(str(image_path))
    if not b64:
        return f'<div class="no-img">image not found: {html.escape(str(image_path))}</div>'
    return f'<div class="image"><img src="data:image/jpeg;base64,{b64}" alt=""></div>'


def _render_html(
    groups: list[dict[str, Any]],
    stats: dict[str, Any],
    output_path: Path,
    scannet_roots: list[Path] | None = None,
    scannetpp_roots: list[Path] | None = None,
    scannetpp_sensor: str = "iphone",
    max_image_width: int = 520,
    image_stats: dict[str, Any] | None = None,
) -> None:
    scannet_roots = scannet_roots or []
    scannetpp_roots = scannetpp_roots or []
    parts = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        "<title>Loose Match Candidate Groups</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}",
        ".summary{background:white;border:1px solid #d7dde5;border-radius:8px;padding:14px;margin-bottom:18px}",
        ".group{background:white;border:1px solid #cfd7e3;border-radius:8px;margin:18px 0;padding:14px}",
        ".html-card{border-left:4px solid #2f6fed;background:#f8fbff;padding:12px;margin-bottom:12px}",
        ".candidate{border:1px solid #d8dee8;border-radius:6px;padding:12px;margin:10px 0;background:#fff}",
        ".meta{color:#536171;font-size:13px;margin:4px 0 8px}",
        ".q{font-size:15px;line-height:1.45;margin:8px 0}",
        ".opts{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:4px 16px;margin:8px 0}",
        ".answer{font-weight:700;color:#1b5e20;margin:6px 0}",
        ".image{margin:10px 0}.image img{max-width:100%;height:auto;border:1px solid #d8dee8;border-radius:6px}",
        ".no-img{margin:10px 0;padding:10px;border:1px dashed #c9d2df;color:#7a8797;background:#fafbfc}",
        ".reasons{margin:6px 0}",
        ".tag{display:inline-block;border-radius:4px;padding:2px 6px;margin:2px;background:#e9edf5;font-size:12px}",
        ".relation{background:#e5f4ea}.exact{background:#e7f0ff}.overlap{background:#fff1d6}",
        "table.fields{border-collapse:collapse;margin-top:8px;font-size:12px}",
        ".fields th,.fields td{border:1px solid #dbe1ea;padding:4px 6px;text-align:left;vertical-align:top}",
        ".fields th{background:#f0f3f7;color:#3a4654}",
        ".muted{color:#7a8797;font-size:13px}",
        "</style></head><body>",
        "<h1>Loose Match Candidate Groups</h1>",
        "<div class=\"summary\">",
        f"<div><b>Source HTML:</b> {html.escape(str(stats['html_path']))}</div>",
        f"<div><b>Benchmark:</b> {html.escape(str(stats['full_path']))}</div>",
        f"<div><b>Loose cards:</b> {stats['loose_count']}</div>",
        f"<div><b>Groups exported:</b> {len(groups)}</div>",
        f"<div><b>Candidate questions shown:</b> {stats['candidate_count']}</div>",
        f"<div><b>Images:</b> {int((image_stats or {}).get('embedded', 0))}/"
        f"{int((image_stats or {}).get('total', 0))} embedded</div>" if image_stats is not None else "",
        "</div>",
    ]

    for group_no, group in enumerate(groups, 1):
        card = group["card"]
        card_fields = group["fields"]
        options = [option["text"] for option in card.get("options", [])]
        parts.extend(
            [
                f"<section class=\"group\" id=\"q{card['viewer_index']}\">",
                f"<h2>Group {group_no}: HTML #{card['viewer_index']} ({len(group['candidates'])} candidates)</h2>",
                "<div class=\"html-card\">",
                f"<div class=\"meta\">{html.escape(str(card['scene_id']))} / {html.escape(str(card['image_name']))} | "
                f"{html.escape(_level_from_badges(card))}_{html.escape(_qtype_from_badges(card))}</div>",
                f"<div class=\"q\">{html.escape(str(card.get('question', '')))}</div>",
                _image_html(card, scannet_roots, scannetpp_roots, scannetpp_sensor, max_image_width, image_stats),
                f"<div class=\"opts\">{_format_options(options)}</div>",
                f"<div class=\"answer\">HTML answer: {html.escape(_answer_text(card.get('gold_answer'), options))}</div>",
                _field_table(card_fields),
                "</div>",
            ]
        )
        for candidate_no, item in enumerate(group["candidates"], 1):
            question = item["question"]
            q_options = [str(option) for option in question.get("options", [])]
            reason_tags = []
            for reason in item["reasons"]:
                cls = "relation" if reason.startswith("relation_exact:") else "exact" if reason.startswith("object_exact:") else "overlap"
                reason_tags.append(f"<span class=\"tag {cls}\">{html.escape(reason)}</span>")
            parts.extend(
                [
                    "<div class=\"candidate\">",
                    f"<h3>Candidate {candidate_no}</h3>",
                    f"<div class=\"meta\">{html.escape(str(question.get('scene_id', '')))} / "
                    f"{html.escape(str(question.get('image_name', '')))} | "
                    f"{html.escape(str(question.get('level', '')))}_{html.escape(str(question.get('type', '')))}</div>",
                    f"<div class=\"reasons\">{''.join(reason_tags)}</div>",
                    f"<div class=\"q\">{html.escape(str(question.get('question', '')))}</div>",
                    f"<div class=\"opts\">{_format_options(q_options)}</div>",
                    f"<div class=\"answer\">Benchmark answer: {html.escape(_answer_text(question.get('answer'), q_options))}</div>",
                    _question_field_table(question),
                    "</div>",
                ]
            )
        parts.append("</section>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def collect_candidate_groups(html_path: Path, full_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    cards = parse_viewer_html(html_text, include_deleted=False)
    simple_by_idx = _cards_simple_fields(html_text)
    _, full_questions = _load_payload(full_path)
    by_exact, by_loose = _build_source_indexes(full_questions)

    used_source_ids: set[int] = set()
    loose_cards: list[tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]] = []
    match_counts = Counter()

    for card in cards:
        idx = int(card["viewer_index"])
        fields = simple_by_idx.get(idx, {})
        exact_candidates = [q for q in by_exact.get(_html_exact_key(card), []) if id(q) not in used_source_ids]
        if exact_candidates:
            used_source_ids.add(id(exact_candidates[0]))
            match_counts["exact"] += 1
            continue
        matched = _choose_loose_match(
            by_loose.get(_loose_key_from_card(card), []),
            used_source_ids,
            fields,
            str(card["question"]),
        )
        if matched is None:
            match_counts["unmatched"] += 1
            continue
        used_source_ids.add(id(matched))
        match_counts["loose"] += 1
        loose_cards.append((card, fields, matched))

    groups: list[dict[str, Any]] = []
    for card, fields, matched in loose_cards:
        candidates = []
        for question in by_loose.get(_loose_key_from_card(card), []):
            reasons = _candidate_reasons(question, fields)
            if not reasons:
                continue
            candidates.append({"question": question, "reasons": reasons})
        if len(candidates) < 2:
            continue
        candidates.sort(key=_candidate_sort_key, reverse=True)
        groups.append(
            {
                "card": card,
                "fields": fields,
                "matched_question": matched,
                "candidates": candidates,
            }
        )

    stats = {
        "html_path": str(html_path),
        "full_path": str(full_path),
        "loose_count": len(loose_cards),
        "groups_exported": len(groups),
        "candidate_count": sum(len(group["candidates"]) for group in groups),
        "match_counts": dict(match_counts),
        "groups_by_type": dict(sorted(Counter(_qtype_from_badges(group["card"]) for group in groups).items())),
        "groups_by_level": dict(sorted(Counter(_level_from_badges(group["card"]) for group in groups).items())),
    }
    return groups, stats


def export_candidates(
    html_path: Path,
    full_path: Path,
    output_path: Path,
    report_path: Path,
    scannet_roots: list[Path] | None = None,
    scannetpp_roots: list[Path] | None = None,
    scannetpp_sensor: str = "iphone",
    max_image_width: int = 520,
) -> dict[str, Any]:
    groups, stats = collect_candidate_groups(html_path, full_path)
    stats["output_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_stats = {"total": 0, "embedded": 0, "missing": 0, "missing_paths": []}
    _render_html(
        groups,
        stats,
        output_path,
        scannet_roots=scannet_roots,
        scannetpp_roots=scannetpp_roots,
        scannetpp_sensor=scannetpp_sensor,
        max_image_width=max_image_width,
        image_stats=image_stats,
    )
    report = {
        **stats,
        "image_stats": image_stats,
        "groups": [
            {
                "viewer_index": int(group["card"]["viewer_index"]),
                "scene_id": str(group["card"]["scene_id"]),
                "image_name": str(group["card"]["image_name"]),
                "level": _level_from_badges(group["card"]),
                "type": _qtype_from_badges(group["card"]),
                "candidate_count": len(group["candidates"]),
                "matched_question": str(group["matched_question"].get("question", "")) if group["matched_question"] else None,
            }
            for group in groups
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scannet_data_root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp_data_root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--max-image-width", type=int, default=520)
    args = parser.parse_args()
    report = export_candidates(
        args.html,
        args.full,
        args.output,
        args.report,
        scannet_roots=args.scannet_data_root,
        scannetpp_roots=args.scannetpp_data_root,
        scannetpp_sensor=args.scannetpp_sensor,
        max_image_width=args.max_image_width,
    )
    print(f"Loose cards: {report['loose_count']}")
    print(f"Groups exported: {report['groups_exported']}")
    print(f"Candidate questions shown: {report['candidate_count']}")
    print(f"Images: {report['image_stats']['embedded']}/{report['image_stats']['total']} embedded")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
