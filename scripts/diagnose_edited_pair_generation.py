#!/usr/bin/env python3
"""Diagnose why human-edited attachment pairs did or did not produce questions.

This script is intentionally read-only.  It compares the attachment pair cards
from edited review HTML / referability cache with an existing pilot output
directory, then writes a pair-level JSON report.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


FRAME_CACHE_SIDECAR_DIR_NAME = ".run_vlm_referability_frame_cache"

OBJECT_ID_FIELDS = (
    "obj_a_id",
    "obj_b_id",
    "obj_c_id",
    "obj_ref_id",
    "obj_face_id",
    "obj_target_id",
    "target_obj_id",
    "query_obj_id",
    "moved_obj_id",
    "removed_obj_id",
    "parent_id",
    "child_id",
    "grandparent_id",
    "grandchild_id",
    "neighbor_id",
    "attachment_parent_id",
    "attachment_child_id",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_ids(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in value:
        obj_id = _as_int(item)
        if obj_id is None or obj_id in seen:
            continue
        seen.add(obj_id)
        out.append(obj_id)
    return out


def _normalize_pairs(value: Any) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if not isinstance(value, list):
        return pairs
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        parent_id = _as_int(item[0])
        child_id = _as_int(item[1])
        if parent_id is not None and child_id is not None:
            pairs.add((parent_id, child_id))
    return pairs


def _salvage_review_image_name_with_original_suffix(
    *,
    original_image_name: str,
    image_id: str,
) -> str:
    updated_stem = str(image_id).strip()
    if not updated_stem:
        return ""
    return f"{updated_stem}{Path(str(original_image_name).strip()).suffix}"


class _AttachmentPairSalvageHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict[str, Any]] = []
        self._current_card: dict[str, Any] | None = None
        self._card_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {str(key): ("" if value is None else str(value)) for key, value in attrs}
        classes = {
            item.strip()
            for item in str(attrs_map.get("class", "")).split()
            if item.strip()
        }
        if tag == "article" and "pair-card" in classes and self._current_card is None:
            self._current_card = {
                "scene_id": str(attrs_map.get("data-scene-id", "")).strip(),
                "image_name": str(attrs_map.get("data-image-name", "")).strip(),
                "group_id": str(attrs_map.get("data-group-id", "")).strip(),
                "pair_id": str(attrs_map.get("data-pair-id", "")).strip(),
                "parent_id": str(attrs_map.get("data-parent-id", "")).strip(),
                "parent_label": str(attrs_map.get("data-parent-label", "")).strip(),
                "child_id": str(attrs_map.get("data-child-id", "")).strip(),
                "child_label": str(attrs_map.get("data-child-label", "")).strip(),
                "deleted": str(attrs_map.get("data-deleted", "")).strip().lower() == "true",
                "parent_surface_text": "",
                "child_surface_text": "",
            }
            self._card_depth = 1
            return

        if self._current_card is None:
            return
        if tag == "article":
            self._card_depth += 1
            return
        if tag != "input":
            return

        name = str(attrs_map.get("name", "")).strip().lower()
        value = html.unescape(str(attrs_map.get("value", ""))).strip()
        if name == "image_id":
            self._current_card["image_id"] = value
            self._current_card["image_name"] = _salvage_review_image_name_with_original_suffix(
                original_image_name=str(self._current_card.get("image_name", "")),
                image_id=value,
            )
        elif name == "parent_surface_text":
            self._current_card["parent_surface_text"] = value
        elif name == "child_surface_text":
            self._current_card["child_surface_text"] = value

    def handle_endtag(self, tag: str) -> None:
        if tag != "article" or self._current_card is None:
            return
        self._card_depth -= 1
        if self._card_depth > 0:
            return
        self.cards.append(dict(self._current_card))
        self._current_card = None
        self._card_depth = 0


def _parse_review_html(path: Path) -> list[dict[str, Any]]:
    parser = _AttachmentPairSalvageHtmlParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    cards: list[dict[str, Any]] = []
    for card in parser.cards:
        parent_id = _as_int(card.get("parent_id"))
        child_id = _as_int(card.get("child_id"))
        if parent_id is None or child_id is None:
            continue
        card["parent_id"] = parent_id
        card["child_id"] = child_id
        card["source_html"] = str(path)
        cards.append(card)
    return cards


def _discover_edited_html_paths(cache_path: Path) -> tuple[list[Path], str, list[str]]:
    warnings: list[str] = []
    legacy_paths = sorted(cache_path.parent.glob("edited*.html"))
    if legacy_paths:
        if len(legacy_paths) > 1:
            warnings.append(
                "Multiple legacy edited*.html files exist; pipeline would reject this, "
                "but this diagnostic will read all of them."
            )
        return legacy_paths, "legacy", warnings

    scene_paths = sorted(cache_path.parent.glob(f"{cache_path.stem}_*_edited.html"))
    if scene_paths:
        return scene_paths, "scene-scoped", warnings
    return [], "none", warnings


def _discover_all_edited_html_paths(cache_path: Path) -> tuple[list[Path], str, list[str]]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("edited*.html", f"{cache_path.stem}_*_edited.html"):
        for path in sorted(cache_path.parent.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    warnings = [
        "include_all_edited_html is enabled; this may include HTML files that run_pipeline would not read."
    ] if paths else []
    return paths, "all", warnings


def _iter_frame_entries(cache_doc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    frames = cache_doc.get("frames", cache_doc)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(frames, dict):
        return out
    for key, value in frames.items():
        if isinstance(key, str) and "/" in key and isinstance(value, dict):
            scene_id, image_name = key.split("/", 1)
            out[(scene_id, image_name)] = value
            continue
        if not isinstance(value, dict):
            continue
        scene_id = str(key)
        for image_name, entry in value.items():
            if isinstance(entry, dict):
                out[(scene_id, str(image_name))] = entry
    return out


def _load_sidecar_entry(cache_path: Path, scene_id: str, image_name: str) -> dict[str, Any] | None:
    sidecar_path = cache_path.parent / FRAME_CACHE_SIDECAR_DIR_NAME / f"{scene_id}.json"
    if not sidecar_path.exists():
        return None
    try:
        sidecar_doc = _read_json(sidecar_path)
    except Exception:
        return None
    frames = sidecar_doc.get("frames") if isinstance(sidecar_doc, dict) else None
    if not isinstance(frames, dict):
        return None
    raw_record = frames.get(image_name)
    if not isinstance(raw_record, dict):
        return None
    entry = raw_record.get("referability_entry", raw_record)
    return entry if isinstance(entry, dict) else None


def _merge_cards_into_cache(
    cache_doc: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    cache_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated = copy.deepcopy(cache_doc)
    frames = updated.setdefault("frames", {})
    merge_warnings: list[dict[str, Any]] = []
    if not isinstance(frames, dict):
        return updated, [{"reason": "cache_frames_not_mapping"}]

    for card in cards:
        if bool(card.get("deleted")):
            continue
        scene_id = str(card.get("scene_id", "")).strip()
        image_name = str(card.get("image_name", "")).strip()
        parent_id = _as_int(card.get("parent_id"))
        child_id = _as_int(card.get("child_id"))
        if not scene_id or not image_name or parent_id is None or child_id is None:
            continue

        scene_frames = frames.get(scene_id)
        if not isinstance(scene_frames, dict):
            scene_frames = {}
            frames[scene_id] = scene_frames
        entry = scene_frames.get(image_name)
        if not isinstance(entry, dict):
            entry = _load_sidecar_entry(cache_path, scene_id, image_name)
            if not isinstance(entry, dict):
                merge_warnings.append({
                    "scene_id": scene_id,
                    "image_name": image_name,
                    "pair": [parent_id, child_id],
                    "reason": "frame_missing_from_cache_and_sidecar",
                })
                continue
            scene_frames[image_name] = entry

        review_card = {
            "pair_id": str(card.get("pair_id") or f"{parent_id}->{child_id}"),
            "parent_id": parent_id,
            "parent_label": str(card.get("parent_label", "")).strip(),
            "parent_surface_text": str(card.get("parent_surface_text", "")).strip(),
            "child_id": child_id,
            "child_label": str(card.get("child_label", "")).strip(),
            "child_surface_text": str(card.get("child_surface_text", "")).strip(),
            "source": "human_salvage_html",
        }
        existing_cards = entry.get("attachment_human_review_cards")
        if not isinstance(existing_cards, list):
            existing_cards = []
        existing_key = {
            (int(item.get("parent_id")), int(item.get("child_id")))
            for item in existing_cards
            if isinstance(item, dict)
            and _as_int(item.get("parent_id")) is not None
            and _as_int(item.get("child_id")) is not None
        }
        if (parent_id, child_id) not in existing_key:
            existing_cards.append(review_card)
        entry["attachment_human_review_cards"] = existing_cards

        pairs = _normalize_pairs(entry.get("attachment_referable_pairs"))
        pairs.add((parent_id, child_id))
        entry["attachment_referable_pairs"] = [[a, b] for a, b in sorted(pairs)]
        ids = set(_normalize_ids(entry.get("attachment_referable_object_ids")))
        ids.update([parent_id, child_id])
        entry["attachment_referable_object_ids"] = sorted(ids)

    return updated, merge_warnings


def _cards_from_cache(cache_doc: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for (scene_id, image_name), entry in _iter_frame_entries(cache_doc).items():
        raw_cards = entry.get("attachment_human_review_cards")
        if not isinstance(raw_cards, list):
            continue
        for raw in raw_cards:
            if not isinstance(raw, dict):
                continue
            parent_id = _as_int(raw.get("parent_id"))
            child_id = _as_int(raw.get("child_id"))
            if parent_id is None or child_id is None:
                continue
            card = dict(raw)
            card["scene_id"] = scene_id
            card["image_name"] = image_name
            card["parent_id"] = parent_id
            card["child_id"] = child_id
            card["deleted"] = False
            card["source_html"] = None
            cards.append(card)
    return cards


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for card in cards:
        if bool(card.get("deleted")):
            continue
        scene_id = str(card.get("scene_id", "")).strip()
        image_name = str(card.get("image_name", "")).strip()
        parent_id = _as_int(card.get("parent_id"))
        child_id = _as_int(card.get("child_id"))
        if not scene_id or not image_name or parent_id is None or child_id is None:
            continue
        key = (scene_id, image_name, parent_id, child_id)
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(card)
        normalized["scene_id"] = scene_id
        normalized["image_name"] = image_name
        normalized["parent_id"] = parent_id
        normalized["child_id"] = child_id
        out.append(normalized)
    return out


def _load_frame_debug(pilot_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    frame_debug_dir = pilot_root / "frame_debug"
    if not frame_debug_dir.exists():
        return out
    for path in sorted(frame_debug_dir.glob("*.json")):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        scene_id = str(doc.get("scene_id") or path.stem) if isinstance(doc, dict) else path.stem
        frames = doc.get("frames") if isinstance(doc, dict) else None
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            image_name = str(frame.get("image_name", "")).strip()
            if image_name:
                out[(scene_id, image_name)] = frame
    return out


def _load_scene_question_cache(pilot_root: Path) -> dict[str, list[dict[str, Any]]]:
    raw_dir = pilot_root / "_raw_questions_scene_cache"
    out: dict[str, list[dict[str, Any]]] = {}
    if not raw_dir.exists():
        return out
    for path in sorted(raw_dir.glob("*.json")):
        try:
            data = _read_json(path)
        except Exception:
            continue
        if isinstance(data, list):
            out[path.stem] = [item for item in data if isinstance(item, dict)]
    return out


def _load_benchmark_questions(pilot_root: Path) -> list[dict[str, Any]]:
    path = pilot_root / "benchmark.json"
    if not path.exists():
        return []
    try:
        data = _read_json(path)
    except Exception:
        return []
    if isinstance(data, dict):
        questions = data.get("questions")
        return [item for item in questions if isinstance(item, dict)] if isinstance(questions, list) else []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _question_object_ids(question: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for field in OBJECT_ID_FIELDS:
        obj_id = _as_int(question.get(field))
        if obj_id is not None:
            ids.add(obj_id)
    for mention in question.get("mentioned_objects", []) or []:
        if not isinstance(mention, dict):
            continue
        obj_id = _as_int(mention.get("object_id"))
        if obj_id is None:
            obj_id = _as_int(mention.get("id"))
        if obj_id is None:
            obj_id = _as_int(mention.get("obj_id"))
        if obj_id is not None:
            ids.add(obj_id)
    return ids


def _question_uses_attachment_referability(question: dict[str, Any]) -> bool:
    qtype = str(question.get("type", "")).strip().lower()
    return (
        qtype == "attachment_chain"
        or qtype.startswith("attachment")
        or bool(question.get("attachment_remapped", False))
    )


def _filter_questions(
    questions: list[dict[str, Any]],
    *,
    scene_id: str,
    image_name: str,
) -> list[dict[str, Any]]:
    return [
        q
        for q in questions
        if str(q.get("scene_id", scene_id)) == scene_id
        and str(q.get("image_name", image_name)) == image_name
    ]


def _summarize_question_hits(
    questions: list[dict[str, Any]],
    *,
    parent_id: int,
    child_id: int,
    parent_surface_text: str,
    child_surface_text: str,
    limit: int,
) -> dict[str, Any]:
    parent_hits: list[dict[str, Any]] = []
    child_hits: list[dict[str, Any]] = []
    pair_hits: list[dict[str, Any]] = []
    attachment_hits: list[dict[str, Any]] = []
    parent_label_hits = 0
    child_label_hits = 0
    type_counter: Counter[str] = Counter()
    attachment_type_counter: Counter[str] = Counter()

    parent_label = parent_surface_text.strip().lower()
    child_label = child_surface_text.strip().lower()
    for question in questions:
        ids = _question_object_ids(question)
        qtext = str(question.get("question", ""))
        qtype = str(question.get("type", ""))
        if parent_id in ids or child_id in ids:
            type_counter[qtype] += 1
        if parent_id in ids:
            parent_hits.append(question)
        if child_id in ids:
            child_hits.append(question)
        if parent_id in ids and child_id in ids:
            pair_hits.append(question)
        if _question_uses_attachment_referability(question) and (parent_id in ids or child_id in ids):
            attachment_hits.append(question)
            attachment_type_counter[qtype] += 1
        lower_text = qtext.lower()
        if parent_label and parent_label in lower_text:
            parent_label_hits += 1
        if child_label and child_label in lower_text:
            child_label_hits += 1

    def previews(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for question in items[:limit]:
            out.append({
                "type": question.get("type"),
                "question": question.get("question"),
                "object_ids": sorted(_question_object_ids(question)),
                "attachment_remapped": bool(question.get("attachment_remapped", False)),
                "trace_reason": question.get("trace_reason"),
                "question_referability_decision": (
                    question.get("question_referability_audit", {}) or {}
                ).get("decision") if isinstance(question.get("question_referability_audit"), dict) else None,
            })
        return out

    return {
        "total": len(questions),
        "parent_object_hits": len(parent_hits),
        "child_object_hits": len(child_hits),
        "pair_object_hits": len(pair_hits),
        "attachment_related_object_hits": len(attachment_hits),
        "parent_surface_text_hits": parent_label_hits,
        "child_surface_text_hits": child_label_hits,
        "types_for_parent_or_child": dict(sorted(type_counter.items())),
        "attachment_types_for_parent_or_child": dict(sorted(attachment_type_counter.items())),
        "parent_examples": previews(parent_hits),
        "child_examples": previews(child_hits),
        "pair_examples": previews(pair_hits),
        "attachment_examples": previews(attachment_hits),
    }


def _reason_for_pair(
    *,
    cache_entry_found: bool,
    frame_debug_found: bool,
    frame_debug: dict[str, Any] | None,
    parent_id: int,
    child_id: int,
    cache_pair_present: bool,
    cache_attachment_ids: set[int],
    raw_summary: dict[str, Any],
    final_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
) -> str:
    if not cache_entry_found:
        return "edited_pair_frame_not_found_in_referability_cache"
    if not cache_pair_present:
        return "edited_pair_not_present_in_attachment_referable_pairs_after_cache_merge"
    if not frame_debug_found:
        if raw_summary["child_object_hits"] or raw_summary["pair_object_hits"]:
            return "frame_debug_missing_but_raw_questions_exist"
        return "frame_debug_missing_cannot_confirm_generation_inputs"
    assert frame_debug is not None
    skip_reason = frame_debug.get("pipeline_skip_reason")
    if skip_reason:
        return f"frame_skipped_by_pipeline:{skip_reason}"

    visible_ids = set(_normalize_ids(frame_debug.get("pipeline_visible_object_ids_used_for_generation")))
    attach_used_ids = set(_normalize_ids(frame_debug.get("pipeline_attachment_referable_object_ids_used_for_generation")))
    attachment_rows = frame_debug.get("attachment_rows")
    attachment_row_pairs = {
        (int(row["parent_id"]), int(row["child_id"]))
        for row in attachment_rows
        if isinstance(row, dict)
        and _as_int(row.get("parent_id")) is not None
        and _as_int(row.get("child_id")) is not None
    } if isinstance(attachment_rows, list) else set()

    if parent_id not in visible_ids or child_id not in visible_ids:
        return "parent_or_child_not_in_pipeline_visible_ids"
    if parent_id not in attach_used_ids or child_id not in attach_used_ids:
        return "parent_or_child_not_in_pipeline_attachment_referable_ids"
    pair_in_attachment_rows = (parent_id, child_id) in attachment_row_pairs
    if not pair_in_attachment_rows:
        return "pair_referable_but_not_in_frame_attachment_graph_rows"
    if raw_summary["child_object_hits"] == 0 and raw_summary["pair_object_hits"] == 0:
        if raw_summary["parent_object_hits"] > 0:
            return "pair_in_attachment_graph_but_only_parent_raw_questions_generated"
        if pair_in_attachment_rows:
            return "pair_in_attachment_graph_but_child_generated_zero_raw_questions"
        return "no_raw_question_mentions_child_or_pair"
    if raw_summary["child_object_hits"] > 0 and final_summary["child_object_hits"] == 0:
        return "raw_question_mentions_child_but_not_in_final_questions_after_caps_or_qc"
    if final_summary["child_object_hits"] > 0 and benchmark_summary["child_object_hits"] == 0:
        return "final_question_mentions_child_but_not_in_benchmark"
    if benchmark_summary["child_surface_text_hits"] == 0 and benchmark_summary["child_object_hits"] > 0:
        return "benchmark_mentions_child_object_id_but_not_human_surface_text"
    if benchmark_summary["child_surface_text_hits"] > 0:
        return "benchmark_contains_human_child_surface_text"
    if cache_attachment_ids and (parent_id not in cache_attachment_ids or child_id not in cache_attachment_ids):
        return "pair_present_but_parent_or_child_missing_from_cache_attachment_ids"
    return "pair_available_but_not_selected_by_question_generation"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pilot_root = Path(args.pilot_root)
    cache_path = Path(args.referability_cache)
    cache_doc = _read_json(cache_path)

    html_warnings: list[str] = []
    if args.edited_html:
        html_paths = [Path(path) for path in args.edited_html]
        html_mode = "explicit"
    elif args.include_all_edited_html:
        html_paths, html_mode, html_warnings = _discover_all_edited_html_paths(cache_path)
    else:
        html_paths, html_mode, html_warnings = _discover_edited_html_paths(cache_path)

    html_cards: list[dict[str, Any]] = []
    html_card_counts: dict[str, dict[str, int]] = {}
    for path in html_paths:
        cards = _parse_review_html(path)
        html_cards.extend(cards)
        html_card_counts[str(path)] = {
            "total_cards": len(cards),
            "kept_cards": sum(1 for card in cards if not bool(card.get("deleted"))),
            "deleted_cards": sum(1 for card in cards if bool(card.get("deleted"))),
        }

    merged_cache_doc = cache_doc
    merge_warnings: list[dict[str, Any]] = []
    if html_cards:
        merged_cache_doc, merge_warnings = _merge_cards_into_cache(
            cache_doc,
            html_cards,
            cache_path=cache_path,
        )

    cards = _dedupe_cards(html_cards if html_cards else _cards_from_cache(merged_cache_doc))
    frame_entries = _iter_frame_entries(merged_cache_doc)
    frame_debug_by_key = _load_frame_debug(pilot_root)
    raw_by_scene = _load_scene_question_cache(pilot_root)
    benchmark_questions = _load_benchmark_questions(pilot_root)

    results: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for card in cards:
        scene_id = str(card["scene_id"])
        image_name = str(card["image_name"])
        parent_id = int(card["parent_id"])
        child_id = int(card["child_id"])
        parent_surface_text = str(card.get("parent_surface_text", "")).strip()
        child_surface_text = str(card.get("child_surface_text", "")).strip()

        cache_entry = frame_entries.get((scene_id, image_name))
        cache_entry_found = isinstance(cache_entry, dict)
        cache_pairs = _normalize_pairs((cache_entry or {}).get("attachment_referable_pairs"))
        cache_attachment_ids = set(_normalize_ids((cache_entry or {}).get("attachment_referable_object_ids")))
        cache_human_cards = (cache_entry or {}).get("attachment_human_review_cards", [])
        cache_human_card_count = len(cache_human_cards) if isinstance(cache_human_cards, list) else 0

        frame_debug = frame_debug_by_key.get((scene_id, image_name))
        frame_debug_found = isinstance(frame_debug, dict)
        attachment_rows = (frame_debug or {}).get("attachment_rows", [])
        attachment_row_pairs = [
            [int(row["parent_id"]), int(row["child_id"])]
            for row in attachment_rows
            if isinstance(row, dict)
            and _as_int(row.get("parent_id")) is not None
            and _as_int(row.get("child_id")) is not None
        ] if isinstance(attachment_rows, list) else []

        raw_questions = _filter_questions(
            raw_by_scene.get(scene_id, []),
            scene_id=scene_id,
            image_name=image_name,
        )
        final_questions = [
            item for item in (frame_debug or {}).get("final_questions", [])
            if isinstance(item, dict)
        ] if frame_debug_found else []
        if not final_questions:
            questions_path = pilot_root / "questions" / f"{scene_id}.json"
            if questions_path.exists():
                scene_questions_doc = _read_json(questions_path)
                scene_questions = scene_questions_doc
                if isinstance(scene_questions_doc, dict):
                    scene_questions = scene_questions_doc.get("questions", [])
                if isinstance(scene_questions, list):
                    final_questions = _filter_questions(
                        [item for item in scene_questions if isinstance(item, dict)],
                        scene_id=scene_id,
                        image_name=image_name,
                    )
        bench_frame_questions = _filter_questions(
            benchmark_questions,
            scene_id=scene_id,
            image_name=image_name,
        )

        raw_summary = _summarize_question_hits(
            raw_questions,
            parent_id=parent_id,
            child_id=child_id,
            parent_surface_text=parent_surface_text,
            child_surface_text=child_surface_text,
            limit=args.example_limit,
        )
        final_summary = _summarize_question_hits(
            final_questions,
            parent_id=parent_id,
            child_id=child_id,
            parent_surface_text=parent_surface_text,
            child_surface_text=child_surface_text,
            limit=args.example_limit,
        )
        benchmark_summary = _summarize_question_hits(
            bench_frame_questions,
            parent_id=parent_id,
            child_id=child_id,
            parent_surface_text=parent_surface_text,
            child_surface_text=child_surface_text,
            limit=args.example_limit,
        )
        reason = _reason_for_pair(
            cache_entry_found=cache_entry_found,
            frame_debug_found=frame_debug_found,
            frame_debug=frame_debug,
            parent_id=parent_id,
            child_id=child_id,
            cache_pair_present=(parent_id, child_id) in cache_pairs,
            cache_attachment_ids=cache_attachment_ids,
            raw_summary=raw_summary,
            final_summary=final_summary,
            benchmark_summary=benchmark_summary,
        )
        reason_counter[reason] += 1

        results.append({
            "scene_id": scene_id,
            "image_name": image_name,
            "pair": [parent_id, child_id],
            "pair_id": card.get("pair_id"),
            "parent_label": card.get("parent_label"),
            "child_label": card.get("child_label"),
            "parent_surface_text": parent_surface_text,
            "child_surface_text": child_surface_text,
            "source_html": card.get("source_html"),
            "diagnosis": reason,
            "cache": {
                "frame_entry_found": cache_entry_found,
                "human_review_card_count": cache_human_card_count,
                "pair_in_attachment_referable_pairs": (parent_id, child_id) in cache_pairs,
                "parent_in_attachment_referable_object_ids": parent_id in cache_attachment_ids,
                "child_in_attachment_referable_object_ids": child_id in cache_attachment_ids,
            },
            "frame_debug": {
                "found": frame_debug_found,
                "pipeline_skip_reason": (frame_debug or {}).get("pipeline_skip_reason"),
                "parent_in_pipeline_visible_ids": parent_id in set(_normalize_ids((frame_debug or {}).get("pipeline_visible_object_ids_used_for_generation"))),
                "child_in_pipeline_visible_ids": child_id in set(_normalize_ids((frame_debug or {}).get("pipeline_visible_object_ids_used_for_generation"))),
                "parent_in_pipeline_attachment_referable_ids": parent_id in set(_normalize_ids((frame_debug or {}).get("pipeline_attachment_referable_object_ids_used_for_generation"))),
                "child_in_pipeline_attachment_referable_ids": child_id in set(_normalize_ids((frame_debug or {}).get("pipeline_attachment_referable_object_ids_used_for_generation"))),
                "pair_in_attachment_rows": [parent_id, child_id] in attachment_row_pairs,
                "attachment_row_pairs": attachment_row_pairs,
            },
            "raw_questions": raw_summary,
            "final_questions": final_summary,
            "benchmark_questions": benchmark_summary,
        })

    return {
        "pilot_root": str(pilot_root),
        "referability_cache": str(cache_path),
        "edited_html_mode": html_mode,
        "edited_html_paths": [str(path) for path in html_paths],
        "warnings": html_warnings,
        "merge_warnings": merge_warnings,
        "html_card_counts": html_card_counts,
        "summary": {
            "diagnosed_pair_count": len(results),
            "reason_counts": dict(sorted(reason_counter.items())),
            "raw_scene_cache_found": bool(raw_by_scene),
            "frame_debug_found": bool(frame_debug_by_key),
            "benchmark_question_count": len(benchmark_questions),
        },
        "pairs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why edited attachment pair cards did or did not appear in generated questions."
    )
    parser.add_argument("--pilot_root", required=True, help="Pipeline output directory, e.g. output/pilot/0-9")
    parser.add_argument("--referability_cache", required=True, help="Referability cache JSON used by the pipeline")
    parser.add_argument(
        "--edited_html",
        action="append",
        default=[],
        help="Optional edited review HTML. May be passed multiple times. If omitted, uses pipeline-style auto discovery.",
    )
    parser.add_argument(
        "--include_all_edited_html",
        action="store_true",
        help=(
            "When --edited_html is omitted, read both legacy edited*.html and "
            "<cache-stem>_<scene>_edited.html files. This is diagnostic-only and "
            "may differ from run_pipeline's default selection."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <pilot_root>/edited_pair_generation_debug.json",
    )
    parser.add_argument("--example_limit", type=int, default=3, help="Question examples to include per hit category")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output) if args.output else Path(args.pilot_root) / "edited_pair_generation_debug.json"
    _write_json(output_path, report)
    summary = report["summary"]
    print(f"Wrote {output_path}")
    print(f"diagnosed pairs: {summary['diagnosed_pair_count']}")
    print("reason counts:")
    for reason, count in summary["reason_counts"].items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
