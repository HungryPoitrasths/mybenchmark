#!/usr/bin/env python3
"""Rewrite benchmark_subset.json from the cards in an edited merged viewer HTML."""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from find_strong_duplicate_questions import _iter_card_ranges, _card_index, _simple_fields
from review_viewer_html import parse_viewer_html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = PROJECT_ROOT / "output" / "edited" / "merged_1-3593.html"
DEFAULT_SUBSET = PROJECT_ROOT / "output" / "benchmark_subset.json"
DEFAULT_FULL = PROJECT_ROOT / "output" / "benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "output" / "benchmark_subset.from_merged_1-3593_report.json"


ROLE_FIELD_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "direction_agent": {
        "query": ("obj_b_label",),
        "reference": ("obj_a_label",),
        "direction": ("correct_value",),
    },
    "occlusion": {
        "target": ("obj_a_label",),
        "visibility": ("correct_value",),
    },
    "distance": {
        "obj_a": ("obj_a_label",),
        "obj_b": ("obj_b_label",),
        "distance_bin": ("correct_value",),
        "distance_m": ("distance_m",),
    },
    "direction_object_centric": {
        "query": ("obj_target_label",),
        "reference": ("obj_ref_label",),
        "facing": ("obj_face_label",),
        "direction": ("correct_value",),
    },
    "direction_allocentric": {
        "query": ("obj_a_label",),
        "reference": ("obj_b_label",),
        "camera": ("camera_cardinal",),
        "direction": ("correct_value",),
    },
    "object_move_agent": {
        "moved": ("moved_obj_label",),
        "query": ("obj_b_label", "query_obj_label"),
        "reference": ("obj_c_label",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
    "object_move_distance": {
        "moved": ("moved_obj_label",),
        "obj_b": ("obj_b_label", "query_obj_label"),
        "obj_c": ("obj_c_label",),
        "old_distance_bin": ("old_distance_bin", "old_correct_value"),
        "new_distance_bin": ("new_distance_bin", "new_correct_value", "correct_value"),
        "old_distance_m": ("old_distance_m",),
        "new_distance_m": ("new_distance_m",),
    },
    "object_move_occlusion": {
        "moved": ("moved_obj_label",),
        "target": ("query_obj_label", "target_obj_label"),
        "old_visibility": ("old_visibility", "old_correct_value"),
        "new_visibility": ("new_visibility", "new_correct_value", "correct_value"),
    },
    "object_move_object_centric": {
        "moved": ("moved_obj_label",),
        "query": ("query_obj_label",),
        "reference": ("obj_ref_label",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
    "object_remove": {
        "removed": ("removed_obj_label",),
        "target": ("obj_b_label",),
        "old_visibility": ("old_visibility", "old_correct_value"),
        "new_visibility": ("new_visibility", "new_correct_value", "correct_value"),
    },
    "object_rotate_object_centric": {
        "moved": ("moved_obj_label",),
        "query": ("query_obj_label",),
        "facing": ("obj_face_label",),
        "reference": ("obj_ref_label",),
        "rotation_angle": ("rotation_angle",),
        "rotation_direction": ("rotation_direction",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
    "object_move_allocentric": {
        "moved": ("moved_obj_label",),
        "query": ("query_obj_label",),
        "reference": ("obj_ref_label",),
        "camera": ("camera_cardinal",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
    "attachment_chain": {
        "moved": ("grandparent_label",),
        "child": ("parent_label",),
        "grandchild": ("grandchild_label",),
        "contrast": ("neighbor_label",),
        "chain_depth": ("chain_depth",),
        "displaced": ("correct_value",),
    },
    "attachment_move": {
        "root": ("root_label",),
        "parent": ("parent_label",),
        "grandchild": ("grandchild_label",),
        "query": ("query_obj_label",),
        "reference": ("obj_ref_label",),
        "reference_frame": ("reference_frame",),
        "query_role": ("query_role",),
        "old_direction": ("old_correct_value",),
        "new_direction": ("new_correct_value", "correct_value"),
    },
    "coordinate_rotation_agent": {
        "query": ("obj_a_label",),
        "reference": ("obj_b_label",),
        "rotation_angle": ("rotation_angle",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
    "coordinate_rotation_object_centric": {
        "query": ("obj_target_label",),
        "reference": ("obj_ref_label",),
        "facing": ("obj_face_label",),
        "rotation_angle": ("rotation_angle",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
    "coordinate_rotation_allocentric": {
        "query": ("obj_a_label",),
        "reference": ("obj_b_label",),
        "camera": ("camera_cardinal",),
        "rotation_angle": ("rotation_angle",),
        "old_direction": ("old_direction", "old_correct_value"),
        "new_direction": ("new_direction", "new_correct_value", "correct_value"),
    },
}


def _load_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(payload, dict) or not isinstance(questions, list):
        raise ValueError(f"Expected benchmark dict with questions: {path}")
    return payload, [q for q in questions if isinstance(q, dict)]


def _qtype_from_badges(card: dict[str, Any]) -> str:
    for badge in card.get("badges", []):
        text = str(badge)
        if text.startswith("L") and "_" in text:
            return text.split("_", 1)[1]
    return ""


def _level_from_badges(card: dict[str, Any]) -> str:
    for badge in card.get("badges", []):
        if badge in {"L1", "L2", "L3"}:
            return str(badge)
    return ""


def _html_exact_key(card: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(card["scene_id"]),
        str(card["image_name"]),
        _level_from_badges(card),
        _qtype_from_badges(card),
        str(card["question"]).strip(),
        tuple(str(option["text"]) for option in card["options"]),
    )


def _question_exact_key(question: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(question.get("scene_id", "")),
        str(question.get("image_name", "")),
        str(question.get("level", "")),
        str(question.get("type", "")),
        str(question.get("question", "")).strip(),
        tuple(str(option) for option in question.get("options", [])),
    )


def _loose_key_from_card(card: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(card["scene_id"]),
        str(card["image_name"]),
        _level_from_badges(card),
        _qtype_from_badges(card),
    )


def _loose_key_from_question(question: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(question.get("scene_id", "")),
        str(question.get("image_name", "")),
        str(question.get("level", "")),
        str(question.get("type", "")),
    )


def _cards_simple_fields(html_text: str) -> dict[int, dict[str, str]]:
    fields: dict[int, dict[str, str]] = {}
    for start, end in _iter_card_ranges(html_text):
        card_html = html_text[start:end]
        fields[_card_index(card_html)] = _simple_fields(card_html)
    return fields


def _answer_from_card(card: dict[str, Any]) -> str | list[str]:
    letters = [
        str(option["letter"]).upper()
        for option in card["options"]
        if str(option["letter"]).upper() == str(card.get("gold_answer") or "").upper()
    ]
    # parse_viewer_html stores only one gold_answer. For manually edited multi-select
    # cards, infer all correct labels by looking at option class in a second pass.
    if letters:
        return letters[0]
    return str(card.get("gold_answer") or "")


def _correct_letters_from_card_html(card_html: str) -> list[str]:
    letters = re.findall(
        r'<div class="opt\s+correct">([A-D])\.\&nbsp;',
        card_html,
        flags=re.IGNORECASE,
    )
    return [letter.upper() for letter in letters]


def _html_card_by_index(html_text: str) -> dict[int, str]:
    cards: dict[int, str] = {}
    for start, end in _iter_card_ranges(html_text):
        card = html_text[start:end]
        cards[_card_index(card)] = card
    return cards


def _option_texts(card: dict[str, Any]) -> dict[str, str]:
    return {str(option["letter"]).upper(): str(option["text"]) for option in card["options"]}


def _update_mentioned_objects(question: dict[str, Any], old_to_new: dict[str, str]) -> None:
    mentioned = question.get("mentioned_objects")
    if not isinstance(mentioned, list):
        return
    for item in mentioned:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        if label in old_to_new:
            item["label"] = old_to_new[label]


def _set_first_present(question: dict[str, Any], keys: tuple[str, ...], value: str, old_to_new: dict[str, str]) -> None:
    if value == "":
        return
    target_key = None
    for key in keys:
        if key in question:
            target_key = key
            break
    if target_key is None:
        target_key = keys[0]
    old_value = str(question.get(target_key, ""))
    question[target_key] = value
    if old_value and old_value != value:
        old_to_new[old_value] = value


def _apply_simple_fields(question: dict[str, Any], qtype: str, fields: dict[str, str]) -> None:
    mapping = ROLE_FIELD_MAP.get(qtype, {})
    old_to_new: dict[str, str] = {}
    for label, value in fields.items():
        keys = mapping.get(label)
        if not keys:
            continue
        _set_first_present(question, keys, str(value), old_to_new)
    _update_mentioned_objects(question, old_to_new)


def _refresh_question_uid(question: dict[str, Any]) -> None:
    if "question_uid" not in question:
        return
    payload = {
        "answer": question.get("answer"),
        "dataset": question.get("_dataset") or question.get("dataset"),
        "image_name": question.get("image_name"),
        "level": question.get("level"),
        "options": question.get("options"),
        "question": question.get("question"),
        "scene_id": question.get("scene_id"),
        "type": question.get("type"),
    }
    question["question_uid"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _update_statistics(payload: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    levels = Counter(str(q.get("level", "")) for q in questions)
    types = Counter(str(q.get("type", "")) for q in questions)
    stats = dict(payload.get("statistics", {}))
    stats["total"] = len(questions)
    stats["by_level"] = {level: levels[level] for level in ("L1", "L2", "L3")}
    stats["by_type"] = dict(sorted(types.items()))
    for level in ("L1", "L2", "L3"):
        answers = Counter(json.dumps(q.get("answer"), ensure_ascii=False) for q in questions if q.get("level") == level)
        total = sum(answers.values())
        if total:
            stats[f"{level}_answer_dist"] = {
                str(json.loads(answer)): round(count / total, 3)
                for answer, count in sorted(answers.items())
            }
    payload["statistics"] = stats


def _build_source_indexes(questions: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], dict[tuple[str, str, str, str], list[dict[str, Any]]]]:
    by_exact: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_loose: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_exact[_question_exact_key(question)].append(question)
        by_loose[_loose_key_from_question(question)].append(question)
    return by_exact, by_loose


def _choose_loose_match(
    candidates: list[dict[str, Any]],
    used_ids: set[int],
    card_fields: dict[str, str],
    card_question: str,
) -> dict[str, Any] | None:
    unused = [q for q in candidates if id(q) not in used_ids]
    if len(unused) == 1:
        return unused[0]
    pool = unused if unused else candidates
    if not pool:
        return None

    def _norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _label_score(source: Any, edited: Any) -> int:
        source_norm = _norm(source)
        edited_norm = _norm(edited)
        if not source_norm or not edited_norm:
            return 0
        if source_norm == edited_norm:
            return 6
        if source_norm in edited_norm or edited_norm in source_norm:
            return 4
        source_tokens = set(re.findall(r"[a-z0-9]+", source_norm))
        edited_tokens = set(re.findall(r"[a-z0-9]+", edited_norm))
        if not source_tokens or not edited_tokens:
            return 0
        overlap = len(source_tokens & edited_tokens)
        if overlap == len(source_tokens) or overlap == len(edited_tokens):
            return 3
        return overlap

    def _template(value: Any) -> str:
        text = _norm(value)
        # Keep the wording and remove only the labels that appear as simple
        # fields, so manually renamed objects do not dominate the comparison.
        labels = sorted(
            (_norm(v) for k, v in card_fields.items() if k not in {"direction", "old_direction", "new_direction"}),
            key=len,
            reverse=True,
        )
        for label in labels:
            if label:
                text = text.replace(label, " ")
        text = re.sub(r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b", lambda m: m.group(0), text)
        return re.sub(r"\s+", " ", text).strip()

    card_template = _template(card_question)

    def score(question: dict[str, Any]) -> tuple[int, int, float]:
        qtype = str(question.get("type", ""))
        mapping = ROLE_FIELD_MAP.get(qtype, {})
        value = 0
        relation_value = 0
        for label, field_value in card_fields.items():
            keys = mapping.get(label, ())
            for key in keys:
                key_value = question.get(key, "")
                if str(key_value) == str(field_value):
                    value += 8
                    relation_value += 1
                    break
                if label in {
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
                }:
                    continue
                label_value = _label_score(key_value, field_value)
                if label_value:
                    value += label_value
                    break
        text_ratio = difflib.SequenceMatcher(None, card_template, _template(question.get("question", ""))).ratio()
        return value, relation_value, text_ratio

    ranked = sorted(pool, key=score, reverse=True)
    return ranked[0]


def rewrite_subset(html_path: Path, subset_path: Path, full_path: Path, report_path: Path) -> dict[str, Any]:
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    cards = parse_viewer_html(html_text, include_deleted=False)
    simple_by_idx = _cards_simple_fields(html_text)
    raw_card_by_idx = _html_card_by_index(html_text)

    subset_payload, _ = _load_payload(subset_path)
    _, full_questions = _load_payload(full_path)
    by_exact, by_loose = _build_source_indexes(full_questions)

    used_source_ids: set[int] = set()
    output_questions: list[dict[str, Any]] = []
    match_counts = Counter()
    unmatched: list[dict[str, Any]] = []

    for card in cards:
        idx = int(card["viewer_index"])
        fields = simple_by_idx.get(idx, {})
        qtype = _qtype_from_badges(card)
        level = _level_from_badges(card)
        matched = None
        exact_candidates = [q for q in by_exact.get(_html_exact_key(card), []) if id(q) not in used_source_ids]
        if exact_candidates:
            matched = exact_candidates[0]
            match_counts["exact"] += 1
        else:
            matched = _choose_loose_match(
                by_loose.get(_loose_key_from_card(card), []),
                used_source_ids,
                fields,
                str(card["question"]),
            )
            if matched is not None:
                match_counts["loose"] += 1

        if matched is None:
            matched = {
                "_dataset": "scannetpp" if not str(card["scene_id"]).startswith("scene") else "scannet",
                "scene_id": str(card["scene_id"]),
                "image_name": str(card["image_name"]),
                "level": level,
                "type": qtype,
            }
            match_counts["synthetic"] += 1
            unmatched.append(
                {
                    "viewer_index": idx,
                    "scene_id": str(card["scene_id"]),
                    "image_name": str(card["image_name"]),
                    "level": level,
                    "type": qtype,
                    "question": str(card["question"]),
                }
            )
        else:
            used_source_ids.add(id(matched))
            matched = copy.deepcopy(matched)

        matched["scene_id"] = str(card["scene_id"])
        matched["image_name"] = str(card["image_name"])
        matched["level"] = level
        matched["type"] = qtype
        matched["question"] = str(card["question"])
        matched["options"] = [str(option["text"]) for option in card["options"]]

        correct_letters = _correct_letters_from_card_html(raw_card_by_idx[idx])
        if not correct_letters:
            correct_letters = [str(card.get("gold_answer") or "")]
        correct_letters = [letter for letter in correct_letters if letter]
        matched["answer"] = correct_letters[0] if len(correct_letters) == 1 else correct_letters
        option_texts = _option_texts(card)
        correct_values = [option_texts[letter] for letter in correct_letters if letter in option_texts]
        if len(correct_values) == 1:
            matched["correct_value"] = correct_values[0]
        elif correct_values:
            matched["correct_values"] = correct_values
            matched["correct_value"] = "; ".join(correct_values)
            matched["multi_select"] = True

        _apply_simple_fields(matched, qtype, fields)
        _refresh_question_uid(matched)
        matched["_viewer_index"] = idx
        matched["_source_html"] = str(html_path)
        output_questions.append(matched)

    payload = copy.deepcopy(subset_payload)
    payload["questions"] = output_questions
    metadata = dict(payload.get("metadata", {}))
    metadata["rewritten_from_html"] = {
        "source_html": str(html_path),
        "question_count": len(output_questions),
        "report": str(report_path),
        "match_counts": dict(match_counts),
    }
    payload["metadata"] = metadata
    _update_statistics(payload, output_questions)
    subset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "source_html": str(html_path),
        "subset_path": str(subset_path),
        "full_path": str(full_path),
        "html_cards": len(cards),
        "output_questions": len(output_questions),
        "match_counts": dict(match_counts),
        "counts_by_level": dict(Counter(q.get("level") for q in output_questions)),
        "counts_by_type": dict(sorted(Counter(q.get("type") for q in output_questions).items())),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched[:200],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = rewrite_subset(args.html, args.subset, args.full, args.report)
    print(f"HTML cards: {report['html_cards']}")
    print(f"Output questions: {report['output_questions']}")
    print(f"Match counts: {report['match_counts']}")
    print(f"Unmatched: {report['unmatched_count']}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
