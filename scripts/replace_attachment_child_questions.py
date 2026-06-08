#!/usr/bin/env python3
"""Resample attachment-remapped L2 question types using valid query!=move items."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.make_viewer as make_viewer
from scripts.review_viewer_html import parse_viewer_html


TARGET_TYPES = (
    "object_move_agent",
    "object_move_distance",
    "object_move_object_centric",
    "object_rotate_object_centric",
)
STRICT_CAP_TYPES = (
    "object_move_agent",
    "object_move_distance",
    "object_move_object_centric",
)
ROTATE_TYPE = "object_rotate_object_centric"
OBJECT_KEY_FIELDS = (
    "query_obj_id",
    "obj_a_id",
    "target_obj_id",
    "obj_target_id",
    "removed_obj_id",
    "obj_ref_id",
    "obj_face_id",
    "moved_obj_id",
    "parent_id",
    "root_id",
    "grandchild_id",
    "grandparent_id",
    "neighbor_id",
    "obj_b_id",
)


def _json_key(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _question_uid(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question.get("_dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "level": question.get("level"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
            "answer": question.get("answer"),
        }
    )


def _question_object_id(question: dict[str, Any]) -> str:
    for field in OBJECT_KEY_FIELDS:
        value = question.get(field)
        if value is not None:
            return str(value)
    return f"uid:{question.get('question_uid', '')}"


def _object_key(question: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(question.get("_dataset") or ""),
        str(question.get("scene_id") or ""),
        str(question.get("image_name") or ""),
        str(question.get("type") or ""),
        _question_object_id(question),
    )


def _frame_key(question: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(question.get("_dataset") or ""),
        str(question.get("scene_id") or ""),
        str(question.get("image_name") or ""),
        str(question.get("type") or ""),
    )


def _scene_key(question: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(question.get("_dataset") or ""),
        str(question.get("scene_id") or ""),
        str(question.get("type") or ""),
    )


def _is_target_type(question: dict[str, Any]) -> bool:
    return str(question.get("type") or "") in TARGET_TYPES


def _is_valid_candidate(question: dict[str, Any]) -> bool:
    if str(question.get("level") or "") != "L2":
        return False
    qtype = str(question.get("type") or "")
    if qtype not in TARGET_TYPES:
        return False
    if not bool(question.get("attachment_remapped", False)):
        return False
    moved_obj_id = _coerce_int(question.get("moved_obj_id"))
    query_obj_id = _coerce_int(question.get("query_obj_id"))
    return moved_obj_id is not None and query_obj_id is not None and moved_obj_id != query_obj_id


def _is_invalid_self_query(question: dict[str, Any]) -> bool:
    if str(question.get("level") or "") != "L2":
        return False
    qtype = str(question.get("type") or "")
    if qtype not in TARGET_TYPES:
        return False
    if not bool(question.get("attachment_remapped", False)):
        return False
    moved_obj_id = _coerce_int(question.get("moved_obj_id"))
    query_obj_id = _coerce_int(question.get("query_obj_id"))
    return moved_obj_id is not None and moved_obj_id == query_obj_id


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_image_map(html_path: Path) -> dict[tuple[str, str], str]:
    cards = parse_viewer_html(html_path.read_text(encoding="utf-8"), include_deleted=True)
    image_map: dict[tuple[str, str], str] = {}
    for card in cards:
        key = (str(card["scene_id"]), str(card["image_name"]))
        image_map.setdefault(key, str(card["image_data_url"]))
    return image_map


def _build_image_html_from_map(question: dict[str, Any], image_map: dict[tuple[str, str], str]) -> str:
    key = (str(question.get("scene_id") or ""), str(question.get("image_name") or ""))
    image_data_url = image_map.get(key)
    if not image_data_url:
        return '<div class="no-img">Image not available</div>'
    return f'<img src="{html.escape(image_data_url, quote=True)}" alt="question image">'


def _build_full_html(
    questions: list[dict[str, Any]],
    image_map: dict[tuple[str, str], str],
    *,
    edited_html_filename: str,
) -> str:
    cards: list[str] = []
    for idx, question in enumerate(questions, start=1):
        cards.append(
            make_viewer.CARD.format(
                img=_build_image_html_from_map(question, image_map),
                meta=make_viewer._build_meta_html(question, idx),
                question=html.escape(str(question.get("question", ""))),
                options=make_viewer._build_options_html(question),
                review_notes=make_viewer._build_review_notes_html(
                    question,
                    include_referability_audit=False,
                ),
                footer=make_viewer._build_footer_html(question),
            )
        )

    return make_viewer.PAGE.format(
        title=html.escape("predictive spatial reasoning benchmark"),
        stats=make_viewer.build_stats_bar(questions),
        summary=make_viewer._build_summary_html(questions),
        cards="\n".join(cards),
        edited_html_filename_json=json.dumps(edited_html_filename),
    )


def _build_simple_html(
    questions: list[dict[str, Any]],
    image_map: dict[tuple[str, str], str],
    *,
    edited_html_filename: str,
) -> str:
    cards: list[str] = []
    for idx, question in enumerate(questions, start=1):
        objects, relations = make_viewer._build_simple_sections(question)
        cards.append(
            make_viewer.SIMPLE_CARD.format(
                img=_build_image_html_from_map(question, image_map),
                meta=make_viewer._build_meta_html(question, idx),
                question=(
                    f'<p class="qtext">{html.escape(str(question.get("question", "")))}</p>'
                    if str(question.get("question", "")).strip()
                    else ""
                ),
                options=make_viewer._build_options_html(question),
                objects=make_viewer._render_simple_section("Objects", objects),
                relations=make_viewer._render_simple_section("Relations", relations),
                footer=make_viewer._build_footer_html(question),
            )
        )

    return make_viewer.PAGE.format(
        title=html.escape("predictive spatial reasoning benchmark (simple review)"),
        stats=make_viewer.build_stats_bar(questions),
        summary=make_viewer._build_summary_html(questions),
        cards="\n".join(cards),
        edited_html_filename_json=json.dumps(edited_html_filename),
    )


def _sorted_candidates(questions: list[dict[str, Any]], *, qtype: str) -> list[dict[str, Any]]:
    candidates = [q for q in questions if _is_valid_candidate(q) and str(q.get("type") or "") == qtype]
    candidates.sort(
        key=lambda q: (
            bool(q.get("relation_unchanged", False)),
            str(q.get("_dataset") or ""),
            str(q.get("scene_id") or ""),
            str(q.get("image_name") or ""),
            str(q.get("question_uid") or _question_uid(q)),
        )
    )
    return candidates


def _clone_for_subset(question: dict[str, Any], *, replacement_source_path: str) -> dict[str, Any]:
    cloned = dict(question)
    cloned["attachment_child_replaced"] = True
    cloned["attachment_child_replacement_source_benchmark"] = replacement_source_path
    cloned["question_uid"] = _question_uid(cloned)
    return cloned


def _sample_strict_type(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    replacement_source_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    frame_counts: Counter[tuple[str, str, str, str]] = Counter()
    scene_counts: Counter[tuple[str, str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    skipped_caps: Counter[str] = Counter()

    for candidate in candidates:
        object_key = _object_key(candidate)
        frame_key = _frame_key(candidate)
        scene_key = _scene_key(candidate)
        if object_counts[object_key] >= 1:
            skipped_caps["object"] += 1
            continue
        if frame_counts[frame_key] >= 2:
            skipped_caps["frame"] += 1
            continue
        if scene_counts[scene_key] >= 8:
            skipped_caps["scene"] += 1
            continue
        selected.append(_clone_for_subset(candidate, replacement_source_path=replacement_source_path))
        object_counts[object_key] += 1
        frame_counts[frame_key] += 1
        scene_counts[scene_key] += 1
        if len(selected) >= target_count:
            break

    return selected, {
        "selected_count": len(selected),
        "target_count": target_count,
        "insufficient": len(selected) < target_count,
        "skipped_due_to_caps": dict(skipped_caps),
        "changed_count": sum(not bool(item.get("relation_unchanged", False)) for item in selected),
    }


def _sample_rotate_type(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    replacement_source_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stages = (
        ("strict", 2, 8),
        ("relax_frame", None, 8),
        ("relax_frame_scene", None, None),
    )
    selected_by_uid: dict[str, dict[str, Any]] = {}
    object_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    stage_additions: Counter[str] = Counter()

    for stage_name, frame_cap, scene_cap in stages:
        frame_counts: Counter[tuple[str, str, str, str]] = Counter(_frame_key(q) for q in selected_by_uid.values())
        scene_counts: Counter[tuple[str, str, str]] = Counter(_scene_key(q) for q in selected_by_uid.values())
        for candidate in candidates:
            uid = str(candidate.get("question_uid") or _question_uid(candidate))
            if uid in selected_by_uid:
                continue
            object_key = _object_key(candidate)
            frame_key = _frame_key(candidate)
            scene_key = _scene_key(candidate)
            if object_counts[object_key] >= 1:
                continue
            if frame_cap is not None and frame_counts[frame_key] >= frame_cap:
                continue
            if scene_cap is not None and scene_counts[scene_key] >= scene_cap:
                continue
            cloned = _clone_for_subset(candidate, replacement_source_path=replacement_source_path)
            selected_by_uid[uid] = cloned
            object_counts[object_key] += 1
            frame_counts[frame_key] += 1
            scene_counts[scene_key] += 1
            stage_additions[stage_name] += 1
            if len(selected_by_uid) >= target_count:
                break
        if len(selected_by_uid) >= target_count:
            break

    selected = list(selected_by_uid.values())
    selected.sort(
        key=lambda q: (
            bool(q.get("relation_unchanged", False)),
            str(q.get("_dataset") or ""),
            str(q.get("scene_id") or ""),
            str(q.get("image_name") or ""),
            str(q.get("question_uid") or ""),
        )
    )
    return selected, {
        "selected_count": len(selected),
        "target_count": target_count,
        "insufficient": len(selected) < target_count,
        "changed_count": sum(not bool(item.get("relation_unchanged", False)) for item in selected),
        "relaxation_additions": dict(stage_additions),
    }


def _resample_target_types(
    subset_questions: list[dict[str, Any]],
    source_questions: list[dict[str, Any]],
    *,
    replacement_source_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_counts = Counter(
        str(question.get("type") or "")
        for question in subset_questions
        if _is_target_type(question)
    )

    selected_by_type: dict[str, list[dict[str, Any]]] = {}
    sampling_report: dict[str, Any] = {}

    for qtype in STRICT_CAP_TYPES:
        candidates = _sorted_candidates(source_questions, qtype=qtype)
        selected, type_report = _sample_strict_type(
            candidates,
            target_count=int(target_counts[qtype]),
            replacement_source_path=replacement_source_path,
        )
        selected_by_type[qtype] = selected
        sampling_report[qtype] = {
            "candidate_count": len(candidates),
            **type_report,
        }

    rotate_candidates = _sorted_candidates(source_questions, qtype=ROTATE_TYPE)
    rotate_selected, rotate_report = _sample_rotate_type(
        rotate_candidates,
        target_count=int(target_counts[ROTATE_TYPE]),
        replacement_source_path=replacement_source_path,
    )
    selected_by_type[ROTATE_TYPE] = rotate_selected
    sampling_report[ROTATE_TYPE] = {
        "candidate_count": len(rotate_candidates),
        **rotate_report,
    }

    updated_questions: list[dict[str, Any]] = []
    inserted_type_counts: Counter[str] = Counter()
    for question in subset_questions:
        qtype = str(question.get("type") or "")
        if qtype not in TARGET_TYPES:
            updated_questions.append(question)
            continue
        idx = inserted_type_counts[qtype]
        selected_items = selected_by_type[qtype]
        if idx >= len(selected_items):
            continue
        updated_questions.append(selected_items[idx])
        inserted_type_counts[qtype] += 1

    dropped_counts = {
        qtype: int(target_counts[qtype]) - int(inserted_type_counts[qtype])
        for qtype in TARGET_TYPES
    }
    report = {
        "target_types": list(TARGET_TYPES),
        "target_counts": {qtype: int(target_counts[qtype]) for qtype in TARGET_TYPES},
        "inserted_counts": {qtype: int(inserted_type_counts[qtype]) for qtype in TARGET_TYPES},
        "dropped_counts": dropped_counts,
        "sampling": sampling_report,
        "resampled_total": sum(inserted_type_counts.values()),
    }
    return updated_questions, report


def _scene_type_ratio_stats(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        if not _is_target_type(question):
            continue
        if str(question.get("level") or "") != "L2":
            continue
        if not bool(question.get("attachment_remapped", False)):
            continue
        grouped[
            (
                str(question.get("_dataset") or ""),
                str(question.get("scene_id") or ""),
                str(question.get("type") or ""),
            )
        ].append(question)

    rows: list[dict[str, Any]] = []
    for (dataset, scene_id, qtype), items in sorted(grouped.items()):
        total = len(items)
        changed = sum(not bool(item.get("relation_unchanged", False)) for item in items)
        rows.append(
            {
                "dataset": dataset,
                "scene_id": scene_id,
                "type": qtype,
                "count": total,
                "with_attachment_ratio": 1.0 if total else 0.0,
                "answer_changed_ratio": 0.0 if total == 0 else changed / total,
                "meets_with_attachment_rule": True,
                "meets_answer_changed_rule": total == 0 or (changed / total) >= 0.80,
            }
        )
    return rows


def _constraint_violations(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    object_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    frame_counts: Counter[tuple[str, str, str, str]] = Counter()
    scene_counts: Counter[tuple[str, str, str]] = Counter()

    strict_questions = [
        q
        for q in questions
        if str(q.get("type") or "") in STRICT_CAP_TYPES and str(q.get("level") or "") == "L2"
    ]
    rotate_questions = [
        q
        for q in questions
        if str(q.get("type") or "") == ROTATE_TYPE and str(q.get("level") or "") == "L2"
    ]

    for question in strict_questions:
        object_counts[_object_key(question)] += 1
        frame_counts[_frame_key(question)] += 1
        scene_counts[_scene_key(question)] += 1

    rotate_object_counts: Counter[tuple[str, str, str, str, str]] = Counter(_object_key(q) for q in rotate_questions)
    rotate_frame_counts: Counter[tuple[str, str, str, str]] = Counter(_frame_key(q) for q in rotate_questions)
    rotate_scene_counts: Counter[tuple[str, str, str]] = Counter(_scene_key(q) for q in rotate_questions)

    return {
        "strict_object_cap_violations": [
            {"key": list(key), "count": count} for key, count in object_counts.items() if count > 1
        ],
        "strict_frame_cap_violations": [
            {"key": list(key), "count": count} for key, count in frame_counts.items() if count > 2
        ],
        "strict_scene_cap_violations": [
            {"key": list(key), "count": count} for key, count in scene_counts.items() if count > 8
        ],
        "rotate_object_cap_violations": [
            {"key": list(key), "count": count} for key, count in rotate_object_counts.items() if count > 1
        ],
        "rotate_frame_over_2": [
            {"key": list(key), "count": count} for key, count in rotate_frame_counts.items() if count > 2
        ],
        "rotate_scene_over_8": [
            {"key": list(key), "count": count} for key, count in rotate_scene_counts.items() if count > 8
        ],
    }


def _validate_final_state(original_questions: list[dict[str, Any]], updated_questions: list[dict[str, Any]]) -> dict[str, Any]:
    original_type_counts = Counter(question.get("type") for question in original_questions)
    updated_type_counts = Counter(question.get("type") for question in updated_questions)
    remaining_invalid = [question for question in updated_questions if _is_invalid_self_query(question)]
    target_validity_failures = [
        {
            "type": question.get("type"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "trace_question_id": question.get("trace_question_id"),
        }
        for question in updated_questions
        if _is_target_type(question) and not _is_valid_candidate(question)
    ]
    ratio_stats = _scene_type_ratio_stats(updated_questions)
    violations = _constraint_violations(updated_questions)
    return {
        "original_type_counts": dict(original_type_counts),
        "updated_type_counts": dict(updated_type_counts),
        "remaining_invalid_count": len(remaining_invalid),
        "remaining_invalid_by_type": dict(Counter(question.get("type") for question in remaining_invalid)),
        "target_validity_failure_count": len(target_validity_failures),
        "target_validity_failures_sample": target_validity_failures[:50],
        "scene_type_ratio_stats": ratio_stats,
        "scene_type_ratio_failures": [
            row
            for row in ratio_stats
            if not row["meets_with_attachment_rule"] or not row["meets_answer_changed_rule"]
        ],
        **violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace attachment child questions in benchmark subset")
    parser.add_argument("--subset", default="output/benchmark_subset.json")
    parser.add_argument("--source", default="output/benchmark.attachment_child_recomputed.json")
    parser.add_argument("--bench_html", default="output/bench.html")
    parser.add_argument("--bench_simple_html", default="output/bench_simple.html")
    parser.add_argument(
        "--report_out",
        default="output/benchmark_subset.attachment_child_replaced_report.json",
    )
    args = parser.parse_args()

    subset_path = PROJECT_ROOT / args.subset
    source_path = PROJECT_ROOT / args.source
    bench_html_path = PROJECT_ROOT / args.bench_html
    bench_simple_html_path = PROJECT_ROOT / args.bench_simple_html
    report_out_path = PROJECT_ROOT / args.report_out

    subset_payload = _load_json(subset_path)
    source_payload = _load_json(source_path)

    subset_questions = list(subset_payload.get("questions", []))
    source_questions = list(source_payload.get("questions", []))

    updated_questions, replacement_report = _resample_target_types(
        subset_questions,
        source_questions,
        replacement_source_path=str(source_path),
    )

    validation = _validate_final_state(subset_questions, updated_questions)
    image_map = _load_image_map(bench_html_path)
    missing_frames = sorted(
        {
            (str(question.get("scene_id")), str(question.get("image_name")))
            for question in updated_questions
            if (str(question.get("scene_id")), str(question.get("image_name"))) not in image_map
        }
    )

    metadata = dict(subset_payload.get("metadata", {}))
    metadata["attachment_child_replacement"] = {
        "source": str(source_path),
        "report_path": str(report_out_path),
        "resampled_total": replacement_report["resampled_total"],
        "remaining_invalid_count": validation["remaining_invalid_count"],
        "missing_image_frame_count": len(missing_frames),
    }

    updated_payload = dict(subset_payload)
    updated_payload["metadata"] = metadata
    updated_payload["questions"] = updated_questions

    full_html = _build_full_html(
        updated_questions,
        image_map,
        edited_html_filename="bench_edited.html",
    )
    simple_html = _build_simple_html(
        updated_questions,
        image_map,
        edited_html_filename="bench_simple_edited.html",
    )

    _write_json(subset_path, updated_payload)
    bench_html_path.write_text(full_html, encoding="utf-8")
    bench_simple_html_path.write_text(simple_html, encoding="utf-8")

    final_report = {
        **replacement_report,
        "validation": validation,
        "missing_image_frame_count": len(missing_frames),
        "missing_image_frames_sample": missing_frames[:50],
    }
    _write_json(report_out_path, final_report)

    print(
        json.dumps(
            {
                "subset_path": str(subset_path),
                "bench_html_path": str(bench_html_path),
                "bench_simple_html_path": str(bench_simple_html_path),
                "report_path": str(report_out_path),
                "resampled_total": replacement_report["resampled_total"],
                "remaining_invalid_count": validation["remaining_invalid_count"],
                "remaining_invalid_by_type": validation["remaining_invalid_by_type"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
