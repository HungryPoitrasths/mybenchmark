#!/usr/bin/env python3
"""Fix invalid self-query attachment move questions in a benchmark subset."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any


QUESTION_UID_FIELDS = (
    "scene_id",
    "image_name",
    "level",
    "type",
    "question",
    "options",
    "answer",
)

STABLE_FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "object_move_agent": ("obj_c_id",),
    "object_move_distance": ("obj_c_id",),
    "object_move_occlusion": ("target_obj_id",),
    "object_move_object_centric": ("obj_ref_id",),
    "object_move_allocentric": ("obj_ref_id",),
}

SKIP_MISSING_CHILD = "missing_attachment_child_id"
SKIP_MISSING_SOURCE = "missing_source_benchmark"
SKIP_SOURCE_NOT_READABLE = "source_not_readable"
SKIP_NO_MATCH = "no_exact_child_match"
SKIP_AMBIGUOUS = "ambiguous_child_match"


def _json_key(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _question_uid(question: dict[str, Any]) -> str:
    payload = {"dataset": question.get("_dataset")}
    for field in QUESTION_UID_FIELDS:
        payload[field] = question.get(field)
    return _json_key(payload)


def _stable_rank(seed: int, key: str) -> int:
    return zlib.crc32(f"{seed}|{key}".encode("utf-8")) & 0xFFFFFFFF


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_seed(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _benchmark_questions(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        questions = data.get("questions", [])
    else:
        questions = data
    if not isinstance(questions, list):
        raise ValueError("Unsupported benchmark structure: expected a list of questions.")
    return [question for question in questions if isinstance(question, dict)]


def _is_invalid_self_attachment_query(question: dict[str, Any]) -> bool:
    if str(question.get("level", "")).strip() != "L2":
        return False
    if not str(question.get("type", "")).startswith("object_move"):
        return False
    if not bool(question.get("attachment_remapped", False)):
        return False
    moved_obj_id = _coerce_int(question.get("moved_obj_id"))
    query_obj_id = _coerce_int(question.get("query_obj_id"))
    return moved_obj_id is not None and moved_obj_id == query_obj_id


def _base_candidate_match(question: dict[str, Any], candidate: dict[str, Any], child_id: int) -> bool:
    if str(candidate.get("scene_id", "")) != str(question.get("scene_id", "")):
        return False
    if str(candidate.get("image_name", "")) != str(question.get("image_name", "")):
        return False
    if str(candidate.get("type", "")) != str(question.get("type", "")):
        return False
    if _coerce_int(candidate.get("moved_obj_id")) != _coerce_int(question.get("moved_obj_id")):
        return False
    if _coerce_int(candidate.get("query_obj_id")) != child_id:
        return False
    return tuple(candidate.get("delta") or []) == tuple(question.get("delta") or [])


def _stable_candidate_match(question: dict[str, Any], candidate: dict[str, Any]) -> bool:
    stable_fields = STABLE_FIELDS_BY_TYPE.get(str(question.get("type", "")), ())
    return all(question.get(field) == candidate.get(field) for field in stable_fields)


def _replacement_candidate(
    question: dict[str, Any],
    source_questions: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    child_id = _coerce_int(question.get("attachment_child_id"))
    if child_id is None:
        return None, SKIP_MISSING_CHILD

    matching_children = [
        candidate
        for candidate in source_questions
        if _base_candidate_match(question, candidate, child_id)
    ]
    exact_matches = [
        candidate for candidate in matching_children if _stable_candidate_match(question, candidate)
    ]
    if len(exact_matches) == 1:
        return exact_matches[0], None
    if len(exact_matches) > 1:
        return None, SKIP_AMBIGUOUS
    return None, SKIP_NO_MATCH


def _skip_entry(question: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "type": question.get("type"),
        "trace_question_id": question.get("trace_question_id"),
        "moved_obj_id": question.get("moved_obj_id"),
        "attachment_child_id": question.get("attachment_child_id"),
        "skip_reason": reason,
    }


def _apply_replacement(
    original: dict[str, Any],
    replacement: dict[str, Any],
    *,
    seed: int | None,
) -> dict[str, Any]:
    fixed = dict(original)
    fixed.update(replacement)

    moved_obj_id = _coerce_int(original.get("moved_obj_id"))
    child_id = _coerce_int(original.get("attachment_child_id"))
    if moved_obj_id is not None:
        fixed["attachment_parent_id"] = moved_obj_id
    if child_id is not None:
        fixed["attachment_child_id"] = child_id
        if moved_obj_id is not None:
            fixed["attachment_pair_id"] = f"{moved_obj_id}->{child_id}"

    fixed["attachment_remapped"] = True
    fixed["has_attachment_chain"] = True
    fixed["_dataset"] = original.get("_dataset", fixed.get("_dataset"))
    fixed["_source_benchmark"] = original.get("_source_benchmark", fixed.get("_source_benchmark"))
    fixed["question_uid"] = _question_uid(fixed)

    if seed is not None:
        fixed["_rank"] = _stable_rank(seed, fixed["question_uid"])
    elif "_rank" in original:
        fixed["_rank"] = original["_rank"]

    return fixed


def fix_attachment_child_queries(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = _benchmark_questions(payload)
    source_cache: dict[Path, tuple[list[dict[str, Any]] | None, str | None]] = {}
    seed = _coerce_seed(payload.get("metadata", {}).get("seed"))

    fixed_questions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    target_count = 0
    fixed_count = 0

    for question in questions:
        if not _is_invalid_self_attachment_query(question):
            fixed_questions.append(question)
            continue

        target_count += 1
        child_id = _coerce_int(question.get("attachment_child_id"))
        if child_id is None:
            fixed_questions.append(question)
            skipped.append(_skip_entry(question, SKIP_MISSING_CHILD))
            continue

        source_path_text = str(question.get("_source_benchmark", "")).strip()
        if not source_path_text:
            fixed_questions.append(question)
            skipped.append(_skip_entry(question, SKIP_MISSING_SOURCE))
            continue

        source_path = Path(source_path_text)
        if source_path not in source_cache:
            try:
                source_cache[source_path] = (_benchmark_questions(_load_json(source_path)), None)
            except (OSError, ValueError, json.JSONDecodeError):
                source_cache[source_path] = (None, SKIP_SOURCE_NOT_READABLE)

        source_questions, source_error = source_cache[source_path]
        if source_questions is None:
            fixed_questions.append(question)
            skipped.append(_skip_entry(question, source_error or SKIP_SOURCE_NOT_READABLE))
            continue

        replacement, skip_reason = _replacement_candidate(question, source_questions)
        if replacement is None:
            fixed_questions.append(question)
            skipped.append(_skip_entry(question, skip_reason or SKIP_NO_MATCH))
            continue

        fixed_questions.append(_apply_replacement(question, replacement, seed=seed))
        fixed_count += 1

    output_payload = dict(payload)
    output_payload["questions"] = fixed_questions
    report = {
        "target_count": target_count,
        "fixed_count": fixed_count,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    return output_payload, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix L2 attachment-remapped object_move questions whose query object incorrectly equals the moved object.",
    )
    parser.add_argument("--input", default="output/benchmark_subset.json", help="Input benchmark JSON")
    parser.add_argument(
        "--output",
        default="output/benchmark_subset.attachment_child_fixed.json",
        help="Output fixed benchmark JSON",
    )
    parser.add_argument(
        "--report",
        default="output/benchmark_subset.attachment_child_fix_report.json",
        help="Output JSON report for skipped rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    payload_data = _load_json(input_path)
    payload = payload_data if isinstance(payload_data, dict) else {"questions": _benchmark_questions(payload_data)}
    fixed_payload, report = fix_attachment_child_queries(payload)

    metadata = dict(fixed_payload.get("metadata", {}))
    postprocess = dict(metadata.get("postprocess", {})) if isinstance(metadata.get("postprocess"), dict) else {}
    postprocess["self_attachment_query_fix"] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "target_count": report["target_count"],
        "fixed_count": report["fixed_count"],
        "skipped_count": report["skipped_count"],
    }
    metadata["postprocess"] = postprocess
    fixed_payload["metadata"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(fixed_payload, f, ensure_ascii=False, indent=2)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"target questions: {report['target_count']}")
    print(f"fixed questions : {report['fixed_count']}")
    print(f"skipped questions: {report['skipped_count']}")
    print(f"output json     : {output_path}")
    print(f"report json     : {report_path}")


if __name__ == "__main__":
    main()
