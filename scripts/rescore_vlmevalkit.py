#!/usr/bin/env python3
"""Rescore saved VLM evaluation JSON files from raw model responses.

This is an offline repair utility for result files produced by
``run_sampled_type_vlm_eval.py``. It re-parses answer letters from
``raw_response`` and rewrites ``prediction``, ``correct``, and ``summary``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


QTYPE_ORDER = [
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


def qtype_sort_key(qtype: str) -> tuple[int, str]:
    try:
        return (QTYPE_ORDER.index(qtype), qtype)
    except ValueError:
        return (len(QTYPE_ORDER), qtype)


def allowed_letters(row: dict[str, Any]) -> str:
    options = row.get("options") or []
    if isinstance(options, list) and options:
        return "".join(chr(65 + idx) for idx in range(min(len(options), 26)))
    for key in ("gt_answer", "prediction", "answer"):
        value = row.get(key)
        if isinstance(value, str):
            letters = re.findall(r"[A-Z]", value.upper())
            if letters:
                highest = max(ord(letter) for letter in letters)
                return "".join(chr(code) for code in range(ord("A"), min(highest, ord("Z")) + 1))
    return "ABCD"


def ordered_unique_letters(values: list[str], letters: str) -> list[str]:
    allowed = letters.upper()
    seen = {value.upper() for value in values if value and value.upper() in allowed}
    return [letter for letter in allowed if letter in seen]


def parse_answers(raw: str | None, letters: str) -> list[str]:
    if not raw:
        return []
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()
    answer_line_patterns = [
        rf"(?:FINAL\s+)?ANSWER\s*[:：]\s*([^\r\n]+)",
        rf"(?:CHOICES?|OPTIONS?)\s*[:：]?\s*([^\r\n]+)",
        rf"答案\s*[:：]?\s*([^\r\n]+)",
    ]

    candidates: list[str] = []
    for pattern in answer_line_patterns:
        match = re.search(pattern, upper)
        if match:
            candidates.append(match.group(1))
            break
    candidates.append(upper)

    for candidate in candidates:
        tokens = re.findall(rf"(?<![A-Z0-9])([{allowed}])(?![A-Z0-9])", candidate)
        if tokens:
            return ordered_unique_letters(tokens, letters)
        compact = re.sub(r"[\s,;/&+|，、\-]+", "", candidate)
        if compact and re.fullmatch(rf"[{allowed}]+", compact):
            return ordered_unique_letters(list(compact), letters)
    return []


def parse_answer(raw: str | None, letters: str) -> str | None:
    if not raw:
        return None
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()
    if re.fullmatch(rf"[{allowed}]", upper):
        return upper

    patterns = [
        rf"(?:FINAL\s+)?ANSWER\s*[:：]\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"(?:CHOICE|OPTION)\s*[:：]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"答案\s*[:：]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^[\(\[]?\s*([{allowed}])\s*[\)\].:：]",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def normalize_answer_letters(value: Any, letters: str, *, multi_select: bool) -> list[str]:
    if isinstance(value, list):
        return ordered_unique_letters([str(item).strip().upper() for item in value], letters)
    text = str(value or "").strip()
    if not text:
        return []
    if multi_select:
        return parse_answers(text, letters)
    parsed = parse_answer(text, letters)
    return [parsed] if parsed else []


def is_multi_select(row: dict[str, Any]) -> bool:
    return bool(row.get("multi_select")) or isinstance(row.get("gt_answer"), list) or isinstance(row.get("answer"), list)


def raw_response_for(row: dict[str, Any]) -> str | None:
    for key in ("raw_response", "model_reasoning", "response", "output", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def ground_truth_for(row: dict[str, Any], letters: str, *, multi_select: bool) -> list[str]:
    for key in ("gt_answers", "gt_answer", "answer"):
        if key in row:
            values = normalize_answer_letters(row.get(key), letters, multi_select=multi_select)
            if values:
                return values
    return []


def rescore_row(row: dict[str, Any], *, keep_existing_prediction: bool) -> tuple[dict[str, Any], bool]:
    updated = dict(row)
    letters = allowed_letters(updated)
    multi = is_multi_select(updated)
    gt_answers = ground_truth_for(updated, letters, multi_select=multi)
    raw_response = raw_response_for(updated)

    parsed = parse_answers(raw_response, letters) if multi else []
    prediction = ",".join(parsed) if multi else parse_answer(raw_response, letters)

    if keep_existing_prediction and not prediction and updated.get("prediction"):
        prediction = str(updated.get("prediction"))
        parsed = normalize_answer_letters(prediction, letters, multi_select=multi)

    if multi:
        prediction_values = parsed or normalize_answer_letters(prediction, letters, multi_select=True)
        correct = bool(prediction_values and gt_answers and set(prediction_values) == set(gt_answers))
        updated["prediction"] = ",".join(prediction_values) if prediction_values else None
        updated["predictions"] = prediction_values
        updated["gt_answers"] = gt_answers
        updated["gt_answer"] = ",".join(gt_answers)
    else:
        correct = bool(prediction and gt_answers and prediction == gt_answers[0])
        updated["prediction"] = prediction
        if gt_answers:
            updated["gt_answer"] = gt_answers[0]

    updated["correct"] = correct
    changed = (
        row.get("prediction") != updated.get("prediction")
        or bool(row.get("correct")) != correct
        or row.get("gt_answer") != updated.get("gt_answer")
    )
    return updated, changed


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for row in results:
        qtype = str(row.get("type") or "unknown")
        stats = by_type.setdefault(
            qtype,
            {"total": 0, "answered": 0, "correct": 0, "errors": 0, "missing_images": 0},
        )
        stats["total"] += 1
        if row.get("prediction"):
            stats["answered"] += 1
        if row.get("correct"):
            stats["correct"] += 1
        if row.get("error"):
            stats["errors"] += 1
        if row.get("error") == "image_not_found":
            stats["missing_images"] += 1

    for stats in by_type.values():
        total = int(stats["total"])
        answered = int(stats["answered"])
        stats["accuracy"] = (float(stats["correct"]) / total) if total else None
        stats["answered_accuracy"] = (float(stats["correct"]) / answered) if answered else None

    ordered = {qtype: by_type[qtype] for qtype in sorted(by_type, key=qtype_sort_key)}
    return {"by_type": ordered}


def load_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported result JSON structure: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Result rows must be objects: {path}")
    return rows


def write_payload(path: Path, payload: Any, rows: list[dict[str, Any]], *, summary: dict[str, Any]) -> None:
    if isinstance(payload, dict):
        payload["results"] = rows
        payload["summary"] = summary
        metadata = payload.setdefault("metadata", {})
        metadata["rescored_by"] = "scripts/rescore_vlmevalkit.py"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def rescore_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    rows = load_rows(payload, path)
    rescored: list[dict[str, Any]] = []
    changed = 0
    answer_counts: Counter[str] = Counter()
    for row in rows:
        updated, row_changed = rescore_row(
            row,
            keep_existing_prediction=args.keep_existing_prediction,
        )
        rescored.append(updated)
        changed += int(row_changed)
        answer_counts[str(updated.get("prediction") or "-")] += 1

    summary = summarize_results(rescored)
    if not args.dry_run:
        write_payload(path, payload, rescored, summary=summary)

    return {
        "path": path,
        "total": len(rescored),
        "changed": changed,
        "summary": summary,
        "answer_counts": answer_counts,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-rescore VLM eval result JSON files from raw_response answer letters."
    )
    parser.add_argument("json_files", nargs="+", help="Result JSON file(s) to rescore in place")
    parser.add_argument("--no_llm", action="store_true", help="Accepted for compatibility; rescoring is always offline")
    parser.add_argument("--dry_run", action="store_true", help="Print what would change without writing files")
    parser.add_argument(
        "--keep_existing_prediction",
        action="store_true",
        help="If raw_response cannot be parsed, keep a non-empty existing prediction",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for json_file in args.json_files:
        path = Path(json_file)
        stats = rescore_file(path, args)
        print(f"{stats['path']}")
        print(f"  rows changed: {stats['changed']}/{stats['total']}")
        for qtype, qstats in stats["summary"]["by_type"].items():
            print(
                f"  {qtype:36s} {qstats['correct']:4d}/{qstats['total']:<4d} "
                f"acc={fmt_pct(qstats['accuracy']):>6s} answered={qstats['answered']}"
            )


if __name__ == "__main__":
    main()
