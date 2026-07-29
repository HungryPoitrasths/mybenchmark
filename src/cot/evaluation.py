from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from .sampling import SUPPORTED_TYPE_ORDER, TYPES_BY_LEVEL


STRICT_ANSWER_RE = re.compile(r"^Answer: ([A-Z](?: [A-Z])*)$")
RELAXED_ANSWER_RE = re.compile(
    r"\b(?:answer|option)\s*(?::|is)?\s*([A-D](?:[\s,]+[A-D])*)\b",
    re.IGNORECASE,
)
BARE_ANSWER_RE = re.compile(r"^([A-D](?:[\s,]+[A-D])*)$", re.IGNORECASE)


def _normalize_letters(value: str) -> str:
    return " ".join(re.findall(r"[A-Z]", value.upper()))


def parse_strict_answer(
    response: str,
    *,
    option_count: int,
    multi_select: bool,
) -> str | None:
    if response.count("Answer:") != 1:
        return None
    lines = response.rstrip().splitlines()
    if not lines:
        return None
    match = STRICT_ANSWER_RE.fullmatch(lines[-1])
    if match is None:
        return None
    letters = match.group(1).split()
    if not multi_select and len(letters) != 1:
        return None
    if len(set(letters)) != len(letters):
        return None
    if option_count <= 0:
        return None
    if any(ord(letter) - ord("A") >= option_count for letter in letters):
        return None
    return " ".join(letters)


def parse_relaxed_answer(
    response: str,
    *,
    option_count: int,
    multi_select: bool,
) -> str | None:
    strict = parse_strict_answer(
        response,
        option_count=option_count,
        multi_select=multi_select,
    )
    if strict is not None:
        return strict
    candidates = list(RELAXED_ANSWER_RE.finditer(response[-500:]))
    bare = BARE_ANSWER_RE.fullmatch(response.strip())
    raw = candidates[-1].group(1) if candidates else bare.group(1) if bare else ""
    letters = re.findall(r"[A-D]", raw.upper())
    if not letters or (not multi_select and len(letters) != 1):
        return None
    if len(set(letters)) != len(letters):
        return None
    if any(ord(letter) - ord("A") >= option_count for letter in letters):
        return None
    return " ".join(letters)


def prediction_response(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "output", "generated_text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    strict_correct = sum(bool(row["strict_correct"]) for row in rows)
    relaxed_correct = sum(bool(row["relaxed_correct"]) for row in rows)
    format_success = sum(bool(row["format_success"]) for row in rows)
    return {
        "total": total,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / total if total else 0.0,
        "relaxed_correct": relaxed_correct,
        "relaxed_accuracy": relaxed_correct / total if total else 0.0,
        "format_success": format_success,
        "format_success_rate": format_success / total if total else 0.0,
    }


def evaluate_predictions(
    sidecar: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions_by_uid = {
        str(row.get("question_uid")): row
        for row in predictions
        if str(row.get("question_uid") or "").strip()
    }
    use_uid = bool(predictions_by_uid)
    evaluated: list[dict[str, Any]] = []
    for index, gold in enumerate(sidecar):
        uid = str(gold.get("question_uid") or "")
        prediction = predictions_by_uid.get(uid, {}) if use_uid else (
            predictions[index] if index < len(predictions) else {}
        )
        response = prediction_response(prediction)
        option_count = int(gold.get("option_count") or 0)
        multi_select = bool(gold.get("multi_select"))
        strict_answer = parse_strict_answer(
            response,
            option_count=option_count,
            multi_select=multi_select,
        )
        relaxed_answer = parse_relaxed_answer(
            response,
            option_count=option_count,
            multi_select=multi_select,
        )
        expected = " ".join(str(value) for value in gold.get("answer_letters") or [])
        facts = gold.get("facts") if isinstance(gold.get("facts"), dict) else {}
        evaluated.append(
            {
                "question_uid": uid,
                "question_type": str(gold.get("question_type") or ""),
                "level": str(facts.get("level") or ""),
                "expected": expected,
                "strict_answer": strict_answer,
                "relaxed_answer": relaxed_answer,
                "format_success": strict_answer is not None,
                "strict_correct": strict_answer == expected,
                "relaxed_correct": relaxed_answer == expected,
                "response": response,
                "generated_words": len(response.split()),
            }
        )

    by_type_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_level_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_type_rows[row["question_type"]].append(row)
        by_level_rows[row["level"]].append(row)

    by_type = {
        question_type: _summarize(by_type_rows.get(question_type, []))
        for question_type in SUPPORTED_TYPE_ORDER
    }
    by_level = {
        level: _summarize(by_level_rows.get(level, []))
        for level in TYPES_BY_LEVEL
    }
    present_type_accuracies = [
        stats["strict_accuracy"] for stats in by_type.values() if stats["total"] > 0
    ]
    present_type_relaxed = [
        stats["relaxed_accuracy"] for stats in by_type.values() if stats["total"] > 0
    ]
    lengths = [int(row["generated_words"]) for row in evaluated]
    overall = _summarize(evaluated)
    overall.update(
        macro_accuracy=(
            sum(present_type_accuracies) / len(present_type_accuracies)
            if present_type_accuracies
            else 0.0
        ),
        relaxed_macro_accuracy=(
            sum(present_type_relaxed) / len(present_type_relaxed)
            if present_type_relaxed
            else 0.0
        ),
        mean_generated_words=sum(lengths) / len(lengths) if lengths else 0.0,
        p95_generated_words=_percentile(lengths, 0.95),
    )
    return {
        "overall": overall,
        "by_level": by_level,
        "by_type": by_type,
        "missing_prediction_count": sum(not row["response"] for row in evaluated),
        "missing_supported_types": [
            question_type
            for question_type, stats in by_type.items()
            if stats["total"] == 0
        ],
        "evaluated": evaluated,
    }
