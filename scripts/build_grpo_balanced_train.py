#!/usr/bin/env python3
"""Build a deterministic, max-min-balanced raw benchmark subset for GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.facts import question_uid


def _digest(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not all(isinstance(row, dict) for row in questions):
        raise ValueError(f"{path}: expected a list of question objects")
    metadata = payload if isinstance(payload, dict) else {}
    return metadata, questions


def allocate_fair_quotas(total: int, capacities: dict[str, int]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("target total must be positive")
    if not capacities or any(value <= 0 for value in capacities.values()):
        raise ValueError("every type must have positive capacity")
    if total > sum(capacities.values()):
        raise ValueError(
            f"requested {total} questions but only {sum(capacities.values())} are available"
        )
    quotas = {question_type: 0 for question_type in sorted(capacities)}
    remaining = total
    while remaining:
        progressed = False
        for question_type in quotas:
            if quotas[question_type] >= capacities[question_type]:
                continue
            quotas[question_type] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("quota allocation stalled")
    return quotas


def _answer_key(question: dict[str, Any]) -> str:
    answer = question.get("answer")
    if isinstance(answer, list):
        return " ".join(str(value) for value in answer)
    return str(answer or "")


def _diverse_order(
    questions: list[dict[str, Any]],
    *,
    question_type: str,
    seed: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        key = (_answer_key(question), str(question.get("scene_id") or ""))
        buckets[key].append(question)
    for key, rows in buckets.items():
        rows.sort(
            key=lambda row: _digest(seed, question_type, key, question_uid(row))
        )
    keys = sorted(buckets, key=lambda key: _digest(seed, question_type, key))
    ordered: list[dict[str, Any]] = []
    while keys:
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            rows = buckets[key]
            if rows:
                ordered.append(rows.pop(0))
            if rows:
                next_keys.append(key)
        keys = next_keys
    return ordered


def select_balanced_questions(
    questions: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_uids: set[str] = set()
    duplicate_uid_count = 0
    for question in questions:
        question_type = str(question.get("type") or "").strip()
        if not question_type:
            raise ValueError("benchmark contains a question without type")
        uid = question_uid(question)
        if uid in seen_uids:
            duplicate_uid_count += 1
            continue
        seen_uids.add(uid)
        by_type[question_type].append(question)

    capacities = {
        question_type: len(rows) for question_type, rows in sorted(by_type.items())
    }
    quotas = allocate_fair_quotas(target, capacities)
    selected: list[dict[str, Any]] = []
    for question_type, rows in sorted(by_type.items()):
        ordered = _diverse_order(rows, question_type=question_type, seed=seed)
        selected.extend(ordered[: quotas[question_type]])
    selected.sort(key=lambda row: _digest(seed, "global", question_uid(row)))

    selected_by_type = Counter(str(row["type"]) for row in selected)
    selected_by_level = Counter(str(row.get("level") or "") for row in selected)
    if len(selected) != target or dict(sorted(selected_by_type.items())) != quotas:
        raise AssertionError("balanced selection did not meet its exact quotas")
    return selected, {
        "strategy": "max_min_type_balance_without_replacement",
        "seed": seed,
        "target": target,
        "available_count": len(questions),
        "unique_available_count": len(seen_uids),
        "duplicate_uid_count": duplicate_uid_count,
        "available_by_type": capacities,
        "target_by_type": quotas,
        "selected_by_type": dict(sorted(selected_by_type.items())),
        "selected_by_level": dict(sorted(selected_by_level.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_train/benchmark.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_train/grpo_balanced_2k.json"),
    )
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    metadata, questions = _load(input_path)
    selected, report = select_balanced_questions(
        questions,
        target=args.target,
        seed=args.seed,
    )
    payload = {
        "name": f"PSR-Bench GRPO balanced train {args.target}",
        "version": metadata.get("version", "1.0"),
        "source": str(args.input),
        "sampling": report,
        "questions": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
