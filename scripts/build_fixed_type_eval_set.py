#!/usr/bin/env python3
"""Build a fixed per-type evaluation set from a benchmark JSON file.

The output is a single benchmark-style JSON file whose question list is fixed.
Sampling favors broad scene coverage by starting with a strict per-scene cap
and relaxing it only when a type would otherwise fall short.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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


def _json_key(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text


def _question_uid(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question.get("_dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
            "answer": question.get("answer"),
        }
    )


def _question_dedupe_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "question": question.get("question"),
        }
    )


def _qtype_sort_key(qtype: str) -> tuple[int, str]:
    try:
        return (QTYPE_ORDER.index(qtype), qtype)
    except ValueError:
        return (len(QTYPE_ORDER), qtype)


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError(f"Unsupported benchmark structure: {path}")
    return [q for q in questions if isinstance(q, dict)]


def load_questions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0

    for q in _load_benchmark(path):
        item = dict(q)
        item["_dataset"] = str(item.get("_dataset") or item.get("dataset") or "unknown")
        item["_source_benchmark"] = str(path)
        item["question_uid"] = _question_uid(item)
        dedupe_key = _question_dedupe_key(item)
        if dedupe_key in seen:
            duplicate_count += 1
            continue
        seen.add(dedupe_key)
        questions.append(item)

    metadata = {
        "source_file": str(path),
        "input_question_count": len(questions),
        "duplicate_question_count": duplicate_count,
        "dedupe_rule": "scene_id + image_name + question",
    }
    return questions, metadata


def sample_questions(
    questions: list[dict[str, Any]],
    *,
    per_type: int,
    scene_cap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_type[str(q.get("type") or "unknown")].append(q)

    sampled: list[dict[str, Any]] = []
    sampling_stats: dict[str, Any] = {}
    for qtype in sorted(by_type, key=_qtype_sort_key):
        group = list(by_type[qtype])
        rng.shuffle(group)

        chosen: list[dict[str, Any]] = []
        chosen_uids: set[str] = set()
        per_scene: Counter[str] = Counter()
        relaxed_added = 0
        relaxed_cap = scene_cap
        max_scene_count = max(
            (Counter(str(q.get("scene_id") or "unknown") for q in group).values()),
            default=0,
        )

        while len(chosen) < per_type and relaxed_cap <= max(1, max_scene_count):
            before = len(chosen)
            for q in group:
                if len(chosen) >= per_type:
                    break
                scene_id = str(q.get("scene_id") or "unknown")
                if per_scene[scene_id] >= relaxed_cap:
                    continue
                uid = str(q["question_uid"])
                if uid in chosen_uids:
                    continue
                chosen.append(q)
                chosen_uids.add(uid)
                per_scene[scene_id] += 1
                if relaxed_cap > scene_cap:
                    relaxed_added += 1
            if len(chosen) == before:
                break
            relaxed_cap += 1

        sampled.extend(chosen)
        sampling_stats[qtype] = {
            "available": len(group),
            "sampled": len(chosen),
            "relaxed_scene_cap_added": relaxed_added,
            "scene_count": len({str(q.get("scene_id") or "unknown") for q in chosen}),
            "initial_scene_cap": scene_cap,
            "final_scene_cap": max(scene_cap, relaxed_cap - 1),
        }

    return sampled, sampling_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed per-type eval set.")
    parser.add_argument("--input", default="output/benchmark_subset.json", help="Input benchmark JSON")
    parser.add_argument("--output", required=True, help="Output fixed benchmark JSON")
    parser.add_argument("--per_type", type=int, default=50, help="Questions sampled per type")
    parser.add_argument("--scene_cap", type=int, default=1, help="Initial max questions per scene within each type")
    parser.add_argument("--seed", type=int, default=20260602, help="Random seed for sampling")
    args = parser.parse_args()
    if args.per_type <= 0:
        parser.error("--per_type must be positive")
    if args.scene_cap <= 0:
        parser.error("--scene_cap must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    questions, metadata = load_questions(input_path)
    selected, sampling_stats = sample_questions(
        questions,
        per_type=args.per_type,
        scene_cap=args.scene_cap,
        seed=args.seed,
    )

    payload = {
        "metadata": {
            **metadata,
            "output_mode": "fixed_type_eval_set",
            "per_type": args.per_type,
            "scene_cap": args.scene_cap,
            "seed": args.seed,
            "sampled_question_count": len(selected),
        },
        "sampling_stats": sampling_stats,
        "questions": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"input questions : {len(questions)}")
    print(f"sampled questions: {len(selected)}")
    print(f"output json     : {output_path}")


if __name__ == "__main__":
    main()
