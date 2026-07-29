#!/usr/bin/env python3
"""Create VLM requests, offline template responses, or reviewed template libraries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.facts import build_fact_record
from src.cot.models import FactExtractionError
from src.cot.templates import (
    DEFAULT_TEMPLATE_PATH,
    generated_templates_for_signature,
    load_template_library,
)


SYSTEM_PROMPT = """You write concise English reasoning templates for a spatial-reasoning SFT dataset.
You receive structured facts, never an image. Produce exactly 12 distinct templates, each 2-4 sentences.
Use only provided facts. Do not invent coordinates, exact internal distances, ray counts, occlusion ratios,
or unseen causes. Preserve the stated camera/object/world reference frame. End with a semantic conclusion,
but do not emit an option letter, <think> tags, or markdown. Use literal placeholders from allowed_slots.
Return one JSON array of 12 strings and no other text."""


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("unsupported benchmark structure")
    return [value for value in values if isinstance(value, dict)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_requests(benchmark: Path) -> list[dict[str, Any]]:
    representatives: dict[str, Any] = {}
    for question in load_questions(benchmark):
        try:
            record = build_fact_record(question)
        except FactExtractionError:
            continue
        representatives.setdefault(record.signature_id, record)
    requests = []
    for signature_id, record in sorted(representatives.items()):
        payload = {
            "question_type": record.question_type,
            "signature_id": signature_id,
            "facts": record.facts,
            "semantic_answer": record.semantic_answer,
            "allowed_slots": ["{observation}", "{transformation}", "{conclusion}"],
        }
        requests.append(
            {
                "signature_id": signature_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
            }
        )
    return requests


def build_offline_responses(benchmark: Path, base_path: Path) -> list[dict[str, Any]]:
    """Return the checked-in deterministic templates for every benchmark signature."""
    load_template_library(base_path)
    representatives: dict[str, Any] = {}
    for question in load_questions(benchmark):
        try:
            record = build_fact_record(question)
        except FactExtractionError:
            continue
        representatives.setdefault(record.signature_id, record)
    return [
        {
            "signature_id": signature_id,
            "templates": [
                template["template"]
                for template in generated_templates_for_signature(signature_id)
            ],
        }
        for signature_id in sorted(representatives)
    ]


def merge_responses(response_path: Path, output_path: Path, base_path: Path) -> None:
    library = load_template_library(base_path)
    overrides = dict(library.get("signature_templates") or {})
    with response_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            signature_id = str(item["signature_id"])
            templates = item.get("templates")
            if not isinstance(templates, list) or len(templates) != 12:
                raise ValueError(f"line {line_number}: expected exactly 12 templates")
            if len({str(value).strip() for value in templates}) != 12:
                raise ValueError(f"line {line_number}: templates must be distinct")
            normalized = []
            for index, template in enumerate(templates):
                text = str(template).strip()
                required = ("{observation}", "{transformation}", "{conclusion}")
                if not all(slot in text for slot in required):
                    raise ValueError(f"line {line_number}, template {index}: missing placeholder")
                if "answer:" in text.lower() or "<think>" in text.lower():
                    raise ValueError(f"line {line_number}, template {index}: forbidden output token")
                normalized.append({"id": f"vlm_{index:02d}", "template": text})
            overrides[signature_id] = normalized
    library["signature_templates"] = overrides
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(library, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("benchmark", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    offline = subparsers.add_parser(
        "offline",
        help="write deterministic response-shaped JSONL without calling a template model",
    )
    offline.add_argument("benchmark", type=Path)
    offline.add_argument("--base", type=Path, default=DEFAULT_TEMPLATE_PATH)
    offline.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("responses", type=Path)
    merge.add_argument("--base", type=Path, default=DEFAULT_TEMPLATE_PATH)
    merge.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        rows = build_requests(args.benchmark)
        write_jsonl(args.output, rows)
        print(f"wrote {len(rows)} signature requests to {args.output}")
    elif args.command == "offline":
        rows = build_offline_responses(args.benchmark, args.base)
        write_jsonl(args.output, rows)
        print(f"wrote {len(rows)} deterministic signature templates to {args.output}")
    else:
        merge_responses(args.responses, args.output, args.base)
        print(f"wrote reviewed template library to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
