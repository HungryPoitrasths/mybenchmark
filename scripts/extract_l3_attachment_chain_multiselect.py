#!/usr/bin/env python3
"""Extract sampled L3 attachment-chain questions as multi-select benchmark input."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sampled_type_vlm_eval import _qtype_sort_key


DEFAULT_INPUT = Path("output/type_sample_vlm_eval/results.json")
DEFAULT_OUTPUT = Path("output/l3_attachment_chain_multiselect/benchmark.json")
DEFAULT_VIEWER_HTML = Path("output/type_sample_vlm_eval/viewer.html")
DEFAULT_IMAGE_DIR = Path("output/l3_attachment_chain_multiselect/images")
MULTI_SELECT_NOTE = "This is a multiple-select question; choose all options that apply."
BOTH_RE = re.compile(r"^\s*Both\s+the\s+(.+?)\s+and\s+the\s+(.+?)\s*$", re.IGNORECASE)
DATA_IMAGE_RE = re.compile(
    r'<img\b[^>]*src="data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<b64>[^"]+)"',
    re.IGNORECASE,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported result structure: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _letter(index: int) -> str:
    return chr(65 + index)


def _normalized(text: object) -> str:
    return str(text or "").strip().lower()


def _sanitize_part(value: object) -> str:
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "unknown"


def _extension_for_mime(mime: str) -> str:
    mime = mime.lower()
    if mime in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    return ".bin"


def attach_embedded_image_paths(
    rows: list[dict[str, Any]],
    *,
    viewer_html: Path,
    image_dir: Path,
) -> None:
    if not viewer_html.exists():
        return

    html_text = viewer_html.read_text(encoding="utf-8")
    images = list(DATA_IMAGE_RE.finditer(html_text))
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _qtype_sort_key(str(row.get("type") or "")),
            str(row.get("scene_id") or ""),
            str(row.get("image_name") or ""),
        ),
    )
    if len(images) != len(sorted_rows):
        raise ValueError(
            f"Embedded image count mismatch: {len(images)} image(s) in {viewer_html}, "
            f"{len(sorted_rows)} result row(s) in input JSON"
        )

    image_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(sorted_rows, start=1):
        if row.get("type") != "attachment_chain":
            continue
        image_match = images[index - 1]
        mime = image_match.group("mime")
        image_bytes = base64.b64decode(image_match.group("b64"))
        scene = _sanitize_part(row.get("scene_id"))
        image_stem = _sanitize_part(Path(str(row.get("image_name") or "image")).stem)
        image_path = image_dir / f"q_{index:04d}__{scene}__{image_stem}{_extension_for_mime(mime)}"
        image_path.write_bytes(image_bytes)
        row["image_path"] = str(image_path.resolve())
        row["checked_image_paths"] = [str(image_path.resolve())]


def convert_attachment_chain_question(row: dict[str, Any]) -> dict[str, Any]:
    options = list(row.get("options") or [])
    both_matches: list[tuple[int, re.Match[str]]] = []
    for idx, option in enumerate(options):
        match = BOTH_RE.match(str(option))
        if match:
            both_matches.append((idx, match))

    if len(both_matches) != 1:
        raise ValueError(
            f"Expected exactly one 'Both the ... and the ...' option for "
            f"{row.get('scene_id')}/{row.get('image_name')}, found {len(both_matches)}"
        )

    both_index, match = both_matches[0]
    first_value = f"the {match.group(1).strip()}"
    second_value = f"the {match.group(2).strip()}"
    target_values = {_normalized(first_value), _normalized(second_value)}

    new_options = [option for idx, option in enumerate(options) if idx != both_index]
    answer_indices = [
        idx for idx, option in enumerate(new_options) if _normalized(option) in target_values
    ]
    if len(answer_indices) != 2:
        raise ValueError(
            f"Could not find both individual answer options for "
            f"{row.get('scene_id')}/{row.get('image_name')}: {first_value!r}, {second_value!r}"
        )

    question_text = str(row.get("question") or "").strip()
    if MULTI_SELECT_NOTE.lower() not in question_text.lower():
        question_text = f"{question_text} {MULTI_SELECT_NOTE}".strip()

    item = copy.deepcopy(row)
    for key in (
        "prediction",
        "predictions",
        "raw_response",
        "correct",
        "error",
        "gt_answer",
        "gt_answers",
    ):
        item.pop(key, None)

    item["question"] = question_text
    item["options"] = new_options
    if row.get("dataset"):
        item["_dataset"] = row.get("dataset")
    item["multi_select"] = True
    item["answer"] = [_letter(idx) for idx in answer_indices]
    item["correct_values"] = [str(new_options[idx]) for idx in answer_indices]
    item["correct_value"] = "; ".join(item["correct_values"])

    return item


def extract_multiselect_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        if row.get("type") == "attachment_chain":
            converted.append(convert_attachment_chain_question(row))
    return converted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sampled L3 attachment-chain questions as multi-select benchmark.json."
    )
    parser.add_argument("--input_json", default=str(DEFAULT_INPUT), help="VLM results.json to read")
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT), help="benchmark.json to write")
    parser.add_argument(
        "--viewer_html",
        default=str(DEFAULT_VIEWER_HTML),
        help="Self-contained viewer.html to export embedded images from; ignored if missing",
    )
    parser.add_argument(
        "--image_dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Directory for extracted local images referenced by the benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    rows = _load_rows(input_path)
    attach_embedded_image_paths(
        rows,
        viewer_html=Path(args.viewer_html),
        image_dir=Path(args.image_dir),
    )
    questions = extract_multiselect_questions(rows)
    if not questions:
        raise SystemExit(f"No attachment_chain questions found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"questions": questions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"extracted questions: {len(questions)}")
    print(f"output benchmark   : {output_path}")


if __name__ == "__main__":
    main()
