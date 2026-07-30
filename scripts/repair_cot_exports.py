#!/usr/bin/env python3
"""Re-render and validate paired CoT sidecar/MS-SWIFT JSONL exports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.models import CotFactRecord
from src.cot.render import render_response
from src.cot.templates import load_template_library
from src.cot.validators import (
    validate_fact_consistency,
    validate_reasoning_consistency,
    validate_response,
    validate_sft_item,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-path", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace both exports after every row passes validation.",
    )
    return parser.parse_args()


def _read_json(line: str, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: expected a JSON object")
    return value


def _record_from_sidecar(row: dict[str, Any]) -> CotFactRecord:
    return CotFactRecord(
        question_uid=str(row["question_uid"]),
        question_type=str(row["question_type"]),
        signature_id=str(row["signature_id"]),
        facts=dict(row["facts"]),
        semantic_answer=row["semantic_answer"],
        answer_letters=[str(value) for value in row["answer_letters"]],
        validation=dict(row.get("validation") or {}),
    )


def _write_row(handle: TextIO | None, row: dict[str, Any]) -> None:
    if handle is not None:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def repair_exports(
    sidecar_path: Path,
    dataset_path: Path,
    *,
    seed: int,
    template_path: Path | None,
    write: bool,
) -> dict[str, int]:
    library = load_template_library(template_path)
    sidecar_temp = sidecar_path.with_name(sidecar_path.name + ".repair.tmp")
    dataset_temp = dataset_path.with_name(dataset_path.name + ".repair.tmp")
    sidecar_output: TextIO | None = None
    dataset_output: TextIO | None = None
    row_count = 0
    response_changes = 0

    try:
        if write:
            sidecar_output = sidecar_temp.open("w", encoding="utf-8", newline="\n")
            dataset_output = dataset_temp.open("w", encoding="utf-8", newline="\n")
        with sidecar_path.open(encoding="utf-8") as sidecar_input, dataset_path.open(
            encoding="utf-8"
        ) as dataset_input:
            for line_number, pair in enumerate(
                zip_longest(sidecar_input, dataset_input), start=1
            ):
                sidecar_line, dataset_line = pair
                if sidecar_line is None or dataset_line is None:
                    raise ValueError(
                        f"row count mismatch between {sidecar_path} and {dataset_path}"
                    )
                sidecar_row = _read_json(
                    sidecar_line, path=sidecar_path, line_number=line_number
                )
                dataset_row = _read_json(
                    dataset_line, path=dataset_path, line_number=line_number
                )
                record = _record_from_sidecar(sidecar_row)
                if dataset_row.get("question_uid") != record.question_uid:
                    raise ValueError(
                        f"{dataset_path}:{line_number}: question_uid does not match sidecar"
                    )

                validate_fact_consistency(record)
                response, template_id = render_response(
                    record, seed=seed, template_library=library
                )
                validate_response(response, record)
                validate_reasoning_consistency(response, record)
                if response != sidecar_row.get("response"):
                    response_changes += 1

                sidecar_row["template_id"] = template_id
                sidecar_row["reasoning"] = response.rsplit("\nAnswer:", 1)[0]
                sidecar_row["response"] = response
                validation = dict(sidecar_row.get("validation") or {})
                validation.update(
                    passed=True,
                    fact_consistency="passed",
                    reasoning_consistency="passed",
                )
                sidecar_row["validation"] = validation

                messages = dataset_row.get("messages")
                if not isinstance(messages, list) or len(messages) != 2:
                    raise ValueError(
                        f"{dataset_path}:{line_number}: expected two messages"
                    )
                messages[1]["content"] = response
                dataset_row["template_id"] = template_id
                validate_sft_item(dataset_row)

                _write_row(sidecar_output, sidecar_row)
                _write_row(dataset_output, dataset_row)
                row_count += 1
    except Exception:
        if sidecar_output is not None:
            sidecar_output.close()
        if dataset_output is not None:
            dataset_output.close()
        sidecar_temp.unlink(missing_ok=True)
        dataset_temp.unlink(missing_ok=True)
        raise
    else:
        if sidecar_output is not None:
            sidecar_output.close()
        if dataset_output is not None:
            dataset_output.close()

    if write:
        os.replace(sidecar_temp, sidecar_path)
        os.replace(dataset_temp, dataset_path)
    return {"rows": row_count, "response_changes": response_changes}


def main() -> int:
    args = parse_args()
    result = repair_exports(
        args.sidecar.resolve(),
        args.dataset.resolve(),
        seed=args.seed,
        template_path=args.template_path.resolve() if args.template_path else None,
        write=args.write,
    )
    result["files_written"] = 2 if args.write else 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
