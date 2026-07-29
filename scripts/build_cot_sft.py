#!/usr/bin/env python3
"""Build deterministic CoT sidecar and MS-SWIFT JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.pipeline import build_dataset
from src.cot.sampling import (
    select_monitor_validation,
    select_pilot_train,
    select_pilot_train_8k,
)


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("benchmark must be a list or an object containing questions")
    return [item for item in questions if isinstance(item, dict)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-path", type=Path)
    parser.add_argument("--scannet-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep candidate image paths for a metadata-only dry run.",
    )
    parser.add_argument("--fail-on-reject", action="store_true")
    parser.add_argument(
        "--preset",
        choices=("all", "pilot-train-8k", "pilot-train-10k", "monitor-val-320"),
        default="all",
        help=(
            "Select a deterministic training or monitoring subset after CoT validation. "
            "The 8k preset uses L1/L2/L3=3669/661/3670; the 10k preset uses "
            "4669/661/4670; the validation preset "
            "uses 20 records for each of the 16 supported types."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = load_questions(args.benchmark)
    result = build_dataset(
        questions,
        benchmark_path=args.benchmark,
        seed=args.seed,
        template_path=args.template_path,
        scannet_roots=[path.resolve() for path in args.scannet_image_root],
        scannetpp_roots=[path.resolve() for path in args.scannetpp_image_root],
        scannetpp_sensor=args.scannetpp_sensor,
        require_images=not args.allow_missing_images,
    )

    if args.preset in {"pilot-train-8k", "pilot-train-10k"} and args.split != "train":
        raise ValueError(f"{args.preset} requires --split train")
    if args.preset == "monitor-val-320" and args.split != "val":
        raise ValueError("monitor-val-320 requires --split val")

    sidecar = result["sidecar"]
    sft = result["sft"]
    selection_report: dict[str, Any] | None = None
    if args.preset == "pilot-train-8k":
        selection = select_pilot_train_8k(sidecar, seed=args.seed)
        selected_indices = selection.indices
        selection_report = selection.report
        sidecar = [sidecar[index] for index in selected_indices]
        sft = [sft[index] for index in selected_indices]
    elif args.preset == "pilot-train-10k":
        selection = select_pilot_train(sidecar, seed=args.seed)
        selected_indices = selection.indices
        selection_report = selection.report
        sidecar = [sidecar[index] for index in selected_indices]
        sft = [sft[index] for index in selected_indices]
    elif args.preset == "monitor-val-320":
        selection = select_monitor_validation(sidecar, seed=args.seed)
        selected_indices = selection.indices
        selection_report = selection.report
        sidecar = [sidecar[index] for index in selected_indices]
        sft = [sft[index] for index in selected_indices]

    prefix = args.output_prefix
    write_jsonl(prefix.with_suffix(".sidecar.jsonl"), sidecar)
    write_jsonl(prefix.with_suffix(".rejected.jsonl"), result["rejected"])
    report = {
        **result["report"],
        "split": args.split,
        "source": str(args.benchmark.resolve()),
        "preset": args.preset,
        "accepted_before_selection": len(result["sidecar"]),
        "selected_count": len(sidecar),
    }
    if selection_report is not None:
        report["selection"] = selection_report
    if args.split == "train" and not args.allow_missing_images:
        write_jsonl(prefix.with_suffix(".ms_swift.jsonl"), sft)
        report["ms_swift_exported"] = len(sft)
    elif args.split == "val" and not args.allow_missing_images:
        write_jsonl(prefix.with_suffix(".ms_swift_eval.jsonl"), sft)
        report["ms_swift_exported"] = 0
        report["ms_swift_eval_exported"] = len(sft)
        report["note"] = "Validation export is for inference/evaluation and is not training data."
    else:
        report["ms_swift_exported"] = 0
        report["note"] = (
            "Metadata-only dry runs do not export an MS-SWIFT file."
            if args.allow_missing_images
            else "Validation and test splits are never exported as SFT training data."
        )
    write_json(prefix.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.fail_on_reject and result["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
