#!/usr/bin/env python3
"""Rebuild bench viewer HTML files with embedded images from dataset roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.make_viewer as make_viewer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild bench.html / bench_simple.html with images from server dataset roots."
    )
    parser.add_argument(
        "--questions",
        default="output/benchmark_subset.json",
        help="Benchmark JSON to render.",
    )
    parser.add_argument(
        "--full_output",
        default="output/bench.html",
        help="Full viewer HTML output path.",
    )
    parser.add_argument(
        "--simple_output",
        default="output/bench_simple.html",
        help="Simple viewer HTML output path.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("scannet", "scannetpp"),
        default=None,
        help="Dataset for the next --data_root entry. Repeat for mixed datasets.",
    )
    parser.add_argument(
        "--data_root",
        action="append",
        default=None,
        help="Image/data root paired by position with --dataset. Repeat once per dataset entry.",
    )
    parser.add_argument(
        "--scannetpp_sensor",
        choices=("iphone", "dslr"),
        default="iphone",
        help="ScanNet++ image layout.",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=480,
        help="Max embedded image width in pixels.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Ordering seed forwarded to make_viewer.",
    )
    parser.add_argument(
        "--include_referability_audit",
        action="store_true",
        help="Render referability audit blocks in the full viewer.",
    )
    parser.add_argument(
        "--apply_auto_filters",
        action="store_true",
        help="Apply legacy viewer auto-filters before rendering.",
    )
    parser.add_argument(
        "--hide_attachment_unchanged",
        action="store_true",
        help="Hide attachment-mediated object_move questions whose answers stay unchanged.",
    )
    args = parser.parse_args(argv)

    if not args.dataset or not args.data_root:
        parser.error("At least one --dataset/--data_root pair is required.")
    if len(args.dataset) != len(args.data_root):
        parser.error("--dataset and --data_root must be provided the same number of times.")

    return args


def _load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError(f"Unsupported benchmark structure: {path}")
    return [question for question in questions if isinstance(question, dict)]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    questions_path = (PROJECT_ROOT / args.questions).resolve()
    full_output_path = (PROJECT_ROOT / args.full_output).resolve()
    simple_output_path = (PROJECT_ROOT / args.simple_output).resolve()

    questions = _load_questions(questions_path)

    scannet_roots: list[Path] = []
    scannetpp_roots: list[Path] = []
    for dataset, data_root in zip(args.dataset, args.data_root):
        root_path = Path(data_root)
        if dataset == "scannet":
            scannet_roots.append(root_path)
        else:
            scannetpp_roots.append(root_path)

    if not scannet_roots and not scannetpp_roots:
        raise ValueError("No dataset roots were resolved.")

    full_html = make_viewer.build_viewer_html(
        questions,
        scannet_roots=scannet_roots,
        scannetpp_roots=scannetpp_roots,
        max_width=args.max_width,
        shuffle_seed=args.shuffle_seed,
        include_attachment_unchanged=not args.hide_attachment_unchanged,
        include_referability_audit=args.include_referability_audit,
        apply_filters=args.apply_auto_filters,
        edited_html_filename=make_viewer._default_edited_html_filename(full_output_path),
        scannetpp_sensor=args.scannetpp_sensor,
    )
    simple_html = make_viewer.build_simple_viewer_html(
        questions,
        scannet_roots=scannet_roots,
        scannetpp_roots=scannetpp_roots,
        max_width=args.max_width,
        shuffle_seed=args.shuffle_seed,
        include_attachment_unchanged=not args.hide_attachment_unchanged,
        apply_filters=args.apply_auto_filters,
        edited_html_filename=make_viewer._default_edited_html_filename(simple_output_path),
        scannetpp_sensor=args.scannetpp_sensor,
    )

    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    simple_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.write_text(full_html, encoding="utf-8")
    simple_output_path.write_text(simple_html, encoding="utf-8")

    print(f"Questions loaded : {len(questions)}")
    print(f"ScanNet roots    : {[str(path) for path in scannet_roots]}")
    print(f"ScanNet++ roots  : {[str(path) for path in scannetpp_roots]}")
    print(f"Full HTML        : {full_output_path}")
    print(f"Simple HTML      : {simple_output_path}")


if __name__ == "__main__":
    main()
