#!/usr/bin/env python3
"""Rebuild bench viewer HTML files with embedded images from dataset roots.

For ScanNet++ iPhone viewers, pass the extracted frame root
(``output/scannetpp_iphone_frames``-style), not the raw dataset root.
"""

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
        help=(
            "Deprecated pairing for --data_root. Prefer --scannet_image_root and "
            "--scannetpp_frame_root."
        ),
    )
    parser.add_argument(
        "--data_root",
        action="append",
        default=None,
        help=(
            "Deprecated root paired by position with --dataset. For "
            "ScanNet++ iPhone, use --scannetpp_frame_root instead of the raw "
            "dataset root."
        ),
    )
    parser.add_argument(
        "--scannet_image_root",
        action="append",
        default=None,
        help="ScanNet image root (for example /path/to/scans). May be repeated.",
    )
    parser.add_argument(
        "--scannetpp_frame_root",
        action="append",
        default=None,
        help=(
            "ScanNet++ extracted iPhone frame root (for example "
            "/path/to/output/scannetpp_iphone_frames). May be repeated."
        ),
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

    has_explicit_roots = bool(args.scannet_image_root or args.scannetpp_frame_root)
    has_legacy_pairs = bool(args.dataset or args.data_root)
    if not has_explicit_roots and not has_legacy_pairs:
        parser.error(
            "Provide at least one of --scannet_image_root, --scannetpp_frame_root, "
            "or a legacy --dataset/--data_root pair."
        )
    if has_legacy_pairs and (not args.dataset or not args.data_root):
        parser.error("--dataset and --data_root must be provided together.")
    if args.dataset and args.data_root and len(args.dataset) != len(args.data_root):
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

    scannet_roots: list[Path] = [Path(path) for path in (args.scannet_image_root or [])]
    scannetpp_roots: list[Path] = [Path(path) for path in (args.scannetpp_frame_root or [])]

    for dataset, data_root in zip(args.dataset or [], args.data_root or []):
        root_path = Path(data_root)
        if dataset == "scannet":
            scannet_roots.append(root_path)
        else:
            scannetpp_roots.append(root_path)
            print(
                "Warning: legacy --dataset scannetpp --data_root is treated as a "
                "ScanNet++ frame root for viewer images. Pass the extracted frame "
                "directory, not the raw dataset root."
            )

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
