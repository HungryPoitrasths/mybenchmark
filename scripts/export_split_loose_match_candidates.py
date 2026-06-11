#!/usr/bin/env python3
"""Export loose-match candidate groups as split HTML files.

The default output is one HTML file per candidate group plus an index.html.
This is intended for server-side review where a single large HTML is hard to
open or annotate.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

from export_loose_match_candidates_html import (
    DEFAULT_FULL,
    DEFAULT_HTML,
    _render_html,
    collect_candidate_groups,
)
from rewrite_subset_from_merged_html import _level_from_badges, _qtype_from_badges


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "edited" / "loose_match_candidate_groups_split"


def _safe_slug(value: Any, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "group")


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _group_filename(group_index: int, groups: list[dict[str, Any]]) -> str:
    first = groups[0]["card"]
    last = groups[-1]["card"]
    if len(groups) == 1:
        return (
            f"group_{group_index:04d}"
            f"__html_q{int(first['viewer_index']):04d}"
            f"__{_safe_slug(first['scene_id'])}"
            f"__{_safe_slug(first['image_name'])}.html"
        )
    return (
        f"group_{group_index:04d}"
        f"__html_q{int(first['viewer_index']):04d}-{int(last['viewer_index']):04d}"
        f"__{len(groups)}_groups.html"
    )


def _write_index(
    output_dir: Path,
    files: list[dict[str, Any]],
    stats: dict[str, Any],
    report_name: str,
) -> None:
    rows = []
    for item in files:
        first = item["groups"][0]["card"]
        last = item["groups"][-1]["card"]
        if item["group_count"] == 1:
            title = f"HTML #{int(first['viewer_index'])}"
            qrange = str(int(first["viewer_index"]))
        else:
            title = f"HTML #{int(first['viewer_index'])}-#{int(last['viewer_index'])}"
            qrange = f"{int(first['viewer_index'])}-{int(last['viewer_index'])}"
        rows.append(
            "<tr>"
            f"<td>{item['file_index']}</td>"
            f"<td><a href=\"{html.escape(item['filename'])}\">{html.escape(title)}</a></td>"
            f"<td>{html.escape(qrange)}</td>"
            f"<td>{item['group_count']}</td>"
            f"<td>{item['candidate_count']}</td>"
            f"<td>{html.escape(str(first['scene_id']))}</td>"
            f"<td>{html.escape(_level_from_badges(first))}_{html.escape(_qtype_from_badges(first))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Split Loose Match Candidate Groups</title>
  <style>
    body{{font-family:Arial,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}}
    .summary{{background:white;border:1px solid #d7dde5;border-radius:8px;padding:14px;margin-bottom:18px}}
    table{{border-collapse:collapse;background:white;width:100%}}
    th,td{{border:1px solid #d7dde5;padding:7px 9px;text-align:left;vertical-align:top}}
    th{{background:#eef2f7}}
    a{{color:#2458c7;text-decoration:none}}
    a:hover{{text-decoration:underline}}
  </style>
</head>
<body>
  <h1>Split Loose Match Candidate Groups</h1>
  <div class="summary">
    <div><b>Source HTML:</b> {html.escape(str(stats["html_path"]))}</div>
    <div><b>Benchmark:</b> {html.escape(str(stats["full_path"]))}</div>
    <div><b>Loose cards:</b> {stats["loose_count"]}</div>
    <div><b>Groups exported:</b> {stats["groups_exported"]}</div>
    <div><b>Candidate questions shown:</b> {stats["candidate_count"]}</div>
    <div><b>Report:</b> <a href="{html.escape(report_name)}">{html.escape(report_name)}</a></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Open</th>
        <th>HTML Q#</th>
        <th>Groups</th>
        <th>Candidates</th>
        <th>First scene</th>
        <th>First type</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(doc, encoding="utf-8")


def export_split(
    html_path: Path,
    full_path: Path,
    output_dir: Path,
    groups_per_file: int,
    scannet_roots: list[Path] | None = None,
    scannetpp_roots: list[Path] | None = None,
    scannetpp_sensor: str = "iphone",
    max_image_width: int = 520,
) -> dict[str, Any]:
    groups, stats = collect_candidate_groups(html_path, full_path)
    scannet_roots = scannet_roots or []
    scannetpp_roots = scannetpp_roots or []
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = _chunked(groups, groups_per_file)
    files: list[dict[str, Any]] = []
    image_stats_total = {"total": 0, "embedded": 0, "missing": 0, "missing_paths": []}
    for file_index, chunk in enumerate(chunks, 1):
        filename = _group_filename(file_index, chunk)
        chunk_stats = {
            **stats,
            "output_path": str(output_dir / filename),
            "groups_exported": len(chunk),
            "candidate_count": sum(len(group["candidates"]) for group in chunk),
        }
        chunk_image_stats = {"total": 0, "embedded": 0, "missing": 0, "missing_paths": []}
        _render_html(
            chunk,
            chunk_stats,
            output_dir / filename,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=scannetpp_sensor,
            max_image_width=max_image_width,
            image_stats=chunk_image_stats,
        )
        for key in ("total", "embedded", "missing"):
            image_stats_total[key] = int(image_stats_total.get(key, 0)) + int(chunk_image_stats.get(key, 0))
        missing_paths = image_stats_total.setdefault("missing_paths", [])
        if isinstance(missing_paths, list):
            for missing_path in chunk_image_stats.get("missing_paths", []):
                if len(missing_paths) >= 20:
                    break
                missing_paths.append(str(missing_path))
        files.append(
            {
                "file_index": file_index,
                "filename": filename,
                "group_count": len(chunk),
                "candidate_count": chunk_stats["candidate_count"],
                "image_stats": chunk_image_stats,
                "groups": chunk,
            }
        )

    report = {
        **stats,
        "output_dir": str(output_dir),
        "groups_per_file": groups_per_file,
        "file_count": len(files),
        "scannet_data_root": [str(path) for path in scannet_roots],
        "scannetpp_data_root": [str(path) for path in scannetpp_roots],
        "scannetpp_sensor": scannetpp_sensor,
        "image_stats": image_stats_total,
        "files": [
            {
                "file_index": item["file_index"],
                "filename": item["filename"],
                "group_count": item["group_count"],
                "candidate_count": item["candidate_count"],
                "image_stats": item["image_stats"],
                "viewer_indices": [int(group["card"]["viewer_index"]) for group in item["groups"]],
            }
            for item in files
        ],
    }
    report_name = "split_report.json"
    (output_dir / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_index(output_dir, files, stats, report_name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scannet_data_root",
        action="append",
        type=Path,
        default=[],
        help="ScanNet scans root containing scene/color; may be repeated.",
    )
    parser.add_argument(
        "--scannetpp_data_root",
        action="append",
        type=Path,
        default=[],
        help="ScanNet++ root or extracted frame root; may be repeated.",
    )
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--max-image-width", type=int, default=520)
    parser.add_argument(
        "--groups-per-file",
        type=int,
        default=1,
        help="Number of candidate groups per output HTML file. Default: 1.",
    )
    args = parser.parse_args()
    if args.groups_per_file < 1:
        raise SystemExit("--groups-per-file must be >= 1")
    report = export_split(
        args.html,
        args.full,
        args.output_dir,
        args.groups_per_file,
        scannet_roots=args.scannet_data_root,
        scannetpp_roots=args.scannetpp_data_root,
        scannetpp_sensor=args.scannetpp_sensor,
        max_image_width=args.max_image_width,
    )
    print(f"Loose cards: {report['loose_count']}")
    print(f"Groups exported: {report['groups_exported']}")
    print(f"Candidate questions shown: {report['candidate_count']}")
    print(f"Images: {report['image_stats']['embedded']}/{report['image_stats']['total']} embedded")
    print(f"Files written: {report['file_count']}")
    print(f"Index: {args.output_dir / 'index.html'}")
    print(f"Report: {args.output_dir / 'split_report.json'}")


if __name__ == "__main__":
    main()
