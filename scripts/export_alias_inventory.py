#!/usr/bin/env python3
"""Export alias inventory and missing-alias candidates from referability caches."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.alias_groups import (
    ALIAS_CONFIG_VERSION,
    get_alias_group_risk_level,
    get_explicit_alias_group_config,
    resolve_alias_metadata,
)
from src.scene_parser import EXCLUDED_LABELS, normalize_label


def _iter_frame_entries(cache_data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    frames = cache_data.get("frames", {})
    if not isinstance(frames, dict):
        return
    for scene_id, scene_frames in frames.items():
        if not isinstance(scene_frames, dict):
            continue
        for image_name, entry in scene_frames.items():
            if not isinstance(entry, dict):
                continue
            yield str(scene_id), str(image_name), entry


def _collect_frame_labels(entry: dict[str, Any]) -> dict[str, str]:
    statuses = entry.get("label_statuses")
    if isinstance(statuses, dict) and statuses:
        return {
            str(label).strip().lower(): str(status).strip().lower()
            for label, status in statuses.items()
            if str(label).strip()
        }

    candidate_labels = entry.get("candidate_labels")
    if isinstance(candidate_labels, list):
        return {
            str(label).strip().lower(): "unknown"
            for label in candidate_labels
            if str(label).strip()
        }
    return {}


def build_alias_inventory(cache_paths: list[str]) -> dict[str, Any]:
    explicit_config = get_explicit_alias_group_config()
    by_canonical: dict[str, dict[str, Any]] = {}
    total_frames = 0

    for cache_path in cache_paths:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cache_data = json.load(handle)

        for scene_id, image_name, entry in _iter_frame_entries(cache_data):
            total_frames += 1
            for raw_label, status in _collect_frame_labels(entry).items():
                canonical_label = normalize_label(raw_label)
                if canonical_label in EXCLUDED_LABELS:
                    continue
                alias = resolve_alias_metadata(
                    raw_label=raw_label,
                    canonical_label=canonical_label,
                )
                record = by_canonical.setdefault(
                    canonical_label,
                    {
                        "canonical_label": canonical_label,
                        "alias_group": alias.alias_group,
                        "alias_source": alias.alias_source,
                        "risk_level": get_alias_group_risk_level(alias.alias_group),
                        "frame_count": 0,
                        "status_counts": Counter(),
                        "raw_labels": Counter(),
                        "cache_files": set(),
                        "scene_ids": set(),
                    },
                )
                record["frame_count"] += 1
                record["status_counts"][status] += 1
                record["raw_labels"][raw_label] += 1
                record["cache_files"].add(Path(cache_path).name)
                record["scene_ids"].add(scene_id)

    observed_labels: list[dict[str, Any]] = []
    missing_explicit_labels: list[dict[str, Any]] = []
    for canonical_label, record in sorted(
        by_canonical.items(),
        key=lambda item: (-int(item[1]["frame_count"]), item[0]),
    ):
        normalized = {
            "canonical_label": canonical_label,
            "alias_group": str(record["alias_group"]),
            "alias_source": str(record["alias_source"]),
            "risk_level": str(record["risk_level"]),
            "frame_count": int(record["frame_count"]),
            "status_counts": dict(sorted(record["status_counts"].items())),
            "raw_labels": dict(
                sorted(
                    record["raw_labels"].items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )
            ),
            "scene_count": len(record["scene_ids"]),
            "cache_files": sorted(record["cache_files"]),
        }
        observed_labels.append(normalized)
        if normalized["alias_source"] != "explicit":
            missing_explicit_labels.append(normalized)

    explicit_by_risk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for alias_group, config in explicit_config.items():
        explicit_by_risk[str(config["risk_level"])].append(
            {
                "alias_group": alias_group,
                "canonical_labels": list(config["canonical_labels"]),
                "variants": list(config["variants"]),
            }
        )
    for risk_level in explicit_by_risk:
        explicit_by_risk[risk_level] = sorted(
            explicit_by_risk[risk_level],
            key=lambda item: item["alias_group"],
        )

    return {
        "alias_config_version": ALIAS_CONFIG_VERSION,
        "cache_paths": [str(Path(path).name) for path in cache_paths],
        "frame_count": total_frames,
        "explicit_alias_groups_by_risk": dict(sorted(explicit_by_risk.items())),
        "observed_labels": observed_labels,
        "missing_explicit_alias_candidates": missing_explicit_labels,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Alias Inventory Report")
    lines.append("")
    lines.append(f"- alias_config_version: `{report['alias_config_version']}`")
    lines.append(f"- cache_files: `{len(report['cache_paths'])}`")
    lines.append(f"- frame_count: `{report['frame_count']}`")
    lines.append("")
    lines.append("## Explicit Alias Groups")
    lines.append("")
    for risk_level, groups in report.get("explicit_alias_groups_by_risk", {}).items():
        lines.append(f"### {risk_level}")
        lines.append("")
        for group in groups:
            labels = ", ".join(group.get("canonical_labels", []))
            variants = ", ".join(group.get("variants", []))
            lines.append(f"- `{group['alias_group']}`")
            lines.append(f"  canonical: {labels}")
            lines.append(f"  variants: {variants}")
        lines.append("")

    lines.append("## Missing Explicit Alias Candidates")
    lines.append("")
    for record in report.get("missing_explicit_alias_candidates", []):
        raw_labels = ", ".join(record.get("raw_labels", {}).keys())
        lines.append(
            f"- `{record['canonical_label']}`: frames={record['frame_count']}, "
            f"risk={record['risk_level']}, raws={raw_labels}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export alias inventory from referability caches")
    parser.add_argument(
        "--cache_glob",
        type=str,
        default="pilot_referability_cache_qwen3_vl_flash*.json",
        help="Glob for referability cache JSON files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="evaluation/alias_inventory_pilot_report.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="evaluation/alias_inventory_pilot_report.md",
        help="Path to write Markdown report",
    )
    args = parser.parse_args()

    cache_paths = sorted(glob.glob(args.cache_glob))
    if not cache_paths:
        raise SystemExit(f"No cache files matched: {args.cache_glob}")

    report = build_alias_inventory(cache_paths)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report to {output_json}")
    print(f"Wrote Markdown report to {output_md}")


if __name__ == "__main__":
    main()
