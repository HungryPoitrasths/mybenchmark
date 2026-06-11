#!/usr/bin/env python3
"""Recompute ONLY the attachment-graph-derived fields in a referability cache.

The "split" attachment-chain fix (scene_parser EXCLUDED_LABELS + support_graph
oversized-parent guard) changes the attachment graph topology. A referability
cache produced before the fix therefore stores stale attachment pairs (a
table-supported object re-parented onto a room-scale "split" blob).

Re-running scripts/run_vlm_referability.py from scratch would redo all the
expensive VLM clarity/referability work, which is UNCHANGED by this fix. The
per-object referability verdict (``attachment_referable_object_ids``) is a VLM
result and is reused verbatim. Only the graph-derived fields are recomputed:

    attachment_referable_pairs
    attachment_referable_pair_count
    attachment_final_referability   (pairs / pair_count mirror)
    attachment_selector_signal      (rebuilt for consistency)
    attachment_final_frame_selection (rebuilt for consistency)

The attachment graph is rebuilt with the CURRENT (fixed) code via the same
``make_data_source(...).load_scene()`` + ``enrich_scene_with_attachment`` path
the pipeline uses, so split blobs are excluded and chains like
carpet->table->plant pot are restored.

Usage (server, raw ScanNet++ present):
    python scripts/rebuild_referability_attachment_pairs.py \
        --input output/scannetpp_flash/0-9/0-9_20260522_084737.json \
        --output output/scannetpp_flash_v2/0-9/0-9_split_fixed.json \
        --dataset scannetpp --scannetpp-sensor iphone \
        --data-root /home/sujinyue/datasets/scannetpp

Local dry check (reuse stored scene_metadata geometry instead of raw data):
    python scripts/rebuild_referability_attachment_pairs.py \
        --input <cache> --output <out> --dataset scannetpp \
        --scene-metadata-dir output/l3_attachment_move_scannetpp/scene_metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import make_data_source
from src.support_graph import enrich_scene_with_attachment, get_scene_attachment_graph

# Reuse the EXACT field-construction helpers from the referability script so the
# rewritten cache is byte-compatible with a fresh run.
from scripts.run_vlm_referability import (
    _apply_attachment_layer_payloads,
    _build_attachment_referable_pairs,
)


def _resolve_scene_dir(data_root: Path, scene_id: str) -> Path:
    for candidate in (data_root / scene_id, data_root / "scans" / scene_id, data_root):
        if candidate.name == scene_id and candidate.is_dir():
            return candidate
        if candidate.is_dir() and (candidate / scene_id).exists():
            return candidate / scene_id
    raise FileNotFoundError(f"scene dir for {scene_id} under {data_root}")


def _graph_from_data_root(
    data_root: Path, dataset: str, scene_id: str, sensor: str
) -> dict[int, list[int]]:
    """Rebuild the attachment graph the same way the pipeline does."""
    scene_dir = _resolve_scene_dir(data_root, scene_id)
    scene = make_data_source(dataset, scene_dir, sensor=sensor).load_scene()
    if scene is None:
        raise RuntimeError(f"failed to parse scene {scene_id}")
    enrich_scene_with_attachment(scene)  # uses fixed support_graph
    return get_scene_attachment_graph(scene)


def _graph_from_scene_metadata(meta_dir: Path, scene_id: str) -> dict[int, list[int]]:
    """Rebuild from a stored scene_metadata dump.

    scene_metadata is a raw dump that still CONTAINS split objects, so apply the
    EXCLUDED_LABELS filter here (parse_scene does this in the data_root path).
    """
    from src.scene_parser import EXCLUDED_LABELS

    meta = json.loads((meta_dir / f"{scene_id}.json").read_text(encoding="utf-8"))
    objects = [
        obj for obj in meta.get("objects", [])
        if str(obj.get("label", "")).strip().lower() not in EXCLUDED_LABELS
    ]
    scene = {"scene_id": scene_id, "objects": objects}
    enrich_scene_with_attachment(scene)  # uses fixed support_graph
    return get_scene_attachment_graph(scene)


def _rewrite_frame_entry(
    entry: dict[str, Any], graph: dict[int, list[int]]
) -> tuple[dict[str, Any], bool]:
    """Recompute graph-derived attachment fields; reuse VLM verdicts. Returns
    (new_entry, changed)."""
    if not isinstance(entry, dict):
        return entry, False
    old_pairs = entry.get("attachment_referable_pairs") or []
    new_pairs = _build_attachment_referable_pairs(
        graph, entry.get("attachment_referable_object_ids")
    )
    updated = dict(entry)
    updated["attachment_referable_pairs"] = new_pairs
    updated["attachment_referable_pair_count"] = len(new_pairs)
    # Rebuild the mirror payloads (attachment_final_referability etc.) exactly
    # as run_vlm_referability would, preserving selector signal / selection rank.
    updated = _apply_attachment_layer_payloads(updated, attachment_pairs=new_pairs)
    changed = _norm(old_pairs) != _norm(new_pairs)
    return updated, changed


def _norm(pairs: Any) -> list[list[int]]:
    out = []
    for p in pairs or []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            out.append([int(p[0]), int(p[1])])
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Existing referability cache JSON to rebuild from.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination for the rebuilt cache (do not overwrite input).")
    ap.add_argument("--dataset", choices=("scannet", "scannetpp"), required=True)
    ap.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Raw dataset root; rebuilds graph via the full pipeline path.")
    ap.add_argument("--scene-metadata-dir", type=Path, default=None,
                    help="Fallback: dir of <scene_id>.json scene_metadata dumps.")
    args = ap.parse_args()

    if not args.data_root and not args.scene_metadata_dir:
        ap.error("provide --data-root or --scene-metadata-dir")

    cache = json.loads(args.input.read_text(encoding="utf-8"))
    frames = cache.get("frames", cache)
    if not isinstance(frames, dict):
        ap.error("unexpected cache structure: 'frames' is not a scene map")

    graph_cache: dict[str, dict[int, list[int]]] = {}
    n_scene_ok = n_scene_fail = n_frames = n_changed = 0
    failures: list[str] = []

    for scene_id, scene_frames in frames.items():
        if not isinstance(scene_frames, dict):
            continue
        try:
            if scene_id not in graph_cache:
                if args.data_root:
                    graph_cache[scene_id] = _graph_from_data_root(
                        args.data_root, args.dataset, scene_id, args.scannetpp_sensor)
                else:
                    graph_cache[scene_id] = _graph_from_scene_metadata(
                        args.scene_metadata_dir, scene_id)
            graph = graph_cache[scene_id]
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            n_scene_fail += 1
            failures.append(f"{scene_id}: {exc}")
            continue
        n_scene_ok += 1
        for image_name, entry in scene_frames.items():
            new_entry, changed = _rewrite_frame_entry(entry, graph)
            scene_frames[image_name] = new_entry
            n_frames += 1
            n_changed += int(changed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    print(f"scenes rebuilt: {n_scene_ok}  failed: {n_scene_fail}")
    print(f"frames processed: {n_frames}  attachment_pairs changed: {n_changed}")
    if failures:
        print("FAILURES:")
        for line in failures:
            print(f"  {line}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
