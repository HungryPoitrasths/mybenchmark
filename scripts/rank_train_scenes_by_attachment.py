#!/usr/bin/env python3
"""Rank ScanNet++ train-split scenes by structural attachment richness.

Ranking priority, all computed locally from mesh/segment/annotation data
(no VLM calls, no scripts/run_vlm_referability.py or scripts/run_pipeline.py
subprocess needed — this script reuses their internal parsing/graph-building
functions directly):
  1. two-hop attachment chain count (A supports B supports C), descending
  2. one-hop attachment edge count (A supports B), descending
  3. unique-label object count (the pre-VLM label-uniqueness prefilter that
     scripts/run_vlm_referability.py applies before ever calling a VLM),
     descending

Example:
    python scripts/rank_train_scenes_by_attachment.py \\
        --data_root /data/zju-151/scannet/data \\
        --split_file scannetpp/nvs_sem_train.txt \\
        --top_n 160 --trim_existing_tail 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.scene_parser import parse_scene
from src.support_graph import enrich_scene_with_attachment, get_attachment_chain
from run_vlm_referability import (
    _build_scene_alias_group_index,
    _build_compat_label_to_object_ids,
)

logger = logging.getLogger("rank_train_scenes")


def _count_attachment_chains(
    objects: list[dict[str, Any]],
    attachment_graph: dict[int, list[int]],
) -> tuple[int, int]:
    """Return (one_hop_edge_count, two_hop_chain_count).

    two_hop_chain_count mirrors the eligibility check in
    src/qa_generator.py's generate_l3_attachment_chain (grandparent/parent/
    grandchild/nearest-non-chain-neighbour labels must be pairwise distinct),
    so it approximates how many attachment_chain questions the scene would
    actually yield once referability filtering runs.
    """
    obj_map = {int(o["id"]): o for o in objects}
    one_hop = sum(len(children) for children in attachment_graph.values())

    two_hop = 0
    for grandparent_id, parent_ids in attachment_graph.items():
        grandparent_id = int(grandparent_id)
        grandparent = obj_map.get(grandparent_id)
        if grandparent is None:
            continue

        for parent_id in parent_ids:
            parent_id = int(parent_id)
            grandchild_ids = attachment_graph.get(parent_id) or []
            if not grandchild_ids:
                continue
            parent = obj_map.get(parent_id)
            if parent is None:
                continue

            this_chain = set(get_attachment_chain(grandparent_id, attachment_graph)) | {grandparent_id}
            non_chain = [o for o in objects if o["id"] not in this_chain]
            if not non_chain:
                continue

            gp_center = grandparent["center"]
            neighbor = min(
                non_chain,
                key=lambda o: sum((a - b) ** 2 for a, b in zip(o["center"], gp_center)),
            )

            for grandchild_id in grandchild_ids:
                grandchild = obj_map.get(int(grandchild_id))
                if grandchild is None:
                    continue
                if len({parent["label"], grandchild["label"], neighbor["label"]}) < 3:
                    continue
                two_hop += 1

    return one_hop, two_hop


def _count_unique_label_objects(objects: list[dict[str, Any]]) -> int:
    """Count objects surviving run_vlm_referability's pre-VLM label filter.

    A label is "unique" when its alias-group family resolves to exactly one
    object id (see _build_scene_alias_group_index / _build_compat_label_to_
    object_ids in scripts/run_vlm_referability.py) -- this is the filter that
    decides which objects are even sent to the VLM for referability review.
    """
    alias_index = _build_scene_alias_group_index(objects)
    label_to_object_ids = _build_compat_label_to_object_ids(objects, alias_index)
    return sum(1 for object_ids in label_to_object_ids.values() if len(object_ids) == 1)


def compute_scene_stats(data_root: Path, scene_id: str) -> dict[str, Any]:
    scene_dir = data_root / scene_id
    scene = parse_scene(scene_dir, dataset="scannetpp")
    if scene is None:
        return {"scene_id": scene_id, "ok": False, "error": "parse_scene returned None (missing data or too few objects)"}

    enrich_scene_with_attachment(scene)
    objects = scene["objects"]
    attachment_graph = {int(k): v for k, v in scene["attachment_graph"].items()}

    one_hop, two_hop = _count_attachment_chains(objects, attachment_graph)
    unique_objects = _count_unique_label_objects(objects)

    return {
        "scene_id": scene_id,
        "ok": True,
        "two_hop_chain_count": two_hop,
        "one_hop_edge_count": one_hop,
        "unique_label_object_count": unique_objects,
        "num_objects": len(objects),
    }


def _worker(payload: tuple[str, str]) -> dict[str, Any]:
    data_root_str, scene_id = payload
    try:
        return compute_scene_stats(Path(data_root_str), scene_id)
    except Exception as exc:  # noqa: BLE001 - one bad scene must not kill an 800+ scene batch
        return {
            "scene_id": scene_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def rank_scenes(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [s for s in stats if s.get("ok")]
    ranked.sort(
        key=lambda s: (
            s["two_hop_chain_count"],
            s["one_hop_edge_count"],
            s["unique_label_object_count"],
        ),
        reverse=True,
    )
    for rank, s in enumerate(ranked, start=1):
        s["rank"] = rank
    return ranked


def write_selected_list(
    ranked: list[dict[str, Any]],
    top_n: int,
    output_path: Path,
    trim_existing_tail: int,
) -> None:
    top_scene_ids = [s["scene_id"] for s in ranked[:top_n]]

    existing_lines: list[str] = []
    if output_path.is_file():
        existing_lines = [
            line.strip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if trim_existing_tail > 0:
            existing_lines = (
                existing_lines[:-trim_existing_tail]
                if len(existing_lines) > trim_existing_tail
                else []
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_lines = existing_lines + top_scene_ids
    output_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %d scene ids to %s (%d kept from existing file + %d newly ranked)",
        len(all_lines), output_path, len(existing_lines), len(top_scene_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=Path("/data/zju-151/scannet/data"),
                         help="Root dir containing one subdirectory per ScanNet++ scene id")
    parser.add_argument("--split_file", type=Path, default=PROJECT_ROOT / "scannetpp" / "nvs_sem_train.txt",
                         help="One scene id per line")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "train_scene_ranking.json",
                         help="Full per-scene stats + ranking, for inspection/debugging")
    parser.add_argument("--selected_output", type=Path,
                         default=Path(os.path.expanduser("~/datasets/scannetpp/train/selected_train_100.txt")))
    parser.add_argument("--top_n", type=int, default=160)
    parser.add_argument("--trim_existing_tail", type=int, default=60,
                         help="Drop this many lines from the end of an existing selected_output before appending the new ranking")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--max_scenes", type=int, default=None, help="Debug: only process the first N scenes from split_file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scene_ids = [
        line.strip()
        for line in args.split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]
    logger.info("Loaded %d candidate scenes from %s", len(scene_ids), args.split_file)

    stats: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, (str(args.data_root), scene_id)): scene_id
            for scene_id in scene_ids
        }
        done = 0
        for future in as_completed(futures):
            result = future.result()
            stats.append(result)
            done += 1
            if not result.get("ok"):
                logger.warning("Scene %s failed: %s", result["scene_id"], result.get("error"))
            if done % 25 == 0 or done == len(scene_ids):
                logger.info("Processed %d/%d scenes", done, len(scene_ids))

    ranked = rank_scenes(stats)
    failed = [s for s in stats if not s.get("ok")]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"ranked": ranked, "failed": failed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote full ranking (%d ok, %d failed) to %s", len(ranked), len(failed), args.output)

    if len(ranked) < args.top_n:
        logger.warning("Only %d scenes ranked successfully, fewer than --top_n=%d", len(ranked), args.top_n)

    write_selected_list(ranked, args.top_n, args.selected_output, args.trim_existing_tail)


if __name__ == "__main__":
    main()
