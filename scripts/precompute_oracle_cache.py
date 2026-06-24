#!/usr/bin/env python3
"""Pre-compute oracle scene cache for a benchmark subset.

Usage:
    python scripts/precompute_oracle_cache.py \
        --subset output/benchmark_subset.json \
        --scannetpp_geometry_root /path/to/scannetpp \
        --output_dir output/oracle_cache \
        --workers 8
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.enrich_oracle_fields import _dataset_kind, _scene_path as _resolve_scene_path
from src.scene_parser import parse_scene


def _collect_obj_ids(q: dict) -> set[int]:
    ids: set[int] = set()
    for m in q.get("mentioned_objects") or []:
        if isinstance(m, dict):
            try:
                ids.add(int(m["obj_id"]))
            except (KeyError, TypeError, ValueError):
                pass
    for key in ("query_obj_id", "obj_ref_id", "obj_face_id"):
        v = q.get(key)
        if v is not None:
            try:
                ids.add(int(v))
            except (TypeError, ValueError):
                pass
    return ids


def _process_scene(args_tuple: tuple) -> tuple[str, str]:
    scene_id, dataset_kind, scene_path_str, obj_ids, out_path_str = args_tuple
    out_path = Path(out_path_str)
    if out_path.is_file():
        return scene_id, "cached"
    try:
        parsed = parse_scene(scene_path_str, dataset=dataset_kind, skip_support_geom=True)
        if not parsed:
            return scene_id, "no_data"
        objects = {int(o["id"]): o for o in parsed.get("objects", [])}
        if obj_ids:
            objects = {k: v for k, v in objects.items() if k in obj_ids}
        if not objects:
            return scene_id, "no_objects"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({"objects": objects, "scene_path": scene_path_str}, f)
        return scene_id, "ok"
    except Exception as exc:
        return scene_id, f"error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute oracle scene cache")
    parser.add_argument("--subset", required=True, help="Path to benchmark subset JSON")
    parser.add_argument("--scannet_root", default="", help="ScanNet scans root")
    parser.add_argument("--scannetpp_geometry_root", default="", help="ScanNet++ geometry root")
    parser.add_argument("--scannetpp_sensor", default="iphone")
    parser.add_argument("--output_dir", default="output/oracle_cache")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.subset) as f:
        raw = json.load(f)
    questions = raw if isinstance(raw, list) else raw.get("questions", list(raw.values())[0] if isinstance(list(raw.values())[0], list) else [])

    scenes: dict[str, dict] = {}
    for q in questions:
        sid = str(q.get("scene_id") or "")
        if not sid:
            continue
        dataset = str(q.get("_dataset") or q.get("dataset") or "")
        if sid not in scenes:
            dk = _dataset_kind(sid, dataset)
            sp = _resolve_scene_path(sid, dataset, args.scannet_root, args.scannetpp_geometry_root)
            scenes[sid] = {"dataset_kind": dk, "scene_path": str(sp), "obj_ids": set()}
        scenes[sid]["obj_ids"].update(_collect_obj_ids(q))

    out_dir = Path(args.output_dir)
    tasks = [
        (sid, info["dataset_kind"], info["scene_path"], info["obj_ids"], str(out_dir / f"{sid}.pkl"))
        for sid, info in scenes.items()
    ]
    print(f"Processing {len(tasks)} scenes with {args.workers} workers...")

    ok = cached = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_scene, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            sid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached += 1
            else:
                errors += 1
                print(f"  {sid}: {status}", file=sys.stderr)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  ok={ok} cached={cached} err={errors}")

    print(f"Done. {ok} computed, {cached} already cached, {errors} errors.")
    print(f"Cache dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
