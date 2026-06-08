#!/usr/bin/env python3
"""Re-run attachment-only referability for a fixed ScanNet++ scene subset.

This wrapper reuses per-scene sidecars from existing scannetpp_flash batches,
executes scripts/run_vlm_referability.py once per scene, and merges the
resulting single-scene caches into one standalone referability cache file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vlm_referability import (
    FRAME_CACHE_SIDECAR_DIR_NAME,
    REFERABILITY_BACKEND,
    REFERABILITY_CACHE_VERSION,
    _build_frame_sidecar_scene_doc,
    _write_json_payload,
)


SCENE_BATCH_BY_ID: dict[str, str] = {
    "09c1414f1b": "0-9",
    "27dd4da69e": "0-9",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run run_vlm_referability for a fixed ScanNet++ scene subset, "
            "reusing existing sidecars and exporting a merged attachment-only cache."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="ScanNet++ raw dataset root containing scene directories.",
    )
    parser.add_argument(
        "--scannetpp_frame_root",
        type=Path,
        default=None,
        help=(
            "Root directory for extracted ScanNet++ iPhone frames. "
            "Passed through to run_vlm_referability."
        ),
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("output/scannetpp_flash"),
        help="Existing scannetpp_flash batch root.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("output/scannetpp_flash_attachment_only_rerun"),
        help="Root directory for per-scene rerun outputs.",
    )
    parser.add_argument(
        "--merged_output",
        type=Path,
        default=Path("output/scannetpp_flash_attachment_only_rerun/selected_attachment_only_cache.json"),
        help="Merged standalone referability cache output path.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=20,
        help="Maximum frames passed through to run_vlm_referability.",
    )
    parser.add_argument(
        "--vlm_workers",
        type=int,
        default=4,
        help="Maximum concurrent VLM requests per scene rerun.",
    )
    parser.add_argument(
        "--frame_clarity_batch_size",
        type=int,
        default=8,
        help="Frame clarity batch size passed through to run_vlm_referability.",
    )
    parser.add_argument(
        "--sensor",
        choices=("iphone", "dslr"),
        default="iphone",
        help="ScanNet++ sensor passed through to run_vlm_referability.",
    )
    parser.add_argument(
        "--vlm_url",
        type=str,
        default=None,
        help="Optional VLM API base URL passed through to run_vlm_referability.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate source inputs and print the planned rerun mapping.",
    )
    parser.add_argument(
        "--allow_frame_expansion",
        action="store_true",
        help=(
            "Allow the rerun to keep MORE frames than the source selection "
            "(e.g. raising --max_frames). The source frame list must remain an "
            "exact ordered prefix of the rerun list; the appended frames are kept. "
            "Without this flag the rerun must reproduce the source list exactly."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return payload


def _find_batch_cache_path(batch_dir: Path) -> Path:
    candidates = sorted(
        path for path in batch_dir.glob("*.json")
        if path.name != "scene_status.json"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one batch cache JSON in {batch_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _scene_source_info(source_root: Path, scene_id: str) -> dict[str, Any]:
    batch_name = SCENE_BATCH_BY_ID[scene_id]
    batch_dir = source_root / batch_name
    batch_cache_path = _find_batch_cache_path(batch_dir)
    batch_cache = _load_json(batch_cache_path)
    frames = batch_cache.get("frames")
    scene_grouping = batch_cache.get("scene_grouping")
    scene_status = batch_cache.get("scene_status")
    if not isinstance(frames, dict) or scene_id not in frames:
        raise RuntimeError(f"Scene {scene_id} not found in frames of {batch_cache_path}")
    if not isinstance(scene_grouping, dict) or scene_id not in scene_grouping:
        raise RuntimeError(f"Scene {scene_id} not found in scene_grouping of {batch_cache_path}")
    if not isinstance(scene_status, dict) or scene_id not in scene_status:
        raise RuntimeError(f"Scene {scene_id} not found in scene_status of {batch_cache_path}")
    sidecar_path = batch_dir / FRAME_CACHE_SIDECAR_DIR_NAME / f"{scene_id}.json"
    if not sidecar_path.exists():
        raise RuntimeError(f"Missing source sidecar for {scene_id}: {sidecar_path}")
    model_name = str(batch_cache.get("model", "")).strip()
    if not model_name:
        raise RuntimeError(f"Missing model in source cache {batch_cache_path}")
    return {
        "scene_id": scene_id,
        "batch_name": batch_name,
        "batch_dir": batch_dir,
        "batch_cache_path": batch_cache_path,
        "batch_cache": batch_cache,
        "model_name": model_name,
        "sidecar_path": sidecar_path,
        "old_frame_names": list((scene_grouping[scene_id] or {}).get("final_cacheable_frame_image_names", [])),
    }


def _validate_source_scene(info: dict[str, Any]) -> None:
    scene_id = str(info["scene_id"])
    batch_cache = info["batch_cache"]
    scene_grouping = (batch_cache.get("scene_grouping") or {}).get(scene_id)
    scene_frames = (batch_cache.get("frames") or {}).get(scene_id)
    if not isinstance(scene_grouping, dict):
        raise RuntimeError(f"Malformed source scene_grouping for {scene_id}")
    if not isinstance(scene_frames, dict):
        raise RuntimeError(f"Malformed source frames for {scene_id}")
    fallback_count = int(scene_grouping.get("selected_after_attachment_slots_count", 0) or 0)
    if fallback_count != 0:
        raise RuntimeError(f"Scene {scene_id} is not attachment-only in source cache; fallback_count={fallback_count}")
    if int(scene_grouping.get("final_cacheable_frame_count", 0) or 0) <= 0:
        raise RuntimeError(f"Scene {scene_id} has no final frames in source cache")
    for image_name, entry in scene_frames.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid source frame entry for {scene_id}/{image_name}")
        if int(entry.get("attachment_pair_ge_50_count", 0) or 0) <= 0:
            raise RuntimeError(f"Source frame {scene_id}/{image_name} is not attachment-qualified")


def _copy_seed_sidecar(source_sidecar: Path, per_scene_output_file: Path) -> None:
    dest_dir = per_scene_output_file.parent / FRAME_CACHE_SIDECAR_DIR_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_sidecar, dest_dir / source_sidecar.name)


def _clean_scene_output_dir(scene_output_dir: Path) -> None:
    """Remove stale per-scene rerun outputs so the scene is reprocessed.

    run_vlm_referability skips scenes already marked complete in
    scene_status.json, which would otherwise leave a previous (failed) run's
    cache in place and re-trigger the same verification error. Wiping the
    directory forces a clean rerun; the seed sidecar is copied back afterwards.
    """
    if scene_output_dir.exists():
        shutil.rmtree(scene_output_dir)


def _run_single_scene(
    *,
    repo_root: Path,
    scene_dir: Path,
    per_scene_output_file: Path,
    model_name: str,
    max_frames: int,
    vlm_workers: int,
    frame_clarity_batch_size: int,
    sensor: str,
    scannetpp_frame_root: Path | None,
    vlm_url: str | None,
) -> None:
    command = [
        sys.executable,
        "scripts/run_vlm_referability.py",
        "--dataset",
        "scannetpp",
        "--data_root",
        str(scene_dir),
        "--output",
        str(per_scene_output_file),
        "--max_frames",
        str(max_frames),
        "--scene_workers",
        "1",
        "--vlm_workers",
        str(vlm_workers),
        "--frame_clarity_batch_size",
        str(frame_clarity_batch_size),
        "--scannetpp_sensor",
        sensor,
        "--vlm_model",
        model_name,
        "--attachment_only",
    ]
    if scannetpp_frame_root is not None:
        command.extend(["--scannetpp_frame_root", str(scannetpp_frame_root)])
    if vlm_url:
        command.extend(["--vlm_url", vlm_url])
    subprocess.run(command, cwd=repo_root, check=True)


def _load_completed_scene_batch_file(scene_output_dir: Path, scene_id: str) -> Path:
    scene_status_path = scene_output_dir / "scene_status.json"
    scene_status_doc = _load_json(scene_status_path)
    completed = scene_status_doc.get("completed_scenes")
    if not isinstance(completed, dict):
        raise RuntimeError(f"Malformed scene_status.json at {scene_status_path}")
    record = completed.get(scene_id)
    if not isinstance(record, dict):
        raise RuntimeError(f"Scene {scene_id} missing from {scene_status_path}")
    batch_file = str(record.get("batch_file", "")).strip()
    if not batch_file:
        raise RuntimeError(f"Scene {scene_id} has no batch_file in {scene_status_path}")
    batch_path = scene_output_dir / batch_file
    if not batch_path.exists():
        raise RuntimeError(f"Completed batch file does not exist: {batch_path}")
    return batch_path


def _verify_rerun_scene(
    *,
    scene_id: str,
    batch_cache_path: Path,
    expected_model_name: str,
    expected_old_frame_names: list[str],
    allow_frame_expansion: bool = False,
) -> dict[str, Any]:
    batch_cache = _load_json(batch_cache_path)
    if str(batch_cache.get("version", "")).strip() != REFERABILITY_CACHE_VERSION:
        raise RuntimeError(f"Version mismatch in rerun cache {batch_cache_path}")
    if str(batch_cache.get("referability_backend", "")).strip() != REFERABILITY_BACKEND:
        raise RuntimeError(f"Referability backend mismatch in rerun cache {batch_cache_path}")
    if str(batch_cache.get("model", "")).strip() != expected_model_name:
        raise RuntimeError(
            f"Model mismatch in rerun cache {batch_cache_path}: "
            f"{batch_cache.get('model')!r} != {expected_model_name!r}"
        )

    frames = batch_cache.get("frames")
    scene_grouping = batch_cache.get("scene_grouping")
    scene_status = batch_cache.get("scene_status")
    if not isinstance(frames, dict) or not isinstance(scene_grouping, dict) or not isinstance(scene_status, dict):
        raise RuntimeError(f"Malformed rerun cache {batch_cache_path}")
    frame_entries = frames.get(scene_id)
    grouping_entry = scene_grouping.get(scene_id)
    status_entry = scene_status.get(scene_id)
    if not isinstance(frame_entries, dict) or not isinstance(grouping_entry, dict) or not isinstance(status_entry, dict):
        raise RuntimeError(f"Scene {scene_id} missing from rerun cache {batch_cache_path}")

    fallback_count = int(grouping_entry.get("selected_after_attachment_slots_count", 0) or 0)
    if fallback_count != 0:
        raise RuntimeError(f"Scene {scene_id} rerun still contains fallback frames: {fallback_count}")
    final_count = int(grouping_entry.get("final_cacheable_frame_count", 0) or 0)
    if final_count <= 0:
        raise RuntimeError(f"Scene {scene_id} rerun produced no final frames")

    new_frame_names = list(grouping_entry.get("final_cacheable_frame_image_names", []))
    if allow_frame_expansion:
        prefix = new_frame_names[: len(expected_old_frame_names)]
        if prefix != expected_old_frame_names:
            raise RuntimeError(
                f"Scene {scene_id} source frames are not an exact ordered prefix of "
                f"the expanded rerun list.\n"
                f"old={expected_old_frame_names}\n"
                f"new_prefix={prefix}\nnew_full={new_frame_names}"
            )
        if len(new_frame_names) < len(expected_old_frame_names):
            raise RuntimeError(
                f"Scene {scene_id} expanded rerun dropped frames: "
                f"old={len(expected_old_frame_names)} new={len(new_frame_names)}"
            )
    elif new_frame_names != expected_old_frame_names:
        raise RuntimeError(
            f"Scene {scene_id} final frame list changed.\n"
            f"old={expected_old_frame_names}\nnew={new_frame_names}"
        )

    for image_name, entry in frame_entries.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid rerun frame entry for {scene_id}/{image_name}")
        if int(entry.get("attachment_pair_ge_50_count", 0) or 0) <= 0:
            raise RuntimeError(f"Rerun frame {scene_id}/{image_name} is not attachment-qualified")

    return {
        "cache": batch_cache,
        "frames": frame_entries,
        "scene_grouping": grouping_entry,
        "scene_status": status_entry,
    }


def _merge_results(
    *,
    merged_output: Path,
    verified_results: list[dict[str, Any]],
    source_model_name: str,
) -> None:
    merged_cache: dict[str, Any] = {
        "version": REFERABILITY_CACHE_VERSION,
        "alias_config_version": None,
        "model": source_model_name,
        "label_batch_size": None,
        "referability_backend": REFERABILITY_BACKEND,
        "frames": {},
        "scene_grouping": {},
        "scene_status": {},
    }

    alias_config_version: object | None = None
    label_batch_size: object | None = None

    for item in verified_results:
        scene_id = str(item["scene_id"])
        cache = item["cache"]
        if alias_config_version is None:
            alias_config_version = cache.get("alias_config_version")
        elif cache.get("alias_config_version") != alias_config_version:
            raise RuntimeError(f"alias_config_version mismatch while merging scene {scene_id}")
        if label_batch_size is None:
            label_batch_size = cache.get("label_batch_size")
        elif cache.get("label_batch_size") != label_batch_size:
            raise RuntimeError(f"label_batch_size mismatch while merging scene {scene_id}")
        merged_cache["frames"][scene_id] = item["frames"]
        merged_cache["scene_grouping"][scene_id] = item["scene_grouping"]
        merged_cache["scene_status"][scene_id] = item["scene_status"]

    merged_cache["alias_config_version"] = alias_config_version
    merged_cache["label_batch_size"] = label_batch_size
    _write_json_payload(merged_output, merged_cache)


def _write_merged_sidecars(
    *,
    merged_output: Path,
    verified_results: list[dict[str, Any]],
    source_model_name: str,
) -> None:
    sidecar_dir = merged_output.parent / FRAME_CACHE_SIDECAR_DIR_NAME
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for item in verified_results:
        scene_id = str(item["scene_id"])
        sidecar_path = sidecar_dir / f"{scene_id}.json"
        sidecar_doc = _build_frame_sidecar_scene_doc(
            scene_id=scene_id,
            model_name=source_model_name,
            referability_backend=REFERABILITY_BACKEND,
            frame_records=item["sidecar_frames"],
        )
        _write_json_payload(sidecar_path, sidecar_doc)


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT
    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    merged_output = args.merged_output.resolve()

    source_infos: list[dict[str, Any]] = []
    for scene_id in sorted(SCENE_BATCH_BY_ID):
        info = _scene_source_info(source_root, scene_id)
        _validate_source_scene(info)
        scene_dir = data_root / scene_id
        if not scene_dir.exists():
            raise RuntimeError(f"Missing scene directory for {scene_id}: {scene_dir}")
        info["scene_dir"] = scene_dir
        source_infos.append(info)

    source_model_names = {str(info["model_name"]) for info in source_infos}
    if len(source_model_names) != 1:
        raise RuntimeError(f"Expected one shared source model, found: {sorted(source_model_names)}")
    source_model_name = next(iter(source_model_names))

    if args.validate_only:
        print("Validated source inputs for scenes:")
        for info in source_infos:
            print(
                f"- {info['scene_id']} from {info['batch_name']} "
                f"frames={len(info['old_frame_names'])} sidecar={info['sidecar_path']}"
            )
        print(f"Shared model: {source_model_name}")
        print(f"Merged output: {merged_output}")
        return

    verified_results: list[dict[str, Any]] = []
    for info in source_infos:
        scene_id = str(info["scene_id"])
        scene_dir = Path(info["scene_dir"])
        scene_output_dir = output_root / scene_id
        per_scene_output_file = scene_output_dir / f"{scene_id}.json"
        _clean_scene_output_dir(scene_output_dir)
        _copy_seed_sidecar(Path(info["sidecar_path"]), per_scene_output_file)
        _run_single_scene(
            repo_root=repo_root,
            scene_dir=scene_dir,
            per_scene_output_file=per_scene_output_file,
            model_name=source_model_name,
            max_frames=int(args.max_frames),
            vlm_workers=int(args.vlm_workers),
            frame_clarity_batch_size=int(args.frame_clarity_batch_size),
            sensor=str(args.sensor),
            scannetpp_frame_root=(
                None if args.scannetpp_frame_root is None else args.scannetpp_frame_root.resolve()
            ),
            vlm_url=args.vlm_url,
        )
        completed_batch_path = _load_completed_scene_batch_file(scene_output_dir, scene_id)
        verified = _verify_rerun_scene(
            scene_id=scene_id,
            batch_cache_path=completed_batch_path,
            expected_model_name=source_model_name,
            expected_old_frame_names=list(info["old_frame_names"]),
            allow_frame_expansion=bool(args.allow_frame_expansion),
        )
        rerun_sidecar_path = scene_output_dir / FRAME_CACHE_SIDECAR_DIR_NAME / f"{scene_id}.json"
        rerun_sidecar_doc = _load_json(rerun_sidecar_path)
        rerun_sidecar_frames = rerun_sidecar_doc.get("frames")
        if not isinstance(rerun_sidecar_frames, dict):
            raise RuntimeError(f"Malformed rerun sidecar for {scene_id}: {rerun_sidecar_path}")
        verified_results.append(
            {
                "scene_id": scene_id,
                "cache": verified["cache"],
                "frames": verified["frames"],
                "scene_grouping": verified["scene_grouping"],
                "scene_status": verified["scene_status"],
                "sidecar_frames": rerun_sidecar_frames,
            }
        )

    _merge_results(
        merged_output=merged_output,
        verified_results=verified_results,
        source_model_name=source_model_name,
    )
    _write_merged_sidecars(
        merged_output=merged_output,
        verified_results=verified_results,
        source_model_name=source_model_name,
    )
    print(f"Wrote merged attachment-only cache to {merged_output}")


if __name__ == "__main__":
    main()
