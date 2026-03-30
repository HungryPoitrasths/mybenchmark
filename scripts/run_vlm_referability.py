#!/usr/bin/env python3
"""VLM-based frame and label-count referability prefilter.

This script runs *before* QA generation. For each selected frame it asks a VLM:
  1. whether the frame is usable for spatial reasoning;
  2. for batches of candidate labels, how many visible instances of each
     label appear in the image.

The output is a cache that can be consumed by scripts/run_pipeline.py via
--referability_cache.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import select_frames
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_poses,
)
from src.utils.depth_occlusion import compute_depth_occlusion, load_depth_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_referability")

DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
DEFAULT_VLM_MODEL = "Qwen2.5-VL-72B-Instruct"
EXCLUDED_LABELS: set[str] = set()
LABEL_BATCH_SIZE = 5
REFERABILITY_CACHE_VERSION = "5.0"
DEPTH_DISAMBIGUATION_MIN_WINNER_RATIO = 0.20
DEPTH_DISAMBIGUATION_MIN_GAP = 0.15


def _image_to_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_vlm_json(
    client,
    model: str,
    content: list[dict],
    default: dict[str, Any],
    max_tokens: int = 512,
) -> dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_object(text)
        if parsed is None:
            return default
        return parsed
    except Exception as e:
        logger.warning("VLM call failed: %s", e)
        return default


def _frame_prompt() -> str:
    return (
        "You are given one original scene image. "
        "Decide whether this frame is usable for object-level visual spatial-reasoning questions. "
        "A usable frame should allow several scene objects to be recognized and referred to reliably from the image alone. "
        "Reject frames that are too blurry, too dark, too unclear, or where most candidate objects are hard to identify or distinguish. "
        'Answer with strict JSON only: {"frame_usable": true, "reason": "clear_scene"}'
    )


def _count_prompt(candidate_labels: list[str]) -> str:
    labels_json = json.dumps(candidate_labels, ensure_ascii=False)
    return (
        "You are given one original scene image and a candidate label list extracted from scene metadata. "
        "Only use the image and this candidate label list. Do not invent new labels. "
        "For each candidate label, return the exact number of instances that are visually recognizable enough to support stable label-based reference in this image. "
        "Count an instance if it is visible enough to identify as that label from the image, even if it is partially occluded or partially outside the frame. "
        "Do not require the whole object to be visible. "
        "Do not count tiny fragments, extremely blurry objects, or ambiguous cases where the label cannot be identified reliably. "
        "If a label has no visible instance, return 0. "
        "Every candidate label must appear exactly once in the output. "
        f"Candidate labels: {labels_json}. "
        'Answer with strict JSON only using this schema: {"counts": {"chair": 3, "lamp": 1, "book": 0}}'
    )


def _chunk_labels(labels: list[str], size: int = LABEL_BATCH_SIZE) -> list[list[str]]:
    return [labels[i:i + size] for i in range(0, len(labels), size)]


def _normalize_count_map(value: Any, expected_labels: list[str]) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    expected = {label.lower() for label in expected_labels}
    out: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str):
            continue
        label = key.strip().lower()
        if label not in expected:
            continue
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        if count_int < 0:
            continue
        out[label] = count_int
    return out


def _frame_decision(
    client,
    model: str,
    image: np.ndarray,
) -> dict[str, Any]:
    full_b64 = _image_to_base64(image)
    default = {
        "frame_usable": True,
        "reason": "vlm_parse_fallback",
    }
    parsed = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
            {"type": "text", "text": _frame_prompt()},
        ],
        default=default,
        max_tokens=128,
    )
    return {
        "frame_usable": bool(parsed.get("frame_usable", default["frame_usable"])),
        "reason": str(parsed.get("reason", default["reason"])),
    }


def _label_count_decision(
    client,
    model: str,
    image_b64: str,
    candidate_labels: list[str],
) -> dict[str, int]:
    if not candidate_labels:
        return {}
    default = {"counts": {}}
    parsed = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": _count_prompt(candidate_labels)},
        ],
        default=default,
        max_tokens=256,
    )
    return _normalize_count_map(parsed.get("counts"), candidate_labels)


def _build_frame_label_candidates(
    visible_object_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
) -> tuple[list[str], dict[str, list[int]]]:
    label_to_ids: dict[str, list[int]] = defaultdict(list)
    for obj_id in visible_object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if not label or label in EXCLUDED_LABELS:
            continue
        label_to_ids[label].append(int(obj_id))

    candidate_labels = sorted(label_to_ids.keys())
    return candidate_labels, dict(label_to_ids)


def _resolve_referable_object_ids(
    label_counts: dict[str, int],
    label_to_ids: dict[str, list[int]],
) -> tuple[list[int], dict[str, list[int]]]:
    referable_ids: list[int] = []
    ambiguous_labels_to_ids: dict[str, list[int]] = {}
    for label, count in label_counts.items():
        if count != 1:
            continue
        obj_ids = label_to_ids.get(label, [])
        if len(obj_ids) == 1:
            referable_ids.append(obj_ids[0])
            continue
        if len(obj_ids) > 1:
            ambiguous_labels_to_ids[str(label)] = [int(obj_id) for obj_id in obj_ids]
    return sorted(referable_ids), dict(sorted(ambiguous_labels_to_ids.items()))


def _disambiguate_by_depth(
    obj_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
    camera_pose,
    depth_image: np.ndarray,
    depth_intrinsics,
    min_winner_ratio: float = DEPTH_DISAMBIGUATION_MIN_WINNER_RATIO,
    min_gap: float = DEPTH_DISAMBIGUATION_MIN_GAP,
) -> tuple[int | None, dict[str, Any]]:
    scores: list[tuple[float, int]] = []
    for obj_id in obj_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        try:
            _status, ratio = compute_depth_occlusion(
                bbox_min=np.array(obj["bbox_min"], dtype=np.float64),
                bbox_max=np.array(obj["bbox_max"], dtype=np.float64),
                camera_pose=camera_pose,
                intrinsics=depth_intrinsics,
                depth_image=depth_image,
            )
        except Exception as exc:
            logger.warning("Depth disambiguation failed for object %s: %s", obj_id, exc)
            continue
        scores.append((float(ratio), int(obj_id)))

    scores.sort(reverse=True)
    candidate_scores = [
        {
            "object_id": int(obj_id),
            "visible_ratio": float(ratio),
        }
        for ratio, obj_id in scores
    ]
    meta: dict[str, Any] = {
        "decision": "no_valid_scores",
        "selected_object_id": None,
        "candidate_scores": candidate_scores,
    }
    if not scores:
        return None, meta

    best_ratio, best_id = scores[0]
    if best_ratio < min_winner_ratio:
        meta["decision"] = "winner_below_min_ratio"
        return None, meta
    if len(scores) >= 2 and (best_ratio - scores[1][0]) < min_gap:
        meta["decision"] = "gap_too_small"
        return None, meta

    meta["decision"] = "selected"
    meta["selected_object_id"] = int(best_id)
    return int(best_id), meta


def _augment_with_depth_disambiguation(
    ambiguous_labels_to_ids: dict[str, list[int]],
    objects_by_id: dict[int, dict[str, Any]],
    camera_pose,
    depth_image: np.ndarray | None,
    depth_intrinsics,
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    if not ambiguous_labels_to_ids:
        return [], {}

    if depth_image is None or depth_intrinsics is None:
        return [], {
            str(label): {
                "decision": "missing_depth",
                "selected_object_id": None,
                "candidate_scores": [],
            }
            for label in sorted(ambiguous_labels_to_ids.keys())
        }

    extra_ids: list[int] = []
    depth_disambiguation: dict[str, dict[str, Any]] = {}
    for label, obj_ids in sorted(ambiguous_labels_to_ids.items()):
        best_id, meta = _disambiguate_by_depth(
            obj_ids=obj_ids,
            objects_by_id=objects_by_id,
            camera_pose=camera_pose,
            depth_image=depth_image,
            depth_intrinsics=depth_intrinsics,
        )
        depth_disambiguation[str(label)] = meta
        if best_id is not None:
            extra_ids.append(int(best_id))
    return sorted(extra_ids), depth_disambiguation


def _count_labels_for_object_ids(
    object_ids: list[int],
    objects_by_id: dict[int, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for obj_id in object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        counts[label] += 1
    return dict(sorted(counts.items()))


def _frame_entry_has_debug_fields(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    required_keys = {
        "candidate_labels",
        "label_to_object_ids",
        "selector_visible_object_ids",
        "selector_visible_label_counts",
        "vlm_count_batches",
        "vlm_unique_object_ids",
    }
    return required_keys.issubset(entry.keys())


def main():
    parser = argparse.ArgumentParser(description="Precompute VLM frame/object referability cache")
    parser.add_argument(
        "--data_root", type=str,
        default=os.getenv("SCANNET_PATH", "/home/lihongxing/datasets/ScanNet/data/scans"),
        help="Root directory of ScanNet scans (contains scene subdirectories)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON cache path",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=300,
        help="Maximum number of scenes to process",
    )
    parser.add_argument(
        "--max_frames", type=int, default=5,
        help="Maximum frames per scene",
    )
    parser.add_argument(
        "--label_map", type=str, default=None,
        help="Path to scannetv2-labels.combined.tsv for raw_category normalization",
    )
    parser.add_argument(
        "--vlm_url", type=str, default=DEFAULT_VLM_URL,
        help="OpenAI-compatible VLM API base URL",
    )
    parser.add_argument(
        "--vlm_model", type=str, default=None,
        help="Model name to use; if omitted, auto-detect from /v1/models",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing output cache if present",
    )
    args = parser.parse_args()

    global EXCLUDED_LABELS
    from src.scene_parser import EXCLUDED_LABELS as SCENE_EXCLUDED_LABELS
    from src.scene_parser import load_scannet_label_map, parse_scene
    from src.support_graph import (
        enrich_scene_with_attachment,
        get_scene_attachment_graph,
        has_nontrivial_attachment,
    )

    EXCLUDED_LABELS = set(SCENE_EXCLUDED_LABELS)

    if args.label_map:
        load_scannet_label_map(args.label_map)

    from openai import OpenAI

    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    client = OpenAI(api_key=api_key, base_url=args.vlm_url)
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        logger.info("VLM available models: %s", available)
    except Exception as e:
        logger.error("Cannot reach VLM at %s: %s", args.vlm_url, e)
        sys.exit(1)

    model_name = args.vlm_model if args.vlm_model else available[0]
    logger.info("Using model: %s", model_name)

    output_path = Path(args.output)
    cache: dict[str, Any] = {
        "version": REFERABILITY_CACHE_VERSION,
        "model": model_name,
        "depth_disambiguation_config": {
            "min_winner_ratio": DEPTH_DISAMBIGUATION_MIN_WINNER_RATIO,
            "min_gap": DEPTH_DISAMBIGUATION_MIN_GAP,
        },
        "frames": {},
    }
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and "frames" in loaded:
            loaded_version = str(loaded.get("version", ""))
            if loaded_version != REFERABILITY_CACHE_VERSION:
                raise RuntimeError(
                    f"Cannot resume referability cache version {loaded_version or '<missing>'}; "
                    f"expected {REFERABILITY_CACHE_VERSION}. Regenerate the cache from scratch."
                )
            cache = loaded
            logger.info("Resuming from %s", output_path)
    cache["depth_disambiguation_config"] = {
        "min_winner_ratio": DEPTH_DISAMBIGUATION_MIN_WINNER_RATIO,
        "min_gap": DEPTH_DISAMBIGUATION_MIN_GAP,
    }

    data_root = Path(args.data_root)
    scene_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and (p / "pose").exists()
    )
    logger.info("Found %d candidate scenes", len(scene_dirs))

    processed = 0
    for scene_dir in scene_dirs:
        if processed >= args.max_scenes:
            break

        scene_id = scene_dir.name
        logger.info(
            "=== Referability scene %s (%d/%d) ===",
            scene_id,
            processed + 1,
            args.max_scenes,
        )

        scene = parse_scene(scene_dir)
        if scene is None:
            continue

        enrich_scene_with_attachment(scene)
        attachment_graph = get_scene_attachment_graph(scene, scene_id=scene_id)
        if not has_nontrivial_attachment(attachment_graph):
            logger.info("Scene %s has no attachment relations -> skipping", scene_id)
            continue

        frames = select_frames(scene_dir, scene["objects"], attachment_graph, args.max_frames)
        if not frames:
            continue

        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        try:
            depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
        except Exception as e:
            logger.warning("Depth intrinsics load failed for %s: %s", scene_id, e)
            depth_intrinsics = None

        scene_cache = cache.setdefault("frames", {}).setdefault(scene_id, {})
        objects_by_id = {int(o["id"]): o for o in scene["objects"]}
        pending_frames = [
            frame
            for frame in frames
            if not _frame_entry_has_debug_fields(scene_cache.get(frame["image_name"]))
        ]
        if not pending_frames:
            logger.info("Scene %s already cached -> skipping", scene_id)
            continue
        logger.info(
            "Processing referability scene %s (%d/%d) with %d pending frames",
            scene_id,
            processed + 1,
            args.max_scenes,
            len(pending_frames),
        )

        for frame in pending_frames:
            image_name = frame["image_name"]
            if image_name not in poses:
                continue

            image_path = scene_dir / "color" / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Cannot read image %s", image_path)
                continue

            candidate_labels, label_to_ids = _build_frame_label_candidates(
                frame["visible_object_ids"],
                objects_by_id,
            )

            frame_info = _frame_decision(client, model_name, image)
            label_counts: dict[str, int] = {}
            referable_object_ids: list[int] = []
            depth_disambiguation: dict[str, dict[str, Any]] = {}
            vlm_count_batches: list[dict[str, Any]] = []
            selector_visible_object_ids = [
                int(obj_id)
                for obj_id in frame["visible_object_ids"]
                if int(obj_id) in objects_by_id
            ]
            selector_visible_label_counts = _count_labels_for_object_ids(
                selector_visible_object_ids,
                objects_by_id,
            )

            if frame_info["frame_usable"]:
                camera_pose = poses[image_name]
                image_b64 = _image_to_base64(image)
                for label_batch in _chunk_labels(candidate_labels):
                    batch_counts = _label_count_decision(client, model_name, image_b64, label_batch)
                    vlm_count_batches.append({
                        "labels": list(label_batch),
                        "raw_counts": dict(sorted(batch_counts.items())),
                    })
                    label_counts.update(batch_counts)

                referable_object_ids, ambiguous_labels_to_ids = _resolve_referable_object_ids(
                    label_counts,
                    label_to_ids,
                )
                if ambiguous_labels_to_ids:
                    depth_image = None
                    frame_id = Path(image_name).stem
                    depth_path = scene_dir / "depth" / f"{frame_id}.png"
                    if depth_intrinsics is not None and depth_path.exists():
                        try:
                            depth_image = load_depth_image(depth_path)
                        except Exception as e:
                            logger.warning("Depth load failed for %s/%s: %s", scene_id, image_name, e)
                    extra_ids, depth_disambiguation = _augment_with_depth_disambiguation(
                        ambiguous_labels_to_ids=ambiguous_labels_to_ids,
                        objects_by_id=objects_by_id,
                        camera_pose=camera_pose,
                        depth_image=depth_image,
                        depth_intrinsics=depth_intrinsics,
                    )
                    referable_object_ids = sorted(set(referable_object_ids) | set(extra_ids))

            frame_entry: dict[str, Any] = {
                "frame_usable": frame_info["frame_usable"],
                "frame_reject_reason": None if frame_info["frame_usable"] else frame_info["reason"],
                "selector_score": int(frame.get("score", 0)),
                "selector_visible_object_ids": selector_visible_object_ids,
                "selector_visible_label_counts": selector_visible_label_counts,
                "candidate_visible_object_ids": selector_visible_object_ids,
                "candidate_labels": list(candidate_labels),
                "label_to_object_ids": {
                    str(label): [int(obj_id) for obj_id in obj_ids]
                    for label, obj_ids in sorted(label_to_ids.items())
                },
                "vlm_count_batches": vlm_count_batches,
                "label_counts": dict(sorted(label_counts.items())),
                "referable_object_ids": referable_object_ids,
                "depth_disambiguation": depth_disambiguation,
                "vlm_unique_object_ids": list(referable_object_ids),
            }
            scene_cache[image_name] = frame_entry

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    logger.info("Saved referability cache to %s", output_path)


if __name__ == "__main__":
    main()
