#!/usr/bin/env python3
"""VLM-based frame and label-count referability prefilter.

This script runs *before* QA generation. For each selected frame it asks a VLM:
  1. whether the frame is usable for spatial reasoning;
  2. for batches of candidate labels, how many clearly visible instances of
     each label appear in the image.

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
    load_scannet_poses,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_referability")

DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
DEFAULT_VLM_MODEL = "Qwen2.5-VL-72B-Instruct"
EXCLUDED_LABELS: set[str] = set()
LABEL_BATCH_SIZE = 5


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
        "Decide whether this frame is usable for visual spatial-reasoning questions. "
        "Reject frames that are too blurry, too dark, too unclear, or where most objects are hard to recognize. "
        'Answer with strict JSON only: {"frame_usable": true, "reason": "clear_scene"}'
    )


def _count_prompt(candidate_labels: list[str]) -> str:
    labels_json = json.dumps(candidate_labels, ensure_ascii=False)
    return (
        "You are given one original scene image and a candidate label list extracted from scene metadata. "
        "Only use the image and this candidate label list. Do not invent new labels. "
        "For each candidate label, return the exact number of clearly visible instances in the image. "
        "Count only instances that are clear enough to support a spatial-reasoning question. "
        "If a label is not clearly visible, return 0. "
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
    image: np.ndarray,
    candidate_labels: list[str],
) -> dict[str, int]:
    if not candidate_labels:
        return {}
    full_b64 = _image_to_base64(image)
    default = {"counts": {}}
    parsed = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
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
) -> list[int]:
    referable_ids: list[int] = []
    for label, count in label_counts.items():
        if count != 1:
            continue
        obj_ids = label_to_ids.get(label, [])
        if len(obj_ids) != 1:
            continue
        referable_ids.append(obj_ids[0])
    return sorted(referable_ids)


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
    from src.support_graph import enrich_scene_with_support, has_nontrivial_support

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
        "version": "3.0",
        "model": model_name,
        "frames": {},
    }
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and "frames" in loaded:
            cache = loaded
            logger.info("Resuming from %s", output_path)

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

        enrich_scene_with_support(scene)
        support_graph = {
            int(parent_id): child_ids
            for parent_id, child_ids in scene.get("support_graph", {}).items()
        }
        if not has_nontrivial_support(support_graph):
            logger.info("Scene %s has no support relations -> skipping", scene_id)
            continue

        frames = select_frames(scene_dir, scene["objects"], None, args.max_frames)
        if not frames:
            continue

        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)

        scene_cache = cache.setdefault("frames", {}).setdefault(scene_id, {})
        objects_by_id = {int(o["id"]): o for o in scene["objects"]}

        for frame in frames:
            image_name = frame["image_name"]
            if image_name in scene_cache:
                continue
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

            if frame_info["frame_usable"]:
                for label_batch in _chunk_labels(candidate_labels):
                    batch_counts = _label_count_decision(client, model_name, image, label_batch)
                    label_counts.update(batch_counts)

                referable_object_ids = _resolve_referable_object_ids(
                    label_counts,
                    label_to_ids,
                )

            frame_entry: dict[str, Any] = {
                "frame_usable": frame_info["frame_usable"],
                "frame_reject_reason": None if frame_info["frame_usable"] else frame_info["reason"],
                "candidate_visible_object_ids": [
                    int(obj_id)
                    for obj_id in frame["visible_object_ids"]
                    if int(obj_id) in objects_by_id
                ],
                "label_counts": label_counts,
                "referable_object_ids": referable_object_ids,
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
