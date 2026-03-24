#!/usr/bin/env python3
"""VLM-based frame and object referability prefilter.

This script runs *before* QA generation. For each selected frame it asks a VLM:
  1. whether the frame is usable for spatial reasoning;
  2. which visible object_ids are clearly visible and referable in text.

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
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import select_frames
from src.utils.colmap_loader import (
    CameraIntrinsics,
    CameraPose,
    load_axis_alignment,
    load_scannet_intrinsics,
    load_scannet_poses,
)
from src.utils.coordinate_transform import project_to_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_referability")

DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
DEFAULT_VLM_MODEL = "Qwen2.5-VL-72B-Instruct"
EXCLUDED_LABELS: set[str] = set()


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


def _project_object_bbox(
    obj: dict,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    bbox_min = np.array(obj["bbox_min"], dtype=np.float64)
    bbox_max = np.array(obj["bbox_max"], dtype=np.float64)
    mid = (bbox_min + bbox_max) / 2.0

    sample_points = []
    for x in [bbox_min[0], bbox_max[0]]:
        for y in [bbox_min[1], bbox_max[1]]:
            for z in [bbox_min[2], bbox_max[2]]:
                sample_points.append(np.array([x, y, z], dtype=np.float64))
    sample_points.append(mid)

    projected: list[tuple[float, float]] = []
    in_frame = 0
    valid = 0
    for pt in sample_points:
        uv, depth = project_to_image(pt, pose, intrinsics)
        if uv is None or depth <= 0:
            continue
        valid += 1
        u, v = float(uv[0]), float(uv[1])
        projected.append((u, v))
        if 0 <= u < intrinsics.width and 0 <= v < intrinsics.height:
            in_frame += 1

    if len(projected) < 2:
        return {
            "projected_bbox": None,
            "roi_bounds": None,
            "bbox_in_frame_ratio": 0.0 if valid == 0 else in_frame / valid,
            "projected_area_px": 0.0,
        }

    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    u_min = max(0, int(np.floor(min(us))))
    u_max = min(intrinsics.width - 1, int(np.ceil(max(us))))
    v_min = max(0, int(np.floor(min(vs))))
    v_max = min(intrinsics.height - 1, int(np.ceil(max(vs))))
    area = float(max(0, u_max - u_min) * max(0, v_max - v_min))
    return {
        "projected_bbox": [u_min, v_min, u_max, v_max],
        "roi_bounds": (u_min, u_max, v_min, v_max),
        "bbox_in_frame_ratio": 0.0 if valid == 0 else in_frame / valid,
        "projected_area_px": area,
    }


def _draw_box(image: np.ndarray, bbox: list[int], label: str) -> np.ndarray:
    annotated = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        annotated,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _crop_with_padding(image: np.ndarray, bbox: list[int], pad_ratio: float = 0.15) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return image.copy()
    return crop


def _bbox_iou(box_a: list[int] | None, box_b: list[int] | None) -> float | None:
    if not box_a or not box_b:
        return None
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return None
    return float(inter / union)


def _normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None


def _call_vlm_json(client, model: str, content: list[dict], default: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
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
        "Decide whether this image is usable for visual spatial-reasoning questions. "
        "Reject frames that are too blurry, too dark, too unclear, or where most objects are hard to recognize. "
        "Answer with strict JSON only: "
        '{"frame_usable": true, "reason": "clear_scene"}'
    )


def _object_prompt(label: str) -> str:
    return (
        f'Image 1 is the full scene with one candidate "{label}" highlighted by a red box. '
        f'Image 2 is a crop around the same boxed region. '
        f'Decide whether the boxed "{label}" is clearly visible and whether a question can refer to it simply as '
        f'"the {label}" without confusing the reader. '
        "If multiple same-label instances form one tight cluster in one place, referable may still be true. "
        "If multiple separated same-label instances would make the phrase ambiguous, referable should be false. "
        "Also provide an optional grounding bbox in the full image if you can localize the intended object. "
        "Answer with strict JSON only: "
        '{"visible": true, "clear": true, "referable": true, '
        '"reason": "single_clear_instance", "grounding_bbox": [x1, y1, x2, y2]}'
    )


def _frame_decision(client, model: str, image: np.ndarray) -> dict[str, Any]:
    full_b64 = _image_to_base64(image)
    default = {"frame_usable": True, "reason": "vlm_parse_fallback"}
    result = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
            {"type": "text", "text": _frame_prompt()},
        ],
        default=default,
    )
    return {
        "frame_usable": bool(result.get("frame_usable", True)),
        "reason": str(result.get("reason", default["reason"])),
    }


def _object_decision(
    client,
    model: str,
    image: np.ndarray,
    obj: dict,
    projected_bbox: list[int],
) -> dict[str, Any]:
    label = obj["label"]
    annotated = _draw_box(image, projected_bbox, label)
    crop = _crop_with_padding(image, projected_bbox)
    full_b64 = _image_to_base64(annotated)
    crop_b64 = _image_to_base64(crop)
    default = {
        "visible": True,
        "clear": True,
        "referable": True,
        "reason": "vlm_parse_fallback",
        "grounding_bbox": None,
    }
    result = _call_vlm_json(
        client,
        model,
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": _object_prompt(label)},
        ],
        default=default,
    )
    grounding_bbox = _normalize_bbox(result.get("grounding_bbox"))
    return {
        "visible": bool(result.get("visible", default["visible"])),
        "clear": bool(result.get("clear", default["clear"])),
        "referable": bool(result.get("referable", default["referable"])),
        "reason": str(result.get("reason", default["reason"])),
        "grounding_bbox": grounding_bbox,
    }


def _enforce_single_referable_per_label(
    decisions: dict[str, dict[str, Any]],
    objects_by_id: dict[int, dict[str, Any]],
) -> None:
    winners: dict[str, tuple[str, float]] = {}
    for obj_id, decision in decisions.items():
        if not decision.get("referable", False):
            continue
        label = objects_by_id[int(obj_id)]["label"]
        area = float(decision.get("audit", {}).get("projected_area_px", 0.0))
        if label not in winners or area > winners[label][1]:
            winners[label] = (obj_id, area)

    for obj_id, decision in decisions.items():
        if not decision.get("referable", False):
            continue
        label = objects_by_id[int(obj_id)]["label"]
        winner_id, _ = winners[label]
        if obj_id != winner_id:
            decision["referable"] = False
            if decision.get("reason") in {"single_clear_instance", "vlm_parse_fallback"}:
                decision["reason"] = "same_label_competition"


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
        "version": "1.0",
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
        logger.info("=== Referability scene %s (%d/%d) ===", scene_id, processed + 1, args.max_scenes)

        scene = parse_scene(scene_dir)
        if scene is None:
            continue

        frames = select_frames(scene_dir, scene["objects"], None, args.max_frames)
        if not frames:
            continue

        axis_align = load_axis_alignment(scene_dir)
        poses = load_scannet_poses(scene_dir, axis_alignment=axis_align)
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception as e:
            logger.warning("Color intrinsics load failed for %s: %s", scene_id, e)
            continue

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

            frame_info = _frame_decision(client, model_name, image)
            frame_entry: dict[str, Any] = {
                "frame_usable": frame_info["frame_usable"],
                "frame_reject_reason": None if frame_info["frame_usable"] else frame_info["reason"],
                "referable_object_ids": [],
                "object_decisions": {},
            }
            if not frame_info["frame_usable"]:
                scene_cache[image_name] = frame_entry
                continue

            pose = poses[image_name]
            for obj_id in frame["visible_object_ids"]:
                obj = objects_by_id.get(int(obj_id))
                if obj is None:
                    continue
                if obj["label"].lower() in EXCLUDED_LABELS:
                    continue

                roi_info = _project_object_bbox(obj, pose, color_intrinsics)
                projected_bbox = roi_info["projected_bbox"]
                if projected_bbox is None:
                    frame_entry["object_decisions"][str(obj_id)] = {
                        "label": obj["label"],
                        "visible": False,
                        "clear": False,
                        "referable": False,
                        "reason": "projection_failed",
                        "audit": {
                            "projected_bbox": None,
                            "grounding_bbox": None,
                            "iou": None,
                            "projected_area_px": 0.0,
                            "bbox_in_frame_ratio": roi_info["bbox_in_frame_ratio"],
                        },
                    }
                    continue

                decision = _object_decision(client, model_name, image, obj, projected_bbox)
                audit = {
                    "projected_bbox": projected_bbox,
                    "grounding_bbox": decision["grounding_bbox"],
                    "iou": _bbox_iou(projected_bbox, decision["grounding_bbox"]),
                    "projected_area_px": roi_info["projected_area_px"],
                    "bbox_in_frame_ratio": roi_info["bbox_in_frame_ratio"],
                }
                frame_entry["object_decisions"][str(obj_id)] = {
                    "label": obj["label"],
                    "visible": decision["visible"],
                    "clear": decision["clear"],
                    "referable": decision["referable"],
                    "reason": decision["reason"],
                    "audit": audit,
                }

            _enforce_single_referable_per_label(frame_entry["object_decisions"], objects_by_id)
            frame_entry["referable_object_ids"] = [
                int(obj_id)
                for obj_id, d in frame_entry["object_decisions"].items()
                if d.get("visible", False) and d.get("clear", False) and d.get("referable", False)
            ]
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
