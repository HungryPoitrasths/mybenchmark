#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frame_selector import compute_frame_object_visibility, get_visible_objects, select_frames
from src.scene_parser import (
    ALWAYS_EXCLUDED,
    QUESTION_ONLY_EXCLUDED,
    load_scannet_label_map,
    normalize_label,
    parse_scene,
)
from src.support_graph import enrich_scene_with_support, get_scene_attachment_graph
from src.utils.colmap_loader import load_axis_alignment, load_scannet_depth_intrinsics, load_scannet_intrinsics
from src.utils.depth_occlusion import load_depth_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scene-by-scene HTML audit page.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene_dir", type=Path, help="Raw ScanNet scene directory.")
    source.add_argument("--scene_metadata", type=Path, help="Saved scene metadata JSON.")
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--referability_cache", type=Path, default=None)
    parser.add_argument("--frame_debug_dir", type=Path, default=None)
    parser.add_argument("--label_map", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("scene_debug_viewer.html"))
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--question_limit_per_frame", type=int, default=200)
    parser.add_argument("--image_mode", choices=("inline", "file_uri"), default="inline")
    parser.add_argument("--strict_mode", action="store_true")
    parser.add_argument("--show_rejected_frames", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def h(value: Any) -> str:
    return html.escape(str(value))


def n(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return h(value)
    return f"{number:.0f}" if digits == 0 else f"{number:.{digits}f}"


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def normalize_object_ids(value: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(value, list):
        return out
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def normalize_label_counts(value: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(value, dict):
        return out
    for key, count in value.items():
        if not isinstance(key, str):
            continue
        try:
            out[key] = int(count)
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def normalize_label_to_object_ids(value: Any) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    if not isinstance(value, dict):
        return out
    for key, object_ids in value.items():
        if isinstance(key, str):
            out[key] = normalize_object_ids(object_ids)
    return dict(sorted(out.items()))


def zh_visibility_value(value: Any) -> str:
    mapping = {
        "unknown": "未知",
        "fully visible": "完全可见",
        "partially occluded": "部分遮挡",
        "not visible": "不可见",
        "center_out_of_frame": "中心点不在画面内",
        "missing_depth": "缺少深度图",
        "depth_occluded": "被深度遮挡",
        "low_visible_ratio": "可见比例过低",
        "small_projection": "投影面积过小",
        "bbox_cut_off": "投影框被裁切",
        "too_close_to_edge": "过于靠近图像边缘",
        "missing_roi_sharpness": "缺少 ROI 清晰度",
        "blurry_roi": "ROI 模糊",
        "duplicate_label": "标签重复",
    }
    text = str(value)
    return mapping.get(text, text)


def zh_relation_value(value: Any) -> str:
    mapping = {
        "attachment": "附着",
        "support": "支撑",
        "supported_by": "支撑",
        "resting_on_soft_surface": "放置在软表面上",
        "contained_in": "包含于",
        "affixed_to": "固定于",
    }
    text = str(value)
    return mapping.get(text, text)


def zh_skip_reason(value: Any) -> str:
    mapping = {
        None: "正常生成",
        "strict_mode_missing_depth": "strict_mode 跳过：缺少深度图",
        "strict_mode_missing_color": "strict_mode 跳过：缺少彩色图或内参",
        "strict_mode_too_few_objects": "strict_mode 跳过：严格可用物体少于 3 个",
        "no_referable_objects_or_l1_candidates": "跳过：没有可指代物体，也没有 L1 候选",
        "missing_pose": "跳过：缺少 pose",
    }
    return mapping.get(value, str(value))


def image_path_to_uri(image_path: Path | None, image_mode: str) -> str | None:
    if image_path is None or not image_path.exists():
        return None
    if image_mode == "file_uri":
        return image_path.resolve().as_uri()
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def resolve_scene_id(scene_dir: Path | None, scene_metadata: Path | None) -> str:
    if scene_dir is not None:
        return scene_dir.name
    assert scene_metadata is not None
    data = load_json(scene_metadata)
    return str(data.get("scene_id") or scene_metadata.stem)


def infer_scene_dir(scene_id: str, scene_dir: Path | None, data_root: Path | None) -> Path | None:
    if scene_dir is not None:
        return scene_dir
    if data_root is None:
        return None
    for candidate in (data_root / scene_id, data_root / "scans" / scene_id):
        if candidate.exists():
            return candidate
    return None


def resolve_scene_json_path(requested: Path | None, scene_id: str, scene_metadata: Path | None, default_dir_name: str) -> Path | None:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested / f"{scene_id}.json" if requested.is_dir() else requested)
    if scene_metadata is not None and scene_metadata.parent.name == "scene_metadata":
        candidates.append(scene_metadata.parent.parent / default_dir_name / f"{scene_id}.json")
    candidates.append(PROJECT_ROOT / "output" / default_dir_name / f"{scene_id}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_scene_with_fallback(scene_id: str, scene_dir: Path | None, scene_metadata: Path | None) -> tuple[dict[str, Any], str, str, list[str]]:
    notes: list[str] = []
    if scene_dir is not None:
        try:
            scene = parse_scene(scene_dir)
            if scene is None:
                raise ValueError(f"parse_scene() returned no usable objects for {scene_dir}")
            enrich_scene_with_support(scene)
            return scene, "原始场景", str(scene_dir), notes
        except Exception as exc:
            if scene_metadata is None:
                raise
            notes.append(f"原始场景解析失败，已回退到元数据快照: {exc}")
    assert scene_metadata is not None
    scene = load_json(scene_metadata)
    if "attachment_graph" not in scene and "support_graph" not in scene:
        try:
            enrich_scene_with_support(scene)
            notes.append("元数据里缺少附着图，已在内存中重建")
        except Exception as exc:
            notes.append(f"仅凭元数据无法重建附着图: {exc}")
    return scene, "场景元数据", str(scene_metadata), notes


def load_predictions_map(predictions_path: Path | None, scene_id: str) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    if predictions_path is None or not predictions_path.exists():
        return buckets
    data = load_json(predictions_path)
    if not isinstance(data, list):
        return buckets
    for item in data:
        if str(item.get("scene_id", "")) != scene_id:
            continue
        key = (str(item.get("image_name", "")), str(item.get("question", "")), str(item.get("gt_answer", item.get("answer", ""))))
        buckets[key].append({"prediction": item.get("prediction"), "raw_response": item.get("raw_response")})
    return buckets


def load_scene_questions(scene_id: str, questions_path: Path | None, scene_metadata: Path | None, predictions_path: Path | None, question_limit_per_frame: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path | None]:
    resolved = resolve_scene_json_path(questions_path, scene_id, scene_metadata, "questions")
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary = {"total": 0, "wrong": 0, "by_type": {}, "by_level": {}}
    if resolved is None:
        return summary, by_frame, None
    data = load_json(resolved)
    questions = data.get("questions", []) if isinstance(data, dict) else data
    questions = [q for q in questions if not q.get("scene_id") or str(q.get("scene_id")) == scene_id]
    buckets = load_predictions_map(predictions_path, scene_id)
    type_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    wrong = 0
    for q in questions:
        key = (str(q.get("image_name", "")), str(q.get("question", "")), str(q.get("answer", "")))
        match_list = buckets.get(key, [])
        match = match_list.pop(0) if match_list else None
        pred = match.get("prediction") if match else None
        pred_ok = None if pred is None else str(pred).upper() == str(q.get("answer", "")).upper()
        item = {"level": q.get("level"), "type": q.get("type"), "question": q.get("question"), "options": list(q.get("options", [])), "answer": q.get("answer"), "prediction": pred, "prediction_correct": pred_ok}
        by_frame[str(q.get("image_name", ""))].append(item)
        type_counter[str(item.get("type") or "unknown")] += 1
        level_counter[str(item.get("level") or "unknown")] += 1
        if pred_ok is False:
            wrong += 1
    for frame_name, items in by_frame.items():
        items.sort(key=lambda item: (item.get("prediction_correct") is True, str(item.get("level") or ""), str(item.get("type") or "")))
        if question_limit_per_frame > 0:
            by_frame[frame_name] = items[:question_limit_per_frame]
    summary["total"] = len(questions)
    summary["wrong"] = wrong
    summary["by_type"] = counter_dict(type_counter)
    summary["by_level"] = counter_dict(level_counter)
    return summary, by_frame, resolved


def load_referability_frames(cache_path: Path | None, scene_id: str) -> tuple[dict[str, dict[str, Any]], Path | None]:
    if cache_path is None or not cache_path.exists():
        return {}, None
    cache = load_json(cache_path)
    frames = cache.get("frames", cache)
    if isinstance(frames.get(scene_id), dict):
        return frames[scene_id], cache_path
    prefix = f"{scene_id}/"
    out: dict[str, dict[str, Any]] = {}
    for key, value in frames.items():
        if isinstance(key, str) and key.startswith(prefix) and isinstance(value, dict):
            out[key[len(prefix):]] = value
    return out, cache_path


def load_frame_debug_doc(frame_debug_path: Path | None, scene_id: str, scene_metadata: Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    resolved = resolve_scene_json_path(frame_debug_path, scene_id, scene_metadata, "frame_debug")
    if resolved is None:
        return None, None
    data = load_json(resolved)
    return (data if isinstance(data, dict) else None), resolved


def count_labels(object_ids: list[int], objects_by_id: dict[int, dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for obj_id in object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is not None:
            counter[str(obj.get("label", ""))] += 1
    return counter_dict(counter)


def build_label_to_object_ids(object_ids: list[int], objects_by_id: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for obj_id in object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue
        label = str(obj.get("label", "")).strip()
        if label:
            buckets[label].append(int(obj_id))
    return {label: sorted(ids) for label, ids in sorted(buckets.items())}


def build_annotation_audit(scene: dict[str, Any], scene_dir: Path | None, scene_metadata: Path | None) -> list[dict[str, Any]]:
    base_dir = scene_dir
    scene_name = scene_dir.name if scene_dir is not None else str(scene.get("scene_id", ""))
    if base_dir is None and scene_metadata is not None:
        base_dir = scene_metadata.parent
    if base_dir is None:
        return []
    agg_path = next((path for path in [base_dir / f"{scene_name}_vh_clean.aggregation.json", base_dir / f"{scene_name}.aggregation.json"] if path.exists()), None)
    segs_path = next((path for path in [base_dir / f"{scene_name}_vh_clean.segs.json", base_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json"] if path.exists()), None)
    if agg_path is None:
        return []
    groups = load_json(agg_path).get("segGroups", [])
    seg_indices = np.asarray(load_json(segs_path).get("segIndices", []), dtype=np.int64) if segs_path is not None else None
    scene_ids = {int(obj["id"]) for obj in scene.get("objects", [])}
    audit: list[dict[str, Any]] = []
    for group in groups:
        object_id = int(group.get("objectId", group.get("id", -1)))
        raw_label = str(group.get("label", "unknown"))
        normalized = normalize_label(raw_label)
        segments = [int(seg) for seg in group.get("segments", [])]
        vertex_count = int(np.isin(seg_indices, segments).sum()) if seg_indices is not None and segments else None
        if object_id in scene_ids:
            status = "保留"
        elif normalized in ALWAYS_EXCLUDED:
            status = "始终排除"
        elif normalized in QUESTION_ONLY_EXCLUDED:
            status = "仅题目上下文"
        else:
            status = "已过滤或缺失"
        audit.append({"object_id": object_id, "raw_label": raw_label, "normalized_label": normalized, "status": status, "segment_count": len(segments), "vertex_count": vertex_count})
    audit.sort(key=lambda row: (row["status"], row["normalized_label"], row["object_id"]))
    return audit


def build_scene_attachment_rows(scene: dict[str, Any]) -> list[dict[str, Any]]:
    obj_map = {int(obj["id"]): obj for obj in scene.get("objects", [])}
    rows: list[dict[str, Any]] = []
    edges = scene.get("attachment_edges")
    if isinstance(edges, list) and edges:
        for edge in edges:
            try:
                parent_id = int(edge["parent_id"])
                child_id = int(edge["child_id"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"parent_id": parent_id, "parent_label": str(obj_map.get(parent_id, {}).get("label", "object")), "child_id": child_id, "child_label": str(obj_map.get(child_id, {}).get("label", "object")), "relation_type": str(edge.get("type") or edge.get("relation_type") or "attachment"), "confidence": edge.get("confidence", edge.get("score"))})
        rows.sort(key=lambda row: (row["parent_label"], row["child_label"], row["parent_id"], row["child_id"]))
        return rows
    try:
        graph = get_scene_attachment_graph(scene, scene_id=str(scene.get("scene_id", "<unknown>")))
    except Exception:
        return rows
    for parent_id, child_ids in graph.items():
        for child_id in child_ids:
            rows.append({"parent_id": int(parent_id), "parent_label": str(obj_map.get(int(parent_id), {}).get("label", "object")), "child_id": int(child_id), "child_label": str(obj_map.get(int(child_id), {}).get("label", "object")), "relation_type": "support", "confidence": None})
    rows.sort(key=lambda row: (row["parent_label"], row["child_label"], row["parent_id"], row["child_id"]))
    return rows


def filter_frame_attachment_rows(scene_attachment_rows: list[dict[str, Any]], relevant_object_ids: set[int]) -> list[dict[str, Any]]:
    return [row for row in scene_attachment_rows if int(row["parent_id"]) in relevant_object_ids and int(row["child_id"]) in relevant_object_ids]


def attachment_summary_for_object(obj_id: int, frame_attachment_rows: list[dict[str, Any]]) -> str:
    attached_to = [f'{row["parent_label"]} #{row["parent_id"]}' for row in frame_attachment_rows if int(row["child_id"]) == obj_id]
    carries = [f'{row["child_label"]} #{row["child_id"]}' for row in frame_attachment_rows if int(row["parent_id"]) == obj_id]
    parts: list[str] = []
    if attached_to:
        parts.append("附着于 " + ", ".join(attached_to))
    if carries:
        parts.append("承载 " + ", ".join(carries))
    return "；".join(parts) if parts else "-"


def load_selected_scannet_poses(scene_dir: Path, image_names: list[str], axis_alignment: np.ndarray | None = None) -> dict[str, Any]:
    from src.utils.colmap_loader import CameraPose

    pose_dir = scene_dir / "pose"
    color_dir = scene_dir / "color"
    if not pose_dir.exists():
        return {}
    alignment = axis_alignment if axis_alignment is not None else np.eye(4, dtype=np.float64)
    poses: dict[str, CameraPose] = {}
    for image_name in image_names:
        if image_name in poses:
            continue
        pose_file = pose_dir / f'{str(image_name).replace(".jpg", "")}.txt'
        if not pose_file.exists() or not (color_dir / image_name).exists():
            continue
        t_c2w = np.loadtxt(str(pose_file))
        if not np.isfinite(t_c2w).all():
            continue
        aligned = alignment @ t_c2w
        r_c2w = aligned[:3, :3]
        t_c2w_vec = aligned[:3, 3]
        poses[image_name] = CameraPose(image_name=image_name, rotation=r_c2w.T.astype(np.float64), translation=(-r_c2w.T @ t_c2w_vec).astype(np.float64))
    return poses


def sort_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items.sort(key=lambda item: (item.get("prediction_correct") is True, str(item.get("level") or ""), str(item.get("type") or "")))
    return items


def normalize_question_items(items: list[dict[str, Any]], question_limit_per_frame: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        pred = item.get("prediction")
        answer = item.get("answer")
        pred_ok = item.get("prediction_correct")
        if pred_ok is None and pred is not None and answer is not None:
            pred_ok = str(pred).upper() == str(answer).upper()
        normalized.append({"level": item.get("level"), "type": item.get("type"), "question": item.get("question"), "options": list(item.get("options", [])), "answer": item.get("answer"), "prediction": pred, "prediction_correct": pred_ok})
    sort_questions(normalized)
    return normalized[:question_limit_per_frame] if question_limit_per_frame > 0 else normalized


def normalize_object_rows(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            obj_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        out.append({
            "id": obj_id,
            "label": str(row.get("label", "")),
            "tags": list(row.get("tags", [])),
            "attachment_summary": str(row.get("attachment_summary", "-")),
            "occlusion_status": row.get("occlusion_status"),
            "visible_ratio": row.get("visible_ratio"),
            "projected_area_px": row.get("projected_area_px"),
            "bbox_in_frame_ratio": row.get("bbox_in_frame_ratio"),
            "roi_sharpness": row.get("roi_sharpness"),
            "center_uv_px": row.get("center_uv_px"),
            "eligible_as_reference": bool(row.get("eligible_as_reference", False)),
            "rejection_reasons": list(row.get("rejection_reasons", [])),
        })
    out.sort(key=lambda row: ("VLM唯一" not in row["tags"], "Pipeline可用" not in row["tags"], str(row["label"]), int(row["id"])))
    return out


def normalize_attachment_rows(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            parent_id = int(row.get("parent_id"))
            child_id = int(row.get("child_id"))
        except (TypeError, ValueError):
            continue
        out.append({"parent_id": parent_id, "parent_label": str(row.get("parent_label", "object")), "child_id": child_id, "child_label": str(row.get("child_label", "object")), "relation_type": str(row.get("relation_type", "attachment")), "confidence": row.get("confidence")})
    out.sort(key=lambda row: (row["parent_label"], row["child_label"], row["parent_id"], row["child_id"]))
    return out


def build_rejected_frames(referability_frames: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rejected: list[dict[str, str]] = []
    for image_name, entry in sorted(referability_frames.items()):
        if isinstance(entry, dict) and entry.get("frame_usable") is False:
            rejected.append({"image_name": image_name, "reason": str(entry.get("frame_reject_reason") or "未提供原因")})
    return rejected


def build_frames_from_debug_doc(frame_debug_doc: dict[str, Any], scene_dir: Path | None, scene_questions: dict[str, list[dict[str, Any]]], question_limit_per_frame: int, image_mode: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for entry in frame_debug_doc.get("frames", []):
        if not isinstance(entry, dict):
            continue
        image_name = str(entry.get("image_name", ""))
        questions = list(scene_questions.get(image_name, [])) or normalize_question_items(list(entry.get("final_questions") or entry.get("generated_questions") or []), question_limit_per_frame)
        wrong = sum(1 for item in questions if item.get("prediction_correct") is False)
        image_path = scene_dir / "color" / image_name if scene_dir is not None else None
        frames.append({
            "image_name": image_name,
            "image_uri": image_path_to_uri(image_path, image_mode),
            "frame_usable": bool(entry.get("frame_usable", True)),
            "frame_reject_reason": entry.get("frame_reject_reason"),
            "pipeline_skip_reason": entry.get("pipeline_skip_reason"),
            "selector_visible_object_ids": normalize_object_ids(entry.get("selector_visible_object_ids")),
            "selector_visible_label_counts": normalize_label_counts(entry.get("selector_visible_label_counts")),
            "pipeline_visible_ids": normalize_object_ids(entry.get("pipeline_visible_object_ids_used_for_generation")),
            "pipeline_visible_label_counts": normalize_label_counts(entry.get("pipeline_visible_label_counts")),
            "vlm_label_counts": normalize_label_counts(entry.get("vlm_label_counts")),
            "referable_object_ids": normalize_object_ids(entry.get("referable_object_ids")),
            "candidate_labels": list(entry.get("candidate_labels", [])),
            "label_to_object_ids": normalize_label_to_object_ids(entry.get("label_to_object_ids")),
            "object_rows": normalize_object_rows(entry.get("object_rows")),
            "attachment_rows": normalize_attachment_rows(entry.get("attachment_rows")),
            "questions": questions,
            "question_wrong": wrong,
        })
    return [frame for frame in frames if frame["frame_usable"] is not False]


def build_fallback_frames(scene: dict[str, Any], scene_dir: Path | None, max_frames: int, referability_frames: dict[str, dict[str, Any]], scene_questions: dict[str, list[dict[str, Any]]], question_limit_per_frame: int, image_mode: str) -> list[dict[str, Any]]:
    objects = list(scene.get("objects") or [])
    objects_by_id = {int(obj["id"]): obj for obj in objects}
    scene_attachment_rows = build_scene_attachment_rows(scene)
    try:
        attachment_graph = get_scene_attachment_graph(scene, scene_id=str(scene.get("scene_id", "<unknown>")))
    except Exception:
        attachment_graph = {}
    ordered_names = [image_name for image_name, entry in sorted(referability_frames.items()) if not isinstance(entry, dict) or entry.get("frame_usable") is not False]
    for image_name in sorted(scene_questions.keys()):
        if image_name not in ordered_names:
            ordered_names.append(image_name)
    selected_frames: list[dict[str, Any]] = []
    if scene_dir is not None and not ordered_names:
        selected_frames = select_frames(scene_dir, objects, attachment_graph, max_frames=max_frames)
        ordered_names = [frame["image_name"] for frame in selected_frames]
    selected_by_name = {frame["image_name"]: frame for frame in selected_frames}
    poses = {}
    color_intrinsics = None
    depth_intrinsics = None
    if scene_dir is not None:
        axis_align = load_axis_alignment(scene_dir)
        poses = load_selected_scannet_poses(scene_dir, ordered_names, axis_alignment=axis_align)
        try:
            color_intrinsics = load_scannet_intrinsics(scene_dir)
        except Exception:
            color_intrinsics = None
        try:
            depth_intrinsics = load_scannet_depth_intrinsics(scene_dir)
        except Exception:
            depth_intrinsics = None
    frames: list[dict[str, Any]] = []
    for image_name in ordered_names:
        ref = referability_frames.get(image_name) or {}
        selector_visible_ids = normalize_object_ids(ref.get("selector_visible_object_ids") or ref.get("candidate_visible_object_ids") or selected_by_name.get(image_name, {}).get("visible_object_ids") or [])
        pipeline_visible_ids = list(selector_visible_ids)
        label_to_object_ids = normalize_label_to_object_ids(ref.get("label_to_object_ids")) or build_label_to_object_ids(selector_visible_ids, objects_by_id)
        object_rows: list[dict[str, Any]] = []
        attachment_rows: list[dict[str, Any]] = []
        if scene_dir is not None and color_intrinsics is not None and image_name in poses:
            pose = poses[image_name]
            depth_image = None
            if depth_intrinsics is not None:
                depth_path = scene_dir / "depth" / image_name.replace(".jpg", ".png")
                if depth_path.exists():
                    try:
                        depth_image = load_depth_image(depth_path)
                    except Exception:
                        depth_image = None
            image_path = scene_dir / "color" / image_name
            visibility_table = compute_frame_object_visibility(objects, pose, color_intrinsics, image_path=image_path if image_path.exists() else None, depth_image=depth_image, depth_intrinsics=depth_intrinsics, strict_mode=True)
            if not pipeline_visible_ids:
                pipeline_visible_ids = [int(obj["id"]) for obj in get_visible_objects(objects, pose, color_intrinsics)]
            relevant_ids = set(selector_visible_ids) | set(pipeline_visible_ids) | set(normalize_object_ids(ref.get("referable_object_ids")))
            attachment_rows = filter_frame_attachment_rows(scene_attachment_rows, relevant_ids)
            for obj in objects:
                obj_id = int(obj["id"])
                if relevant_ids and obj_id not in relevant_ids:
                    continue
                meta = visibility_table.get(obj_id, {})
                tags: list[str] = []
                if obj_id in selector_visible_ids:
                    tags.append("VLM候选")
                if obj_id in normalize_object_ids(ref.get("referable_object_ids")):
                    tags.append("VLM唯一")
                if obj_id in pipeline_visible_ids:
                    tags.append("Pipeline可用")
                if any(obj_id in (int(row["parent_id"]), int(row["child_id"])) for row in attachment_rows):
                    tags.append("被attachment约束")
                object_rows.append({"id": obj_id, "label": str(obj.get("label", "")), "tags": tags, "attachment_summary": attachment_summary_for_object(obj_id, attachment_rows), "occlusion_status": meta.get("occlusion_status"), "visible_ratio": meta.get("visible_ratio"), "projected_area_px": meta.get("projected_area_px"), "bbox_in_frame_ratio": meta.get("bbox_in_frame_ratio"), "roi_sharpness": meta.get("roi_sharpness"), "center_uv_px": meta.get("center_uv_px"), "eligible_as_reference": bool(meta.get("eligible_as_reference", False)), "rejection_reasons": list(meta.get("rejection_reasons", []))})
            object_rows = normalize_object_rows(object_rows)
        questions = list(scene_questions.get(image_name, [])) or normalize_question_items([], question_limit_per_frame)
        wrong = sum(1 for item in questions if item.get("prediction_correct") is False)
        image_path = scene_dir / "color" / image_name if scene_dir is not None else None
        frames.append({"image_name": image_name, "image_uri": image_path_to_uri(image_path, image_mode), "frame_usable": None if not ref else bool(ref.get("frame_usable", True)), "frame_reject_reason": ref.get("frame_reject_reason"), "pipeline_skip_reason": None, "selector_visible_object_ids": selector_visible_ids, "selector_visible_label_counts": normalize_label_counts(ref.get("selector_visible_label_counts")) or count_labels(selector_visible_ids, objects_by_id), "pipeline_visible_ids": pipeline_visible_ids, "pipeline_visible_label_counts": count_labels(pipeline_visible_ids, objects_by_id), "vlm_label_counts": normalize_label_counts(ref.get("label_counts")), "referable_object_ids": normalize_object_ids(ref.get("referable_object_ids")), "candidate_labels": list(ref.get("candidate_labels", [])) or sorted(label_to_object_ids.keys()), "label_to_object_ids": label_to_object_ids, "object_rows": object_rows, "attachment_rows": attachment_rows, "questions": questions, "question_wrong": wrong})
    return [frame for frame in frames if frame["frame_usable"] is not False]


def summarize_questions_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    type_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    total = 0
    wrong = 0
    for frame in frames:
        for question in frame.get("questions", []):
            total += 1
            type_counter[str(question.get("type") or "unknown")] += 1
            level_counter[str(question.get("level") or "unknown")] += 1
            if question.get("prediction_correct") is False:
                wrong += 1
    return {"total": total, "wrong": wrong, "by_type": counter_dict(type_counter), "by_level": counter_dict(level_counter)}


def render_vlm_count_table(frame: dict[str, Any]) -> str:
    labels = sorted(set(frame["candidate_labels"]) | set(frame["label_to_object_ids"].keys()) | set(frame["vlm_label_counts"].keys()))
    if not labels:
        return '<div class="muted">这一帧没有 VLM 类别计数数据。</div>'
    referable_set = set(frame["referable_object_ids"])
    rows = []
    for label in labels:
        candidate_ids = frame["label_to_object_ids"].get(label, [])
        vlm_count = int(frame["vlm_label_counts"].get(label, 0))
        unique = vlm_count == 1 and len(candidate_ids) == 1 and candidate_ids[0] in referable_set
        rows.append(f"<tr><td>{h(label)}</td><td>{vlm_count}</td><td>{h(', '.join(str(obj_id) for obj_id in candidate_ids) or '-')}</td><td>{'是' if unique else '否'}</td></tr>")
    return "<table><thead><tr><th>类别</th><th>VLM 数量</th><th>候选对象 ID</th><th>是否唯一可指代</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_object_table(frame: dict[str, Any]) -> str:
    if not frame["object_rows"]:
        return '<div class="muted">这一帧没有逐物体审计数据。</div>'
    rows = []
    for row in frame["object_rows"]:
        flags = " / ".join(row.get("tags", [])) or "-"
        reasons = "，".join(zh_visibility_value(reason) for reason in row.get("rejection_reasons", [])) or "-"
        rows.append(
            "<tr>"
            f"<td>{row['id']}</td><td>{h(row['label'])}</td><td>{h(flags)}</td><td>{h(row.get('attachment_summary', '-'))}</td>"
            f"<td>{h(zh_visibility_value(row.get('occlusion_status', '-')))}</td><td>{n(row.get('visible_ratio'), 2)}</td>"
            f"<td>{n(row.get('projected_area_px'), 0)}</td><td>{n(row.get('bbox_in_frame_ratio'), 2)}</td>"
            f"<td>{n(row.get('roi_sharpness'), 1)}</td><td>{h(reasons)}</td></tr>"
        )
    head = "<table><thead><tr><th>ID</th><th>类别</th><th>标记</th><th>attachment 关系</th><th>遮挡状态</th><th>可见比例</th><th>投影面积</th><th>框内比例</th><th>清晰度</th><th>过滤原因</th></tr></thead>"
    return head + "<tbody>" + "".join(rows) + "</tbody></table>"


def render_attachment_table(rows: list[dict[str, Any]], empty_text: str) -> str:
    if not rows:
        return f'<div class="muted">{h(empty_text)}</div>'
    body = "".join(f"<tr><td>{row['parent_id']}</td><td>{h(row['parent_label'])}</td><td>{row['child_id']}</td><td>{h(row['child_label'])}</td><td>{h(zh_relation_value(row['relation_type']))}</td><td>{n(row.get('confidence'), 3)}</td></tr>" for row in rows)
    return "<table><thead><tr><th>父 ID</th><th>父类别</th><th>子 ID</th><th>子类别</th><th>类型</th><th>confidence</th></tr></thead><tbody>" + body + "</tbody></table>"


def render_questions(frame: dict[str, Any]) -> str:
    if not frame["questions"]:
        return '<div class="muted">这一帧没有题目。</div>'
    cards = []
    for question in frame["questions"]:
        klass = "bad" if question.get("prediction_correct") is False else "ok" if question.get("prediction_correct") is True else ""
        if question.get("prediction_correct") is False:
            status = f'<span class="pill bad">预测 {h(question["prediction"])} / 真值 {h(question["answer"])}</span>'
        elif question.get("prediction_correct") is True:
            status = '<span class="pill ok">正确</span>'
        else:
            status = f'<span class="pill warn">答案 {h(question["answer"])}</span>'
        options = "<br>".join(f"{chr(65 + idx)}) {h(option)}" for idx, option in enumerate(question["options"]))
        cards.append(f'<div class="qcard {klass}"><div class="qmeta"><span class="pill">{h(question["level"])}</span><span class="pill">{h(question["type"])}</span>{status}</div><div class="qtext">{h(question["question"])}</div><div class="qopts">{options}</div></div>')
    return "".join(cards)


def render_rejected_frames(rejected_frames: list[dict[str, str]]) -> str:
    if not rejected_frames:
        return ""
    rows = "".join(f"<tr><td>{h(item['image_name'])}</td><td>{h(item['reason'])}</td></tr>" for item in rejected_frames)
    return f'<details class="card"><summary>VLM 拒绝帧 ({len(rejected_frames)})</summary><table><thead><tr><th>帧名</th><th>拒绝原因</th></tr></thead><tbody>{rows}</tbody></table></details>'


def render_metric_notes(strict_mode: bool) -> str:
    strict_text = "开" if strict_mode else "关"
    return """
<section class="card"><h2>指标说明</h2>
<div class="muted">页面默认只展示 VLM 通过帧；错位的 2D 叠框已经移除。过滤原因一列始终按严格审计规则计算，用来排错。</div>
<table><thead><tr><th>字段</th><th>含义</th></tr></thead><tbody>
<tr><td>VLM候选</td><td>referability 阶段认为在这帧里可见的候选对象。</td></tr>
<tr><td>VLM唯一</td><td>VLM 判断该类别数量为 1，且候选对象也只有 1 个。</td></tr>
<tr><td>Pipeline可用</td><td>这次 run_pipeline 实际传给 generate_all_questions() 的对象集合。</td></tr>
<tr><td>被attachment约束</td><td>该物体在当前帧相关对象里参与了 attachment 边。</td></tr>
<tr><td>可见比例</td><td>visible_in_frame_count / valid_in_frame_count。</td></tr>
<tr><td>投影面积</td><td>3D bbox 采样点投影后的 2D 包围区域面积，单位像素。</td></tr>
<tr><td>框内比例</td><td>投影采样点中落在图像范围内的比例。</td></tr>
<tr><td>清晰度</td><td>投影 ROI 灰度图的 Laplacian variance。</td></tr>
<tr><td>strict 阈值</td><td>visible_ratio>=0.6, projected_area_px>=800, bbox_in_frame_ratio>=0.6, edge_margin_px>=12, roi_sharpness>=45, 且标签唯一。当前运行 strict_mode=""" + strict_text + """。</td></tr>
</tbody></table></section>
"""


def render_frame_section(frame: dict[str, Any], index: int, strict_mode: bool) -> str:
    image_block = f'<div class="imgwrap"><img src="{h(frame["image_uri"])}" alt="{h(frame["image_name"])}"></div>' if frame["image_uri"] else '<div class="imgwrap img-missing">这一帧缺少图像文件。</div>'
    return (
        f'<section class="card frame" id="frame-{index}"><h2>{h(frame["image_name"])}</h2>'
        f'<div class="metrics"><div class="metric"><div class="k">VLM</div><div class="v">通过</div><div class="s">{h(frame["frame_reject_reason"] or "该帧已通过 referability")}</div></div>'
        f'<div class="metric"><div class="k">Pipeline</div><div class="v">{len(frame["pipeline_visible_ids"])}</div><div class="s">{h(zh_skip_reason(frame["pipeline_skip_reason"]))} | strict_mode={"开" if strict_mode else "关"}</div></div>'
        f'<div class="metric"><div class="k">VLM 唯一对象</div><div class="v">{len(frame["referable_object_ids"])}</div><div class="s">selector 可见={len(frame["selector_visible_object_ids"])}</div></div>'
        f'<div class="metric"><div class="k">题目数</div><div class="v">{len(frame["questions"])}</div><div class="s">错题={frame["question_wrong"]}</div></div></div>'
        f'<div class="twocol"><div><h3>原始帧</h3>{image_block}</div><div><h3>VLM 类别计数</h3>{render_vlm_count_table(frame)}</div></div>'
        f'<h3>逐物体审计</h3>{render_object_table(frame)}<h3>这一帧的 attachment</h3>{render_attachment_table(frame["attachment_rows"], "这一帧没有相关 attachment 对。")}<h3>题目</h3>{render_questions(frame)}</section>'
    )


def render_annotation_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">没有原始 aggregation 审计数据。使用 --scene_dir 可启用这张表。</div>'
    body = "".join(f"<tr><td>{row['object_id']}</td><td>{h(row['raw_label'])}</td><td>{h(row['normalized_label'])}</td><td>{h(row['status'])}</td><td>{row['segment_count']}</td><td>{h(row['vertex_count']) if row['vertex_count'] is not None else '-'}</td></tr>" for row in rows)
    return "<table><thead><tr><th>ID</th><th>原始标签</th><th>归一化标签</th><th>状态</th><th>分段数</th><th>顶点数</th></tr></thead><tbody>" + body + "</tbody></table>"


def render_html(scene_id: str, source_mode: str, source_detail: str, notes: list[str], question_summary: dict[str, Any], frames: list[dict[str, Any]], rejected_frames: list[dict[str, str]], scene_attachment_rows: list[dict[str, Any]], annotation_rows: list[dict[str, Any]], open3d_cmd: str, strict_mode: bool) -> str:
    note_html = "".join(f"<li>{h(note)}</li>" for note in notes) or "<li>无额外说明。</li>"
    frame_nav = "".join(f'<a class="pill" href="#frame-{index}">{h(frame["image_name"])}</a>' for index, frame in enumerate(frames))
    frame_sections = "".join(render_frame_section(frame, index, strict_mode) for index, frame in enumerate(frames))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h(scene_id)} 场景调试页</title><style>
body{{margin:0;background:#edf1f5;color:#132030;font:14px/1.5 "Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif}} .page{{max-width:1500px;margin:0 auto;padding:16px;display:grid;gap:16px}}
.card{{background:#fff;border:1px solid rgba(19,32,48,.08);border-radius:18px;box-shadow:0 14px 32px rgba(19,32,48,.06);padding:16px}} h1,h2,h3{{margin:0 0 10px}} h1{{font-size:28px}} h2{{font-size:22px}} h3{{font-size:17px;margin-top:16px}}
.muted{{color:#617387}} .metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}} .metric{{background:#f7fafc;border:1px solid rgba(19,32,48,.06);border-radius:14px;padding:10px 12px}} .k{{font-size:12px;color:#667788}} .v{{font-size:22px;font-weight:700}} .s{{font-size:12px;color:#6f8192;margin-top:4px}}
.pill{{display:inline-block;padding:4px 9px;border-radius:999px;background:#eef4fa;border:1px solid rgba(19,32,48,.08);margin:0 6px 6px 0;color:#24476a;text-decoration:none}} .pill.ok{{background:#e9f7ef;color:#1e7a41}} .pill.bad{{background:#fff1f1;color:#9b2424}} .pill.warn{{background:#fff6e8;color:#8a5400}}
.twocol{{display:grid;grid-template-columns:minmax(0,1fr) minmax(340px,.95fr);gap:16px;align-items:start}} .imgwrap{{background:#111a26;border-radius:14px;overflow:hidden;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center}} .imgwrap img{{width:100%;height:100%;object-fit:contain;display:block}} .img-missing{{color:#d7dfe8}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:8px 10px;border-bottom:1px solid rgba(19,32,48,.08);text-align:left;vertical-align:top}} th{{font-size:12px;color:#607284;background:#fff;position:sticky;top:0}}
.qcard{{border:1px solid rgba(19,32,48,.08);border-radius:14px;padding:10px 12px;margin-bottom:10px;background:#fff}} .qcard.bad{{border-color:rgba(180,58,58,.18);background:#fff8f8}} .qcard.ok{{border-color:rgba(34,139,80,.18);background:#f8fffb}} .qmeta{{margin-bottom:6px}} .qtext{{margin-bottom:8px}} .qopts{{color:#42576b}}
pre{{background:#101722;color:#d7e0ea;padding:14px;border-radius:14px;overflow:auto}} ul{{margin:8px 0 0 18px}} details summary{{cursor:pointer;font-weight:700;margin-bottom:10px}} @media(max-width:1180px){{.metrics,.twocol{{grid-template-columns:1fr}}}}
</style></head><body><div class="page">
<section class="card"><h1>{h(scene_id)}</h1><div class="muted">主页面默认只显示 VLM 通过帧。页面优先读取 frame_debug/<scene_id>.json 的真实运行结果；如果没有，再回退到 viewer 现场重算。</div>
<div class="metrics" style="margin-top:12px"><div class="metric"><div class="k">数据来源</div><div class="v">{h(source_mode)}</div><div class="s">{h(source_detail)}</div></div><div class="metric"><div class="k">展示帧数</div><div class="v">{len(frames)}</div><div class="s">VLM 拒绝帧={len(rejected_frames)}</div></div><div class="metric"><div class="k">题目统计</div><div class="v">{question_summary['total']}</div><div class="s">错题={question_summary['wrong']}</div></div><div class="metric"><div class="k">场景 attachment</div><div class="v">{len(scene_attachment_rows)}</div><div class="s">strict_mode={"开" if strict_mode else "关"}</div></div></div>
<h3>Open3D 几何检查</h3><pre>{h(open3d_cmd)}</pre><h3>备注</h3><ul>{note_html}</ul><h3>帧索引</h3>{frame_nav or '<div class="muted">没有可展示的 VLM 通过帧。</div>'}</section>
{render_rejected_frames(rejected_frames)}{render_metric_notes(strict_mode)}{frame_sections}
<section class="card"><h2>全场景 attachment</h2>{render_attachment_table(scene_attachment_rows, "整个场景没有可用的 attachment 边。")}</section>
<section class="card"><h2>原始标注审计</h2>{render_annotation_table(annotation_rows)}</section>
</div></body></html>"""


def main() -> None:
    args = parse_args()
    if args.label_map is not None:
        load_scannet_label_map(args.label_map)
    scene_id = resolve_scene_id(args.scene_dir, args.scene_metadata)
    scene_dir = infer_scene_dir(scene_id, args.scene_dir, args.data_root)
    scene, source_mode, source_detail, notes = load_scene_with_fallback(scene_id, scene_dir, args.scene_metadata)
    question_summary, scene_questions, question_path = load_scene_questions(scene_id, args.questions, args.scene_metadata, args.predictions, args.question_limit_per_frame)
    referability_frames, cache_path = load_referability_frames(args.referability_cache, scene_id)
    frame_debug_doc, frame_debug_path = load_frame_debug_doc(args.frame_debug_dir, scene_id, args.scene_metadata)
    notes.append(f"已加载真实运行调试 JSON: {frame_debug_path}" if frame_debug_path is not None else "没有找到 frame_debug JSON，已回退为 viewer 现场重算")
    if cache_path is None:
        notes.append("未加载 referability cache，无法展示 VLM 拒绝帧摘要")
    if question_path is None and frame_debug_path is None:
        notes.append("未找到题目文件，题目卡片将为空")
    if args.question_limit_per_frame > 0:
        notes.append(f"每帧题目卡片最多显示前 {args.question_limit_per_frame} 条")
    frames = build_frames_from_debug_doc(frame_debug_doc, scene_dir, scene_questions, args.question_limit_per_frame, args.image_mode) if frame_debug_doc is not None else build_fallback_frames(scene, scene_dir, args.max_frames, referability_frames, scene_questions, args.question_limit_per_frame, args.image_mode)
    if question_summary["total"] == 0:
        question_summary = summarize_questions_from_frames(frames)
    rejected_frames = build_rejected_frames(referability_frames) if args.show_rejected_frames else []
    scene_attachment_rows = build_scene_attachment_rows(scene)
    annotation_rows = build_annotation_audit(scene, scene_dir, args.scene_metadata)
    open3d_cmd = f'python scripts/view_scannet_bbox_open3d.py --scene_dir "{scene_dir}" --render_mode mesh --show_centers' if scene_dir is not None else f'python scripts/view_scannet_bbox_open3d.py --scene_metadata "{args.scene_metadata}" --render_mode mesh --show_centers'
    html_text = render_html(scene_id, source_mode, source_detail, notes, question_summary, frames, rejected_frames, scene_attachment_rows, annotation_rows, open3d_cmd, args.strict_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"wrote debug viewer to {args.output}")


if __name__ == "__main__":
    main()
