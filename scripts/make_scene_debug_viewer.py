#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
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
from src.utils.colmap_loader import (
    load_axis_alignment,
    load_scannet_depth_intrinsics,
    load_scannet_intrinsics,
)
from src.utils.depth_occlusion import load_depth_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static scene-by-scene debug HTML for frame selection, VLM counts, and question outputs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene_dir", type=Path, help="Raw ScanNet scene directory.")
    source.add_argument("--scene_metadata", type=Path, help="Saved scene metadata JSON.")
    parser.add_argument("--data_root", type=Path, default=None, help="Optional root used to infer scene_dir from scene_id.")
    parser.add_argument("--questions", type=Path, default=None, help="Scene question JSON or directory containing <scene_id>.json.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional VLM predictions JSON.")
    parser.add_argument("--referability_cache", type=Path, default=None, help="Optional run_vlm_referability.py cache JSON.")
    parser.add_argument("--label_map", type=Path, default=None, help="Optional scannetv2-labels.combined.tsv used to match pipeline normalization.")
    parser.add_argument("--output", type=Path, default=Path("scene_debug_viewer.html"))
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--question_limit_per_frame", type=int, default=200, help="0 means no limit.")
    parser.add_argument("--strict_mode", action="store_true")
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
    if digits == 0:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


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


def load_scene_with_fallback(
    scene_id: str,
    scene_dir: Path | None,
    scene_metadata: Path | None,
) -> tuple[dict[str, Any], str, str, list[str]]:
    notes: list[str] = []
    if scene_dir is not None:
        try:
            scene = parse_scene(scene_dir)
            if scene is None:
                raise ValueError(f"parse_scene() returned no usable objects for {scene_dir}")
            enrich_scene_with_support(scene)
            return scene, "raw_scene", str(scene_dir), notes
        except Exception as exc:
            if scene_metadata is None:
                raise
            notes.append(f"raw scene parse failed; using metadata snapshot instead: {exc}")
    assert scene_metadata is not None
    scene = load_json(scene_metadata)
    if "attachment_graph" not in scene and "support_graph" not in scene:
        try:
            enrich_scene_with_support(scene)
            notes.append("metadata had no support graph; rebuilt support/attachment fields in memory")
        except Exception as exc:
            notes.append(f"could not rebuild support graph from metadata only: {exc}")
    return scene, "scene_metadata", str(scene_metadata), notes


def resolve_scene_json_path(
    requested: Path | None,
    scene_id: str,
    scene_metadata: Path | None,
    default_dir_name: str,
) -> Path | None:
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


def load_scene_questions(
    scene_id: str,
    questions_path: Path | None,
    scene_metadata: Path | None,
    predictions_path: Path | None,
    question_limit_per_frame: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path | None]:
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
        item = {
            "level": q.get("level"),
            "type": q.get("type"),
            "question": q.get("question"),
            "options": list(q.get("options", [])),
            "answer": q.get("answer"),
            "prediction": pred,
            "prediction_correct": pred_ok,
        }
        by_frame[str(q.get("image_name", ""))].append(item)
        type_counter[str(item.get("type") or "unknown")] += 1
        level_counter[str(item.get("level") or "unknown")] += 1
        if pred_ok is False:
            wrong += 1
    for frame_name, items in by_frame.items():
        items.sort(key=lambda x: (x.get("prediction_correct") is True, str(x.get("level") or ""), str(x.get("type") or "")))
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


def count_labels(object_ids: list[int], objects_by_id: dict[int, dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for obj_id in object_ids:
        obj = objects_by_id.get(int(obj_id))
        if obj is not None:
            counter[str(obj.get("label", ""))] += 1
    return counter_dict(counter)


def build_annotation_audit(scene: dict[str, Any], scene_dir: Path | None, scene_metadata: Path | None) -> list[dict[str, Any]]:
    base_dir = scene_dir
    scene_name = scene_dir.name if scene_dir is not None else str(scene.get("scene_id", ""))
    if base_dir is None and scene_metadata is not None:
        base_dir = scene_metadata.parent
    if base_dir is None:
        return []
    candidates = [base_dir / f"{scene_name}_vh_clean.aggregation.json", base_dir / f"{scene_name}.aggregation.json"]
    agg_path = next((p for p in candidates if p.exists()), None)
    segs_candidates = [base_dir / f"{scene_name}_vh_clean.segs.json", base_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json"]
    segs_path = next((p for p in segs_candidates if p.exists()), None)
    if agg_path is None:
        return []
    groups = load_json(agg_path).get("segGroups", [])
    seg_indices = None
    if segs_path is not None:
        seg_indices = np.asarray(load_json(segs_path).get("segIndices", []), dtype=np.int64)
    scene_ids = {int(obj["id"]) for obj in scene.get("objects", [])}
    audit: list[dict[str, Any]] = []
    for group in groups:
        object_id = int(group.get("objectId", group.get("id", -1)))
        raw_label = str(group.get("label", "unknown"))
        normalized = normalize_label(raw_label)
        segments = [int(seg) for seg in group.get("segments", [])]
        vertex_count = int(np.isin(seg_indices, segments).sum()) if seg_indices is not None and segments else None
        if object_id in scene_ids:
            status = "kept"
        elif normalized in ALWAYS_EXCLUDED:
            status = "always_excluded"
        elif normalized in QUESTION_ONLY_EXCLUDED:
            status = "question_only_context"
        else:
            status = "filtered_or_missing"
        audit.append({"object_id": object_id, "raw_label": raw_label, "normalized_label": normalized, "status": status, "segment_count": len(segments), "vertex_count": vertex_count})
    audit.sort(key=lambda x: (x["status"], x["normalized_label"], x["object_id"]))
    return audit


def build_support_rows(scene: dict[str, Any]) -> list[dict[str, Any]]:
    obj_map = {int(obj["id"]): obj for obj in scene.get("objects", [])}
    rows: list[dict[str, Any]] = []
    edges = scene.get("attachment_edges")
    if isinstance(edges, list) and edges:
        for edge in edges:
            rows.append({
                "parent_id": int(edge["parent_id"]),
                "child_id": int(edge["child_id"]),
                "parent_label": str(obj_map.get(int(edge["parent_id"]), {}).get("label", "object")),
                "child_label": str(obj_map.get(int(edge["child_id"]), {}).get("label", "object")),
                "relation_type": str(edge.get("relation_type", "attachment")),
                "score": edge.get("score"),
                "source": "attachment_edges",
            })
        return rows
    try:
        graph = get_scene_attachment_graph(scene, scene_id=str(scene.get("scene_id", "<unknown>")))
    except Exception:
        return []
    for parent_id, child_ids in graph.items():
        for child_id in child_ids:
            rows.append({
                "parent_id": int(parent_id),
                "child_id": int(child_id),
                "parent_label": str(obj_map.get(int(parent_id), {}).get("label", "object")),
                "child_label": str(obj_map.get(int(child_id), {}).get("label", "object")),
                "relation_type": "support",
                "score": None,
                "source": "support_graph",
            })
    return rows


def load_selected_scannet_poses(
    scene_dir: Path,
    image_names: list[str],
    axis_alignment: np.ndarray | None = None,
) -> dict[str, Any]:
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
        frame_id = str(image_name).replace(".jpg", "")
        pose_file = pose_dir / f"{frame_id}.txt"
        if not pose_file.exists():
            continue
        if not (color_dir / image_name).exists():
            continue
        T_c2w = np.loadtxt(str(pose_file))
        if not np.isfinite(T_c2w).all():
            continue
        T_c2w_aligned = alignment @ T_c2w
        R_c2w = T_c2w_aligned[:3, :3]
        t_c2w = T_c2w_aligned[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        poses[image_name] = CameraPose(
            image_name=image_name,
            rotation=R_w2c.astype(np.float64),
            translation=t_w2c.astype(np.float64),
        )
    return poses


def build_frames(
    scene: dict[str, Any],
    scene_dir: Path | None,
    max_frames: int,
    strict_mode: bool,
    referability_frames: dict[str, dict[str, Any]],
    scene_questions: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    objects = list(scene.get("objects") or [])
    objects_by_id = {int(obj["id"]): obj for obj in objects}
    try:
        support_graph = get_scene_attachment_graph(scene, scene_id=str(scene.get("scene_id", "<unknown>")))
    except Exception:
        support_graph = {}
    support_ids = {int(k) for k in support_graph.keys()}
    for child_ids in support_graph.values():
        support_ids.update(int(child) for child in child_ids)
    ordered_names: list[str] = []
    for name in sorted(referability_frames.keys()):
        if name not in ordered_names:
            ordered_names.append(name)
    for name in sorted(scene_questions.keys()):
        if name not in ordered_names:
            ordered_names.append(name)
    selected = []
    if scene_dir and not ordered_names:
        selected = select_frames(scene_dir, objects, support_graph, max_frames=max_frames)
        ordered_names = [frame["image_name"] for frame in selected]
    selected_by_name = {frame["image_name"]: frame for frame in selected}
    poses = {}
    color_intrinsics = None
    depth_intrinsics = None
    if scene_dir:
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
        sel = selected_by_name.get(image_name)
        ref = referability_frames.get(image_name) or {}
        frame_selector_ids = [int(x) for x in sel.get("visible_object_ids", [])] if sel else []
        frame_selector_score = int(sel.get("score", 0)) if sel else 0
        pipeline_ids = list(frame_selector_ids)
        visibility_rows = []
        if scene_dir and color_intrinsics is not None and image_name in poses:
            pose = poses[image_name]
            image_path = scene_dir / "color" / image_name
            depth_image = None
            if depth_intrinsics is not None:
                depth_path = scene_dir / "depth" / image_name.replace(".jpg", ".png")
                if depth_path.exists():
                    try:
                        depth_image = load_depth_image(depth_path)
                    except Exception:
                        depth_image = None
            vis = compute_frame_object_visibility(objects, pose, color_intrinsics, image_path=image_path if image_path.exists() else None, depth_image=depth_image, depth_intrinsics=depth_intrinsics, strict_mode=True)
            if not frame_selector_ids:
                frame_selector_ids = [int(obj["id"]) for obj in get_visible_objects(objects, pose, color_intrinsics)]
            if not frame_selector_score:
                frame_selector_score = len(frame_selector_ids) * (1 + sum(1 for x in frame_selector_ids if x in support_ids))
            if ref:
                pipeline_ids = [int(obj["id"]) for obj in get_visible_objects(objects, pose, color_intrinsics)]
            if strict_mode:
                pipeline_ids = [obj_id for obj_id, meta in vis.items() if meta.get("eligible_as_reference", False)]
            frame_set = set(frame_selector_ids)
            pipe_set = set(pipeline_ids)
            ref_set = {int(x) for x in ref.get("referable_object_ids", [])}
            cand_set = {int(x) for x in ref.get("candidate_visible_object_ids", [])}
            for obj in objects:
                obj_id = int(obj["id"])
                meta = dict(vis.get(obj_id, {}))
                meta.update({"id": obj_id, "label": str(obj.get("label", "")), "is_frame_selector_candidate": obj_id in frame_set, "is_pipeline_visible": obj_id in pipe_set, "is_vlm_candidate": obj_id in cand_set, "is_vlm_referable": obj_id in ref_set})
                visibility_rows.append(meta)
        counts_union = sorted(set(count_labels(frame_selector_ids, objects_by_id)) | set(count_labels(pipeline_ids, objects_by_id)) | set((ref.get("label_counts") or {}).keys()))
        frame_questions = list(scene_questions.get(image_name, []))
        wrong = sum(1 for q in frame_questions if q.get("prediction_correct") is False)
        frames.append({
            "image_name": image_name,
            "image_uri": (scene_dir / "color" / image_name).resolve().as_uri() if scene_dir and (scene_dir / "color" / image_name).exists() else None,
            "image_width": getattr(color_intrinsics, "width", 640) if color_intrinsics is not None else 640,
            "image_height": getattr(color_intrinsics, "height", 480) if color_intrinsics is not None else 480,
            "frame_selector_score": frame_selector_score,
            "frame_selector_visible_ids": frame_selector_ids,
            "frame_selector_label_counts": count_labels(frame_selector_ids, objects_by_id),
            "support_visible_count": sum(1 for x in frame_selector_ids if x in support_ids),
            "pipeline_visible_ids": pipeline_ids,
            "pipeline_label_counts": count_labels(pipeline_ids, objects_by_id),
            "vlm_present": bool(ref),
            "frame_usable": bool(ref.get("frame_usable", False)) if ref else None,
            "frame_reject_reason": ref.get("frame_reject_reason") if ref else None,
            "vlm_label_counts": {str(k): int(v) for k, v in (ref.get("label_counts") or {}).items()} if ref else {},
            "referable_object_ids": [int(x) for x in ref.get("referable_object_ids", [])] if ref else [],
            "visibility_rows": visibility_rows,
            "count_labels_union": counts_union,
            "questions": frame_questions,
            "question_wrong": wrong,
        })
    return frames


def render_count_table(frame: dict[str, Any]) -> str:
    rows = []
    for label in frame["count_labels_union"]:
        fs = int(frame["frame_selector_label_counts"].get(label, 0))
        pipe = int(frame["pipeline_label_counts"].get(label, 0))
        vlm = int(frame["vlm_label_counts"].get(label, 0))
        delta = vlm - fs
        rows.append(f"<tr><td>{h(label)}</td><td>{fs}</td><td>{pipe}</td><td>{vlm}</td><td>{delta:+d}</td></tr>")
    if not rows:
        return '<div class="muted">No VLM count data for this frame.</div>'
    return (
        '<table><thead><tr><th>Label</th><th>Frame Selector</th><th>Pipeline</th><th>VLM</th><th>Delta(VLM-FS)</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_overlay_svg(frame: dict[str, Any]) -> str:
    if not frame["image_uri"] or not frame["visibility_rows"]:
        return '<div class="img-missing">Image or overlay data unavailable for this frame.</div>'
    width = int(frame["image_width"])
    height = int(frame["image_height"])
    parts = [f'<svg class="overlay" viewBox="0 0 {width} {height}" preserveAspectRatio="none">']
    for row in frame["visibility_rows"]:
        bounds = row.get("roi_bounds_px")
        if not bounds:
            continue
        x0, x1, y0, y1 = [int(v) for v in bounds]
        w = max(0, x1 - x0)
        hgt = max(0, y1 - y0)
        if w <= 0 or hgt <= 0:
            continue
        classes = []
        if row.get("is_frame_selector_candidate"):
            classes.append("cand")
        if row.get("is_pipeline_visible"):
            classes.append("pipe")
        if row.get("is_vlm_referable"):
            classes.append("ref")
        if not classes:
            continue
        cls = " ".join(classes)
        parts.append(f'<rect class="{cls}" x="{x0}" y="{y0}" width="{w}" height="{hgt}"></rect>')
        if row.get("is_vlm_referable"):
            parts.append(f'<text x="{x0 + 4}" y="{max(12, y0 - 4)}">{h(row["label"])} #{row["id"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_visibility_table(frame: dict[str, Any]) -> str:
    if not frame["visibility_rows"]:
        return '<div class="muted">No per-object visibility table for this frame.</div>'
    rows = []
    for row in sorted(frame["visibility_rows"], key=lambda x: (not x.get("is_vlm_referable", False), not x.get("is_pipeline_visible", False), x["label"], x["id"])):
        flags = []
        if row.get("is_frame_selector_candidate"):
            flags.append("candidate")
        if row.get("is_pipeline_visible"):
            flags.append("pipeline")
        if row.get("is_vlm_referable"):
            flags.append("referable")
        if row.get("eligible_as_reference"):
            flags.append("strict-ok")
        reasons = ", ".join(row.get("rejection_reasons", [])) or "-"
        rows.append(
            "<tr>"
            f"<td>{row['id']}</td><td>{h(row['label'])}</td><td>{h(' | '.join(flags) or '-')}</td>"
            f"<td>{h(row.get('occlusion_status', '-'))}</td><td>{n(row.get('visible_ratio'), 2)}</td>"
            f"<td>{n(row.get('projected_area_px'), 0)}</td><td>{n(row.get('bbox_in_frame_ratio'), 2)}</td>"
            f"<td>{n(row.get('roi_sharpness'), 1)}</td><td>{h(reasons)}</td></tr>"
        )
    return (
        '<table><thead><tr><th>ID</th><th>Label</th><th>Flags</th><th>Occlusion</th><th>Visible Ratio</th>'
        f"<th>Proj Area</th><th>In Frame</th><th>Sharpness</th><th>Reasons</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_questions(frame: dict[str, Any]) -> str:
    questions = frame["questions"]
    if not questions:
        return '<div class="muted">No loaded questions for this frame.</div>'
    cards = []
    for q in questions:
        klass = "bad" if q.get("prediction_correct") is False else "ok" if q.get("prediction_correct") is True else ""
        status = (
            f'<span class="pill bad">pred {h(q["prediction"])} / gt {h(q["answer"])}</span>'
            if q.get("prediction_correct") is False else
            '<span class="pill ok">correct</span>'
            if q.get("prediction_correct") is True else
            f'<span class="pill warn">answer {h(q["answer"])}</span>'
        )
        options = "<br>".join(f"{chr(65 + i)}) {h(opt)}" for i, opt in enumerate(q["options"]))
        cards.append(
            f'<div class="qcard {klass}"><div class="qmeta"><span class="pill">{h(q["level"])}</span>'
            f'<span class="pill">{h(q["type"])}</span>{status}</div><div class="qtext">{h(q["question"])}</div>'
            f'<div class="qopts">{options}</div></div>'
        )
    return "".join(cards)


def render_support_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No attachment or support relations available.</div>'
    body = "".join(
        f"<tr><td>{h(r['parent_label'])} #{r['parent_id']}</td><td>{h(r['child_label'])} #{r['child_id']}</td>"
        f"<td>{h(r['relation_type'])}</td><td>{n(r.get('score'), 3)}</td><td>{h(r['source'])}</td></tr>"
        for r in rows
    )
    return f'<table><thead><tr><th>Parent</th><th>Child</th><th>Type</th><th>Score</th><th>Source</th></tr></thead><tbody>{body}</tbody></table>'


def render_annotation_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No raw aggregation audit was available. Use `--scene_dir` to enable this table.</div>'
    body = "".join(
        f"<tr><td>{r['object_id']}</td><td>{h(r['raw_label'])}</td><td>{h(r['normalized_label'])}</td>"
        f"<td>{h(r['status'])}</td><td>{r['segment_count']}</td><td>{h(r['vertex_count']) if r['vertex_count'] is not None else '-'}</td></tr>"
        for r in rows
    )
    return f'<table><thead><tr><th>ID</th><th>Raw Label</th><th>Normalized</th><th>Status</th><th>Segments</th><th>Vertices</th></tr></thead><tbody>{body}</tbody></table>'


def render_html(
    scene_id: str,
    source_mode: str,
    source_detail: str,
    notes: list[str],
    scene_question_summary: dict[str, Any],
    frames: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    open3d_cmd: str,
    strict_mode: bool,
) -> str:
    note_html = "".join(f'<li>{h(note)}</li>' for note in notes) or "<li>No warnings.</li>"
    frame_nav = "".join(f'<a class="pill" href="#frame-{i}">{h(frame["image_name"])}</a>' for i, frame in enumerate(frames))
    frame_sections = []
    for i, frame in enumerate(frames):
        vlm_status = (
            "usable"
            if frame["frame_usable"]
            else "rejected"
            if frame["frame_usable"] is False
            else "not loaded"
        )
        vlm_detail = frame["frame_reject_reason"]
        if not vlm_detail:
            vlm_detail = "referable=" + str(len(frame["referable_object_ids"]))
        image_block = (
            f'<div class="imgwrap"><img src="{h(frame["image_uri"])}" alt="{h(frame["image_name"])}">{render_overlay_svg(frame)}</div>'
            if frame["image_uri"] else '<div class="imgwrap img-missing">Image missing for this frame.</div>'
        )
        frame_sections.append(
            f'<section class="card frame" id="frame-{i}"><h2>{h(frame["image_name"])}</h2>'
            f'<div class="metrics"><div class="metric"><div class="k">Frame Score</div><div class="v">{frame["frame_selector_score"]}</div><div class="s">n_visible={len(frame["frame_selector_visible_ids"])}, support_visible={frame["support_visible_count"]}</div></div>'
            f'<div class="metric"><div class="k">Pipeline Visible</div><div class="v">{len(frame["pipeline_visible_ids"])}</div><div class="s">strict_mode={"on" if strict_mode else "off"}</div></div>'
            f'<div class="metric"><div class="k">VLM</div><div class="v">{h(vlm_status)}</div><div class="s">{h(vlm_detail)}</div></div>'
            f'<div class="metric"><div class="k">Questions</div><div class="v">{len(frame["questions"])}</div><div class="s">wrong={frame["question_wrong"]}</div></div></div>'
            f'<div class="twocol"><div><h3>Image Overlay</h3>{image_block}<div class="legend"><span class="cand">frame selector candidate</span><span class="pipe">pipeline visible</span><span class="ref">VLM referable</span></div></div>'
            f'<div><h3>Label Counts</h3>{render_count_table(frame)}</div></div>'
            f'<h3>Per-Object Visibility</h3>{render_visibility_table(frame)}<h3>Questions</h3>{render_questions(frame)}</section>'
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h(scene_id)} debug viewer</title>
<style>
body{{margin:0;background:#edf2f7;color:#102030;font:14px/1.45 "Segoe UI",Arial,sans-serif}} .page{{max-width:1600px;margin:0 auto;padding:16px;display:grid;gap:16px}}
.card{{background:#fff;border:1px solid rgba(16,32,48,.08);border-radius:18px;box-shadow:0 14px 32px rgba(16,32,48,.07);padding:16px}} h1,h2,h3{{margin:0 0 10px}} h1{{font-size:26px}} h2{{font-size:21px}} h3{{font-size:17px;margin-top:16px}}
.muted{{color:#607284}} .metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}} .metric{{background:#f7fafc;border:1px solid rgba(16,32,48,.06);border-radius:14px;padding:10px 12px}} .k{{font-size:12px;color:#607284}} .v{{font-size:22px;font-weight:700}} .s{{font-size:12px;color:#708396;margin-top:4px}}
.pill{{display:inline-block;padding:4px 9px;border-radius:999px;background:#eef4fa;border:1px solid rgba(16,32,48,.08);margin:0 6px 6px 0;color:#24476a;text-decoration:none}} .pill.ok{{background:#e9f7ef;color:#1e7a41}} .pill.bad{{background:#fff1f1;color:#9b2424}} .pill.warn{{background:#fff6e8;color:#8a5400}}
.twocol{{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(360px,.9fr);gap:16px;align-items:start}} .imgwrap{{position:relative;background:#0f1b28;border-radius:14px;overflow:hidden;aspect-ratio:4/3}} .imgwrap img{{width:100%;height:100%;object-fit:contain;display:block}} .img-missing{{display:flex;align-items:center;justify-content:center;color:#d6dde6}}
.overlay{{position:absolute;inset:0;width:100%;height:100%}} .overlay rect{{fill:none;stroke-width:2}} .overlay rect.cand{{stroke:#d0d7de}} .overlay rect.pipe{{stroke:#43a047}} .overlay rect.ref{{stroke:#1d6fd1;stroke-width:3}} .overlay text{{fill:#1d6fd1;font:12px "Segoe UI",Arial,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}} .legend span{{padding:4px 8px;border-radius:999px;background:#f7fafc;border:1px solid rgba(16,32,48,.06)}} .legend .cand{{color:#54606c}} .legend .pipe{{color:#2f7d32}} .legend .ref{{color:#1d5fb0}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:8px 10px;border-bottom:1px solid rgba(16,32,48,.08);text-align:left;vertical-align:top}} th{{font-size:12px;color:#607284;background:#fff;position:sticky;top:0}}
.qcard{{border:1px solid rgba(16,32,48,.08);border-radius:14px;padding:10px 12px;margin-bottom:10px;background:#fff}} .qcard.bad{{border-color:rgba(180,58,58,.18);background:#fff8f8}} .qcard.ok{{border-color:rgba(34,139,80,.18);background:#f8fffb}} .qmeta{{margin-bottom:6px}} .qtext{{margin-bottom:8px}} .qopts{{color:#42576b}}
pre{{background:#0f1722;color:#d7e0ea;padding:14px;border-radius:14px;overflow:auto}} ul{{margin:8px 0 0 18px}} @media(max-width:1180px){{.metrics,.twocol{{grid-template-columns:1fr}}}}
</style></head><body><div class="page">
<section class="card"><h1>{h(scene_id)}</h1><div class="muted">Static scene debug report for frame selection, VLM referability, and question generation outputs.</div>
<div class="metrics" style="margin-top:12px"><div class="metric"><div class="k">Source</div><div class="v">{h(source_mode)}</div><div class="s">{h(source_detail)}</div></div>
<div class="metric"><div class="k">Frames</div><div class="v">{len(frames)}</div><div class="s">scene questions={scene_question_summary['total']}</div></div>
<div class="metric"><div class="k">Wrong Predictions</div><div class="v">{scene_question_summary['wrong']}</div><div class="s">if predictions file was provided</div></div>
<div class="metric"><div class="k">3D Viewer</div><div class="v">Use Open3D</div><div class="s">command below uses aligned raw scene/metadata</div></div></div>
<h3>Open3D Geometry Check</h3><pre>{h(open3d_cmd)}</pre><h3>Notes</h3><ul>{note_html}</ul><h3>Frame Index</h3>{frame_nav}</section>
{''.join(frame_sections)}
<section class="card"><h2>Attachment / Support Relations</h2>{render_support_table(support_rows)}</section>
<section class="card"><h2>Raw Annotation Audit</h2>{render_annotation_table(annotation_rows)}</section>
</div></body></html>"""


def main() -> None:
    args = parse_args()
    if args.label_map is not None:
        load_scannet_label_map(args.label_map)
    scene_id = resolve_scene_id(args.scene_dir, args.scene_metadata)
    scene_dir = infer_scene_dir(scene_id, args.scene_dir, args.data_root)
    scene, source_mode, source_detail, notes = load_scene_with_fallback(scene_id, scene_dir, args.scene_metadata)
    scene_questions_summary, scene_questions, question_path = load_scene_questions(scene_id, args.questions, args.scene_metadata, args.predictions, args.question_limit_per_frame)
    referability_frames, cache_path = load_referability_frames(args.referability_cache, scene_id)
    if args.question_limit_per_frame > 0:
        notes.append(f"question cards are truncated to the first {args.question_limit_per_frame} items per frame")
    if question_path is None:
        notes.append("no scene question JSON found; question cards will be empty")
    if cache_path is None:
        notes.append("no referability cache loaded; VLM frame/object diagnostics will be empty")
    frames = build_frames(scene, scene_dir, args.max_frames, args.strict_mode, referability_frames, scene_questions)
    support_rows = build_support_rows(scene)
    annotation_rows = build_annotation_audit(scene, scene_dir, args.scene_metadata)
    open3d_cmd = (
        f'python scripts/view_scannet_bbox_open3d.py --scene_dir "{scene_dir}" --render_mode mesh --show_centers'
        if scene_dir is not None else
        f'python scripts/view_scannet_bbox_open3d.py --scene_metadata "{args.scene_metadata}" --render_mode mesh --show_centers'
    )
    html_text = render_html(scene_id, source_mode, source_detail, notes, scene_questions_summary, frames, support_rows, annotation_rows, open3d_cmd, args.strict_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"wrote debug viewer to {args.output}")


if __name__ == "__main__":
    main()
