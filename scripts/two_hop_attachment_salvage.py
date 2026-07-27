#!/usr/bin/env python3
"""Create an interactive two-hop attachment salvage review.

The generated JSON is intentionally shaped like a referability cache so it can
be passed directly to ``scripts/run_pipeline.py --referability_cache``.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alias_groups import ALIAS_CONFIG_VERSION
from src.datasets.scannetpp import ScanNetPPDataSource
from src.utils.coordinate_transform import project_to_image

REFERABILITY_CACHE_VERSION = "20.0"
TOOL_VERSION = "1.0"
ROLE_NAMES = ("moved", "child", "grandchild", "contrast")


def parse_image_spec(spec: str) -> tuple[str, list[str]]:
    """Parse ``scene_id:frame_id\\frame_id`` into scene and image names."""
    text = str(spec or "").strip()
    if ":" not in text:
        raise ValueError(f"Image spec must be scene_id:frame\\frame, got {spec!r}")
    scene_id, raw_frames = text.split(":", 1)
    scene_id = scene_id.strip()
    if not scene_id or not raw_frames.strip():
        raise ValueError(f"Image spec is missing scene or frames: {spec!r}")
    names: list[str] = []
    for token in re.split(r"[\\,;\s]+", raw_frames):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"frame_{int(token):06d}.jpg"
        elif not Path(token).suffix:
            token = f"{token}.jpg"
        names.append(token)
    if not names:
        raise ValueError(f"Image spec has no frame names: {spec!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"Image spec contains duplicate frames: {spec!r}")
    return scene_id, names


def _bbox_corners(obj: dict[str, Any]) -> np.ndarray:
    minimum = np.asarray(obj["bbox_min"], dtype=np.float64)
    maximum = np.asarray(obj["bbox_max"], dtype=np.float64)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def project_object_box(
    obj: dict[str, Any],
    pose: Any,
    intrinsics: Any,
    width: int,
    height: int,
) -> list[int] | None:
    """Project a 3D object bbox and return a clipped ``[x1,y1,x2,y2]``."""
    points: list[np.ndarray] = []
    for point in _bbox_corners(obj):
        pixel, depth = project_to_image(point, pose, intrinsics)
        if pixel is None or depth <= 0 or not np.isfinite(pixel).all():
            continue
        points.append(np.asarray(pixel, dtype=np.float64))
    if not points:
        return None
    projected = np.asarray(points)
    x1 = max(0, int(np.floor(projected[:, 0].min())))
    y1 = max(0, int(np.floor(projected[:, 1].min())))
    x2 = min(width, int(np.ceil(projected[:, 0].max())))
    y2 = min(height, int(np.ceil(projected[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _label_maps(objects: list[dict[str, Any]], object_ids: list[int]) -> tuple[dict[str, str], dict[str, int], dict[str, list[int]]]:
    labels: dict[str, str] = {}
    counts: dict[str, int] = {}
    label_to_ids: dict[str, list[int]] = {}
    for obj in objects:
        obj_id = int(obj["id"])
        if obj_id not in object_ids:
            continue
        label = str(obj.get("label", "object")).strip().lower() or "object"
        labels[label] = "unique" if counts.get(label, 0) == 0 else "multiple"
        counts[label] = counts.get(label, 0) + 1
        label_to_ids.setdefault(label, []).append(obj_id)
    for label, count in counts.items():
        labels[label] = "unique" if count == 1 else "multiple"
    return labels, counts, label_to_ids


def build_frame_record(
    *,
    scene_id: str,
    image_name: str,
    image_path: Path,
    scene: dict[str, Any],
    pose: Any,
    intrinsics: Any,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    projected_objects: list[dict[str, Any]] = []
    for obj in scene.get("objects", []):
        try:
            box = project_object_box(obj, pose, intrinsics, width, height)
        except (KeyError, TypeError, ValueError):
            box = None
        if box is None:
            continue
        projected_objects.append(
            {"id": int(obj["id"]), "label": str(obj.get("label", "object")), "box": box}
        )
    visible_ids = sorted({int(item["id"]) for item in projected_objects})
    label_statuses, label_counts, label_to_ids = _label_maps(scene.get("objects", []), visible_ids)
    return {
        "scene_id": scene_id,
        "image_name": image_name,
        "image_width": int(width),
        "image_height": int(height),
        "image_path": str(image_path),
        "image_data_url": _image_data_url(image_path),
        "objects": projected_objects,
        "visible_object_ids": visible_ids,
        "label_statuses": label_statuses,
        "label_counts": label_counts,
        "label_to_object_ids": label_to_ids,
    }


def render_review_html(frames: list[dict[str, Any]], output_json_name: str) -> str:
    cards: list[str] = []
    for index, frame in enumerate(frames):
        boxes: list[str] = []
        for obj in frame["objects"]:
            x1, y1, x2, y2 = obj["box"]
            left = 100 * x1 / frame["image_width"]
            top = 100 * y1 / frame["image_height"]
            width = 100 * (x2 - x1) / frame["image_width"]
            height = 100 * (y2 - y1) / frame["image_height"]
            text = html.escape(f'{obj["label"]} #{obj["id"]}')
            boxes.append(
                f'<div class="box" style="left:{left:.5f}%;top:{top:.5f}%;width:{width:.5f}%;height:{height:.5f}%">'
                f'<span>{text}</span></div>'
            )
        role_fields = "".join(
            f'<div class="role" data-role="{role}"><strong>{role}</strong>'
            f'<label>object ID<input name="{role}_id" type="number" step="1" inputmode="numeric"></label>'
            f'<label>label<input name="{role}_label" type="text" autocomplete="off"></label></div>'
            for role in ROLE_NAMES
        )
        cards.append(
            f'<article class="card" data-index="{index}" data-scene-id="{html.escape(frame["scene_id"])}" '
            f'data-image-name="{html.escape(frame["image_name"])}">'
            f'<h2>{html.escape(frame["scene_id"])} / {html.escape(frame["image_name"])}</h2>'
            f'<div class="visual"><img src="{frame["image_data_url"]}"><div class="boxes">{"".join(boxes)}</div></div>'
            f'<div class="roles">{role_fields}</div>'
            '<button type="button" class="delete">Delete card</button>'
            '</article>'
        )
    initial = json.dumps(
        [
            {
                "scene_id": frame["scene_id"],
                "image_name": frame["image_name"],
                "visible_object_ids": frame["visible_object_ids"],
                "objects": frame["objects"],
                "label_statuses": frame["label_statuses"],
                "label_counts": frame["label_counts"],
                "label_to_object_ids": frame["label_to_object_ids"],
            }
            for frame in frames
        ],
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Two-hop Attachment Salvage</title>
<style>
body{{font:14px system-ui,sans-serif;background:#f3f4f6;color:#111827;margin:0;padding:24px}}h1{{margin-top:0}}
.card{{background:white;border:1px solid #d1d5db;border-radius:12px;padding:16px;margin:0 0 24px;display:grid;gap:14px}}
.visual{{position:relative;max-width:1100px;width:max-content;max-width:100%}}.visual img{{display:block;max-width:100%;height:auto}}
.boxes{{position:absolute;inset:0}}.box{{position:absolute;border:2px solid #ef4444;box-sizing:border-box;pointer-events:none}}.box span{{background:#ef4444;color:#fff;font-size:11px;padding:2px 4px;white-space:nowrap}}
.roles{{display:grid;grid-template-columns:repeat(4,minmax(190px,1fr));gap:10px}}.role{{display:grid;gap:8px;padding:10px;border:1px solid #d1d5db;border-radius:8px}}label{{display:grid;gap:5px;font-weight:600}}input{{padding:8px;border:1px solid #9ca3af;border-radius:6px;background:white}}
button{{padding:9px 14px;border:0;border-radius:7px;background:#2563eb;color:white;cursor:pointer}}button.delete{{background:#b91c1c;width:max-content}}.deleted{{opacity:.35}}
@media(max-width:720px){{.roles{{grid-template-columns:1fr 1fr}}}}
</style></head><body><h1>Two-hop Attachment Salvage</h1>
<p>Read the projected bbox labels, then type one object ID and label for each role.</p>
<div id="cards">{"".join(cards)}</div>
<button id="export">Export JSON</button>
<script>
const initial = {initial};
function collect() {{
  const rows=[]; document.querySelectorAll('.card:not(.deleted)').forEach(card=>{{
    const roles={{}}; card.querySelectorAll('.role').forEach(row=>{{
      const role=row.dataset.role; const idText=row.querySelector(`[name="${{role}}_id"]`).value.trim();
      const label=row.querySelector(`[name="${{role}}_label"]`).value.trim();
      roles[role]={{id:Number(idText),label}};
    }});
    rows.push({{scene_id:card.dataset.sceneId,image_name:card.dataset.imageName,roles}});
  }}); return rows;
}}
document.querySelectorAll('.card').forEach(card=>card.querySelector('.delete').addEventListener('click',()=>{{card.classList.toggle('deleted')}}));
document.getElementById('export').addEventListener('click',()=>{{
  try {{
  const selected=collect(); const byKey=new Map(initial.map(f=>[f.scene_id+'/'+f.image_name,f]));
  const frames={{}}; const graph={{}};
  selected.forEach(item=>{{
    const source=byKey.get(item.scene_id+'/'+item.image_name); if(!source) return;
    const roleIds=Object.fromEntries(Object.entries(item.roles).map(([k,v])=>[k,v.id]));
    const ids=[roleIds.moved,roleIds.child,roleIds.grandchild,roleIds.contrast];
    if(ids.some(v=>!Number.isInteger(v))) throw new Error('Every role must have an integer object ID');
    if(new Set(ids).size!==4) throw new Error('Roles must be distinct on every image');
    const projectedIds=new Set(source.objects.map(object=>Number(object.id)));
    if(ids.some(id=>!projectedIds.has(id))) throw new Error('Every object ID must belong to a projected bbox on its image');
    if(Object.values(item.roles).some(role=>!role.label.trim())) throw new Error('Every role must have a non-empty label');
    const optionLabels=['child','grandchild','contrast'].map(role=>item.roles[role].label.trim().toLowerCase());
    if(new Set(optionLabels).size!==3) throw new Error('Child, grandchild and contrast labels must be distinct');
    const sceneFrames=frames[item.scene_id] || (frames[item.scene_id]={{}});
    sceneFrames[item.image_name]={{scene_id:item.scene_id,image_name:item.image_name,frame_usable:true,
      candidate_visible_object_ids:source.visible_object_ids,pipeline_visible_object_ids_used_for_generation:source.visible_object_ids,
      visible_object_ids:source.visible_object_ids,referable_object_ids:ids,attachment_referable_object_ids:ids,
      attachment_referable_pairs:[[roleIds.moved,roleIds.child],[roleIds.child,roleIds.grandchild]],attachment_referable_pair_count:2,
      label_statuses:source.label_statuses,label_counts:source.label_counts,label_to_object_ids:source.label_to_object_ids,
      manual_attachment_roles:item.roles}};
    const sceneGraph=graph[item.scene_id] || (graph[item.scene_id]={{}});
    sceneGraph[String(roleIds.moved)]=Array.from(new Set([...(sceneGraph[String(roleIds.moved)]||[]),roleIds.child])).sort((a,b)=>a-b);
    sceneGraph[String(roleIds.child)]=Array.from(new Set([...(sceneGraph[String(roleIds.child)]||[]),roleIds.grandchild])).sort((a,b)=>a-b);
  }});
  const payload={{version:{json.dumps(REFERABILITY_CACHE_VERSION)},alias_config_version:{json.dumps(ALIAS_CONFIG_VERSION)},referability_backend:'two_hop_attachment_salvage',model:'human',schema:'two_hop_attachment_salvage_v1',tool_version:{json.dumps(TOOL_VERSION)},frames,manual_attachment_graph:graph}};
  const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}); const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download={json.dumps(output_json_name)}; a.click(); URL.revokeObjectURL(a.href);
  }} catch(error) {{ alert(error && error.message ? error.message : String(error)); }}
}});
</script></body></html>"""


def _normalize_roles(raw: Any, object_map: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("Each frame must contain moved/child/grandchild/contrast roles")
    roles: dict[str, dict[str, Any]] = {}
    ids: list[int] = []
    for role in ROLE_NAMES:
        value = raw.get(role)
        if not isinstance(value, dict):
            raise ValueError(f"Role {role!r} must contain a manually entered id and label")
        try:
            obj_id = int(value.get("id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Role {role!r} is not selected") from exc
        if obj_id not in object_map:
            raise ValueError(f"Role {role!r} references unavailable object id {obj_id}")
        label = str(value.get("label", "")).strip()
        if not label:
            raise ValueError(f"Role {role!r} has an empty label")
        ids.append(obj_id)
        roles[role] = {"id": obj_id, "label": label}
    if len(set(ids)) != len(ids):
        raise ValueError("moved, child, grandchild and contrast must be distinct objects")
    option_labels = [roles[name]["label"].strip().lower() for name in ("child", "grandchild", "contrast")]
    if len(set(option_labels)) != len(option_labels):
        raise ValueError("child, grandchild and contrast must have distinct labels")
    return roles


def build_cache(frames: list[dict[str, Any]], selections: list[dict[str, Any]]) -> dict[str, Any]:
    frame_lookup = {(f["scene_id"], f["image_name"]): f for f in frames}
    output_frames: dict[str, dict[str, Any]] = {}
    manual_graph: dict[str, dict[str, list[int]]] = {}
    for selection in selections:
        key = (str(selection.get("scene_id", "")).strip(), str(selection.get("image_name", "")).strip())
        frame = frame_lookup.get(key)
        if frame is None:
            raise ValueError(f"Selection references unknown frame {key[0]}/{key[1]}")
        object_map = {int(obj["id"]): obj for obj in frame["objects"]}
        roles = _normalize_roles(selection.get("roles"), object_map)
        moved, child, grandchild = (roles[name]["id"] for name in ("moved", "child", "grandchild"))
        manual_graph.setdefault(key[0], {}).setdefault(str(moved), [])
        manual_graph[key[0]][str(moved)] = sorted(set(manual_graph[key[0]][str(moved)] + [child]))
        manual_graph[key[0]].setdefault(str(child), [])
        manual_graph[key[0]][str(child)] = sorted(set(manual_graph[key[0]][str(child)] + [grandchild]))
        visible_ids = list(frame["visible_object_ids"])
        role_ids = [roles[name]["id"] for name in ROLE_NAMES]
        output_frames.setdefault(key[0], {})[key[1]] = {
            "scene_id": key[0],
            "image_name": key[1],
            "frame_usable": True,
            "candidate_visible_object_ids": visible_ids,
            "pipeline_visible_object_ids_used_for_generation": visible_ids,
            "visible_object_ids": visible_ids,
            "referable_object_ids": role_ids,
            "attachment_referable_object_ids": role_ids,
            "attachment_referable_pairs": [[moved, child], [child, grandchild]],
            "attachment_referable_pair_count": 2,
            "label_statuses": frame["label_statuses"],
            "label_counts": frame["label_counts"],
            "label_to_object_ids": frame["label_to_object_ids"],
            "manual_attachment_roles": roles,
        }
    return {
        "version": REFERABILITY_CACHE_VERSION,
        "alias_config_version": ALIAS_CONFIG_VERSION,
        "referability_backend": "two_hop_attachment_salvage",
        "model": "human",
        "schema": "two_hop_attachment_salvage_v1",
        "tool_version": TOOL_VERSION,
        "frames": output_frames,
        "manual_attachment_graph": manual_graph,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_root", required=True, type=Path)
    parser.add_argument("--frame_root", type=Path, default=PROJECT_ROOT / "output" / "scannetpp_iphone_frames")
    parser.add_argument("--image", action="append", default=[], help="scene_id:frame_id\\frame_id")
    parser.add_argument("--images", default=None, help="One or more specs separated by commas")
    parser.add_argument("--output_html", type=Path, default=Path("two_hop_attachment_salvage.html"))
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("two_hop_attachment_salvage.json"),
        help="Suggested browser download filename; the browser chooses the final directory",
    )
    args = parser.parse_args()
    specs = list(args.image)
    if args.images:
        specs.extend(item.strip() for item in str(args.images).split(",") if item.strip())
    if not specs:
        parser.error("at least one --image or --images value is required")

    frames: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        scene_id, image_names = parse_image_spec(spec)
        scene_dir = args.scene_root / scene_id
        data_source = ScanNetPPDataSource(scene_dir, sensor="iphone", frame_root=args.frame_root)
        scene = data_source.load_scene()
        intrinsics = data_source.load_intrinsics()
        poses = data_source.load_poses()
        for image_name in image_names:
            key = (scene_id, image_name)
            if key in seen:
                raise ValueError(f"Duplicate image selection: {scene_id}/{image_name}")
            seen.add(key)
            pose = poses.get(image_name)
            if pose is None:
                raise ValueError(f"Pose not found for {scene_id}/{image_name}")
            image_path = data_source.image_path(image_name)
            if not image_path.exists():
                raise ValueError(f"Image not found: {image_path}")
            frames.append(
                build_frame_record(
                    scene_id=scene_id,
                    image_name=image_name,
                    image_path=image_path,
                    scene=scene,
                    pose=pose,
                    intrinsics=intrinsics,
                )
            )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render_review_html(frames, args.output_json.name), encoding="utf-8")
    print(f"Wrote review HTML to {args.output_html}")
    print("Open the HTML, select four roles per image, then click Export JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
