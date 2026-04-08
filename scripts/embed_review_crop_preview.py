#!/usr/bin/env python3
"""Embed real question-review crop previews into a flagged review HTML card."""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


CARD_START = '<div class="card">'
REVIEW_BLOCK_START = '<div class="review-block">'
CROP_PREVIEW_START = "<!-- crop-preview:start -->"
CROP_PREVIEW_END = "<!-- crop-preview:end -->"
DIV_TOKEN_RE = re.compile(r"<div\b[^>]*>|</div>", flags=re.IGNORECASE)
REVIEW_TITLE_RE = re.compile(r'<div class="review-title">([^<]+)</div>')
REVIEW_LINE_RE = re.compile(r'<div class="review-line">([^<]*)</div>')
FOOTER_RE = re.compile(r'<div class="footer">\s*([^<]+?)\s*&nbsp;/&nbsp;\s*([^<]+?)\s*</div>')
EMBEDDED_IMAGE_RE = re.compile(
    r'<div class="img-wrap"><img src="data:(?P<mime>[^;]+);base64,(?P<b64>[^"]+)"',
    flags=re.IGNORECASE,
)
OBJECT_REVIEW_RE = re.compile(
    r"^(?P<label>.+?)#(?P<obj_id>\d+)"
    r"(?: \[(?P<roles>[^\]]+)\])?: "
    r"(?P<status>[a-z_]+)"
    r"(?: \((?P<reason>.*)\))?$"
)
AUDIT_OBJECT_RE = re.compile(
    r"^(?P<role>[a-zA-Z0-9_]+): "
    r"label=(?P<label>.*?), "
    r"obj_id=(?P<obj_id>\d+|-), "
    r"label_status=(?P<label_status>.*?), "
    r"candidates=(?P<candidates>.*?), "
    r"referable=(?P<referable>.*?), "
    r"result=(?P<result>[a-z_]+), "
    r"reasons=(?P<reasons>.*)$"
)

PREVIEW_STYLE = """
.crop-preview{margin:14px 0 12px;padding:12px 14px;border-radius:10px;background:#fff;
              border:1px solid #e5e7eb}
.crop-preview-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.crop-preview-title{font-size:13px;font-weight:700;color:#111}
.crop-preview-subtitle{font-size:12px;color:#6b7280}
.crop-preview-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
                   gap:12px;margin-top:12px}
.crop-preview-item{border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:#f8fafc}
.crop-preview-meta{font-size:12px;font-weight:700;color:#111}
.crop-preview-state{font-size:12px;color:#7c2d12;margin-top:4px}
.crop-preview-item img{width:100%;display:block;border-radius:8px;margin-top:8px;background:#111}
.crop-preview-missing{margin-top:8px;border-radius:8px;padding:18px 12px;background:#fff7ed;
                      border:1px dashed #fdba74;font-size:12px;color:#9a3412;line-height:1.5}
.crop-preview-bounds{font-size:11px;color:#6b7280;line-height:1.5;margin-top:8px}
"""


@dataclass(frozen=True)
class ReviewTarget:
    label: str
    obj_id: int
    roles: tuple[str, ...]
    status: str
    reason: str


def _iter_div_ranges(text: str, start_marker: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    search_from = 0
    while True:
        start = text.find(start_marker, search_from)
        if start < 0:
            return ranges
        depth = 0
        for match in DIV_TOKEN_RE.finditer(text, start):
            token = match.group(0).lower()
            if token.startswith("<div"):
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    ranges.append((start, match.end()))
                    search_from = match.end()
                    break
        else:
            raise ValueError(f"Unbalanced div structure after marker: {start_marker}")


def _find_matching_card(
    html_text: str,
    *,
    scene_id: str,
    image_name: str,
    match_text: str,
) -> tuple[int, int, str]:
    footer_text = f"{scene_id} &nbsp;/&nbsp; {image_name}"
    for start, end in _iter_div_ranges(html_text, CARD_START):
        card_html = html_text[start:end]
        if footer_text in card_html and match_text in html.unescape(card_html):
            return start, end, card_html
    raise ValueError(
        f"Could not find a card for {scene_id}/{image_name} containing {match_text!r}"
    )


def _extract_review_blocks(card_html: str) -> dict[str, list[str]]:
    notes_start = card_html.find('<div class="review-notes">')
    if notes_start < 0:
        return {}
    notes_html = card_html[notes_start:]
    blocks: dict[str, list[str]] = {}
    for start, end in _iter_div_ranges(notes_html, REVIEW_BLOCK_START):
        block_html = notes_html[start:end]
        title_match = REVIEW_TITLE_RE.search(block_html)
        if not title_match:
            continue
        title = html.unescape(title_match.group(1)).strip()
        lines = [
            html.unescape(line).strip()
            for line in REVIEW_LINE_RE.findall(block_html)
            if html.unescape(line).strip()
        ]
        blocks[title] = lines
    return blocks


def parse_review_targets(card_html: str) -> list[ReviewTarget]:
    review_blocks = _extract_review_blocks(card_html)
    lines = review_blocks.get("VLM Review", [])
    targets: list[ReviewTarget] = []
    for line in lines:
        match = OBJECT_REVIEW_RE.match(line)
        if not match:
            continue
        roles_raw = match.group("roles") or ""
        roles = tuple(
            role.strip() for role in roles_raw.split(",") if role.strip()
        )
        targets.append(
            ReviewTarget(
                label=match.group("label").strip(),
                obj_id=int(match.group("obj_id")),
                roles=roles,
                status=match.group("status").strip(),
                reason=(match.group("reason") or "").strip(),
            )
        )
    if not targets:
        raise ValueError("No object review lines found in the matched card")
    return targets


def _format_bounds(bounds: object) -> str:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return "-"
    try:
        values = [int(value) for value in bounds]
    except (TypeError, ValueError):
        return "-"
    return f"[{values[0]}, {values[1]}, {values[2]}, {values[3]}]"


def build_crop_preview_html(
    targets: list[ReviewTarget],
    crop_by_obj_id: dict[int, dict[str, object]],
    *,
    image_name: str,
    subtitle: str | None = None,
) -> str:
    items: list[str] = []
    for target in targets:
        crop_entry = crop_by_obj_id.get(target.obj_id, {})
        label = f"{target.label}#{target.obj_id}"
        roles = ", ".join(target.roles) if target.roles else "-"
        state = target.status
        if target.reason:
            state = f"{state} ({target.reason})"
        crop_reason = str(crop_entry.get("reason", "") or "").strip()
        crop_b64 = str(crop_entry.get("image_b64", "") or "")
        crop_mime = str(crop_entry.get("mime", "") or "image/jpeg")

        if crop_b64:
            media_html = (
                f'<img src="data:{html.escape(crop_mime)};base64,{crop_b64}" '
                f'alt="{html.escape(label)} crop">'
            )
        else:
            missing_reason = crop_reason or "missing_crop"
            media_html = (
                '<div class="crop-preview-missing">'
                f"Crop unavailable: {html.escape(missing_reason)}"
                "</div>"
            )

        items.append(
            '<div class="crop-preview-item">'
            f'<div class="crop-preview-meta">{html.escape(label)} | {html.escape(roles)}</div>'
            f'<div class="crop-preview-state">Review status: {html.escape(state)}</div>'
            f"{media_html}"
            '<div class="crop-preview-bounds">'
            f'roi_bounds_px: {html.escape(_format_bounds(crop_entry.get("roi_bounds_px")))}<br>'
            f'crop_bounds_px: {html.escape(_format_bounds(crop_entry.get("crop_bounds_px")))}'
            "</div>"
            "</div>"
        )

    return (
        f"{CROP_PREVIEW_START}"
        '<div class="crop-preview">'
        '<div class="crop-preview-head">'
        '<div>'
        '<div class="crop-preview-title">Crop Preview</div>'
        '<div class="crop-preview-subtitle">'
        f'{html.escape(subtitle or f"Exact per-object crops used by question presence review for {image_name}.")}'
        '</div>'
        '</div>'
        '</div>'
        f'<div class="crop-preview-grid">{"".join(items)}</div>'
        '</div>'
        f"{CROP_PREVIEW_END}"
    )


def inject_crop_preview(card_html: str, preview_html: str) -> str:
    if CROP_PREVIEW_START in card_html and CROP_PREVIEW_END in card_html:
        start = card_html.index(CROP_PREVIEW_START)
        end = card_html.index(CROP_PREVIEW_END) + len(CROP_PREVIEW_END)
        return card_html[:start] + preview_html + card_html[end:]

    anchor = '<div class="review-notes">'
    anchor_idx = card_html.find(anchor)
    if anchor_idx < 0:
        raise ValueError("Could not find review-notes block inside the matched card")
    return card_html[:anchor_idx] + preview_html + card_html[anchor_idx:]


def ensure_preview_style(html_text: str) -> str:
    if ".crop-preview{" in html_text:
        return html_text
    style_end = html_text.find("</style>")
    if style_end < 0:
        return html_text
    return html_text[:style_end] + PREVIEW_STYLE + "\n" + html_text[style_end:]


def parse_card_footer(card_html: str) -> tuple[str, str]:
    match = FOOTER_RE.search(card_html)
    if not match:
        raise ValueError("Could not parse scene/image footer from card")
    return html.unescape(match.group(1)).strip(), html.unescape(match.group(2)).strip()


def parse_card_targets(card_html: str) -> list[ReviewTarget]:
    merged: dict[int, ReviewTarget] = {}
    try:
        merged = {
            target.obj_id: target for target in parse_review_targets(card_html)
        }
    except ValueError:
        merged = {}

    review_blocks = _extract_review_blocks(card_html)
    for line in review_blocks.get("Referability Audit", []):
        match = AUDIT_OBJECT_RE.match(line)
        if not match:
            continue
        obj_id_raw = match.group("obj_id").strip()
        if not obj_id_raw.isdigit():
            continue
        obj_id = int(obj_id_raw)
        role = match.group("role").strip()
        label = match.group("label").strip()
        status = match.group("result").strip() or "pass"
        reasons = match.group("reasons").strip()
        reason = "" if reasons == "-" else reasons

        existing = merged.get(obj_id)
        if existing is None:
            merged[obj_id] = ReviewTarget(
                label=label,
                obj_id=obj_id,
                roles=(role,) if role else (),
                status=status,
                reason=reason,
            )
            continue

        merged[obj_id] = ReviewTarget(
            label=existing.label or label,
            obj_id=obj_id,
            roles=tuple(dict.fromkeys((*existing.roles, role) if role else existing.roles)),
            status=existing.status or status,
            reason=existing.reason or reason,
        )

    if not merged:
        raise ValueError("No review targets found in card")
    return list(merged.values())


def _decode_embedded_card_image(card_html: str):
    import cv2
    import numpy as np

    match = EMBEDDED_IMAGE_RE.search(card_html)
    if not match:
        return None, None
    try:
        raw = base64.b64decode(match.group("b64"))
    except Exception as exc:
        raise RuntimeError(f"Failed to decode embedded image: {exc}") from exc
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode embedded image bytes with OpenCV")
    return image, (match.group("mime").strip() or "image/jpeg")


def _scale_bounds(
    bounds: object,
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> list[int] | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        u_min, u_max, v_min, v_max = [float(value) for value in bounds]
    except (TypeError, ValueError):
        return None
    scale_x = 1.0 if src_width <= 0 else dst_width / float(src_width)
    scale_y = 1.0 if src_height <= 0 else dst_height / float(src_height)
    scaled = [
        int(round(u_min * scale_x)),
        int(round(u_max * scale_x)),
        int(round(v_min * scale_y)),
        int(round(v_max * scale_y)),
    ]
    scaled[0] = max(0, min(dst_width, scaled[0]))
    scaled[1] = max(0, min(dst_width, scaled[1]))
    scaled[2] = max(0, min(dst_height, scaled[2]))
    scaled[3] = max(0, min(dst_height, scaled[3]))
    return scaled


def _rescale_visibility_meta(
    visibility_meta: dict[str, object],
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> dict[str, object]:
    scaled = dict(visibility_meta)
    roi_bounds = _scale_bounds(
        visibility_meta.get("roi_bounds_px"),
        src_width=src_width,
        src_height=src_height,
        dst_width=dst_width,
        dst_height=dst_height,
    )
    scaled["roi_bounds_px"] = roi_bounds
    if roi_bounds is not None:
        width = max(0, roi_bounds[1] - roi_bounds[0])
        height = max(0, roi_bounds[3] - roi_bounds[2])
        scaled["projected_area_px"] = float(width * height)
        scaled["edge_margin_px"] = float(
            min(
                roi_bounds[0],
                roi_bounds[2],
                dst_width - roi_bounds[1],
                dst_height - roi_bounds[3],
            )
        )
    return scaled


def _load_scene_context(
    *,
    scene_id: str,
    scene_dir: Path,
    output_dir: Path,
    scene_metadata: Path | None,
) -> dict[str, object]:
    from scripts import run_pipeline as rp

    scene = None
    errors: list[str] = []
    metadata_path = scene_metadata
    if metadata_path is None:
        candidate = rp._question_review_scene_metadata_path(output_dir, scene_id)
        if candidate.exists():
            metadata_path = candidate

    if metadata_path is not None and metadata_path.exists():
        try:
            scene = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read scene metadata {metadata_path}: {exc}") from exc
    elif scene_dir.exists():
        scene = rp.parse_scene(scene_dir)
    else:
        raise FileNotFoundError(
            f"Could not resolve scene data for {scene_id}. "
            "Provide --scene_dir and optionally --scene_metadata."
        )

    objects = scene.get("objects", []) if isinstance(scene, dict) else []
    objects_by_id: dict[int, dict[str, object]] = {}
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_id = rp._coerce_object_id(obj.get("id"))
            if obj_id is None:
                continue
            objects_by_id[obj_id] = obj

    poses: dict[str, object] = {}
    color_intrinsics = None
    if scene_dir.exists():
        try:
            axis_align = rp.load_axis_alignment(scene_dir)
            poses = rp.load_scannet_poses(scene_dir, axis_alignment=axis_align)
        except Exception as exc:
            errors.append(f"missing_pose_data:{exc}")
        try:
            color_intrinsics = rp.load_scannet_intrinsics(scene_dir)
        except Exception as exc:
            errors.append(f"missing_color_intrinsics:{exc}")

    return {
        "objects": objects,
        "objects_by_id": objects_by_id,
        "poses": poses,
        "color_intrinsics": color_intrinsics,
        "errors": errors,
    }


def build_crop_map(
    *,
    scene_id: str,
    image_name: str,
    scene_dir: Path,
    image_path: Path | None,
    output_dir: Path,
    scene_metadata: Path | None,
    image_override=None,
) -> dict[int, dict[str, object]]:
    import cv2

    from scripts import run_pipeline as rp

    scene_context = _load_scene_context(
        scene_id=scene_id,
        scene_dir=scene_dir,
        output_dir=output_dir,
        scene_metadata=scene_metadata,
    )
    pose = scene_context["poses"].get(image_name) if isinstance(scene_context["poses"], dict) else None
    if pose is None:
        raise RuntimeError(f"Missing camera pose for {scene_id}/{image_name}")
    color_intrinsics = scene_context.get("color_intrinsics")
    if color_intrinsics is None:
        raise RuntimeError(f"Missing color intrinsics for {scene_id}")
    objects = scene_context.get("objects", [])
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"Missing scene objects for {scene_id}")
    image = None
    image_for_crops = image_override
    image_path_for_visibility = image_path if image_path is not None and image_path.exists() else None

    if image_path_for_visibility is not None:
        image = cv2.imread(str(image_path_for_visibility))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path_for_visibility}")
        image_for_crops = image
    elif image_override is None:
        raise FileNotFoundError(
            f"Image not found for {scene_id}/{image_name}. "
            "Provide a readable source image or an embedded image override."
        )

    raw_visibility = rp.compute_frame_object_visibility(
        objects=objects,
        pose=pose,
        color_intrinsics=color_intrinsics,
        image_path=image_path_for_visibility,
        depth_image=None,
        depth_intrinsics=None,
        strict_mode=False,
    )
    crop_by_obj_id: dict[int, dict[str, object]] = {}
    src_width = int(getattr(color_intrinsics, "width", 0) or 0)
    src_height = int(getattr(color_intrinsics, "height", 0) or 0)
    dst_height, dst_width = int(image_for_crops.shape[0]), int(image_for_crops.shape[1])
    for obj_id, meta in raw_visibility.items():
        crop_meta = meta
        if image is None:
            crop_meta = _rescale_visibility_meta(
                meta,
                src_width=src_width,
                src_height=src_height,
                dst_width=dst_width,
                dst_height=dst_height,
            )
        crop_by_obj_id[int(obj_id)] = rp._build_question_review_crop(image_for_crops, crop_meta)
    return crop_by_obj_id


def _resolve_paths(
    *,
    scene_id: str,
    image_name: str,
    data_root: Path | None,
    scene_dir: Path | None,
    image_path: Path | None,
) -> tuple[Path, Path]:
    resolved_scene_dir = scene_dir
    if resolved_scene_dir is None:
        if data_root is None:
            raise ValueError("Provide either --scene_dir or --data_root")
        resolved_scene_dir = data_root / scene_id

    resolved_image_path = image_path or (resolved_scene_dir / "color" / image_name)
    return resolved_scene_dir, resolved_image_path


def embed_all_card_crop_previews(
    *,
    html_path: Path,
    data_root: Path,
    output_dir: Path,
    output_html: Path,
) -> Path:
    html_text = html_path.read_text(encoding="utf-8")
    updated_parts: list[str] = []
    cursor = 0
    crop_cache: dict[tuple[str, str], dict[int, dict[str, object]]] = {}

    for card_start, card_end in _iter_div_ranges(html_text, CARD_START):
        card_html = html_text[card_start:card_end]
        updated_parts.append(html_text[cursor:card_start])

        try:
            scene_id, image_name = parse_card_footer(card_html)
            targets = parse_card_targets(card_html)
        except ValueError:
            updated_parts.append(card_html)
            cursor = card_end
            continue

        frame_key = (scene_id, image_name)
        crop_by_obj_id = crop_cache.get(frame_key)
        if crop_by_obj_id is None:
            resolved_scene_dir, resolved_image_path = _resolve_paths(
                scene_id=scene_id,
                image_name=image_name,
                data_root=data_root,
                scene_dir=None,
                image_path=None,
            )
            embedded_image, _embedded_mime = _decode_embedded_card_image(card_html)
            crop_by_obj_id = build_crop_map(
                scene_id=scene_id,
                image_name=image_name,
                scene_dir=resolved_scene_dir,
                image_path=resolved_image_path,
                output_dir=output_dir,
                scene_metadata=None,
                image_override=embedded_image,
            )
            crop_cache[frame_key] = crop_by_obj_id

        preview_html = build_crop_preview_html(
            targets,
            crop_by_obj_id,
            image_name=image_name,
            subtitle=(
                f"Per-object crops for the mentioned instances in {image_name}. "
                "If the original color frame is unavailable, crops are scaled from the embedded viewer image."
            ),
        )
        updated_parts.append(inject_crop_preview(card_html, preview_html))
        cursor = card_end

    updated_parts.append(html_text[cursor:])
    updated_html = ensure_preview_style("".join(updated_parts))
    output_html.write_text(updated_html, encoding="utf-8")
    return output_html


def embed_review_crop_preview(
    *,
    html_path: Path,
    scene_id: str,
    image_name: str,
    match_text: str,
    data_root: Path | None,
    scene_dir: Path | None,
    image_path: Path | None,
    output_dir: Path,
    scene_metadata: Path | None,
    output_html: Path,
) -> Path:
    html_text = html_path.read_text(encoding="utf-8")
    card_start, card_end, card_html = _find_matching_card(
        html_text,
        scene_id=scene_id,
        image_name=image_name,
        match_text=match_text,
    )
    targets = parse_card_targets(card_html)
    resolved_scene_dir, resolved_image_path = _resolve_paths(
        scene_id=scene_id,
        image_name=image_name,
        data_root=data_root,
        scene_dir=scene_dir,
        image_path=image_path,
    )
    embedded_image, _embedded_mime = _decode_embedded_card_image(card_html)
    crop_by_obj_id = build_crop_map(
        scene_id=scene_id,
        image_name=image_name,
        scene_dir=resolved_scene_dir,
        image_path=resolved_image_path,
        output_dir=output_dir,
        scene_metadata=scene_metadata,
        image_override=embedded_image,
    )
    preview_html = build_crop_preview_html(
        targets,
        crop_by_obj_id,
        image_name=image_name,
    )
    updated_card = inject_crop_preview(card_html, preview_html)
    updated_html = html_text[:card_start] + updated_card + html_text[card_end:]
    updated_html = ensure_preview_style(updated_html)
    output_html.write_text(updated_html, encoding="utf-8")
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed crop previews into one review card or all cards in a viewer HTML."
    )
    parser.add_argument("--html", type=Path, required=True, help="Existing flagged review HTML")
    parser.add_argument("--scene_id")
    parser.add_argument("--image_name")
    parser.add_argument(
        "--match_text",
        help="Text that uniquely identifies the target card, e.g. sofa#26=unsure",
    )
    parser.add_argument(
        "--all_cards",
        action="store_true",
        help="Embed crop previews for every card in the HTML using scene/image footer info",
    )
    parser.add_argument("--data_root", type=Path, help="Parent directory of ScanNet scene dirs")
    parser.add_argument("--scene_dir", type=Path, help="Direct path to the scene directory")
    parser.add_argument("--image_path", type=Path, help="Direct path to the source image")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Pipeline output dir used to resolve scene_metadata by default",
    )
    parser.add_argument("--scene_metadata", type=Path, help="Optional explicit scene metadata JSON")
    parser.add_argument("--output_html", type=Path, help="Write result to this HTML path")
    parser.add_argument("--inplace", action="store_true", help="Modify the input HTML in place")
    args = parser.parse_args()

    if args.output_html is not None and args.inplace:
        parser.error("--output_html and --inplace cannot be combined")
    if args.all_cards and args.scene_dir is not None:
        parser.error("--all_cards does not support --scene_dir; use --data_root")
    if args.all_cards and args.data_root is None:
        parser.error("--all_cards requires --data_root")
    if not args.all_cards and (not args.scene_id or not args.image_name or not args.match_text):
        parser.error("--scene_id, --image_name, and --match_text are required unless --all_cards is set")

    output_html = args.html
    if args.output_html is not None:
        output_html = args.output_html
    elif not args.inplace:
        output_html = args.html.with_name(f"{args.html.stem}_with_crops{args.html.suffix}")

    if args.all_cards:
        result_path = embed_all_card_crop_previews(
            html_path=args.html,
            data_root=args.data_root,
            output_dir=args.output_dir,
            output_html=output_html,
        )
        print(f"Saved crop preview HTML: {result_path}")
        return

    result_path = embed_review_crop_preview(
        html_path=args.html,
        scene_id=args.scene_id,
        image_name=args.image_name,
        match_text=args.match_text,
        data_root=args.data_root,
        scene_dir=args.scene_dir,
        image_path=args.image_path,
        output_dir=args.output_dir,
        scene_metadata=args.scene_metadata,
        output_html=output_html,
    )
    print(f"Saved crop preview HTML: {result_path}")


if __name__ == "__main__":
    main()
