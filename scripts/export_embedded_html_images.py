#!/usr/bin/env python3
"""Batch export embedded Base64 images from benchmark HTML files.

Supports two common HTML layouts in this repository:
1. Question viewer cards (`<div class="card">`)
2. Attachment-pair review cards (`<article class="pair-card">`)

For each parsed item, writes one image file plus a manifest that preserves the
question id or relation ids so a VLM can match the image to its metadata.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.review_viewer_html import parse_viewer_html


DATA_IMAGE_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,")


def _sanitize_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "unknown"


def _extension_for_mime(mime: str) -> str:
    mime = (mime or "").lower()
    if mime == "image/png":
        return ".png"
    if mime in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mime == "image/webp":
        return ".webp"
    return ".bin"


def _split_data_url(data_url: str) -> tuple[str, bytes]:
    if "," not in data_url:
        raise ValueError("Malformed data URL: missing comma separator")
    header, b64_data = data_url.split(",", 1)
    mime = header.removeprefix("data:").split(";", 1)[0].strip().lower()
    return mime, base64.b64decode(b64_data)


class _PairCardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[dict[str, Any]] = []
        self._current_card: dict[str, Any] | None = None
        self._card_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {str(key): ("" if value is None else str(value)) for key, value in attrs}
        classes = {
            item.strip()
            for item in str(attrs_map.get("class", "")).split()
            if item.strip()
        }

        if tag == "article" and "pair-card" in classes and self._current_card is None:
            self._current_card = {
                "scene_id": str(attrs_map.get("data-scene-id", "")).strip(),
                "image_name": str(attrs_map.get("data-image-name", "")).strip(),
                "group_id": html.unescape(str(attrs_map.get("data-group-id", "")).strip()),
                "pair_id": html.unescape(str(attrs_map.get("data-pair-id", "")).strip()),
                "parent_id": str(attrs_map.get("data-parent-id", "")).strip(),
                "parent_label": str(attrs_map.get("data-parent-label", "")).strip(),
                "child_id": str(attrs_map.get("data-child-id", "")).strip(),
                "child_label": str(attrs_map.get("data-child-label", "")).strip(),
                "deleted": str(attrs_map.get("data-deleted", "")).strip().lower() == "true",
                "image_data_url": "",
                "mime": "",
            }
            self._card_depth = 1
            return

        if self._current_card is None:
            return

        if tag == "article":
            self._card_depth += 1
            return

        if tag == "img":
            src = str(attrs_map.get("src", "")).strip()
            if src.startswith("data:image/") and not self._current_card.get("image_data_url"):
                mime = src[5:].split(";", 1)[0].strip().lower()
                self._current_card["image_data_url"] = src
                self._current_card["mime"] = mime

    def handle_endtag(self, tag: str) -> None:
        if tag != "article" or self._current_card is None:
            return
        self._card_depth -= 1
        if self._card_depth > 0:
            return
        self.cards.append(dict(self._current_card))
        self._current_card = None
        self._card_depth = 0


def _parse_pair_cards(html_text: str) -> list[dict[str, Any]]:
    parser = _PairCardParser()
    parser.feed(html_text)
    parser.close()
    records: list[dict[str, Any]] = []
    for index, card in enumerate(parser.cards, start=1):
        if not card.get("image_data_url"):
            continue
        parent_id = str(card.get("parent_id", "")).strip() or "unknown"
        child_id = str(card.get("child_id", "")).strip() or "unknown"
        records.append(
            {
                "source_type": "pair_card",
                "item_index": index,
                "item_id": f"pair_{index:04d}",
                "scene_id": str(card.get("scene_id", "")).strip(),
                "image_name": str(card.get("image_name", "")).strip(),
                "group_id": str(card.get("group_id", "")).strip(),
                "pair_id": str(card.get("pair_id", "")).strip() or f"{parent_id}->{child_id}",
                "parent_id": parent_id,
                "parent_label": str(card.get("parent_label", "")).strip(),
                "child_id": child_id,
                "child_label": str(card.get("child_label", "")).strip(),
                "deleted": bool(card.get("deleted", False)),
                "image_data_url": str(card.get("image_data_url", "")).strip(),
                "mime": str(card.get("mime", "")).strip(),
            }
        )
    return records


def _parse_question_cards(
    html_text: str,
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for card in parse_viewer_html(html_text, include_deleted=include_deleted):
        viewer_index = int(card["viewer_index"])
        records.append(
            {
                "source_type": "question_card",
                "item_index": viewer_index,
                "item_id": f"q_{viewer_index:04d}",
                "viewer_index": viewer_index,
                "scene_id": str(card["scene_id"]),
                "image_name": str(card["image_name"]),
                "question": str(card["question"]),
                "options": card.get("options", []),
                "gold_answer": card.get("gold_answer"),
                "gold_option": card.get("gold_option"),
                "badges": list(card.get("badges", [])),
                "deleted": bool(card.get("deleted", False)),
                "image_data_url": str(card["image_data_url"]),
                "mime": str(card["mime"]),
            }
        )
    return records


def _parse_generic_images(html_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, match in enumerate(
        re.finditer(r'<img\b[^>]*src="(?P<src>data:image/[^"]+)"', html_text, flags=re.IGNORECASE),
        start=1,
    ):
        src = str(match.group("src")).strip()
        mime = src[5:].split(";", 1)[0].strip().lower()
        records.append(
            {
                "source_type": "raw_img",
                "item_index": index,
                "item_id": f"img_{index:04d}",
                "image_data_url": src,
                "mime": mime,
            }
        )
    return records


def _collect_records(
    html_text: str,
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    question_records = _parse_question_cards(
        html_text,
        include_deleted=include_deleted,
    )
    if question_records:
        return question_records

    pair_records = _parse_pair_cards(html_text)
    if pair_records:
        return pair_records

    return _parse_generic_images(html_text)


def _build_image_name(record: dict[str, Any], mime: str) -> str:
    ext = _extension_for_mime(mime)
    source_type = str(record.get("source_type", "")).strip()
    scene_id = _sanitize_part(str(record.get("scene_id", "")).strip())
    image_stem = _sanitize_part(Path(str(record.get("image_name", "")).strip() or "embedded").stem)

    if source_type == "question_card":
        item_id = _sanitize_part(str(record["item_id"]))
        return f"{item_id}__{scene_id}__{image_stem}{ext}"

    if source_type == "pair_card":
        pair_id = _sanitize_part(str(record.get("pair_id", "")) or "pair")
        parent_id = _sanitize_part(str(record.get("parent_id", "")) or "parent")
        child_id = _sanitize_part(str(record.get("child_id", "")) or "child")
        return f"{pair_id}__p{parent_id}__c{child_id}__{scene_id}__{image_stem}{ext}"

    item_id = _sanitize_part(str(record["item_id"]))
    return f"{item_id}{ext}"


def _flatten_for_csv(record: dict[str, Any]) -> dict[str, str]:
    options = record.get("options")
    badges = record.get("badges")
    return {
        "item_id": str(record.get("item_id", "")),
        "source_type": str(record.get("source_type", "")),
        "scene_id": str(record.get("scene_id", "")),
        "image_name": str(record.get("image_name", "")),
        "image_path": str(record.get("image_path", "")),
        "viewer_index": str(record.get("viewer_index", "")),
        "question": str(record.get("question", "")),
        "options_json": json.dumps(options, ensure_ascii=False) if options is not None else "",
        "gold_answer": str(record.get("gold_answer", "") or ""),
        "gold_option": str(record.get("gold_option", "") or ""),
        "badges_json": json.dumps(badges, ensure_ascii=False) if badges is not None else "",
        "group_id": str(record.get("group_id", "")),
        "pair_id": str(record.get("pair_id", "")),
        "parent_id": str(record.get("parent_id", "")),
        "parent_label": str(record.get("parent_label", "")),
        "child_id": str(record.get("child_id", "")),
        "child_label": str(record.get("child_label", "")),
        "deleted": str(record.get("deleted", "")),
    }


def _write_manifest(output_dir: Path, html_path: Path, records: list[dict[str, Any]]) -> None:
    manifest = {
        "source_html": str(html_path),
        "item_count": len(records),
        "items": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_rows = [_flatten_for_csv(record) for record in records]
    fieldnames = list(csv_rows[0].keys()) if csv_rows else [
        "item_id",
        "source_type",
        "scene_id",
        "image_name",
        "image_path",
        "viewer_index",
        "question",
        "options_json",
        "gold_answer",
        "gold_option",
        "badges_json",
        "group_id",
        "pair_id",
        "parent_id",
        "parent_label",
        "child_id",
        "child_label",
        "deleted",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def _export_html(
    html_path: Path,
    output_root: Path,
    relative_to: Path,
    *,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    html_text = html_path.read_text(encoding="utf-8")
    if not DATA_IMAGE_RE.search(html_text):
        return None

    records = _collect_records(html_text, include_deleted=include_deleted)
    if not records:
        return None

    relative_html = html_path.resolve().relative_to(relative_to.resolve())
    target_dir = output_root / relative_html.parent / html_path.stem
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    exported_records: list[dict[str, Any]] = []
    for record in records:
        mime, image_bytes = _split_data_url(str(record["image_data_url"]))
        image_name = _build_image_name(record, mime)
        image_path = images_dir / image_name
        image_path.write_bytes(image_bytes)

        exported_record = dict(record)
        exported_record.pop("image_data_url", None)
        exported_record["mime"] = mime
        exported_record["image_path"] = str(image_path.resolve())
        exported_records.append(exported_record)

    _write_manifest(target_dir, html_path.resolve(), exported_records)
    return {
        "source_html": str(html_path.resolve()),
        "output_dir": str(target_dir.resolve()),
        "item_count": len(exported_records),
        "source_type": str(exported_records[0].get("source_type", "")) if exported_records else "",
    }


def _discover_html_files(html_args: list[Path], scan_root: Path) -> list[Path]:
    if html_args:
        return sorted(path.resolve() for path in html_args if path.is_file())
    return sorted(scan_root.rglob("*.html"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export embedded Base64 images from HTML files and preserve question/relation ids.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        action="append",
        default=[],
        help="Specific HTML file(s) to process. If omitted, scans --scan_root.",
    )
    parser.add_argument(
        "--scan_root",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Root directory to scan for HTML files when --html is omitted.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "output" / "extracted_embedded_images",
        help="Root directory for exported images and manifests.",
    )
    parser.add_argument(
        "--include_deleted",
        action="store_true",
        help="Also export question cards marked data-deleted=\"true\".",
    )
    args = parser.parse_args()

    scan_root = args.scan_root.resolve()
    output_root = args.output_root.resolve()
    html_files = _discover_html_files(args.html, scan_root)
    if not html_files:
        raise RuntimeError("No HTML files found to process")

    output_root.mkdir(parents=True, exist_ok=True)

    processed: list[dict[str, Any]] = []
    skipped = 0
    for html_path in html_files:
        result = _export_html(
            html_path=html_path,
            output_root=output_root,
            relative_to=scan_root if not args.html else PROJECT_ROOT,
            include_deleted=args.include_deleted,
        )
        if result is None:
            skipped += 1
            continue
        processed.append(result)

    index_doc = {
        "scan_root": str(scan_root),
        "output_root": str(output_root),
        "processed_count": len(processed),
        "skipped_count": skipped,
        "processed": processed,
    }
    (output_root / "index.json").write_text(
        json.dumps(index_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Processed {len(processed)} HTML files with embedded images; "
        f"skipped {skipped} without supported records."
    )
    print(f"Index: {output_root / 'index.json'}")


if __name__ == "__main__":
    main()
