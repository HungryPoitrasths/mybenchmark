#!/usr/bin/env python3
"""Standalone multi-stage image quality filter with HTML gallery output.

Pipeline:
1. Compute grayscale Laplacian variance and Tenengrad focus metrics.
2. Keep only images that pass both stage-1 thresholds.
3. Score stage-1 survivors with BRISQUE.
4. Keep only images whose BRISQUE score is at or below the threshold.
5. Export JSON, CSV, and an HTML gallery for the final selected images.

Example:
    python scripts/filter_image_quality.py ^
        --input_dir scene0000_02/color ^
        --output_dir output/scene0000_02_quality ^
        --laplacian-threshold 120 ^
        --tenengrad-threshold 15 ^
        --brisque-threshold 35

BRISQUE dependency:
    This script intentionally does not change the repo's existing pipeline.
    It expects the optional `brisque` package to be installed separately:

        python -m pip install brisque
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
DEFAULT_LAPLACIAN_THRESHOLD = 120.0
DEFAULT_TENENGRAD_THRESHOLD = 15.0
DEFAULT_BRISQUE_THRESHOLD = 35.0
DEFAULT_BRISQUE_MAX_SIDE = 0
DEFAULT_REPORT_IMAGE_MAX_SIDE = 0
DEFAULT_REPORT_JPEG_QUALITY = 85


@dataclass(slots=True)
class ImageQualityRecord:
    image_path: Path
    width: int
    height: int
    laplacian_variance: float
    tenengrad: float
    stage1_pass: bool
    brisque_score: float | None = None
    brisque_input_width: int | None = None
    brisque_input_height: int | None = None
    stage2_pass: bool | None = None
    final_pass: bool = False

    def to_dict(self, *, output_dir: Path) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "relative_to_output": _html_relpath(self.image_path, output_dir),
            "width": int(self.width),
            "height": int(self.height),
            "laplacian_variance": round(float(self.laplacian_variance), 6),
            "tenengrad": round(float(self.tenengrad), 6),
            "stage1_pass": bool(self.stage1_pass),
            "brisque_score": None if self.brisque_score is None else round(float(self.brisque_score), 6),
            "brisque_input_width": self.brisque_input_width,
            "brisque_input_height": self.brisque_input_height,
            "stage2_pass": None if self.stage2_pass is None else bool(self.stage2_pass),
            "final_pass": bool(self.final_pass),
        }


class BrisqueScorer:
    def __init__(self) -> None:
        try:
            from brisque import BRISQUE
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "BRISQUE scorer is unavailable. Install it with `python -m pip install brisque` "
                "and rerun this script."
            ) from exc

        try:
            self._model = BRISQUE(url=False)
        except TypeError:
            self._model = BRISQUE()

    def score(self, image_bgr: np.ndarray) -> float:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            score = self._model.score(image_rgb)
        except TypeError:
            try:
                score = self._model.score(img=image_rgb)
            except TypeError:
                score = self._model.score(image=image_rgb)
        if score is None:
            raise RuntimeError("BRISQUE scorer returned no score")
        return float(score)


def _collect_image_paths(
    input_dir: Path,
    *,
    patterns: Sequence[str],
    recursive: bool,
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    image_paths: set[Path] = set()
    for pattern in patterns:
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        for path in iterator:
            if path.is_file():
                image_paths.add(path.resolve())
    return sorted(image_paths)


def select_image_paths_in_order(
    image_paths: Sequence[Path],
    *,
    max_images: int | None,
    sample_start: int = 0,
) -> list[Path]:
    if sample_start < 0:
        raise ValueError(f"sample_start must be >= 0, got {sample_start}")
    if max_images is not None and max_images <= 0:
        raise ValueError(f"max_images must be > 0, got {max_images}")

    ordered = list(image_paths)
    if sample_start >= len(ordered):
        return []
    if max_images is None:
        return ordered[sample_start:]
    return ordered[sample_start:sample_start + max_images]


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def compute_tenengrad(gray_image: np.ndarray) -> float:
    gray_float = gray_image.astype(np.float32)
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    return float(np.mean(gradient_magnitude))


def resize_for_brisque(image_bgr: np.ndarray, *, max_side: int | None) -> np.ndarray:
    if max_side is None or int(max_side) <= 0:
        return image_bgr

    height, width = image_bgr.shape[:2]
    longest_side = max(int(width), int(height))
    if longest_side <= int(max_side):
        return image_bgr

    scale = float(max_side) / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def read_image(image_path: Path) -> np.ndarray:
    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read image bytes: {image_path}") from exc
    if buffer.size == 0:
        raise ValueError(f"Image file is empty or unreadable: {image_path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return image


def encode_jpeg_data_url(image_bgr: np.ndarray, *, quality: int) -> str:
    normalized_quality = max(1, min(100, int(quality)))
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), normalized_quality],
    )
    if not ok:
        raise ValueError("Cannot encode embedded JPEG preview")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def encode_file_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    resolved_mime = mime_type or "application/octet-stream"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{resolved_mime};base64,{payload}"


def evaluate_stage1(
    image_path: Path,
    *,
    laplacian_threshold: float,
    tenengrad_threshold: float,
) -> tuple[ImageQualityRecord, np.ndarray]:
    image = read_image(image_path)

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = compute_laplacian_variance(gray)
    tenengrad = compute_tenengrad(gray)
    stage1_pass = (
        laplacian_variance >= float(laplacian_threshold)
        and tenengrad >= float(tenengrad_threshold)
    )
    record = ImageQualityRecord(
        image_path=image_path,
        width=int(width),
        height=int(height),
        laplacian_variance=laplacian_variance,
        tenengrad=tenengrad,
        stage1_pass=stage1_pass,
    )
    return record, image


def apply_brisque_filter(
    records_and_images: Iterable[tuple[ImageQualityRecord, np.ndarray]],
    *,
    brisque_threshold: float,
    brisque_max_side: int | None,
    scorer: BrisqueScorer,
    show_progress: bool = False,
) -> list[ImageQualityRecord]:
    results: list[ImageQualityRecord] = []
    iterable: Iterable[tuple[ImageQualityRecord, np.ndarray]] = records_and_images
    if show_progress:
        iterable = tqdm(list(records_and_images), desc="Stage 2/2 BRISQUE", unit="img")
    for record, image in iterable:
        if not record.stage1_pass:
            record.stage2_pass = False
            record.final_pass = False
            results.append(record)
            continue

        brisque_image = resize_for_brisque(image, max_side=brisque_max_side)
        record.brisque_input_width = int(brisque_image.shape[1])
        record.brisque_input_height = int(brisque_image.shape[0])
        score = scorer.score(brisque_image)
        record.brisque_score = float(score)
        record.stage2_pass = score <= float(brisque_threshold)
        record.final_pass = bool(record.stage2_pass)
        results.append(record)
    return results


def summarize_records(records: Sequence[ImageQualityRecord]) -> dict[str, Any]:
    total = len(records)
    stage1_pass = sum(1 for item in records if item.stage1_pass)
    stage2_scored = sum(1 for item in records if item.brisque_score is not None)
    final_selected = sum(1 for item in records if item.final_pass)
    return {
        "total_images": total,
        "stage1_pass": stage1_pass,
        "stage1_reject": total - stage1_pass,
        "stage2_scored": stage2_scored,
        "final_selected": final_selected,
        "final_reject": total - final_selected,
    }


def _html_relpath(path: Path, output_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path, output_dir)).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def build_html_report(
    *,
    records: Sequence[ImageQualityRecord],
    output_dir: Path,
    title: str,
    summary: dict[str, Any],
    thresholds: dict[str, float],
    copied_image_map: dict[Path, str] | None = None,
) -> str:
    copied_image_map = copied_image_map or {}
    ordered_records = sorted(records, key=lambda item: str(item.image_path.name))

    summary_items = "".join(
        f'<div class="stat"><div class="stat-value">{html.escape(str(value))}</div>'
        f'<div class="stat-label">{html.escape(str(key).replace("_", " "))}</div></div>'
        for key, value in summary.items()
    )

    threshold_items = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{value:.4f}</td>"
        "</tr>"
        for name, value in thresholds.items()
    )

    if ordered_records:
        cards = []
        for index, record in enumerate(ordered_records, start=1):
            image_src = copied_image_map.get(record.image_path)
            if image_src is None:
                image_src = _html_relpath(record.image_path, output_dir)
            status_text = "kept" if record.final_pass else "filtered out"
            stage1_text = "pass" if record.stage1_pass else "reject"
            stage2_text = (
                "-"
                if record.stage2_pass is None
                else ("pass" if record.stage2_pass else "reject")
            )
            status_class = "status-kept" if record.final_pass else "status-filtered"
            cards.append(
                f"""
                <article class="card">
                  <div class="image-wrap">
                    <img src="{html.escape(image_src)}" loading="lazy" alt="{html.escape(record.image_path.name)}" width="{record.width}" height="{record.height}" />
                  </div>
                  <div class="card-body">
                    <div class="card-title">#{index} {html.escape(record.image_path.name)}</div>
                    <div class="status-row">
                      <span class="status-pill {status_class}">{html.escape(status_text)}</span>
                      <span class="status-detail">stage1: {html.escape(stage1_text)} / stage2: {html.escape(stage2_text)}</span>
                    </div>
                    <div class="meta">{html.escape(str(record.image_path))}</div>
                    <div class="metrics">
                      <span>Laplacian: {record.laplacian_variance:.2f}</span>
                      <span>Tenengrad: {record.tenengrad:.2f}</span>
                      <span>BRISQUE: {'-' if record.brisque_score is None else f'{float(record.brisque_score):.2f}'}</span>
                      <span>BRISQUE input: {record.brisque_input_width or record.width} × {record.brisque_input_height or record.height}</span>
                      <span>Size: {record.width} × {record.height}</span>
                    </div>
                  </div>
                </article>
                """
            )
        gallery_html = "\n".join(cards)
    else:
        gallery_html = '<div class="empty">没有图片通过三阶段筛选，请先放宽阈值后重试。</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f1115;
      --panel: #171a21;
      --panel-2: #202530;
      --text: #e8ebf2;
      --muted: #a6adbb;
      --accent: #6ea8fe;
      --border: #313847;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 28px;
    }}
    .lead {{
      color: var(--muted);
      margin-bottom: 20px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    .stat-value {{
      font-size: 24px;
      font-weight: 700;
      color: var(--accent);
    }}
    .stat-label {{
      color: var(--muted);
      margin-top: 4px;
      text-transform: capitalize;
    }}
    .thresholds {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 24px;
    }}
    .thresholds td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }}
    .thresholds tr:last-child td {{
      border-bottom: none;
    }}
    .gallery {{
      display: block;
    }}
    .card {{
      display: block;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      margin-bottom: 18px;
    }}
    .image-wrap {{
      background: #000;
      overflow: auto;
    }}
    .image-wrap img {{
      display: block;
      width: auto;
      height: auto;
      max-width: none;
    }}
    .card-body {{
      padding: 18px 20px;
    }}
    .card-title {{
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
      margin-bottom: 10px;
    }}
    .status-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .status-pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .status-kept {{
      background: rgba(72, 187, 120, 0.18);
      color: #7ee787;
      border: 1px solid rgba(72, 187, 120, 0.35);
    }}
    .status-filtered {{
      background: rgba(248, 81, 73, 0.16);
      color: #ff9b9b;
      border: 1px solid rgba(248, 81, 73, 0.35);
    }}
    .status-detail {{
      color: var(--muted);
      font-size: 12px;
    }}
    .metrics {{
      display: grid;
      gap: 6px;
      font-size: 15px;
    }}
    .empty {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{html.escape(title)}</h1>
    <div class="lead">先用 Laplacian variance 和 Tenengrad 做一级筛选，再用 BRISQUE 做二级筛选。下面按图片编号顺序展示全部图片，并标记每张图是否被筛掉。</div>
    <section class="stats">{summary_items}</section>
    <table class="thresholds">
      <tbody>{threshold_items}</tbody>
    </table>
    <section class="gallery">{gallery_html}</section>
  </div>
</body>
</html>
"""


def _write_csv(
    csv_path: Path,
    *,
    records: Sequence[ImageQualityRecord],
) -> None:
    fieldnames = [
        "image_path",
        "width",
        "height",
        "laplacian_variance",
        "tenengrad",
        "stage1_pass",
        "brisque_score",
        "brisque_input_width",
        "brisque_input_height",
        "stage2_pass",
        "final_pass",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "width": int(record.width),
                    "height": int(record.height),
                    "laplacian_variance": f"{record.laplacian_variance:.6f}",
                    "tenengrad": f"{record.tenengrad:.6f}",
                    "stage1_pass": int(record.stage1_pass),
                    "brisque_score": "" if record.brisque_score is None else f"{record.brisque_score:.6f}",
                    "brisque_input_width": "" if record.brisque_input_width is None else int(record.brisque_input_width),
                    "brisque_input_height": "" if record.brisque_input_height is None else int(record.brisque_input_height),
                    "stage2_pass": "" if record.stage2_pass is None else int(bool(record.stage2_pass)),
                    "final_pass": int(record.final_pass),
                }
            )


def _build_embedded_report_images(
    records: Sequence[ImageQualityRecord],
    *,
    report_image_max_side: int | None,
    report_jpeg_quality: int,
    show_progress: bool = False,
) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    ordered = sorted(records, key=lambda item: str(item.image_path.name))
    iterable: Iterable[ImageQualityRecord] = ordered
    if show_progress:
        iterable = tqdm(ordered, desc="Embed report images", unit="img")
    for index, record in enumerate(iterable, start=1):
        if report_image_max_side is None or int(report_image_max_side) <= 0:
            mapping[record.image_path] = encode_file_data_url(record.image_path)
            continue
        image = read_image(record.image_path)
        preview = resize_for_brisque(image, max_side=report_image_max_side)
        mapping[record.image_path] = encode_jpeg_data_url(preview, quality=report_jpeg_quality)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Laplacian + Tenengrad + BRISQUE image filter with HTML output."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory that contains input images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for JSON/CSV/HTML outputs.")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=list(DEFAULT_IMAGE_PATTERNS),
        help="Glob patterns used to collect images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories under input_dir.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many images, using sorted filename order.",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Skip this many sorted images before taking the sequential sample window.",
    )
    parser.add_argument(
        "--laplacian-threshold",
        type=float,
        default=DEFAULT_LAPLACIAN_THRESHOLD,
        help="Minimum Laplacian variance required to pass stage 1.",
    )
    parser.add_argument(
        "--tenengrad-threshold",
        type=float,
        default=DEFAULT_TENENGRAD_THRESHOLD,
        help="Minimum mean Tenengrad magnitude required to pass stage 1.",
    )
    parser.add_argument(
        "--brisque-threshold",
        type=float,
        default=DEFAULT_BRISQUE_THRESHOLD,
        help="Maximum BRISQUE score allowed to pass stage 2. Lower is better.",
    )
    parser.add_argument(
        "--brisque-max-side",
        type=int,
        default=DEFAULT_BRISQUE_MAX_SIDE,
        help=(
            "Resize stage-1 survivors so the longer side is at most this many pixels before "
            "BRISQUE scoring. Use 0 to disable resizing."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Image Quality Filter Report",
        help="HTML report title.",
    )
    parser.add_argument(
        "--copy-selected-images",
        action="store_true",
        help=(
            "Embed compressed JPEG previews directly into the HTML report so the gallery remains "
            "viewable even when original image paths are inaccessible."
        ),
    )
    parser.add_argument(
        "--report-image-max-side",
        type=int,
        default=DEFAULT_REPORT_IMAGE_MAX_SIDE,
        help=(
            "When --copy-selected-images is set, embed previews whose longer side is at most this many "
            "pixels. Use 0 to embed the original image bytes at original resolution."
        ),
    )
    parser.add_argument(
        "--report-jpeg-quality",
        type=int,
        default=DEFAULT_REPORT_JPEG_QUALITY,
        help="JPEG quality for HTML report preview images when --copy-selected-images is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_image_paths = _collect_image_paths(
        args.input_dir,
        patterns=args.patterns,
        recursive=bool(args.recursive),
    )
    image_paths = select_image_paths_in_order(
        all_image_paths,
        max_images=args.max_images,
        sample_start=int(args.sample_start),
    )
    if not image_paths:
        raise ValueError(
            f"No images selected under {args.input_dir}; "
            f"available={len(all_image_paths)}, sample_start={args.sample_start}, max_images={args.max_images}"
        )

    stage1_records_and_images = [
        evaluate_stage1(
            image_path,
            laplacian_threshold=float(args.laplacian_threshold),
            tenengrad_threshold=float(args.tenengrad_threshold),
        )
        for image_path in tqdm(image_paths, desc="Stage 1/2 Sharpness", unit="img")
    ]

    brisque_scorer = BrisqueScorer()
    records = apply_brisque_filter(
        stage1_records_and_images,
        brisque_threshold=float(args.brisque_threshold),
        brisque_max_side=int(args.brisque_max_side),
        scorer=brisque_scorer,
        show_progress=True,
    )

    summary = summarize_records(records)
    thresholds = {
        "laplacian_threshold": float(args.laplacian_threshold),
        "tenengrad_threshold": float(args.tenengrad_threshold),
        "brisque_threshold": float(args.brisque_threshold),
        "brisque_max_side": float(args.brisque_max_side),
        "report_image_max_side": float(args.report_image_max_side),
        "report_jpeg_quality": float(args.report_jpeg_quality),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_image_map = (
        _build_embedded_report_images(
            records,
            report_image_max_side=int(args.report_image_max_side),
            report_jpeg_quality=int(args.report_jpeg_quality),
            show_progress=True,
        )
        if args.copy_selected_images
        else None
    )

    report_html = build_html_report(
        records=records,
        output_dir=args.output_dir,
        title=args.title,
        summary=summary,
        thresholds=thresholds,
        copied_image_map=copied_image_map,
    )

    json_path = args.output_dir / "quality_results.json"
    csv_path = args.output_dir / "quality_results.csv"
    html_path = args.output_dir / "quality_gallery.html"

    json_payload = {
        "input_dir": str(args.input_dir),
        "patterns": list(args.patterns),
        "recursive": bool(args.recursive),
        "sampling": {
            "mode": "sequential",
            "available_images": len(all_image_paths),
            "selected_images": len(image_paths),
            "sample_start": int(args.sample_start),
            "max_images": args.max_images,
        },
        "thresholds": thresholds,
        "summary": summary,
        "records": [record.to_dict(output_dir=args.output_dir) for record in records],
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, records=records)
    html_path.write_text(report_html, encoding="utf-8")

    print(f"Selected {len(image_paths)} / {len(all_image_paths)} image(s) in sequential order")
    print(f"Scanned {summary['total_images']} image(s)")
    print(f"Stage-1 pass: {summary['stage1_pass']}")
    print(f"BRISQUE max side: {args.brisque_max_side}")
    print(f"Final selected: {summary['final_selected']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
