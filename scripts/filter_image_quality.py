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
import csv
import html
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

DEFAULT_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
DEFAULT_LAPLACIAN_THRESHOLD = 120.0
DEFAULT_TENENGRAD_THRESHOLD = 15.0
DEFAULT_BRISQUE_THRESHOLD = 35.0


@dataclass(slots=True)
class ImageQualityRecord:
    image_path: Path
    width: int
    height: int
    laplacian_variance: float
    tenengrad: float
    stage1_pass: bool
    brisque_score: float | None = None
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


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def compute_tenengrad(gray_image: np.ndarray) -> float:
    gray_float = gray_image.astype(np.float32)
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    return float(np.mean(gradient_magnitude))


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
    scorer: BrisqueScorer,
) -> list[ImageQualityRecord]:
    results: list[ImageQualityRecord] = []
    for record, image in records_and_images:
        if not record.stage1_pass:
            record.stage2_pass = False
            record.final_pass = False
            results.append(record)
            continue

        score = scorer.score(image)
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
    selected = sorted(
        (record for record in records if record.final_pass),
        key=lambda item: (
            float(item.brisque_score if item.brisque_score is not None else float("inf")),
            -float(item.laplacian_variance),
            -float(item.tenengrad),
            str(item.image_path),
        ),
    )

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

    if selected:
        cards = []
        for index, record in enumerate(selected, start=1):
            image_src = copied_image_map.get(record.image_path)
            if image_src is None:
                image_src = _html_relpath(record.image_path, output_dir)
            cards.append(
                f"""
                <article class="card">
                  <div class="image-wrap">
                    <img src="{html.escape(image_src)}" loading="lazy" alt="{html.escape(record.image_path.name)}" />
                  </div>
                  <div class="card-body">
                    <div class="card-title">#{index} {html.escape(record.image_path.name)}</div>
                    <div class="meta">{html.escape(str(record.image_path))}</div>
                    <div class="metrics">
                      <span>Laplacian: {record.laplacian_variance:.2f}</span>
                      <span>Tenengrad: {record.tenengrad:.2f}</span>
                      <span>BRISQUE: {float(record.brisque_score or 0.0):.2f}</span>
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
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}
    .image-wrap {{
      background: #000;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .image-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .card-body {{
      padding: 14px;
    }}
    .card-title {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
      margin-bottom: 10px;
    }}
    .metrics {{
      display: grid;
      gap: 6px;
      font-size: 13px;
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
    <div class="lead">先用 Laplacian variance 和 Tenengrad 做一级筛选，再用 BRISQUE 做二级筛选。下面只展示最终保留的图片。</div>
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
                    "stage2_pass": "" if record.stage2_pass is None else int(bool(record.stage2_pass)),
                    "final_pass": int(record.final_pass),
                }
            )


def _copy_selected_images(
    records: Sequence[ImageQualityRecord],
    *,
    output_dir: Path,
) -> dict[Path, str]:
    copied_dir = output_dir / "selected_images"
    copied_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, str] = {}
    selected = [record for record in records if record.final_pass]
    for index, record in enumerate(selected, start=1):
        destination = copied_dir / f"{index:04d}_{record.image_path.name}"
        shutil.copy2(record.image_path, destination)
        mapping[record.image_path] = destination.relative_to(output_dir).as_posix()
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
        "--title",
        type=str,
        default="Image Quality Filter Report",
        help="HTML report title.",
    )
    parser.add_argument(
        "--copy-selected-images",
        action="store_true",
        help="Copy final selected images into output_dir/selected_images for a self-contained report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_paths = _collect_image_paths(
        args.input_dir,
        patterns=args.patterns,
        recursive=bool(args.recursive),
    )
    if not image_paths:
        raise ValueError(f"No images matched under {args.input_dir}")

    stage1_records_and_images = (
        evaluate_stage1(
            image_path,
            laplacian_threshold=float(args.laplacian_threshold),
            tenengrad_threshold=float(args.tenengrad_threshold),
        )
        for image_path in image_paths
    )

    brisque_scorer = BrisqueScorer()
    records = apply_brisque_filter(
        stage1_records_and_images,
        brisque_threshold=float(args.brisque_threshold),
        scorer=brisque_scorer,
    )

    summary = summarize_records(records)
    thresholds = {
        "laplacian_threshold": float(args.laplacian_threshold),
        "tenengrad_threshold": float(args.tenengrad_threshold),
        "brisque_threshold": float(args.brisque_threshold),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_image_map = (
        _copy_selected_images(records, output_dir=args.output_dir)
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
        "thresholds": thresholds,
        "summary": summary,
        "records": [record.to_dict(output_dir=args.output_dir) for record in records],
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, records=records)
    html_path.write_text(report_html, encoding="utf-8")

    print(f"Scanned {summary['total_images']} image(s)")
    print(f"Stage-1 pass: {summary['stage1_pass']}")
    print(f"Final selected: {summary['final_selected']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
