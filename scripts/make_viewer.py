#!/usr/bin/env python3
"""Generate a self-contained HTML viewer for QA validation.

Each question is shown next to its source image.
The output is a single HTML file with base64-embedded images —
no server required, just open in any browser.

Usage:
    python scripts/make_viewer.py \
        --questions output/pilot/human_validation_sample.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --output output/pilot/viewer.html
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def img_to_b64(path: Path, max_width: int = 480) -> str | None:
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_width:
            img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CausalSpatial-Bench Viewer</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f0f2f5;margin:0;padding:20px}}
h1{{text-align:center;color:#333;margin-bottom:4px}}
.stats{{text-align:center;color:#666;font-size:14px;margin-bottom:24px}}
.card{{display:flex;background:#fff;border-radius:10px;
       box-shadow:0 2px 6px rgba(0,0,0,.12);margin-bottom:18px;overflow:hidden}}
.img-wrap{{flex:0 0 auto;width:480px;background:#222;display:flex;
           align-items:center;justify-content:center}}
.img-wrap img{{width:480px;display:block}}
.no-img{{width:480px;height:200px;display:flex;align-items:center;
         justify-content:center;color:#999;font-size:13px}}
.body{{padding:18px 20px;flex:1;min-width:0}}
.meta{{font-size:12px;color:#888;margin-bottom:10px}}
.badge{{display:inline-block;padding:2px 9px;border-radius:12px;
        font-weight:bold;font-size:11px;margin-right:6px}}
.L1{{background:#dbeafe;color:#1d4ed8}}
.L2{{background:#fce7f3;color:#9d174d}}
.L3{{background:#ede9fe;color:#5b21b6}}
.qtext{{font-size:15px;font-weight:600;color:#111;margin:0 0 14px}}
.opt{{padding:7px 12px;margin:4px 0;border-radius:6px;font-size:14px;
      background:#f8f9fa;border:1px solid #e5e7eb}}
.opt.correct{{background:#dcfce7;border-color:#86efac;font-weight:700}}
.footer{{margin-top:14px;font-size:11px;color:#aaa}}
.idx{{float:right;color:#ccc;font-size:12px}}
</style>
</head>
<body>
<h1>CausalSpatial-Bench — QA Viewer</h1>
<div class="stats">{n} questions &nbsp;·&nbsp; {levels}</div>
{cards}
</body>
</html>
"""

CARD = """\
<div class="card">
  <div class="img-wrap">{img}</div>
  <div class="body">
    <div class="meta">
      <span class="badge {level}">{level}</span>
      <span class="badge" style="background:#f3f4f6;color:#374151">{qtype}</span>
      <span class="idx">#{idx}</span>
    </div>
    <p class="qtext">{question}</p>
    {options}
    <div class="footer">{scene_id} &nbsp;/&nbsp; {image_name}</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build HTML QA viewer")
    parser.add_argument("--questions", required=True,
                        help="Path to questions JSON (e.g. human_validation_sample.json)")
    parser.add_argument("--image_root", required=True,
                        help="Root of ScanNet scans (parent of scene dirs)")
    parser.add_argument("--output", default="viewer.html")
    parser.add_argument("--max_width", type=int, default=480,
                        help="Max image width in pixels (default 480)")
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) and "questions" in data else data

    image_root = Path(args.image_root)
    level_counter: Counter = Counter()
    cards: list[str] = []

    for idx, q in enumerate(questions, 1):
        level   = q.get("level", "?")
        qtype   = q.get("type", "")
        scene   = q.get("scene_id", "")
        frame   = q.get("image_name", "")
        answer  = q.get("answer", "")
        opts    = q.get("options", [])

        level_counter[level] += 1

        # Image
        img_path = image_root / scene / "color" / frame
        b64 = img_to_b64(img_path, args.max_width)
        img_html = (
            f'<img src="data:image/jpeg;base64,{b64}">'
            if b64 else '<div class="no-img">image not found</div>'
        )

        # Options
        opt_html = ""
        for i, opt in enumerate(opts):
            letter = chr(65 + i)
            cls = "opt correct" if letter == answer else "opt"
            opt_html += f'<div class="{cls}">{letter}.&nbsp; {opt}</div>\n    '

        cards.append(CARD.format(
            img=img_html,
            level=level,
            qtype=qtype,
            idx=idx,
            question=q.get("question", ""),
            options=opt_html,
            scene_id=scene,
            image_name=frame,
        ))

        if idx % 20 == 0:
            print(f"  {idx}/{len(questions)} processed…", flush=True)

    levels_str = " &nbsp;·&nbsp; ".join(
        f'{k}: {v}' for k, v in sorted(level_counter.items())
    )
    html = PAGE.format(n=len(questions), levels=levels_str, cards="\n".join(cards))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"Saved: {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
