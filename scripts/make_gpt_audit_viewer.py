#!/usr/bin/env python3
"""Build a self-contained human-review viewer for GPT audit findings."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from io import BytesIO
import html
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_benchmark_gpt import (
    CHECK_NAMES,
    load_benchmark,
    ordered_image_names,
    image_roles,
    question_fingerprint,
)
from scripts.make_viewer import _resolve_image_path


CHECK_LABELS = {
    "referability": "物体可指代性",
    "occlusion_visibility": "遮挡可见性",
    "attachment_pair": "依附关系",
    "continuity": "多图连续性",
    "fairness": "题目公平性",
}
ROLE_LABELS = {
    "first_main_view": "首张主图",
    "bridge_view": "过渡图",
    "last_main_view": "末张主图",
}


def _load_audit(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError(f"Audit file must contain a results list: {path}")
    results = [item for item in payload["results"] if isinstance(item, dict)]
    return payload, results


def join_flagged_questions(
    questions: list[dict[str, Any]],
    audit_results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[int] = set()
    for result in audit_results:
        if result.get("final_status") != "flagged":
            continue
        source_index = result.get("source_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(questions):
            raise RuntimeError(f"Invalid audit source_index: {source_index!r}")
        if source_index in seen:
            raise RuntimeError(f"Duplicate audit result for source_index {source_index}")
        question = questions[source_index]
        actual = question_fingerprint(question, source_index)
        expected = result.get("question_fingerprint")
        if expected != actual:
            raise RuntimeError(
                f"Question fingerprint mismatch at source_index {source_index}; "
                "the benchmark changed after the audit"
            )
        seen.add(source_index)
        joined.append((question, result))
    return joined


def _image_data_url(path: Path, max_width: int) -> str | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            if max_width > 0 and image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=90, optimize=True)
    except (OSError, ValueError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _render_images(
    question: dict[str, Any],
    *,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
    max_image_width: int,
) -> str:
    names = ordered_image_names(question)
    roles = image_roles(question, names)
    blocks: list[str] = []
    for index, (name, role) in enumerate(zip(names, roles), start=1):
        image_question = {**question, "image_name": name}
        path = _resolve_image_path(
            image_question, scannet_roots, scannetpp_roots, scannetpp_sensor
        )
        url = _image_data_url(path, max_image_width)
        label = f"{index}. {ROLE_LABELS.get(role, role)}"
        media = (
            f'<img src="{url}" alt="{html.escape(label)}" loading="lazy">'
            if url
            else f'<div class="missing">图片不存在<br><code>{html.escape(str(path))}</code></div>'
        )
        blocks.append(
            '<figure class="frame">'
            f"{media}<figcaption><strong>{html.escape(label)}</strong>"
            f"<span>{html.escape(name)}</span></figcaption></figure>"
        )
    if not blocks:
        blocks.append('<div class="missing">题目未提供图片</div>')
    return '<div class="frames">' + "".join(blocks) + "</div>"


def _correct_option_index(question: dict[str, Any]) -> int | None:
    answer = question.get("answer")
    if isinstance(answer, int):
        return answer if answer >= 0 else None
    text = str(answer or "").strip().upper()
    if len(text) == 1 and "A" <= text <= "Z":
        return ord(text) - ord("A")
    return None


def _render_question(question: dict[str, Any]) -> str:
    options = question.get("options") if isinstance(question.get("options"), list) else []
    correct_index = _correct_option_index(question)
    option_rows = []
    for index, option in enumerate(options):
        correct = index == correct_index or option == question.get("correct_value")
        cls = "option correct" if correct else "option"
        option_rows.append(
            f'<li class="{cls}"><span>{chr(ord("A") + index)}</span>{html.escape(str(option))}</li>'
        )
    correct_value = html.escape(str(question.get("correct_value") or ""))
    answer = html.escape(str(question.get("answer") or ""))
    answer_html = f'<div class="answer">正确答案：<strong>{answer}</strong>'
    if correct_value:
        answer_html += f' <span>{correct_value}</span>'
    answer_html += "</div>"
    return (
        f'<p class="question-text">{html.escape(str(question.get("question") or ""))}</p>'
        + ('<ol class="options">' + "".join(option_rows) + "</ol>" if option_rows else "")
        + answer_html
    )


def _stage_checks(stage: Any) -> dict[str, Any]:
    if not isinstance(stage, dict):
        return {}
    result = stage.get("result")
    return result.get("checks", {}) if isinstance(result, dict) and isinstance(result.get("checks"), dict) else {}


def _collect_final_issues(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    stage = result.get("final_result")
    rows: list[tuple[str, str, str]] = []
    if not isinstance(stage, dict):
        return [("审核流程", "error", "缺少最终审核结果")]
    if stage.get("status") != "ok":
        details = stage.get("error") or "; ".join(stage.get("validation_errors") or [])
        return [("审核流程", str(stage.get("status") or "error"), str(details or "审核输出无效"))]
    checks = _stage_checks(stage)
    for name in result.get("applicable_checks", []):
        check = checks.get(name) if isinstance(checks.get(name), dict) else {}
        verdict = str(check.get("verdict") or "missing")
        if verdict == "pass":
            continue
        label = CHECK_LABELS.get(name, name)
        issues = check.get("issues") if isinstance(check.get("issues"), list) else []
        if issues:
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                message = str(issue.get("message_zh") or check.get("summary_zh") or "需人工检查")
                code = str(issue.get("code") or verdict)
                refs: list[str] = []
                if issue.get("image_indices"):
                    refs.append("图片 " + ", ".join(str(v) for v in issue["image_indices"]))
                if issue.get("object_labels"):
                    refs.append("物体 " + ", ".join(str(v) for v in issue["object_labels"]))
                if refs:
                    message += "（" + "；".join(refs) + "）"
                rows.append((label, code, message))
        else:
            rows.append((label, verdict, str(check.get("summary_zh") or "需人工检查")))
    return rows or [("审核流程", "unknown", "最终结果未通过，但没有返回具体问题")]


def _render_issues(result: dict[str, Any]) -> str:
    rows = _collect_final_issues(result)
    return '<div class="issues">' + "".join(
        '<div class="issue">'
        f'<div><strong>{html.escape(label)}</strong><code>{html.escape(code)}</code></div>'
        f'<p>{html.escape(message)}</p></div>'
        for label, code, message in rows
    ) + "</div>"


def _render_stage(title: str, stage: Any, *, open_by_default: bool = False) -> str:
    if not isinstance(stage, dict):
        return f'<details class="trace"><summary>{html.escape(title)}：未调用</summary></details>'
    model = str(stage.get("model") or "本地输入检查")
    status = str(stage.get("status") or "unknown")
    checks = _stage_checks(stage)
    rows: list[str] = []
    for name in CHECK_NAMES:
        item = checks.get(name)
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "missing")
        summary = str(item.get("summary_zh") or "")
        rows.append(
            '<tr>'
            f'<th>{html.escape(CHECK_LABELS.get(name, name))}</th>'
            f'<td><span class="verdict {html.escape(verdict)}">{html.escape(verdict)}</span></td>'
            f'<td>{html.escape(summary)}</td></tr>'
        )
    diagnostics: list[str] = []
    if stage.get("error"):
        diagnostics.append("错误：" + str(stage["error"]))
    if stage.get("validation_errors"):
        diagnostics.append("格式校验：" + "; ".join(str(v) for v in stage["validation_errors"]))
    diag_html = "".join(f'<p class="diagnostic">{html.escape(value)}</p>' for value in diagnostics)
    opened = " open" if open_by_default else ""
    table = f'<table><tbody>{"".join(rows)}</tbody></table>' if rows else ""
    return (
        f'<details class="trace"{opened}><summary>{html.escape(title)}：'
        f'{html.escape(model)} <span class="stage-status">{html.escape(status)}</span></summary>'
        f"{diag_html}{table}</details>"
    )


def _render_card(
    question: dict[str, Any],
    result: dict[str, Any],
    *,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
    max_image_width: int,
) -> str:
    source_index = int(result["source_index"])
    qtype = str(question.get("type") or "unknown")
    categories = [str(value) for value in result.get("problem_checks", [])] or ["audit_error"]
    category_value = " ".join(categories)
    tags = "".join(
        f'<span class="tag">{html.escape(CHECK_LABELS.get(value, value))}</span>' for value in categories
    )
    metadata = (
        f'<span>#{source_index}</span><span>{html.escape(str(question.get("level") or ""))}</span>'
        f'<span>{html.escape(qtype)}</span><span>scene {html.escape(str(question.get("scene_id") or ""))}</span>'
    )
    return (
        f'<article class="audit-card" data-source-index="{source_index}" data-deleted="false" data-type="{html.escape(qtype, quote=True)}" '
        f'data-categories="{html.escape(category_value, quote=True)}">'
        '<header class="card-head"><div class="metadata">' + metadata + '</div><div class="tags">' + tags + '</div></header>'
        + _render_images(
            question,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=scannetpp_sensor,
            max_image_width=max_image_width,
        )
        + '<section class="question-block"><h2>原题</h2>' + _render_question(question) + "</section>"
        + '<section class="problem-block"><h2>需人工检查</h2>' + _render_issues(result) + "</section>"
        + '<section class="trace-block"><h2>审核轨迹</h2>'
        + _render_stage("GPT-4.1-mini 初审", result.get("primary_result"))
        + _render_stage("GPT-5.2 复核", result.get("review_result"), open_by_default=True)
        + "</section>"
        + '<footer><button type="button" class="delete-toggle">Delete</button></footer></article>'
    )


def build_viewer_html(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    title: str,
    output_filename: str,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str = "iphone",
    max_image_width: int = 1200,
) -> str:
    types = sorted({str(question.get("type") or "unknown") for question, _ in joined})
    category_counts = Counter(
        category
        for _, result in joined
        for category in (result.get("problem_checks") or ["audit_error"])
    )
    category_options = "".join(
        f'<option value="{html.escape(name, quote=True)}">{html.escape(CHECK_LABELS.get(name, name))} ({count})</option>'
        for name, count in sorted(category_counts.items())
    )
    type_options = "".join(
        f'<option value="{html.escape(name, quote=True)}">{html.escape(name)}</option>' for name in types
    )
    cards = "".join(
        _render_card(
            question,
            result,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=scannetpp_sensor,
            max_image_width=max_image_width,
        )
        for question, result in joined
    )
    edited_name = f"{Path(output_filename).stem}_edited.html"
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f4f5f6;--paper:#fff;--text:#17191c;--muted:#687079;--line:#d9dde1;--red:#a52620;--red-bg:#fff1ef;--green:#176b45;--green-bg:#eaf7f0;--blue:#215d8f;--amber:#8a5a0a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 Arial,"Microsoft YaHei",sans-serif;letter-spacing:0}}
.topbar{{position:sticky;top:0;z-index:20;background:#202428;color:#fff;border-bottom:1px solid #000;padding:14px 22px}}
.topbar-inner{{max-width:1680px;margin:auto;display:flex;align-items:center;gap:18px;flex-wrap:wrap}}
h1{{font-size:18px;margin:0 12px 0 0;white-space:nowrap}} .controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
label{{font-size:12px;color:#c9ced3}} select,button{{font:inherit;border:1px solid #aeb5bc;border-radius:6px;background:#fff;color:#202428;padding:7px 10px}}
button{{cursor:pointer;font-weight:700}} button:hover{{border-color:#69737c;background:#f3f5f6}} #export{{margin-left:auto}}
.counter{{color:#e4e7e9;min-width:170px}} main{{max-width:1680px;margin:22px auto;padding:0 18px 80px;display:grid;gap:18px}}
.audit-card{{background:var(--paper);border:1px solid var(--line);border-radius:8px;overflow:hidden;box-shadow:0 1px 2px #0000000d}}
.audit-card[data-deleted="true"]{{opacity:.48;border-color:#8d98a1}} .audit-card.filtered{{display:none}}
.card-head{{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;align-items:center}}
.metadata,.tags{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}} .metadata span{{color:#4e5861;border-right:1px solid var(--line);padding-right:7px}}
.tag{{font-size:12px;color:var(--red);background:var(--red-bg);border:1px solid #efb9b4;border-radius:999px;padding:2px 8px}}
.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr));gap:1px;background:var(--line);border-bottom:1px solid var(--line)}}
.frame{{margin:0;background:#181a1c;min-width:0;display:flex;flex-direction:column}} .frame img{{width:100%;height:auto;max-height:68vh;object-fit:contain;display:block;flex:1}}
figcaption{{background:#fff;padding:7px 10px;display:flex;justify-content:space-between;gap:12px;color:var(--muted)}} figcaption strong{{color:var(--text)}}
.missing{{min-height:220px;padding:24px;background:#f1f2f3;color:var(--red);display:grid;place-content:center;text-align:center;overflow-wrap:anywhere}}
.question-block,.problem-block,.trace-block{{padding:16px 18px;border-bottom:1px solid var(--line)}} h2{{font-size:13px;text-transform:uppercase;color:#515a62;margin:0 0 10px}}
.question-text{{font-size:16px;margin:0 0 12px;max-width:1200px}} .options{{list-style:none;padding:0;margin:0;display:grid;gap:5px;max-width:900px}}
.option{{padding:7px 10px;border-left:3px solid #c6ccd1;background:#f7f8f9}} .option span{{display:inline-block;width:28px;font-weight:700}} .option.correct{{border-color:var(--green);background:var(--green-bg)}}
.answer{{margin-top:10px;color:var(--green)}} .answer span{{margin-left:8px;color:#384047}} .issues{{display:grid;gap:8px}}
.issue{{border-left:4px solid var(--red);background:var(--red-bg);padding:9px 12px;max-width:1200px}} .issue div{{display:flex;gap:10px;align-items:center}} .issue code{{color:var(--red)}} .issue p{{margin:4px 0 0}}
.trace{{border-top:1px solid var(--line);padding:8px 0}} .trace summary{{cursor:pointer;font-weight:700}} .stage-status{{font-size:12px;color:var(--blue);margin-left:6px}}
table{{border-collapse:collapse;width:100%;margin-top:9px}} th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} th{{width:150px;background:#f6f7f8}}
.verdict{{font-weight:700}} .verdict.pass{{color:var(--green)}} .verdict.fail,.verdict.uncertain{{color:var(--red)}} .verdict.not_applicable{{color:var(--muted)}} .diagnostic{{color:var(--red);overflow-wrap:anywhere}}
footer{{padding:12px 16px;display:flex;justify-content:flex-end}} .delete-toggle{{color:#fff;background:var(--red);border-color:var(--red)}} .audit-card[data-deleted="true"] .delete-toggle{{background:var(--blue);border-color:var(--blue)}}
@media(max-width:700px){{.topbar{{position:static;padding:12px}} #export{{margin-left:0}} main{{padding:0 8px}} .card-head{{align-items:flex-start;flex-direction:column}} figcaption{{flex-direction:column;gap:2px}}}}
</style>
</head>
<body>
<div class="topbar"><div class="topbar-inner">
<h1>{html.escape(title)}</h1>
<div class="controls"><label>问题类别 <select id="category"><option value="">全部</option>{category_options}</select></label>
<label>题型 <select id="qtype"><option value="">全部</option>{type_options}</select></label>
<label><input type="checkbox" id="hide-deleted"> 隐藏已删除</label></div>
<span class="counter" id="counter"></span><button type="button" id="export">Export Edited HTML</button>
</div></div>
<main>{cards or '<p>没有需要人工复核的题目。</p>'}</main>
<script>
(() => {{
  const cards = [...document.querySelectorAll('.audit-card')];
  const category = document.getElementById('category');
  const qtype = document.getElementById('qtype');
  const hideDeleted = document.getElementById('hide-deleted');
  const counter = document.getElementById('counter');
  function updateButton(card) {{
    const deleted = card.dataset.deleted === 'true';
    const button = card.querySelector('.delete-toggle');
    button.textContent = deleted ? 'Restore' : 'Delete';
    button.setAttribute('aria-pressed', deleted ? 'true' : 'false');
  }}
  function applyFilters() {{
    let visible = 0;
    for (const card of cards) {{
      const categories = card.dataset.categories.split(' ');
      const matches = (!category.value || categories.includes(category.value)) &&
        (!qtype.value || card.dataset.type === qtype.value) &&
        (!hideDeleted.checked || card.dataset.deleted !== 'true');
      card.classList.toggle('filtered', !matches);
      if (matches) visible += 1;
    }}
    const deleted = cards.filter(card => card.dataset.deleted === 'true').length;
    counter.textContent = `显示 ${{visible}} / ${{cards.length}}，已删除 ${{deleted}}`;
  }}
  cards.forEach(card => updateButton(card));
  document.addEventListener('click', event => {{
    const button = event.target.closest('.delete-toggle');
    if (!button) return;
    const card = button.closest('.audit-card');
    card.dataset.deleted = card.dataset.deleted === 'true' ? 'false' : 'true';
    updateButton(card); applyFilters();
  }});
  [category, qtype, hideDeleted].forEach(control => control.addEventListener('change', applyFilters));
  document.getElementById('export').addEventListener('click', () => {{
    const blob = new Blob(['<!DOCTYPE html>\n' + document.documentElement.outerHTML], {{type:'text/html;charset=utf-8'}});
    const url = URL.createObjectURL(blob); const link = document.createElement('a');
    link.href = url; link.download = {json.dumps(edited_name)}; document.body.appendChild(link);
    link.click(); link.remove(); URL.revokeObjectURL(url);
  }});
  applyFilters();
}})();
</script>
</body></html>'''


def generate_viewer(
    *,
    benchmark_path: Path,
    audit_path: Path,
    output_path: Path,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str = "iphone",
    max_image_width: int = 1200,
) -> dict[str, int]:
    _, questions = load_benchmark(benchmark_path)
    _, audit_results = _load_audit(audit_path)
    joined = join_flagged_questions(questions, audit_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_viewer_html(
            joined,
            title="GPT 审核问题复核",
            output_filename=output_path.name,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=scannetpp_sensor,
            max_image_width=max_image_width,
        ),
        encoding="utf-8",
    )
    return {"flagged": len(joined), "source_questions": len(questions)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a self-contained viewer for GPT audit findings")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path, help="flagged_questions.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scannet_image_root", action="append", default=[])
    parser.add_argument("--scannetpp_image_root", action="append", default=[])
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--max_image_width", type=int, default=1200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = generate_viewer(
        benchmark_path=args.benchmark.resolve(),
        audit_path=args.audit.resolve(),
        output_path=args.output.resolve(),
        scannet_roots=[Path(value).resolve() for value in args.scannet_image_root],
        scannetpp_roots=[Path(value).resolve() for value in args.scannetpp_image_root],
        scannetpp_sensor=args.scannetpp_sensor,
        max_image_width=args.max_image_width,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
