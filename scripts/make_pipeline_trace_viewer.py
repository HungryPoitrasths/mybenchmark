#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from collections import Counter
from pathlib import Path
from typing import Any


def _h(value: Any) -> str:
    return html.escape(str(value))


def _question_card(question: dict[str, Any]) -> str:
    options = question.get("options") if isinstance(question.get("options"), list) else []
    options_html = "".join(
        f'<div class="option">{chr(65 + idx)}. {_h(option)}</div>'
        for idx, option in enumerate(options)
    )
    if not options_html:
        options_html = '<div class="muted">No options</div>'
    answer = _h(question.get("answer", "-"))
    return (
        '<div class="qcard">'
        f'<div class="qmeta"><span class="pill">{_h(question.get("level", "?"))}</span>'
        f'<span class="pill">{_h(question.get("type", "?"))}</span>'
        f'<span class="pill">{_h(question.get("trace_question_id", "-"))}</span>'
        f'<span class="pill answer">Answer {answer}</span></div>'
        f'<div class="qtext">{_h(question.get("question", ""))}</div>'
        f'<div class="options">{options_html}</div>'
        '</div>'
    )


def _render_stage_table(stage_summaries: list[dict[str, Any]]) -> str:
    if not stage_summaries:
        return '<div class="muted">No stage summaries recorded.</div>'
    rows = []
    for entry in stage_summaries:
        details = entry.get("details")
        detail_text = json.dumps(details, ensure_ascii=False, sort_keys=True) if isinstance(details, dict) else ""
        rows.append(
            "<tr>"
            f"<td>{_h(entry.get('stage', '-'))}</td>"
            f"<td>{_h(entry.get('status', '-'))}</td>"
            f"<td>{_h(entry.get('elapsed_ms', '-'))}</td>"
            f"<td>{_h(detail_text)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>Status</th><th>Elapsed ms</th><th>Details</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_object_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No object audit rows.</div>'
    body = []
    for row in rows:
        tags = ", ".join(str(tag) for tag in row.get("tags", [])) or "-"
        body.append(
            "<tr>"
            f"<td>{_h(row.get('id', '-'))}</td>"
            f"<td>{_h(row.get('label', '-'))}</td>"
            f"<td>{_h(tags)}</td>"
            f"<td>{_h(row.get('attachment_summary', '-'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Label</th><th>Tags</th><th>Attachment</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _render_attachment_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No attachment rows.</div>'
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{_h(row.get('parent_id', '-'))}</td>"
            f"<td>{_h(row.get('parent_label', '-'))}</td>"
            f"<td>{_h(row.get('child_id', '-'))}</td>"
            f"<td>{_h(row.get('child_label', '-'))}</td>"
            f"<td>{_h(row.get('relation_type', '-'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Parent ID</th><th>Parent</th><th>Child ID</th><th>Child</th><th>Type</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _render_lifecycle(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="muted">No question lifecycle rows.</div>'
    body = []
    for row in rows:
        removal = row.get("removal_reason") or "-"
        detail = row.get("removal_detail")
        duplicate_of = row.get("duplicate_of_trace_question_id")
        duplicate_of_question = row.get("duplicate_of_question")
        removal_details = row.get("removal_details")
        detail_parts: list[str] = []
        if detail:
            detail_parts.append(f"<div>{_h(detail)}</div>")
        if duplicate_of:
            detail_parts.append(f'<div class="muted">Kept ID: {_h(duplicate_of)}</div>')
        if duplicate_of_question:
            detail_parts.append(f'<div class="muted">Kept Question: {_h(duplicate_of_question)}</div>')
        if isinstance(removal_details, dict) and removal_details:
            detail_parts.append(
                f'<pre class="compact">{_h(json.dumps(removal_details, ensure_ascii=False, indent=2, sort_keys=True))}</pre>'
            )
        detail_html = "".join(detail_parts) or "-"
        body.append(
            "<tr>"
            f"<td>{_h(row.get('trace_question_id', '-'))}</td>"
            f"<td>{_h(row.get('status', '-'))}</td>"
            f"<td>{_h(row.get('trace_source', '-'))}</td>"
            f"<td>{_h(row.get('level', '-'))}</td>"
            f"<td>{_h(row.get('type', '-'))}</td>"
            f"<td>{_h(removal)}</td>"
            f"<td>{detail_html}</td>"
            f"<td>{_h(row.get('question', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Status</th><th>Source</th><th>Level</th><th>Type</th><th>Removal Reason</th><th>Removal Detail</th><th>Question</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _render_json_panel(payload: Any) -> str:
    return f"<pre>{_h(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))}</pre>"


def _image_path_to_data_uri(image_path: Any) -> str | None:
    if not image_path:
        return None
    path = Path(str(image_path))
    if not path.exists() or not path.is_file():
        return None
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime_type};base64,{payload}"


def build_single_frame_trace_html(trace_doc: dict[str, Any]) -> str:
    input_doc = trace_doc.get("input", {}) if isinstance(trace_doc.get("input"), dict) else {}
    frame_context = trace_doc.get("frame_context", {}) if isinstance(trace_doc.get("frame_context"), dict) else {}
    raw_questions = trace_doc.get("raw_questions", []) if isinstance(trace_doc.get("raw_questions"), list) else []
    final_questions = trace_doc.get("final_questions", []) if isinstance(trace_doc.get("final_questions"), list) else []
    lifecycle = trace_doc.get("question_lifecycle", []) if isinstance(trace_doc.get("question_lifecycle"), list) else []
    stage_summaries = trace_doc.get("stage_summaries", []) if isinstance(trace_doc.get("stage_summaries"), list) else []
    qc = trace_doc.get("quality_control", {}) if isinstance(trace_doc.get("quality_control"), dict) else {}
    image_src = (
        frame_context.get("image_src")
        or _image_path_to_data_uri(frame_context.get("image_path"))
        or frame_context.get("image_uri")
    )
    status = trace_doc.get("status", "unknown")
    stop_reason = trace_doc.get("stop_reason") or "completed"
    stop_details = trace_doc.get("stop_details", {}) if isinstance(trace_doc.get("stop_details"), dict) else {}
    raw_type_counts = Counter(str(question.get("type", "unknown")) for question in raw_questions)
    final_type_counts = Counter(str(question.get("type", "unknown")) for question in final_questions)
    count_summary = ", ".join(f"{label}={count}" for label, count in sorted(raw_type_counts.items())) or "-"
    final_summary = ", ".join(f"{label}={count}" for label, count in sorted(final_type_counts.items())) or "-"
    image_html = (
        f'<img src="{_h(image_src)}" alt="{_h(input_doc.get("image_name", ""))}">'
        if image_src else f'<div class="muted">Cannot load image preview. Expected image path: {_h(frame_context.get("image_path", "-"))}</div>'
    )
    final_cards = "".join(_question_card(question) for question in final_questions) or '<div class="muted">No final questions.</div>'
    skipped_steps = qc.get("skipped_steps", []) if isinstance(qc.get("skipped_steps"), list) else []
    skipped_html = "".join(f"<li>{_h(item)}</li>" for item in skipped_steps) or "<li>None</li>"
    outcome_note = (
        f'<p class="warning">Pipeline stopped early at `{_h(stop_reason)}`. Inspect the diagnostics below before checking question outputs.</p>'
        if status != "completed" else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(input_doc.get("scene_id", "single-frame-trace"))} trace</title>
  <style>
    body {{ margin: 0; background: #eef3f7; color: #16202b; font: 14px/1.45 "Segoe UI", Arial, sans-serif; }}
    .page {{ max-width: 1480px; margin: 0 auto; padding: 18px; display: grid; gap: 16px; }}
    .card {{ background: #fff; border: 1px solid rgba(22,32,43,.08); border-radius: 18px; padding: 16px; box-shadow: 0 10px 26px rgba(22,32,43,.06); }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ background: #f6fafc; border: 1px solid rgba(22,32,43,.06); border-radius: 14px; padding: 10px 12px; }}
    .k {{ font-size: 12px; color: #667787; }}
    .v {{ font-size: 24px; font-weight: 700; }}
    .s {{ font-size: 12px; color: #6c7d8c; margin-top: 4px; }}
    .muted {{ color: #637384; }}
    .pill {{ display: inline-block; padding: 4px 9px; margin: 0 6px 6px 0; border-radius: 999px; background: #edf5fb; border: 1px solid rgba(22,32,43,.08); }}
    .pill.answer {{ background: #eef8ee; }}
    .twocol {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(380px, .95fr); gap: 16px; align-items: start; }}
    .imgbox {{ background: #121a23; border-radius: 14px; overflow: hidden; aspect-ratio: 4 / 3; display: flex; align-items: center; justify-content: center; }}
    .imgbox img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid rgba(22,32,43,.08); text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; color: #60717f; background: #fff; position: sticky; top: 0; }}
    .qcard {{ border: 1px solid rgba(22,32,43,.08); border-radius: 14px; padding: 12px; margin-bottom: 10px; }}
    .qmeta {{ margin-bottom: 8px; }}
    .qtext {{ font-size: 15px; margin-bottom: 8px; }}
    .option {{ color: #3e5468; margin-bottom: 4px; }}
    pre {{ background: #101722; color: #dce6ef; padding: 14px; border-radius: 14px; overflow: auto; }}
    ul {{ margin: 8px 0 0 18px; }}
    .warning {{ margin: 12px 0 0; padding: 12px 14px; border-radius: 12px; background: #fff4dd; color: #7a5511; border: 1px solid rgba(196,145,33,.28); }}
    .compact {{ margin: 8px 0 0; font-size: 12px; }}
    @media (max-width: 1180px) {{ .metrics, .twocol {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="card">
      <h1>Single-Frame Pipeline Trace</h1>
      <div class="metrics">
        <div class="metric"><div class="k">Status</div><div class="v">{_h(status)}</div><div class="s">{_h(stop_reason)}</div></div>
        <div class="metric"><div class="k">Scene / Frame</div><div class="v">{_h(input_doc.get("scene_id", "-"))}</div><div class="s">{_h(input_doc.get("image_name", "-"))}</div></div>
        <div class="metric"><div class="k">Raw Questions</div><div class="v">{len(raw_questions)}</div><div class="s">{_h(count_summary)}</div></div>
        <div class="metric"><div class="k">Final Questions</div><div class="v">{len(final_questions)}</div><div class="s">{_h(final_summary)}</div></div>
      </div>
      <p class="muted" style="margin-top:12px">Referability source: {_h(frame_context.get("referability_source", input_doc.get("referability_source", "-")))}. Quality control mode: {_h(qc.get("mode", "-"))}.</p>
      {outcome_note}
    </section>

    <section class="card">
      <h2>Stage Timeline</h2>
      {_render_stage_table(stage_summaries)}
    </section>

    <section class="card">
      <h2>Run Diagnostics</h2>
      <h3>Stop Details</h3>
      {_render_json_panel(stop_details)}
      <h3>Inputs</h3>
      {_render_json_panel(input_doc)}
    </section>

    <section class="card">
      <h2>Frame Context</h2>
      <div class="twocol">
        <div>
          <div class="imgbox">{image_html}</div>
        </div>
        <div>
          <div class="metrics">
            <div class="metric"><div class="k">Selector Visible</div><div class="v">{len(frame_context.get("selector_visible_object_ids", []))}</div><div class="s">{_h(frame_context.get("candidate_visibility_source", "-"))}</div></div>
            <div class="metric"><div class="k">Pipeline Visible</div><div class="v">{len(frame_context.get("pipeline_visible_object_ids_used_for_generation", []))}</div><div class="s">{_h(frame_context.get("pipeline_skip_reason") or "generated")}</div></div>
            <div class="metric"><div class="k">Referable Objects</div><div class="v">{len(frame_context.get("referable_object_ids", []))}</div><div class="s">{_h(frame_context.get("frame_reject_reason") or "frame usable")}</div></div>
            <div class="metric"><div class="k">Attachments</div><div class="v">{len(frame_context.get("attachment_rows", []))}</div><div class="s">{_h(frame_context.get("image_path", "-"))}</div></div>
          </div>
          <h3 style="margin-top:16px">Referability Labels</h3>
          <pre>{_h(json.dumps(frame_context.get("vlm_label_statuses", {}), ensure_ascii=False, indent=2, sort_keys=True))}</pre>
        </div>
      </div>
      <h3>Object Audit</h3>
      {_render_object_rows(frame_context.get("object_rows", []))}
      <h3>Attachment Rows</h3>
      {_render_attachment_rows(frame_context.get("attachment_rows", []))}
    </section>

    <section class="card">
      <h2>Quality Control</h2>
      <p class="muted">Skipped benchmark-level steps in single-frame mode.</p>
      <ul>{skipped_html}</ul>
    </section>

    <section class="card">
      <h2>Question Lifecycle</h2>
      {_render_lifecycle(lifecycle)}
    </section>

    <section class="card">
      <h2>Final Questions</h2>
      {final_cards}
    </section>
  </div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML viewer from a single-frame trace JSON")
    parser.add_argument("--trace_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pipeline_trace.html"))
    args = parser.parse_args()

    trace_doc = json.loads(args.trace_json.read_text(encoding="utf-8"))
    html_text = build_single_frame_trace_html(trace_doc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"wrote trace viewer to {args.output}")


if __name__ == "__main__":
    main()
