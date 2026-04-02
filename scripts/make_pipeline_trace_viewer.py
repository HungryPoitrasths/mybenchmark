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


def _load_json_file(path: Any) -> Any:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_audit_docs(trace_doc: dict[str, Any], inline_audits: dict[str, Any] | None = None) -> dict[str, Any]:
    audit_docs = dict(inline_audits or {})
    artifacts = trace_doc.get("artifacts", {}) if isinstance(trace_doc.get("artifacts"), dict) else {}
    audit_paths = artifacts.get("audits", {}) if isinstance(artifacts.get("audits"), dict) else {}
    for key, path in audit_paths.items():
        if key in audit_docs:
            continue
        payload = _load_json_file(path)
        if payload is not None:
            audit_docs[key] = payload

    if "frame_gate" not in audit_docs:
        audit_docs["frame_gate"] = {
            "status": trace_doc.get("status"),
            "stop_reason": trace_doc.get("stop_reason"),
            "stop_details": trace_doc.get("stop_details", {}),
            "stage_summaries": trace_doc.get("stage_summaries", []),
            "input": trace_doc.get("input", {}),
            "frame_context": trace_doc.get("frame_context", {}),
        }
    if "question_lifecycle" not in audit_docs:
        audit_docs["question_lifecycle"] = trace_doc.get("question_lifecycle", [])
    if "quality_filter" not in audit_docs:
        audit_docs["quality_filter"] = {
            "quality_control": trace_doc.get("quality_control", {}),
            "removed_questions": [
                event
                for event in trace_doc.get("trace_events", [])
                if event.get("event") == "question_removed" and str(event.get("stage")) == "quality_filter"
            ],
        }
    return audit_docs


def _question_card(question: dict[str, Any]) -> str:
    options = question.get("options") if isinstance(question.get("options"), list) else []
    options_html = "".join(
        f'<div class="option">{chr(65 + idx)}. {_h(option)}</div>'
        for idx, option in enumerate(options)
    ) or '<div class="muted">No options</div>'
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


def _render_reason_index(reason_index: dict[str, Any]) -> str:
    top_blockers = reason_index.get("top_blockers", []) if isinstance(reason_index.get("top_blockers"), list) else []
    stage_stops = reason_index.get("stage_stops", []) if isinstance(reason_index.get("stage_stops"), list) else []
    generator_reason_counts = reason_index.get("generator_reason_counts", {}) if isinstance(reason_index.get("generator_reason_counts"), dict) else {}
    quality_filter_removals = reason_index.get("quality_filter_removals", {}) if isinstance(reason_index.get("quality_filter_removals"), dict) else {}
    stage_html = "".join(
        f"<li>{_h(item.get('stage', '-'))}: {_h(item.get('reason', '-'))}</li>"
        for item in stage_stops
    ) or "<li>None</li>"
    blocker_html = "".join(
        f"<li>{_h(item.get('reason', '-'))}: {_h(item.get('count', 0))}</li>"
        for item in top_blockers
    ) or "<li>None</li>"
    generator_html = "".join(
        "<tr>"
        f"<td>{_h(generator)}</td>"
        f"<td>{_h(json.dumps(counts, ensure_ascii=False, sort_keys=True))}</td>"
        "</tr>"
        for generator, counts in sorted(generator_reason_counts.items())
    ) or '<tr><td colspan="2" class="muted">No generator reasons recorded.</td></tr>'
    qc_html = "".join(
        f"<li>{_h(reason)}: {_h(count)}</li>"
        for reason, count in sorted(quality_filter_removals.items())
    ) or "<li>None</li>"
    return (
        '<div class="grid2">'
        '<div>'
        '<h3>Stage Stops</h3>'
        f'<ul>{stage_html}</ul>'
        '<h3>Top Blockers</h3>'
        f'<ul>{blocker_html}</ul>'
        '<h3>Quality Filter Removals</h3>'
        f'<ul>{qc_html}</ul>'
        '</div>'
        '<div>'
        '<h3>Generator Reason Counts</h3>'
        '<table><thead><tr><th>Generator</th><th>Reasons</th></tr></thead>'
        f'<tbody>{generator_html}</tbody></table>'
        '</div>'
        '</div>'
    )


def _render_object_pool(object_pool_audit: dict[str, Any]) -> str:
    details = object_pool_audit.get("details", {}) if isinstance(object_pool_audit.get("details"), dict) else {}
    rows = details.get("rows", []) if isinstance(details.get("rows"), list) else []
    summary = details.get("summary", {}) if isinstance(details.get("summary"), dict) else {}
    if not rows:
        return '<div class="muted">No object pool snapshot recorded.</div>'
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{_h(row.get('id', '-'))}</td>"
            f"<td>{_h(row.get('label', '-'))}</td>"
            f"<td>{_h(', '.join(str(reason) for reason in row.get('reasons', [])) or '-')}</td>"
            f"<td>{_h(', '.join(str(tag) for tag in row.get('tags', [])) or '-')}</td>"
            "</tr>"
        )
    return (
        f'<p class="muted">Summary: {_h(json.dumps(summary, ensure_ascii=False, sort_keys=True))}</p>'
        "<table><thead><tr><th>ID</th><th>Label</th><th>Reasons</th><th>Tags</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _render_generator_summary_table(generator_docs: dict[str, dict[str, Any]]) -> str:
    if not generator_docs:
        return '<div class="muted">No generator audits recorded.</div>'
    rows = []
    for generator, doc in sorted(generator_docs.items()):
        summary = doc.get("summary", {}) if isinstance(doc.get("summary"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{_h(generator)}</td>"
            f"<td>{_h(doc.get('generated_count', summary.get('generated_count', doc.get('output_count', 0))))}</td>"
            f"<td>{_h(doc.get('candidate_count', summary.get('candidate_count', len(doc.get('candidate_events', [])))))}</td>"
            f"<td>{_h(doc.get('audit_mode', '-'))}</td>"
            f"<td>{_h(json.dumps(doc.get('reason_counts', summary.get('reason_counts', {})), ensure_ascii=False, sort_keys=True))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Generator</th><th>Generated</th><th>Candidates</th><th>Audit Mode</th><th>Reason Counts</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_candidate_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<div class="muted">No candidate-level audit for this generator.</div>'
    rows = []
    for event in events:
        evidence = event.get("evidence")
        evidence_html = (
            f'<pre class="compact">{_h(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))}</pre>'
            if isinstance(evidence, dict) and evidence
            else "-"
        )
        rows.append(
            "<tr>"
            f"<td>{_h(event.get('candidate_kind', '-'))}</td>"
            f"<td>{_h(event.get('candidate_key', '-'))}</td>"
            f"<td>{_h(event.get('status', '-'))}</td>"
            f"<td>{_h(event.get('reason_code', '-'))}</td>"
            f"<td>{_h(event.get('reason_detail', '-'))}</td>"
            f"<td>{evidence_html}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Kind</th><th>Candidate</th><th>Status</th><th>Reason</th><th>Detail</th><th>Evidence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_generator_details(generator_docs: dict[str, dict[str, Any]]) -> str:
    if not generator_docs:
        return '<div class="muted">No generator audit docs.</div>'
    blocks = []
    for generator, doc in sorted(generator_docs.items()):
        contexts = doc.get("context_events", [])
        summary = doc.get("summary", {}) if isinstance(doc.get("summary"), dict) else {}
        blocks.append(
            "<details class='detail-card'>"
            f"<summary>{_h(generator)} | generated={_h(doc.get('generated_count', 0))} | candidates={_h(doc.get('candidate_count', 0))}</summary>"
            "<div class='detail-body'>"
            "<h4>Context</h4>"
            f"{_render_json_panel(contexts)}"
            "<h4>Summary</h4>"
            f"{_render_json_panel(summary)}"
            "<h4>Candidate Audit</h4>"
            f"{_render_candidate_table(doc.get('candidate_events', []))}"
            "<h4>Outputs</h4>"
            f"{_render_json_panel(doc.get('output_events', []))}"
            "</div>"
            "</details>"
        )
    return "".join(blocks)


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


def build_single_frame_trace_html(
    trace_doc: dict[str, Any],
    audit_docs: dict[str, Any] | None = None,
) -> str:
    audit_docs = _load_audit_docs(trace_doc, audit_docs)
    frame_gate = audit_docs.get("frame_gate", {}) if isinstance(audit_docs.get("frame_gate"), dict) else {}
    reason_index = audit_docs.get("reason_index", {}) if isinstance(audit_docs.get("reason_index"), dict) else {}
    object_pool = audit_docs.get("object_pool", {}) if isinstance(audit_docs.get("object_pool"), dict) else {}
    referability = audit_docs.get("referability", {}) if isinstance(audit_docs.get("referability"), dict) else {}
    lifecycle = audit_docs.get("question_lifecycle", []) if isinstance(audit_docs.get("question_lifecycle"), list) else []
    quality_filter = audit_docs.get("quality_filter", {}) if isinstance(audit_docs.get("quality_filter"), dict) else {}
    input_doc = frame_gate.get("input", {}) if isinstance(frame_gate.get("input"), dict) else {}
    frame_context = frame_gate.get("frame_context", {}) if isinstance(frame_gate.get("frame_context"), dict) else {}
    raw_questions = trace_doc.get("raw_questions", []) if isinstance(trace_doc.get("raw_questions"), list) else []
    final_questions = trace_doc.get("final_questions", []) if isinstance(trace_doc.get("final_questions"), list) else []
    stage_summaries = frame_gate.get("stage_summaries", []) if isinstance(frame_gate.get("stage_summaries"), list) else []
    qc = quality_filter.get("quality_control", {}) if isinstance(quality_filter.get("quality_control"), dict) else {}
    generator_docs = {
        key.split("generator:", 1)[1]: value
        for key, value in audit_docs.items()
        if key.startswith("generator:")
    }
    if not generator_docs:
        generator_docs = {
            key[len("generator_"):]: value
            for key, value in audit_docs.items()
            if key.startswith("generator_")
        }

    image_src = (
        frame_context.get("image_src")
        or _image_path_to_data_uri(frame_context.get("image_path"))
        or frame_context.get("image_uri")
    )
    status = trace_doc.get("status", "unknown")
    stop_reason = trace_doc.get("stop_reason") or "completed"
    raw_type_counts = Counter(str(question.get("type", "unknown")) for question in raw_questions)
    final_type_counts = Counter(str(question.get("type", "unknown")) for question in final_questions)
    count_summary = ", ".join(f"{label}={count}" for label, count in sorted(raw_type_counts.items())) or "-"
    final_summary = ", ".join(f"{label}={count}" for label, count in sorted(final_type_counts.items())) or "-"
    image_html = (
        f'<img src="{_h(image_src)}" alt="{_h(input_doc.get("image_name", ""))}">'
        if image_src else f'<div class="muted">Cannot load image preview. Expected image path: {_h(frame_context.get("image_path", "-"))}</div>'
    )
    final_cards = "".join(_question_card(question) for question in final_questions) or '<div class="muted">No final questions.</div>'
    removed_questions = quality_filter.get("removed_questions", []) if isinstance(quality_filter.get("removed_questions"), list) else []
    outcome_note = (
        f'<p class="warning">Pipeline stopped early at `{_h(stop_reason)}`. Inspect the root-cause sections below.</p>'
        if status != "completed" else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(input_doc.get("scene_id", "single-frame-trace"))} trace</title>
  <style>
    body {{ margin: 0; background: #edf2f6; color: #16202b; font: 14px/1.45 "Segoe UI", Arial, sans-serif; }}
    .page {{ max-width: 1540px; margin: 0 auto; padding: 18px; display: grid; gap: 16px; }}
    .card {{ background: #fff; border: 1px solid rgba(22,32,43,.08); border-radius: 18px; padding: 16px; box-shadow: 0 10px 26px rgba(22,32,43,.06); }}
    h1, h2, h3, h4 {{ margin: 0 0 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ background: #f6fafc; border: 1px solid rgba(22,32,43,.06); border-radius: 14px; padding: 10px 12px; }}
    .k {{ font-size: 12px; color: #667787; }}
    .v {{ font-size: 24px; font-weight: 700; }}
    .s {{ font-size: 12px; color: #6c7d8c; margin-top: 4px; }}
    .muted {{ color: #637384; }}
    .pill {{ display: inline-block; padding: 4px 9px; margin: 0 6px 6px 0; border-radius: 999px; background: #edf5fb; border: 1px solid rgba(22,32,43,.08); }}
    .pill.answer {{ background: #eef8ee; }}
    .twocol {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(420px, .95fr); gap: 16px; align-items: start; }}
    .grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
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
    .detail-card {{ border: 1px solid rgba(22,32,43,.08); border-radius: 14px; padding: 10px 12px; margin-bottom: 10px; background: #fbfdff; }}
    .detail-card summary {{ cursor: pointer; font-weight: 600; }}
    .detail-body {{ margin-top: 12px; }}
    @media (max-width: 1180px) {{ .metrics, .twocol, .grid2 {{ grid-template-columns: 1fr; }} }}
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
      <p class="muted" style="margin-top:12px">Trace detail: {_h(input_doc.get("trace_detail", "-"))}. Referability source: {_h(frame_context.get("referability_source", input_doc.get("referability_source", "-")))}.</p>
      {outcome_note}
    </section>

    <section class="card">
      <h2>Root Cause Summary</h2>
      {_render_reason_index(reason_index)}
    </section>

    <section class="card">
      <h2>Stage-Centric Audit</h2>
      <h3>Stage Timeline</h3>
      {_render_stage_table(stage_summaries)}
      <h3>Inputs</h3>
      {_render_json_panel(input_doc)}
      <h3>Stop Details</h3>
      {_render_json_panel(frame_gate.get("stop_details", {}))}
    </section>

    <section class="card">
      <h2>Frame And Referability</h2>
      <div class="twocol">
        <div>
          <div class="imgbox">{image_html}</div>
        </div>
        <div>
          <div class="metrics">
            <div class="metric"><div class="k">Selector Visible</div><div class="v">{len(frame_context.get("selector_visible_object_ids", []))}</div><div class="s">{_h(frame_context.get("candidate_visibility_source", "-"))}</div></div>
            <div class="metric"><div class="k">Pipeline Visible</div><div class="v">{len(frame_context.get("pipeline_visible_object_ids_used_for_generation", []))}</div><div class="s">{_h(frame_context.get("pipeline_skip_reason") or "generated")}</div></div>
            <div class="metric"><div class="k">Referable Objects</div><div class="v">{len(frame_context.get("referable_object_ids", []))}</div><div class="s">{_h(frame_context.get("frame_reject_reason") or "focus check passed")}</div></div>
            <div class="metric"><div class="k">Attachments</div><div class="v">{len(frame_context.get("attachment_rows", []))}</div><div class="s">{_h(frame_context.get("image_path", "-"))}</div></div>
          </div>
          <h3 style="margin-top:16px">Referability Audit</h3>
          {_render_json_panel(referability)}
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Object Pool Audit</h2>
      {_render_object_pool(object_pool)}
    </section>

    <section class="card">
      <h2>Generator Overview</h2>
      {_render_generator_summary_table(generator_docs)}
    </section>

    <section class="card">
      <h2>Generator Audit Details</h2>
      {_render_generator_details(generator_docs)}
    </section>

    <section class="card">
      <h2>Quality Filter</h2>
      <p class="muted">Removed questions in single-frame mode after generation.</p>
      {_render_json_panel(qc)}
      <h3>Removed Questions</h3>
      {_render_json_panel(removed_questions)}
    </section>

    <section class="card">
      <h2>Question-Centric Audit</h2>
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
