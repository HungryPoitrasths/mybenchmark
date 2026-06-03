#!/usr/bin/env python3
"""Audit benchmark question quality with Anthropic vision models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.review_viewer_html import parse_viewer_html

logger = logging.getLogger("question_audit")

DEFAULT_HTML_PATH = PROJECT_ROOT / "output" / "bench.html"
DEFAULT_BENCHMARK_JSON_PATH = PROJECT_ROOT / "output" / "benchmark_subset.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "qa_audit"
DEFAULT_MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-20250514")

CHECK_OBJECT_IDENTIFIABILITY = "object_identifiability"
CHECK_OCCLUSION_GT = "occlusion_gt"
CHECK_ATTACHMENT_CHAIN = "attachment_chain"
ALL_CHECKS = (
    CHECK_OBJECT_IDENTIFIABILITY,
    CHECK_OCCLUSION_GT,
    CHECK_ATTACHMENT_CHAIN,
)
CHECK_ALIASES = {
    "all": "all",
    CHECK_OBJECT_IDENTIFIABILITY: CHECK_OBJECT_IDENTIFIABILITY,
    "identifiability": CHECK_OBJECT_IDENTIFIABILITY,
    "object_identifiability": CHECK_OBJECT_IDENTIFIABILITY,
    "object": CHECK_OBJECT_IDENTIFIABILITY,
    CHECK_OCCLUSION_GT: CHECK_OCCLUSION_GT,
    "occlusion": CHECK_OCCLUSION_GT,
    "occlusion_gt": CHECK_OCCLUSION_GT,
    CHECK_ATTACHMENT_CHAIN: CHECK_ATTACHMENT_CHAIN,
    "attachment": CHECK_ATTACHMENT_CHAIN,
    "attachment_chain": CHECK_ATTACHMENT_CHAIN,
}

QUESTION_AUDIT_MAX_RETRIES = 4
QUESTION_AUDIT_RETRY_DELAY_SECONDS = 2.0

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", flags=re.IGNORECASE | re.DOTALL)

CHECK_DISPLAY_NAMES = {
    CHECK_OBJECT_IDENTIFIABILITY: "Object Identifiability",
    CHECK_OCCLUSION_GT: "L1 Occlusion GT",
    CHECK_ATTACHMENT_CHAIN: "L3 Attachment Chain",
}


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _normalize_option_texts(options: list[Any]) -> tuple[str, ...]:
    normalized: list[str] = []
    for option in options:
        if isinstance(option, dict):
            normalized.append(_normalize_text(option.get("text")))
        else:
            normalized.append(_normalize_text(str(option)))
    return tuple(normalized)


def _join_key(question: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        _normalize_text(question.get("scene_id")),
        _normalize_text(question.get("image_name")),
        _normalize_text(question.get("question")),
        _normalize_option_texts(question.get("options", [])),
    )


def _split_data_url(data_url: str) -> tuple[str, str]:
    if "," not in data_url:
        raise ValueError("Malformed data URL: missing comma separator")
    header, data = data_url.split(",", 1)
    if not header.startswith("data:") or ";base64" not in header:
        raise ValueError("Malformed data URL: expected data:<mime>;base64,...")
    mime = header.removeprefix("data:").split(";", 1)[0].strip().lower()
    return mime, data


def _resolve_api_key(env_name: str) -> str:
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    raise RuntimeError(f"Missing API key. Set environment variable {env_name}.")


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "too many requests",
            "concurrent_request_limit_exceeded",
            "too many concurrent requests",
            "overloaded",
            "timeout",
            "timed out",
            "connection reset",
            "temporarily unavailable",
            "529",
        )
    )


def _is_authentication_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "401",
            "403",
            "unauthorized",
            "authentication",
            "invalid api key",
            "invalid x-api-key",
            "permission denied",
        )
    )


def _call_with_retries(create_fn, *, context: str):
    last_exc: Exception | None = None
    for attempt in range(1, QUESTION_AUDIT_MAX_RETRIES + 1):
        try:
            return create_fn()
        except Exception as exc:
            last_exc = exc
            if _is_authentication_error(exc):
                raise RuntimeError(f"{context} failed with an authentication error: {exc}") from exc
            if not _is_retryable_error(exc) or attempt >= QUESTION_AUDIT_MAX_RETRIES:
                raise
            delay_seconds = QUESTION_AUDIT_RETRY_DELAY_SECONDS * attempt
            logger.warning(
                "%s hit a transient error (%d/%d). Retrying in %.1fs: %s",
                context,
                attempt,
                QUESTION_AUDIT_MAX_RETRIES,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)
    if last_exc is None:
        raise RuntimeError(f"{context} failed without raising an error")
    raise last_exc


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_json_response(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not raw_text.strip():
        return None, "empty response"

    candidates = [raw_text.strip()]
    fence_match = JSON_FENCE_RE.search(raw_text)
    if fence_match is not None:
        candidates.insert(0, fence_match.group("body").strip())
    balanced = _extract_balanced_json(raw_text)
    if balanced is not None:
        candidates.insert(0, balanced)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "response did not contain a valid JSON object"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "pass", "passed", "correct", "valid"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "incorrect", "invalid"}:
            return False
    return None


def _ensure_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = _normalize_text(value)
        return [normalized] if normalized else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            normalized = _normalize_text(str(item))
            if normalized:
                result.append(normalized)
        return result
    normalized = _normalize_text(str(value))
    return [normalized] if normalized else []


def _option_letter_labels(question: dict[str, Any]) -> str:
    lines: list[str] = []
    for index, text in enumerate(question.get("options", [])):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}) {text}")
    return "\n".join(lines)


def _format_mentioned_objects(question: dict[str, Any]) -> str:
    objects = question.get("mentioned_objects") or []
    lines: list[str] = []
    for obj in objects:
        role = _normalize_text(obj.get("role")) or "object"
        label = _normalize_text(obj.get("label")) or "unknown"
        obj_id = obj.get("obj_id")
        obj_id_text = f", obj_id={obj_id}" if obj_id is not None else ""
        lines.append(f"- {role}: {label}{obj_id_text}")
    if lines:
        return "\n".join(lines)

    fallback_labels = [
        _normalize_text(question.get("obj_a_label")),
        _normalize_text(question.get("obj_b_label")),
        _normalize_text(question.get("grandparent_label")),
        _normalize_text(question.get("parent_label")),
        _normalize_text(question.get("grandchild_label")),
        _normalize_text(question.get("neighbor_label")),
    ]
    deduped = [label for label in fallback_labels if label]
    return "\n".join(f"- object: {label}" for label in deduped)


def _build_object_identifiability_prompt(question: dict[str, Any]) -> tuple[str, str]:
    mentioned_objects = _format_mentioned_objects(question)
    prompt = (
        "Audit whether every object referenced by the question is visually identifiable in the image.\n\n"
        f"Question:\n{question['question']}\n\n"
        f"Mentioned objects:\n{mentioned_objects}\n\n"
        "Be strict.\n"
        "Flag the question if any mentioned object is not clearly visible, if multiple plausible instances make the reference ambiguous, "
        "or if the image does not let you reliably confirm the object's presence.\n"
        "Ignore whether the benchmark's current answer is correct; judge only identifiability.\n\n"
        "Respond with exactly one JSON object and no markdown:\n"
        "{\n"
        '  "passed": true,\n'
        '  "issues": [],\n'
        '  "objects": [\n'
        '    {"role": "reference", "label": "chair", "visible": true, "unique": true, "certain_present": true, "notes": ""}\n'
        "  ],\n"
        '  "summary": "brief explanation"\n'
        "}"
    )
    system = (
        "You are a spatial reasoning benchmark quality auditor. "
        "Judge object identifiability conservatively and return strict JSON only."
    )
    return system, prompt


def _build_occlusion_prompt(question: dict[str, Any]) -> tuple[str, str]:
    target_label = _normalize_text(question.get("obj_a_label")) or "the target object"
    prompt = (
        "Verify whether the benchmark ground-truth occlusion label is correct.\n\n"
        f"Question:\n{question['question']}\n\n"
        f"Target object: {target_label}\n"
        f"Ground-truth answer: {question.get('correct_value')}\n\n"
        "Use these labels only: unoccluded, occluded, not visible, uncertain.\n"
        "If the image does not support a reliable judgment, use uncertain and mark the audit as failed.\n\n"
        "Respond with exactly one JSON object and no markdown:\n"
        "{\n"
        '  "passed": true,\n'
        '  "gt_correct": true,\n'
        '  "observed_status": "not visible",\n'
        '  "issues": [],\n'
        '  "summary": "brief explanation"\n'
        "}"
    )
    system = (
        "You are verifying ground-truth occlusion labels for a visual spatial benchmark. "
        "Return strict JSON only."
    )
    return system, prompt


def _build_attachment_prompt(question: dict[str, Any]) -> tuple[str, str]:
    grandparent = _normalize_text(question.get("grandparent_label")) or "grandparent"
    parent = _normalize_text(question.get("parent_label")) or "parent"
    grandchild = _normalize_text(question.get("grandchild_label")) or "grandchild"
    neighbor = _normalize_text(question.get("neighbor_label")) or "neighbor"
    gt_text = _normalize_text(question.get("correct_value"))
    options_text = _option_letter_labels(question)
    prompt = (
        "Verify whether the claimed support / attachment chain is visually plausible.\n\n"
        f"Question:\n{question['question']}\n\n"
        f"Question options:\n{options_text}\n\n"
        f"Ground-truth selected options: {gt_text}\n"
        f"Expected chain: {grandparent} -> {parent} -> {grandchild}\n"
        f"Distractor / non-chain object: {neighbor}\n\n"
        "Judge only what can be supported by visible evidence in the image. "
        "A pair counts as supported only if the physical support or attachment is visually plausible.\n\n"
        "Respond with exactly one JSON object and no markdown:\n"
        "{\n"
        '  "passed": true,\n'
        '  "chain_supported": true,\n'
        '  "gt_supported": true,\n'
        '  "issues": [],\n'
        '  "pair_checks": [\n'
        '    {"parent": "cabinet", "child": "counter", "supported": true, "notes": ""},\n'
        '    {"parent": "counter", "child": "sink", "supported": true, "notes": ""}\n'
        "  ],\n"
        '  "summary": "brief explanation"\n'
        "}"
    )
    system = (
        "You are verifying physical support and attachment relationships in benchmark images. "
        "Return strict JSON only."
    )
    return system, prompt


def build_audit_prompt(question: dict[str, Any], check_type: str) -> tuple[str, str]:
    if check_type == CHECK_OBJECT_IDENTIFIABILITY:
        return _build_object_identifiability_prompt(question)
    if check_type == CHECK_OCCLUSION_GT:
        return _build_occlusion_prompt(question)
    if check_type == CHECK_ATTACHMENT_CHAIN:
        return _build_attachment_prompt(question)
    raise ValueError(f"Unsupported check type: {check_type}")


def make_anthropic_caller(
    *,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    base_url: str | None = None,
):
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is not installed. Run `pip install anthropic`.") from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = Anthropic(**kwargs)

    def call(image_data_url: str, prompt: str, *, system_prompt: str, context: str) -> str:
        mime, data = _split_data_url(image_data_url)
        response = _call_with_retries(
            lambda: client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            ),
            context=context,
        )
        parts: list[str] = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "\n".join(part for part in parts if part).strip()

    return client, call


def _normalize_check_result(
    *,
    question: dict[str, Any],
    check_type: str,
    raw_response: str,
    parsed_response: dict[str, Any] | None,
    parse_error: str | None,
    prompt_error: str | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if prompt_error:
        issues.append(prompt_error)

    if parsed_response is None:
        if parse_error:
            issues.append(parse_error)
        passed = False
    else:
        issues.extend(_ensure_string_list(parsed_response.get("issues")))
        passed = _coerce_bool(parsed_response.get("passed"))

        if check_type == CHECK_OBJECT_IDENTIFIABILITY:
            if passed is None:
                object_entries = parsed_response.get("objects")
                failures: list[str] = []
                if isinstance(object_entries, list):
                    for entry in object_entries:
                        if not isinstance(entry, dict):
                            continue
                        label = _normalize_text(entry.get("label")) or "object"
                        visible = _coerce_bool(entry.get("visible"))
                        unique = _coerce_bool(entry.get("unique"))
                        certain_present = _coerce_bool(entry.get("certain_present"))
                        if visible is False:
                            failures.append(f"{label} not clearly visible")
                        if unique is False:
                            failures.append(f"{label} not uniquely identifiable")
                        if certain_present is False:
                            failures.append(f"{label} cannot be reliably confirmed")
                issues.extend(failure for failure in failures if failure not in issues)
                passed = not failures and not prompt_error and not parse_error

        elif check_type == CHECK_OCCLUSION_GT:
            gt_correct = _coerce_bool(parsed_response.get("gt_correct"))
            observed_status = _normalize_text(parsed_response.get("observed_status"))
            if passed is None and gt_correct is not None:
                passed = gt_correct
            if observed_status == "uncertain":
                passed = False
                if "model marked the image as uncertain" not in issues:
                    issues.append("model marked the image as uncertain")
            if passed is None:
                passed = False

        elif check_type == CHECK_ATTACHMENT_CHAIN:
            chain_supported = _coerce_bool(parsed_response.get("chain_supported"))
            gt_supported = _coerce_bool(parsed_response.get("gt_supported"))
            if passed is None:
                if chain_supported is not None and gt_supported is not None:
                    passed = chain_supported and gt_supported
                elif chain_supported is not None:
                    passed = chain_supported
                else:
                    pair_checks = parsed_response.get("pair_checks")
                    supported_values: list[bool] = []
                    if isinstance(pair_checks, list):
                        for entry in pair_checks:
                            if isinstance(entry, dict):
                                supported = _coerce_bool(entry.get("supported"))
                                if supported is not None:
                                    supported_values.append(supported)
                    passed = bool(supported_values) and all(supported_values)
            if passed is None:
                passed = False
        else:
            passed = False

    summary = ""
    if parsed_response is not None:
        summary = _normalize_text(parsed_response.get("summary"))

    status = "passed" if passed else "flagged"
    if prompt_error or parsed_response is None:
        status = "error" if prompt_error or parse_error else status

    return {
        "trace_question_id": question.get("trace_question_id"),
        "viewer_index": question.get("viewer_index"),
        "check_type": check_type,
        "check_label": CHECK_DISPLAY_NAMES.get(check_type, check_type),
        "passed": bool(passed),
        "status": status,
        "issues": issues,
        "summary": summary,
        "question": question.get("question"),
        "options": question.get("options", []),
        "correct_value": question.get("correct_value"),
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "level": question.get("level"),
        "type": question.get("type"),
        "ambiguity_score": question.get("ambiguity_score"),
        "viewer_badges": question.get("viewer_badges", []),
        "image_data_url": question.get("image_data_url"),
        "claude_response": parsed_response,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "metadata": {
            "trace_question_id": question.get("trace_question_id"),
            "viewer_index": question.get("viewer_index"),
            "level": question.get("level"),
            "type": question.get("type"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "answer": question.get("answer"),
            "correct_value": question.get("correct_value"),
            "mentioned_objects": question.get("mentioned_objects"),
            "question_referability_decision": (question.get("question_referability_audit") or {}).get("decision"),
            "obj_a_label": question.get("obj_a_label"),
            "obj_b_label": question.get("obj_b_label"),
            "grandparent_label": question.get("grandparent_label"),
            "parent_label": question.get("parent_label"),
            "grandchild_label": question.get("grandchild_label"),
            "neighbor_label": question.get("neighbor_label"),
            "multi_select": question.get("multi_select"),
        },
    }


def review_question(question: dict[str, Any], *, check_type: str, call_model) -> dict[str, Any]:
    system_prompt, prompt = build_audit_prompt(question, check_type)
    try:
        raw_response = call_model(
            question["image_data_url"],
            prompt,
            system_prompt=system_prompt,
            context=f"{check_type} viewer#{question['viewer_index']} {question.get('scene_id')}/{question.get('image_name')}",
        )
        parsed_response, parse_error = _parse_json_response(raw_response)
        return _normalize_check_result(
            question=question,
            check_type=check_type,
            raw_response=raw_response,
            parsed_response=parsed_response,
            parse_error=parse_error,
        )
    except Exception as exc:
        return _normalize_check_result(
            question=question,
            check_type=check_type,
            raw_response="",
            parsed_response=None,
            parse_error=None,
            prompt_error=f"API call failed: {exc}",
        )


def load_benchmark_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        questions = payload.get("questions")
        if isinstance(questions, list):
            return [dict(item) for item in questions]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise RuntimeError(f"Unsupported benchmark JSON structure in {path}")


def load_viewer_cards(path: Path, *, include_deleted: bool = False) -> list[dict[str, Any]]:
    html_text = path.read_text(encoding="utf-8")
    return parse_viewer_html(html_text, include_deleted=include_deleted)


def merge_questions(
    *,
    cards: list[dict[str, Any]],
    benchmark_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    item_map: dict[tuple[str, str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in benchmark_questions:
        item_map[_join_key(item)].append(item)

    duplicates = {key: len(values) for key, values in item_map.items() if len(values) > 1}
    if duplicates:
        duplicate_examples = list(duplicates.items())[:3]
        raise RuntimeError(f"Benchmark join keys are not unique; examples: {duplicate_examples}")

    merged: list[dict[str, Any]] = []
    for card in cards:
        join_key = _join_key(card)
        matches = item_map.get(join_key)
        if not matches:
            raise RuntimeError(
                "Could not match viewer card to benchmark JSON for "
                f"viewer_index={card.get('viewer_index')} scene={card.get('scene_id')} image={card.get('image_name')}"
            )
        item = matches[0]
        merged_question = dict(item)
        merged_question["viewer_index"] = card["viewer_index"]
        merged_question["viewer_badges"] = list(card.get("badges", []))
        merged_question["image_data_url"] = card["image_data_url"]
        merged_question["mime"] = card["mime"]
        merged_question["deleted"] = bool(card.get("deleted", False))
        merged_question["options"] = [option["text"] for option in card.get("options", [])]
        merged_question["viewer_options"] = list(card.get("options", []))
        merged_question["html_gold_answer"] = card.get("gold_answer")
        merged_question["html_gold_option"] = card.get("gold_option")
        merged.append(merged_question)

    merged.sort(key=lambda item: int(item.get("viewer_index", 0)))
    return merged


def select_questions_for_check(
    questions: list[dict[str, Any]],
    *,
    check_type: str,
    ambiguity_threshold: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if check_type == CHECK_OBJECT_IDENTIFIABILITY:
        selected = list(questions)
        if ambiguity_threshold is not None:
            selected = [
                question
                for question in selected
                if float(question.get("ambiguity_score") or 0.0) >= ambiguity_threshold
            ]
    elif check_type == CHECK_OCCLUSION_GT:
        selected = [
            question
            for question in questions
            if question.get("level") == "L1" and question.get("type") == "occlusion"
        ]
    elif check_type == CHECK_ATTACHMENT_CHAIN:
        selected = [
            question
            for question in questions
            if question.get("level") == "L3" and question.get("type") == "attachment_chain"
        ]
    else:
        raise ValueError(f"Unsupported check type: {check_type}")

    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def _render_options(options: list[str]) -> str:
    blocks: list[str] = []
    for index, text in enumerate(options):
        letter = chr(ord("A") + index)
        blocks.append(
            '<div class="opt">'
            f'<span class="letter">{html.escape(letter)}</span>'
            f'<span class="text">{html.escape(text)}</span>'
            "</div>"
        )
    return "".join(blocks)


def _render_json_block(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def build_flagged_html(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    summary_lines = "".join(
        f'<div class="summary-line"><strong>{html.escape(CHECK_DISPLAY_NAMES.get(key, key))}:</strong> '
        f"{html.escape(str(value.get('flagged', 0)))} flagged / {html.escape(str(value.get('total', 0)))} total</div>"
        for key, value in sorted(summary.items())
    )

    cards: list[str] = []
    for item in report.get("results", []):
        badges = [
            item.get("check_label") or item.get("check_type"),
            item.get("level"),
            item.get("type"),
            item.get("status"),
        ]
        if item.get("ambiguity_score") is not None:
            badges.append(f"ambiguity={float(item['ambiguity_score']):.3f}")
        badge_html = "".join(
            f'<span class="badge">{html.escape(str(text))}</span>' for text in badges if text
        )
        issues = item.get("issues") or []
        issue_html = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in issues) or "<li>No issue text</li>"
        cards.append(
            f"""
<div class="card">
  <div class="img-wrap"><img src="{item['image_data_url']}" alt="question image"></div>
  <div class="body">
    <div class="meta">{badge_html}<span class="idx">viewer #{html.escape(str(item.get('viewer_index')))}</span></div>
    <div class="footer-mini">{html.escape(str(item.get('scene_id')))} / {html.escape(str(item.get('image_name')))} / {html.escape(str(item.get('trace_question_id')))}</div>
    <p class="qtext">{html.escape(str(item.get('question')))}</p>
    <div class="opts">{_render_options(list(item.get('options') or []))}</div>
    <div class="facts">
      <div><strong>GT:</strong> {html.escape(str(item.get('correct_value') or '-'))}</div>
      <div><strong>Summary:</strong> {html.escape(str(item.get('summary') or '-'))}</div>
    </div>
    <div class="issues"><strong>Issues</strong><ul>{issue_html}</ul></div>
    <details class="raw">
      <summary>Claude JSON</summary>
      <pre>{_render_json_block(item.get('claude_response'))}</pre>
    </details>
    <details class="raw">
      <summary>Raw Response</summary>
      <pre>{html.escape(str(item.get('raw_response') or ''))}</pre>
    </details>
  </div>
</div>
"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Question Audit Flagged Report</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f3f4f6;color:#111827;margin:0;padding:20px}}
h1{{margin:0 0 8px}}
.stats{{margin:0 0 20px;color:#4b5563}}
.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 20px}}
.summary-line{{background:#fff;border:1px solid #e5e7eb;border-radius:999px;padding:8px 12px;font-size:13px}}
.card{{display:flex;gap:0;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:18px;overflow:hidden}}
.img-wrap{{flex:0 0 480px;background:#111827;display:flex;align-items:flex-start;justify-content:center}}
.img-wrap img{{display:block;width:480px;height:auto}}
.body{{padding:18px 20px;flex:1}}
.meta{{font-size:12px;color:#6b7280;margin-bottom:6px}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;background:#e0e7ff;color:#3730a3;font-size:11px;font-weight:700;margin-right:6px;margin-bottom:4px}}
.idx{{float:right;color:#9ca3af}}
.footer-mini{{font-size:12px;color:#6b7280;margin-bottom:10px}}
.qtext{{font-size:15px;font-weight:600;line-height:1.45;margin:0 0 14px}}
.opt{{padding:7px 12px;margin:4px 0;border-radius:6px;font-size:14px;background:#f9fafb;border:1px solid #e5e7eb}}
.letter{{display:inline-block;width:24px;font-weight:700}}
.facts{{margin-top:12px;padding:12px 14px;border-radius:8px;background:#eff6ff;border:1px solid #bfdbfe;font-size:13px;line-height:1.6}}
.issues{{margin-top:12px;padding:12px 14px;border-radius:8px;background:#fff7ed;border:1px solid #fed7aa;font-size:13px}}
.issues ul{{margin:8px 0 0 18px;padding:0}}
.raw{{margin-top:10px}}
pre{{white-space:pre-wrap;word-break:break-word;background:#111827;color:#e5e7eb;padding:12px;border-radius:8px;font-size:12px;line-height:1.5}}
</style>
</head>
<body>
<h1>Question Audit Flagged Report</h1>
<div class="stats">{html.escape(str(report.get('flagged_count', 0)))} flagged / {html.escape(str(report.get('total_count', 0)))} reviewed</div>
<div class="summary">{summary_lines}</div>
{''.join(cards)}
</body>
</html>
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item.get("check_type"))].append(item)
    for check_type, check_results in grouped.items():
        statuses = Counter(str(item.get("status")) for item in check_results)
        summary[check_type] = {
            "total": len(check_results),
            "passed": sum(1 for item in check_results if item.get("passed")),
            "flagged": sum(1 for item in check_results if not item.get("passed")),
            "errors": statuses.get("error", 0),
        }
    return summary


def run_audit(
    *,
    questions_by_check: dict[str, list[dict[str, Any]]],
    call_model,
    workers: int,
) -> list[dict[str, Any]]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for check_type, questions in questions_by_check.items():
        for question in questions:
            jobs.append((check_type, question))

    if not jobs:
        return []

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_to_job = {
            pool.submit(review_question, question, check_type=check_type, call_model=call_model): (check_type, question)
            for check_type, question in jobs
        }
        for future in as_completed(future_to_job):
            check_type, question = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _normalize_check_result(
                    question=question,
                    check_type=check_type,
                    raw_response="",
                    parsed_response=None,
                    parse_error=None,
                    prompt_error=f"Worker failed: {exc}",
                )
            results.append(result)

    results.sort(key=lambda item: (str(item.get("check_type")), int(item.get("viewer_index") or 0)))
    return results


def resolve_checks(raw_checks: list[str]) -> list[str]:
    flattened: list[str] = []
    for value in raw_checks:
        for piece in str(value).split(","):
            normalized = piece.strip()
            if normalized:
                flattened.append(normalized)

    if not flattened or "all" in flattened:
        return list(ALL_CHECKS)

    seen: set[str] = set()
    resolved: list[str] = []
    for check_type in flattened:
        canonical = CHECK_ALIASES.get(check_type)
        if canonical is None or canonical == "all":
            raise ValueError(f"Unsupported check type: {check_type}")
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark question quality with Claude / Anthropic vision models")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML_PATH, help="Input viewer HTML path")
    parser.add_argument("--benchmark_json", type=Path, default=DEFAULT_BENCHMARK_JSON_PATH, help="Structured benchmark JSON path")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for audit artifacts")
    parser.add_argument(
        "--check",
        action="append",
        default=None,
        help=(
            "Audit check to run. Repeatable or comma-separated. "
            "Supported values: all, identifiability, occlusion, attachment_chain"
        ),
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent Anthropic requests")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-check limit after filtering")
    parser.add_argument("--ambiguity_threshold", type=float, default=None, help="Optional minimum ambiguity_score for object_identifiability")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Anthropic model name")
    parser.add_argument("--api_key_env", type=str, default="ANTHROPIC_API_KEY", help="Environment variable containing the Anthropic API key")
    parser.add_argument("--base_url", type=str, default=None, help="Optional Anthropic-compatible base URL")
    parser.add_argument("--max_tokens", type=int, default=768, help="Max output tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--include_deleted", action="store_true", help="Include deleted cards from the HTML viewer")
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    checks = resolve_checks(args.check or ["all"])
    cards = load_viewer_cards(args.html.resolve(), include_deleted=args.include_deleted)
    benchmark_questions = load_benchmark_questions(args.benchmark_json.resolve())
    merged_questions = merge_questions(cards=cards, benchmark_questions=benchmark_questions)

    questions_by_check: dict[str, list[dict[str, Any]]] = {}
    for check_type in checks:
        selected = select_questions_for_check(
            merged_questions,
            check_type=check_type,
            ambiguity_threshold=args.ambiguity_threshold,
            limit=args.limit,
        )
        questions_by_check[check_type] = selected
        logger.info("Selected %d questions for %s", len(selected), check_type)

    total_selected = sum(len(items) for items in questions_by_check.values())
    if total_selected == 0:
        raise RuntimeError("No questions selected for audit. Check filters and inputs.")

    api_key = _resolve_api_key(args.api_key_env)
    _, call_model = make_anthropic_caller(
        api_key=api_key,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    results = run_audit(
        questions_by_check=questions_by_check,
        call_model=call_model,
        workers=args.workers,
    )

    summary = summarize_results(results)
    flagged_results = [item for item in results if not item.get("passed")]
    generated_at = datetime.now(timezone.utc).isoformat()

    full_report = {
        "generated_at": generated_at,
        "model": args.model,
        "checks": checks,
        "html": str(args.html.resolve()),
        "benchmark_json": str(args.benchmark_json.resolve()),
        "total_count": len(results),
        "flagged_count": len(flagged_results),
        "summary": summary,
        "results": results,
    }
    flagged_report = {
        **{key: value for key, value in full_report.items() if key != "results"},
        "results": flagged_results,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_json_path = output_dir / "full_results.json"
    flagged_json_path = output_dir / "flagged_questions.json"
    report_html_path = output_dir / "report.html"
    write_json(full_json_path, full_report)
    write_json(flagged_json_path, flagged_report)
    report_html_path.write_text(build_flagged_html(flagged_report), encoding="utf-8")

    logger.info("Wrote %s", full_json_path)
    logger.info("Wrote %s", flagged_json_path)
    logger.info("Wrote %s", report_html_path)


if __name__ == "__main__":
    main()
