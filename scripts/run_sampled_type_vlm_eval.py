#!/usr/bin/env python3
"""Sample benchmark questions by type, run a VLM, and build an HTML report.

The script scans benchmark.json files under one or more output roots, samples
up to N questions per question type, evaluates them through an OpenAI-compatible
vision API, and writes both a resumable JSON result file and a self-contained
HTML viewer.

Example:
    python scripts/run_sampled_type_vlm_eval.py \
        --root output/pilot \
        --root output/scannetpp_polit \
        --scannet_image_root data/scannet \
        --scannetpp_image_root output/scannetpp_iphone_frames \
        --scannetpp_sensor iphone \
        --vlm_url https://www.packyapi.com/v1 \
        --vlm_model qwen3.5-flash \
        --output_json output/type_sample_eval/results.json \
        --output_html output/type_sample_eval/viewer.html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


SYSTEM_PROMPT = (
    "You are a careful vision-language assistant solving multiple-choice "
    "spatial reasoning questions about an image."
)

PROMPT_SUFFIX = (
    "Choose exactly one option letter, then provide your full reasoning.\n"
    "Keep the whole response within the maximum output token limit.\n"
    "Return this format:\n"
    "Answer: <single letter>\n"
    "Reasoning: <full reasoning>"
)

QTYPE_ORDER = [
    "direction_agent",
    "occlusion",
    "distance",
    "direction_object_centric",
    "direction_allocentric",
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "viewpoint_move",
    "object_remove",
    "attachment_chain",
    "coordinate_rotation_agent",
    "coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric",
]

QTYPE_DISPLAY = {
    "direction_agent": "L1_direction_agent",
    "occlusion": "L1_occlusion",
    "distance": "L1_distance",
    "direction_object_centric": "L1_direction_object_centric",
    "direction_allocentric": "L1_direction_allocentric",
    "object_move_agent": "L2_object_move_agent",
    "object_move_distance": "L2_object_move_distance",
    "object_move_occlusion": "L2_object_move_occlusion",
    "object_move_object_centric": "L2_object_move_object_centric",
    "object_rotate_object_centric": "L2_object_rotate_object_centric",
    "object_move_allocentric": "L2_object_move_allocentric",
    "viewpoint_move": "L2_viewpoint_move",
    "object_remove": "L2_object_remove",
    "attachment_chain": "L3_attachment_chain",
    "coordinate_rotation_agent": "L3_coordinate_rotation_agent",
    "coordinate_rotation_object_centric": "L3_coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric": "L3_coordinate_rotation_allocentric",
}


@dataclass(frozen=True)
class ImageResolution:
    path: Path | None
    checked_paths: tuple[str, ...]


def _json_key(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _question_uid(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "dataset": question.get("_dataset"),
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "type": question.get("type"),
            "question": question.get("question"),
            "options": question.get("options"),
            "answer": question.get("answer"),
        }
    )


def _question_dedupe_key(question: dict[str, Any]) -> str:
    return _json_key(
        {
            "scene_id": question.get("scene_id"),
            "image_name": question.get("image_name"),
            "question": question.get("question"),
        }
    )


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError(f"Unsupported benchmark structure: {path}")
    return [q for q in questions if isinstance(q, dict)]


def _infer_dataset(root: Path, benchmark_path: Path) -> str:
    text = f"{root.as_posix()}/{benchmark_path.as_posix()}".lower()
    return "scannetpp" if "scannetpp" in text else "scannet"


def load_questions(roots: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_files: list[str] = []
    duplicate_count = 0

    for root in roots:
        for benchmark_path in sorted(root.rglob("benchmark.json")):
            source_files.append(str(benchmark_path))
            dataset = _infer_dataset(root, benchmark_path)
            for q in _load_benchmark(benchmark_path):
                item = dict(q)
                item["_dataset"] = dataset
                item["_source_root"] = str(root)
                item["_source_benchmark"] = str(benchmark_path)
                uid = _question_uid(item)
                item["question_uid"] = uid
                dedupe_key = _question_dedupe_key(item)
                if dedupe_key in seen:
                    duplicate_count += 1
                    continue
                seen.add(dedupe_key)
                questions.append(item)

    metadata = {
        "source_files": source_files,
        "source_file_count": len(source_files),
        "deduped_question_count": len(questions),
        "duplicate_question_count": duplicate_count,
        "dedupe_rule": "scene_id + image_name + question",
    }
    return questions, metadata


def _qtype_sort_key(qtype: str) -> tuple[int, str]:
    try:
        return (QTYPE_ORDER.index(qtype), qtype)
    except ValueError:
        return (len(QTYPE_ORDER), qtype)


def _qtype_display(qtype: str) -> str:
    return QTYPE_DISPLAY.get(qtype, qtype)


def sample_questions(
    questions: list[dict[str, Any]],
    *,
    per_type: int,
    scene_cap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_type[str(q.get("type") or "unknown")].append(q)

    sampled: list[dict[str, Any]] = []
    sampling_stats: dict[str, Any] = {}
    for qtype in sorted(by_type, key=_qtype_sort_key):
        group = list(by_type[qtype])
        rng.shuffle(group)

        chosen: list[dict[str, Any]] = []
        chosen_uids: set[str] = set()
        per_scene: Counter[str] = Counter()
        for q in group:
            if len(chosen) >= per_type:
                break
            scene_id = str(q.get("scene_id") or "unknown")
            if per_scene[scene_id] >= scene_cap:
                continue
            chosen.append(q)
            chosen_uids.add(str(q["question_uid"]))
            per_scene[scene_id] += 1

        relaxed_added = 0
        if len(chosen) < per_type:
            for q in group:
                if len(chosen) >= per_type:
                    break
                uid = str(q["question_uid"])
                if uid in chosen_uids:
                    continue
                chosen.append(q)
                chosen_uids.add(uid)
                relaxed_added += 1

        sampled.extend(chosen)
        sampling_stats[qtype] = {
            "available": len(group),
            "sampled": len(chosen),
            "relaxed_scene_cap_added": relaxed_added,
            "scene_count": len({str(q.get("scene_id") or "unknown") for q in chosen}),
        }

    return sampled, sampling_stats


def build_prompt(question: dict[str, Any]) -> str:
    parts = [str(question.get("question") or "").strip(), ""]
    options = question.get("options") or []
    for idx, option in enumerate(options):
        parts.append(f"{chr(65 + idx)}) {option}")
    parts.extend(["", PROMPT_SUFFIX])
    return "\n".join(parts)


def allowed_letters(question: dict[str, Any]) -> str:
    n_options = len(question.get("options") or [])
    if n_options <= 0:
        return "ABCD"
    return "".join(chr(65 + idx) for idx in range(min(n_options, 26)))


def parse_answer(raw: str | None, letters: str) -> str | None:
    if not raw:
        return None
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()
    if re.fullmatch(f"[{allowed}]", upper):
        return upper

    patterns = [
        rf"(?:FINAL\s+)?ANSWER\s*[:：]\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"(?:CHOICE|OPTION)\s*[:：]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^[\(\[]?\s*([{allowed}])\s*[\)\].:：-]",
    ]
    for pattern in patterns:
        m = re.search(pattern, upper)
        if m:
            return m.group(1)

    m = re.search(rf"\b([{allowed}])\b", upper)
    return m.group(1) if m else None


def _mime_for_path(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def _encode_image(path: Path, max_px: int | None = None) -> tuple[str, str]:
    if max_px and max_px > 0 and Image is not None:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_px, max_px))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"

    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii"), _mime_for_path(path)


def resolve_image(
    question: dict[str, Any],
    *,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
) -> ImageResolution:
    dataset = str(question.get("_dataset") or "scannet")
    scene = str(question.get("scene_id") or "")
    image_name = str(question.get("image_name") or "")
    roots = scannetpp_roots if dataset == "scannetpp" else scannet_roots

    candidates: list[Path] = []
    for root in roots:
        if dataset == "scannetpp":
            if scannetpp_sensor == "iphone":
                candidates.append(root / scene / image_name)
            elif scannetpp_sensor == "dslr":
                candidates.append(root / scene / "dslr" / "resized_images" / image_name)
            else:
                raise ValueError(
                    f"scannetpp_sensor must be 'iphone' or 'dslr', got {scannetpp_sensor!r}"
                )
            candidates.extend(
                [
                    root / scene / image_name,
                    root / scene / "dslr" / "resized_images" / image_name,
                    root / scene / "iphone" / "rgb" / image_name,
                ]
            )
        else:
            candidates.extend(
                [
                    root / scene / "color" / image_name,
                    root / scene / image_name,
                ]
            )

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return ImageResolution(candidate, tuple(checked))
    return ImageResolution(None, tuple(checked))


def make_client(api_provider: str, base_url: str, api_key: str, timeout: float, use_curl: bool = False):
    if use_curl:
        return CurlResponsesClient(base_url=base_url, api_key=api_key, timeout=timeout)

    if api_provider == "anthropic":
        from anthropic import Anthropic

        return Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class CurlResponsesClient:
    """Send Responses-API requests through the system ``curl`` binary.

    Some proxies sit behind a TLS layer that blocks Python's TLS stacks
    (stdlib ssl / httpx / curl_cffi all fail the handshake) while the system
    ``curl`` binary connects fine. This client shells out to ``curl`` to reach
    the ``/responses`` endpoint, posting the JSON body via ``--data-binary``.
    """

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def responses_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/responses"
        body = json.dumps(payload, ensure_ascii=False)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(body)
                tmp_path = tmp.name
            cmd = [
                "curl",
                "-s",
                "-S",
                "--fail-with-body",
                "--max-time",
                str(int(self.timeout)),
                url,
                "-H",
                f"Authorization: Bearer {self.api_key}",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                f"@{tmp_path}",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 30,
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"curl exited {proc.returncode}: {detail[:500]}")
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError("curl returned an empty body")
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"non-JSON response: {out[:500]}") from exc


class ThreadLocalOpenAIClientFactory:
    def __init__(self, *, api_provider: str, base_url: str, api_key: str, timeout: float, use_curl: bool = False) -> None:
        self.api_provider = api_provider
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.use_curl = use_curl
        self.local = threading.local()

    def get_client(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is None:
            client = make_client(
                self.api_provider, self.base_url, self.api_key, self.timeout, self.use_curl
            )
            self.local.client = client
        return client


def _extract_responses_text(data: dict[str, Any]) -> str:
    """Extract assistant text from a Responses-API JSON payload."""
    err = data.get("error")
    if err:
        raise RuntimeError(f"api_error: {err}")
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    chunks: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def call_model(
    client: Any,
    *,
    api_provider: str,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_image_max_px: int,
    send_temperature: bool = True,
) -> str:
    b64, mime = _encode_image(image_path, api_image_max_px)
    data_url = f"data:{mime};base64,{b64}"
    if api_provider == "openai_responses":
        input_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": prompt},
                ],
            },
        ]
        if isinstance(client, CurlResponsesClient):
            payload: dict[str, Any] = {
                "model": model,
                "input": input_messages,
                "max_output_tokens": max_tokens,
            }
            if send_temperature:
                payload["temperature"] = temperature
            data = client.responses_create(payload)
            return _extract_responses_text(data)

        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if send_temperature:
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    if api_provider == "anthropic":
        response = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        chunks = [
            str(getattr(block, "text", ""))
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        return "\n".join(chunks).strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("question_uid")): row
        for row in rows
        if isinstance(row, dict) and row.get("question_uid")
    }


def save_results(
    path: Path,
    *,
    metadata: dict[str, Any],
    sampling_stats: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "sampling_stats": sampling_stats,
        "summary": summarize_results(results),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def result_from_question(
    question: dict[str, Any],
    *,
    image_resolution: ImageResolution,
    raw_response: str | None,
    error: str | None,
) -> dict[str, Any]:
    letters = allowed_letters(question)
    prediction = parse_answer(raw_response, letters)
    gt_answer = str(question.get("answer") or "").strip().upper()
    correct = bool(prediction and gt_answer and prediction == gt_answer)
    return {
        "question_uid": question.get("question_uid"),
        "dataset": question.get("_dataset"),
        "source_benchmark": question.get("_source_benchmark"),
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "image_path": str(image_resolution.path) if image_resolution.path else None,
        "checked_image_paths": list(image_resolution.checked_paths),
        "level": question.get("level"),
        "type": question.get("type"),
        "question": question.get("question"),
        "options": question.get("options"),
        "gt_answer": gt_answer,
        "correct_value": question.get("correct_value"),
        "prediction": prediction,
        "raw_response": raw_response,
        "correct": correct,
        "error": error,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for row in results:
        qtype = str(row.get("type") or "unknown")
        stats = by_type.setdefault(
            qtype,
            {"total": 0, "answered": 0, "correct": 0, "errors": 0, "missing_images": 0},
        )
        stats["total"] += 1
        if row.get("prediction"):
            stats["answered"] += 1
        if row.get("correct"):
            stats["correct"] += 1
        if row.get("error"):
            stats["errors"] += 1
        if row.get("error") == "image_not_found":
            stats["missing_images"] += 1

    for stats in by_type.values():
        total = int(stats["total"])
        answered = int(stats["answered"])
        stats["accuracy"] = (float(stats["correct"]) / total) if total else None
        stats["answered_accuracy"] = (
            float(stats["correct"]) / answered if answered else None
        )

    ordered = {
        qtype: by_type[qtype]
        for qtype in sorted(by_type, key=_qtype_sort_key)
    }
    return {"by_type": ordered}


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _option_html(row: dict[str, Any]) -> str:
    gt = str(row.get("gt_answer") or "")
    pred = str(row.get("prediction") or "")
    chunks: list[str] = []
    for idx, option in enumerate(row.get("options") or []):
        letter = chr(65 + idx)
        classes = ["option"]
        if letter == gt:
            classes.append("gold")
        if letter == pred and pred != gt:
            classes.append("predicted")
        chunks.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="letter">{html.escape(letter)}</span>'
            f'<span>{html.escape(str(option))}</span>'
            "</div>"
        )
    return "\n".join(chunks)


def _image_html(row: dict[str, Any], html_image_max_px: int) -> str:
    image_path_text = row.get("image_path")
    if not image_path_text:
        checked = row.get("checked_image_paths") or []
        first_checked = checked[0] if checked else ""
        return (
            '<div class="missing-image">'
            "image not found"
            f"<small>{html.escape(str(first_checked))}</small>"
            "</div>"
        )
    path = Path(str(image_path_text))
    if not path.exists():
        return (
            '<div class="missing-image">'
            "image path no longer exists"
            f"<small>{html.escape(str(path))}</small>"
            "</div>"
        )
    b64, mime = _encode_image(path, html_image_max_px)
    return f'<img src="data:{mime};base64,{b64}" alt="">'


def build_html(results: list[dict[str, Any]], *, title: str, html_image_max_px: int) -> str:
    summary = summarize_results(results)["by_type"]
    summary_rows = []
    for qtype, stats in summary.items():
        summary_rows.append(
            "<tr>"
            f'<td class="qtype-cell">{html.escape(_qtype_display(qtype))}</td>'
            f"<td>{stats['correct']} / {stats['total']}</td>"
            f"<td>{_fmt_pct(stats['accuracy'])}</td>"
            f"<td>{stats['answered']}</td>"
            f"<td>{stats['missing_images']}</td>"
            f"<td>{stats['errors']}</td>"
            "</tr>"
        )

    cards = []
    for idx, row in enumerate(sorted(results, key=lambda r: (_qtype_sort_key(str(r.get("type") or "")), str(r.get("scene_id") or ""), str(r.get("image_name") or ""))), 1):
        status = "correct" if row.get("correct") else "wrong"
        if row.get("error"):
            status = "error"
        pred = row.get("prediction") or "-"
        raw = row.get("raw_response") or row.get("error") or ""
        cards.append(
            f'<article class="card {status}">'
            '<div class="image-wrap">'
            f'{_image_html(row, html_image_max_px)}'
            "</div>"
            '<div class="content">'
            '<div class="meta">'
            f'<span>#{idx}</span>'
            f'<span class="qtype">{html.escape(_qtype_display(str(row.get("type") or "unknown")))}</span>'
            f'<span>{html.escape(str(row.get("dataset") or ""))}</span>'
            f'<span>{html.escape(str(row.get("scene_id") or ""))} / {html.escape(str(row.get("image_name") or ""))}</span>'
            f'<span class="pill">{html.escape(status)}</span>'
            "</div>"
            f'<h2>{html.escape(str(row.get("question") or ""))}</h2>'
            f'<div class="options">{_option_html(row)}</div>'
            '<div class="answer-line">'
            f'<strong>GT:</strong> {html.escape(str(row.get("gt_answer") or "-"))}'
            f'<strong>Model:</strong> {html.escape(str(pred))}'
            f'<strong>Correct value:</strong> {html.escape(str(row.get("correct_value") or "-"))}'
            "</div>"
            "<details open>"
            "<summary>Model reasoning and raw answer</summary>"
            f"<pre>{html.escape(str(raw))}</pre>"
            "</details>"
            "</div>"
            "</article>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f4f1ea;
  --ink: #1c2520;
  --muted: #5f6a62;
  --line: #d7d0c3;
  --paper: #fffdf8;
  --good: #146c43;
  --bad: #a9271b;
  --warn: #9a6700;
  --accent: #245b63;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background:
    linear-gradient(135deg, rgba(36,91,99,0.10), transparent 34rem),
    linear-gradient(315deg, rgba(184,125,42,0.10), transparent 30rem),
    var(--bg);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}}
header {{
  padding: 32px clamp(16px, 4vw, 48px) 18px;
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0 0 16px;
  font-size: clamp(28px, 4vw, 48px);
  font-weight: 700;
  letter-spacing: 0;
}}
.summary-table {{
  width: 100%;
  border-collapse: collapse;
  background: rgba(255,253,248,0.72);
}}
.summary-table th,
.summary-table td {{
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  font-size: 14px;
}}
.summary-table th {{
  color: var(--muted);
  font-family: "Trebuchet MS", sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}}
.summary-table .qtype-cell {{
  color: var(--accent);
  font-weight: 800;
  font-family: "Trebuchet MS", sans-serif;
}}
main {{
  padding: 24px clamp(16px, 4vw, 48px) 48px;
}}
.card {{
  display: grid;
  grid-template-columns: minmax(260px, 38%) 1fr;
  gap: 22px;
  margin: 0 0 18px;
  padding: 16px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-left: 6px solid var(--line);
  box-shadow: 0 12px 28px rgba(40, 34, 23, 0.08);
}}
.card.correct {{ border-left-color: var(--good); }}
.card.wrong {{ border-left-color: var(--bad); }}
.card.error {{ border-left-color: var(--warn); }}
.image-wrap {{
  min-height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e8e2d6;
  border: 1px solid var(--line);
  overflow: hidden;
}}
.image-wrap img {{
  width: 100%;
  height: auto;
  display: block;
}}
.missing-image {{
  padding: 18px;
  color: var(--muted);
  font-family: "Trebuchet MS", sans-serif;
  text-align: center;
}}
.missing-image small {{
  display: block;
  margin-top: 8px;
  overflow-wrap: anywhere;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 10px;
  font-family: "Trebuchet MS", sans-serif;
  color: var(--muted);
  font-size: 13px;
}}
.meta span {{
  border: 1px solid var(--line);
  padding: 3px 7px;
  background: rgba(255,255,255,0.52);
}}
.meta .qtype {{
  color: #ffffff;
  background: var(--accent);
  border-color: var(--accent);
  font-weight: 800;
}}
.pill {{
  color: var(--ink);
  font-weight: 700;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 19px;
  line-height: 1.35;
}}
.options {{
  display: grid;
  gap: 7px;
  margin-bottom: 12px;
}}
.option {{
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  background: #fbf7ef;
}}
.option.gold {{
  border-color: rgba(20,108,67,0.55);
  background: rgba(20,108,67,0.08);
}}
.option.predicted {{
  border-color: rgba(169,39,27,0.55);
  background: rgba(169,39,27,0.08);
}}
.letter {{
  font-family: "Trebuchet MS", sans-serif;
  font-weight: 700;
}}
.answer-line {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 8px 0 12px;
  font-family: "Trebuchet MS", sans-serif;
}}
details {{
  border-top: 1px solid var(--line);
  padding-top: 10px;
}}
summary {{
  cursor: pointer;
  color: var(--accent);
  font-family: "Trebuchet MS", sans-serif;
  font-weight: 700;
}}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 10px 0 0;
  padding: 12px;
  background: #1f2924;
  color: #f2efe8;
  font-family: Consolas, "Courier New", monospace;
  font-size: 13px;
  line-height: 1.45;
}}
@media (max-width: 820px) {{
  .card {{ grid-template-columns: 1fr; }}
  .summary-table {{ display: block; overflow-x: auto; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <table class="summary-table">
    <thead><tr><th>Type</th><th>Correct</th><th>Accuracy</th><th>Answered</th><th>Missing images</th><th>Errors</th></tr></thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
</header>
<main>
  {''.join(cards)}
</main>
</body>
</html>
"""


def _result_dedupe_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("scene_id"), row.get("image_name"), row.get("question"))


def dedupe_results_by_frame_question(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[Any, Any, Any]] = set()
    filtered: list[dict[str, Any]] = []
    dropped = 0
    for row in results:
        key = _result_dedupe_key(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        filtered.append(row)
    return filtered, dropped


def _summary_rows_html(summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for qtype, stats in summary.items():
        rows.append(
            "<tr>"
            f'<td class="qtype-cell">{html.escape(_qtype_display(qtype))}</td>'
            f"<td>{stats['correct']} / {stats['total']}</td>"
            f"<td>{_fmt_pct(stats['accuracy'])}</td>"
            f"<td>{stats['answered']}</td>"
            f"<td>{stats['missing_images']}</td>"
            f"<td>{stats['errors']}</td>"
            "</tr>"
        )
    return "".join(rows)


def _ensure_postprocess_css(text: str) -> str:
    if ".summary-table .qtype-cell" not in text:
        text = text.replace(
            '.summary-table th {\n'
            '  color: var(--muted);\n'
            '  font-family: "Trebuchet MS", sans-serif;\n'
            '  text-transform: uppercase;\n'
            '  letter-spacing: 0.03em;\n'
            '}\n',
            '.summary-table th {\n'
            '  color: var(--muted);\n'
            '  font-family: "Trebuchet MS", sans-serif;\n'
            '  text-transform: uppercase;\n'
            '  letter-spacing: 0.03em;\n'
            '}\n'
            '.summary-table .qtype-cell {\n'
            '  color: var(--accent);\n'
            '  font-weight: 800;\n'
            '  font-family: "Trebuchet MS", sans-serif;\n'
            '}\n',
        )
    if ".meta .qtype" not in text:
        text = text.replace(
            ".meta span {\n"
            "  border: 1px solid var(--line);\n"
            "  padding: 3px 7px;\n"
            "  background: rgba(255,255,255,0.52);\n"
            "}\n",
            ".meta span {\n"
            "  border: 1px solid var(--line);\n"
            "  padding: 3px 7px;\n"
            "  background: rgba(255,255,255,0.52);\n"
            "}\n"
            ".meta .qtype {\n"
            "  color: #ffffff;\n"
            "  background: var(--accent);\n"
            "  border-color: var(--accent);\n"
            "  font-weight: 800;\n"
            "}\n",
        )
    return text


def postprocess_existing_html(
    *,
    html_path: Path,
    json_path: Path,
) -> dict[str, int]:
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported result JSON structure: {json_path}")

    filtered, json_dropped = dedupe_results_by_frame_question(rows)
    summary = summarize_results(filtered)["by_type"]

    if isinstance(payload, dict):
        payload["results"] = filtered
        payload["summary"] = {"by_type": summary}
        metadata = payload.setdefault("metadata", {})
        metadata["result_dedupe_rule"] = "scene_id + image_name + question"
        metadata["result_dedupe_dropped_count"] = json_dropped
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        json_path.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    text = html_path.read_text(encoding="utf-8")
    article_re = re.compile(r'<article class="card ([^"]+)">.*?</article>', re.S)
    meta_re = re.compile(
        r'<div class="meta"><span>#(\d+)</span>'
        r'<span(?: class="qtype")?>([^<]+)</span>'
        r"<span>[^<]*</span>"
        r"<span>([^<]+) / ([^<]+)</span>"
        r'<span class="pill">[^<]+</span></div>'
    )
    h2_re = re.compile(r"<h2>(.*?)</h2>", re.S)

    raw_by_display = {display: raw for raw, display in QTYPE_DISPLAY.items()}
    seen_html: set[tuple[str, str, str]] = set()
    kept_articles: list[str] = []
    html_dropped = 0
    for article_match in article_re.finditer(text):
        article = article_match.group(0)
        meta_match = meta_re.search(article)
        h2_match = h2_re.search(article)
        if not meta_match or not h2_match:
            kept_articles.append(article)
            continue

        qtype_text = html.unescape(meta_match.group(2))
        raw_type = raw_by_display.get(qtype_text, qtype_text)
        scene_id = html.unescape(meta_match.group(3))
        image_name = html.unescape(meta_match.group(4))
        question = html.unescape(re.sub(r"<.*?>", "", h2_match.group(1)))
        dedupe_key = (scene_id, image_name, question)
        if dedupe_key in seen_html:
            html_dropped += 1
            continue
        seen_html.add(dedupe_key)

        display_type = _qtype_display(raw_type)
        next_idx = len(kept_articles) + 1
        article = re.sub(
            r'(<div class="meta"><span>#)\d+(</span>)'
            r'<span(?: class="qtype")?>[^<]+</span>',
            (
                rf"\g<1>{next_idx}\g<2>"
                f'<span class="qtype">{html.escape(display_type)}</span>'
            ),
            article,
            count=1,
        )
        kept_articles.append(article)

    first_article = article_re.search(text)
    last_article: re.Match[str] | None = None
    for last_article in article_re.finditer(text):
        pass
    if first_article and last_article:
        text = (
            text[: first_article.start()]
            + "\n".join(kept_articles)
            + text[last_article.end() :]
        )

    text = re.sub(
        r"<tbody>\s*.*?\s*</tbody>",
        f"<tbody>\n      {_summary_rows_html(summary)}\n    </tbody>",
        text,
        count=1,
        flags=re.S,
    )
    text = _ensure_postprocess_css(text)
    html_path.write_text(text, encoding="utf-8")

    return {
        "json_kept": len(filtered),
        "json_dropped": json_dropped,
        "html_kept": len(kept_articles),
        "html_dropped": html_dropped,
    }


def run_api_question(
    *,
    args: argparse.Namespace,
    client_factory: ThreadLocalOpenAIClientFactory,
    idx: int,
    total: int,
    question: dict[str, Any],
    resolution: ImageResolution,
) -> dict[str, Any]:
    raw_response: str | None = None
    error: str | None = None
    prompt = build_prompt(question)
    print(
        f"[{idx}/{total}] {question.get('type')} "
        f"{question.get('scene_id')}/{question.get('image_name')} -> API",
        flush=True,
    )
    for attempt in range(args.retries + 1):
        try:
            raw_response = call_model(
                client_factory.get_client(),
                api_provider=args.api_provider,
                model=args.model,
                image_path=resolution.path,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                api_image_max_px=args.api_image_max_px,
                send_temperature=not args.no_temperature,
            )
            print(f"[{idx}/{total}] done", flush=True)
            break
        except Exception as exc:  # pragma: no cover - network/API dependent
            if attempt >= args.retries:
                error = f"api_error: {exc}"
                print(f"[{idx}/{total}] failed: {exc}", flush=True)
            else:
                wait = args.retry_delay * (2 ** attempt)
                print(
                    f"[{idx}/{total}] attempt {attempt + 1} failed: {exc}; "
                    f"retrying in {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)

    if args.delay > 0:
        time.sleep(args.delay)

    return result_from_question(
        question,
        image_resolution=resolution,
        raw_response=raw_response,
        error=error,
    )


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    roots = [Path(root) for root in args.root]
    all_questions, metadata = load_questions(roots)
    selected, sampling_stats = sample_questions(
        all_questions,
        per_type=args.per_type,
        scene_cap=args.scene_cap,
        seed=args.seed,
    )

    metadata.update(
        {
            "input_question_count": len(all_questions),
            "sampled_question_count": len(selected),
            "per_type": args.per_type,
            "scene_cap": args.scene_cap,
            "seed": args.seed,
            "model": args.model,
            "base_url": args.base_url,
            "api_provider": args.api_provider,
            "scannetpp_sensor": args.scannetpp_sensor,
            "vlm_workers": args.vlm_workers,
        }
    )

    output_json = Path(args.output_json)
    existing = load_existing_results(output_json)
    results_by_uid: dict[str, dict[str, Any]] = {}

    client_factory: ThreadLocalOpenAIClientFactory | None = None
    if not args.skip_api:
        default_api_key_env = "ANTHROPIC_AUTH_TOKEN" if args.api_provider == "anthropic" else "OPENAI_API_KEY"
        api_key = (
            args.api_key
            or (os.getenv(args.api_key_env) if args.api_key_env else None)
            or os.getenv(default_api_key_env)
            or os.getenv("DASHSCOPE_API_KEY")
            or "EMPTY"
        )
        client_factory = ThreadLocalOpenAIClientFactory(
            api_provider=args.api_provider,
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.timeout,
            use_curl=args.use_curl,
        )

    api_call_count = 0
    api_work: list[tuple[int, dict[str, Any], ImageResolution]] = []
    for idx, question in enumerate(selected, 1):
        uid = str(question["question_uid"])
        cached = existing.get(uid)
        if (
            cached
            and not args.force
            and cached.get("raw_response") is not None
            and cached.get("prediction") is not None
        ):
            results_by_uid[uid] = cached
            continue

        resolution = resolve_image(
            question,
            scannet_roots=[Path(p) for p in args.scannet_image_root],
            scannetpp_roots=[Path(p) for p in args.scannetpp_image_root],
            scannetpp_sensor=args.scannetpp_sensor,
        )

        raw_response: str | None = None
        error: str | None = None
        if resolution.path is None:
            error = "image_not_found"
        elif args.skip_api:
            error = "api_skipped"
        else:
            api_work.append((idx, question, resolution))
            continue

        results_by_uid[uid] = result_from_question(
            question,
            image_resolution=resolution,
            raw_response=raw_response,
            error=error,
        )

        if idx % args.checkpoint_every == 0:
            ordered = [results_by_uid[str(q["question_uid"])] for q in selected if str(q["question_uid"]) in results_by_uid]
            save_results(
                output_json,
                metadata=metadata,
                sampling_stats=sampling_stats,
                results=ordered,
            )
            print(f"checkpoint: {len(ordered)}/{len(selected)} results saved")

    if api_work:
        if client_factory is None:
            raise RuntimeError("client_factory was not initialized")
        workers = max(1, int(args.vlm_workers))
        print(f"running {len(api_work)} VLM request(s) with --vlm_workers {workers}", flush=True)

        def _store_result(row: dict[str, Any]) -> None:
            nonlocal api_call_count
            results_by_uid[str(row["question_uid"])] = row
            api_call_count += 1
            if api_call_count % args.checkpoint_every == 0:
                ordered_rows = [
                    results_by_uid[str(q["question_uid"])]
                    for q in selected
                    if str(q["question_uid"]) in results_by_uid
                ]
                save_results(
                    output_json,
                    metadata=metadata,
                    sampling_stats=sampling_stats,
                    results=ordered_rows,
                )
                print(f"checkpoint: {len(ordered_rows)}/{len(selected)} results saved")

        if workers == 1:
            for idx, question, resolution in api_work:
                _store_result(
                    run_api_question(
                        args=args,
                        client_factory=client_factory,
                        idx=idx,
                        total=len(selected),
                        question=question,
                        resolution=resolution,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_api_question,
                        args=args,
                        client_factory=client_factory,
                        idx=idx,
                        total=len(selected),
                        question=question,
                        resolution=resolution,
                    )
                    for idx, question, resolution in api_work
                ]
                for future in as_completed(futures):
                    _store_result(future.result())

    results = [results_by_uid[str(q["question_uid"])] for q in selected]
    metadata["api_calls_made"] = api_call_count
    return results, metadata, sampling_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample questions by type, run a VLM, and build an HTML report."
    )
    parser.add_argument("--root", action="append", default=None, help="Output root to scan for benchmark.json files")
    parser.add_argument("--per_type", type=int, default=100, help="Questions sampled per type")
    parser.add_argument("--scene_cap", type=int, default=3, help="Max questions per scene within each type before relaxation")
    parser.add_argument("--seed", type=int, default=20260529, help="Random seed for sampling")
    parser.add_argument("--scannet_image_root", action="append", default=None, help="ScanNet image root; can be repeated")
    parser.add_argument("--scannetpp_image_root", action="append", default=None, help="ScanNet++ image root; can be repeated")
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone", help="ScanNet++ image layout, matching scripts/make_viewer.py")
    parser.add_argument("--base_url", "--vlm_url", dest="base_url", default="https://www.packyapi.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", "--vlm_model", dest="model", default="qwen3.5-flash", help="Model name")
    parser.add_argument(
        "--api_provider",
        choices=("openai_chat", "openai_responses", "anthropic"),
        default="openai_chat",
        help="Wire protocol to use for image+text VLM calls",
    )
    parser.add_argument("--api_key", default=None, help="API key; otherwise read from --api_key_env or provider defaults")
    parser.add_argument("--api_key_env", default=None, help="Environment variable for API key")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum model output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api_image_max_px", type=int, default=1280, help="Resize longest image side for API; 0 disables")
    parser.add_argument("--html_image_max_px", type=int, default=720, help="Resize longest image side embedded in HTML; 0 disables")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request API timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per API call")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Initial retry delay in seconds")
    parser.add_argument("--vlm_workers", type=int, default=1, help="Maximum number of concurrent VLM requests")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save JSON every N processed questions")
    parser.add_argument("--output_json", default="output/type_sample_vlm_eval/results.json", help="Resumable JSON result path")
    parser.add_argument("--output_html", default="output/type_sample_vlm_eval/viewer.html", help="HTML report path")
    parser.add_argument("--title", default="Sampled VLM Spatial QA Evaluation", help="HTML report title")
    parser.add_argument("--skip_api", action="store_true", help="Only sample and build a report skeleton; do not call the API")
    parser.add_argument("--force", action="store_true", help="Re-run questions even if cached in output_json")
    parser.add_argument(
        "--use_curl",
        action="store_true",
        help="Send openai_responses requests via the system curl binary (bypasses Python TLS stacks that some proxies block)",
    )
    parser.add_argument(
        "--no_temperature",
        action="store_true",
        help="Do not send the temperature parameter (use for reasoning models that reject non-default temperature)",
    )
    parser.add_argument(
        "--postprocess_existing_html",
        action="store_true",
        help="Deduplicate and relabel an existing embedded HTML report without rebuilding images",
    )
    args = parser.parse_args(argv)

    if args.root is None:
        args.root = ["output/pilot", "output/scannetpp_polit"]
    if args.scannet_image_root is None:
        args.scannet_image_root = ["data/scannet"]
    if args.scannetpp_image_root is None:
        args.scannetpp_image_root = ["output/scannetpp_iphone_frames", "++data"]
    if args.per_type <= 0:
        parser.error("--per_type must be positive")
    if args.scene_cap <= 0:
        parser.error("--scene_cap must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint_every must be positive")
    if args.vlm_workers <= 0:
        parser.error("--vlm_workers must be positive")
    if args.use_curl and args.api_provider != "openai_responses":
        parser.error("--use_curl currently supports only --api_provider openai_responses")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.postprocess_existing_html:
        stats = postprocess_existing_html(
            html_path=Path(args.output_html),
            json_path=Path(args.output_json),
        )
        print(f"json kept       : {stats['json_kept']}")
        print(f"json dropped    : {stats['json_dropped']}")
        print(f"html kept       : {stats['html_kept']}")
        print(f"html dropped    : {stats['html_dropped']}")
        print(f"json output     : {args.output_json}")
        print(f"html output     : {args.output_html}")
        return

    results, metadata, sampling_stats = evaluate(args)

    output_json = Path(args.output_json)
    save_results(
        output_json,
        metadata=metadata,
        sampling_stats=sampling_stats,
        results=results,
    )

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(results, title=args.title, html_image_max_px=args.html_image_max_px),
        encoding="utf-8",
    )

    summary = summarize_results(results)["by_type"]
    print(f"loaded questions : {metadata['input_question_count']}")
    print(f"sampled questions: {len(results)}")
    print(f"json output      : {output_json}")
    print(f"html output      : {output_html}")
    for qtype, stats in summary.items():
        print(
            f"{qtype:36s} {stats['correct']:4d}/{stats['total']:<4d} "
            f"acc={_fmt_pct(stats['accuracy']):>6s} answered={stats['answered']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
