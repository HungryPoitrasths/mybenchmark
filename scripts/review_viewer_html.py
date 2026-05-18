#!/usr/bin/env python3
"""Review a self-contained benchmark viewer HTML with a VLM.

Parses each viewer card, extracts the embedded image, question text, options,
and hidden gold answer (for evaluation only), then asks a VLM to answer
without revealing the gold option.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("viewer_review")

DEFAULT_VLM_URL = "http://183.129.178.195:60029/v1"
VLM_API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
PLACEHOLDER_VLM_API_KEY = "EMPTY"
QUESTION_REVIEW_MAX_RETRIES = 4
QUESTION_REVIEW_RETRY_DELAY_SECONDS = 2.0

DIV_TOKEN_RE = re.compile(r"<div\b[^>]*>|</div>", flags=re.IGNORECASE)
CARD_START_RE = re.compile(
    r'<div\b(?=[^>]*\bclass=(["\'])[^"\']*\bcard\b[^"\']*\1)[^>]*>',
    flags=re.IGNORECASE,
)
DATA_DELETED_RE = re.compile(
    r'\bdata-deleted\s*=\s*(["\'])true\1',
    flags=re.IGNORECASE,
)
IMAGE_RE = re.compile(
    r'<div class="img-wrap"><img src="data:(?P<mime>[^;]+);base64,(?P<b64>[^"]+)"',
    flags=re.IGNORECASE,
)
QTEXT_RE = re.compile(r'<p class="qtext">(?P<text>.*?)</p>', flags=re.IGNORECASE | re.DOTALL)
OPTION_RE = re.compile(
    r'<div class="opt(?P<correct>\s+correct)?">(?P<letter>[A-D])\.&nbsp;(?P<text>.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
FOOTER_RE = re.compile(
    r'<div class="footer">\s*(?P<scene_id>[^<]+?)\s*&nbsp;/&nbsp;\s*(?P<image_name>[^<]+?)\s*</div>',
    flags=re.IGNORECASE,
)
IDX_RE = re.compile(r'<span class="idx">#(?P<idx>\d+)</span>', flags=re.IGNORECASE)
META_RE = re.compile(r'<div class="meta">(?P<meta>.*?)</div>', flags=re.IGNORECASE | re.DOTALL)
BADGE_RE = re.compile(r"<span class=\"badge(?: [^\"]*)?\"[^>]*>(?P<text>.*?)</span>", flags=re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = (
    "You are a visual spatial-reasoning assistant. "
    "Answer multiple-choice questions about spatial relationships in images."
)


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


def _iter_card_div_ranges(text: str) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    search_from = 0
    while True:
        start_match = CARD_START_RE.search(text, search_from)
        if start_match is None:
            return ranges

        start = start_match.start()
        depth = 0
        for match in DIV_TOKEN_RE.finditer(text, start):
            token = match.group(0).lower()
            if token.startswith("<div"):
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    ranges.append((start, match.end(), start_match.group(0)))
                    search_from = match.end()
                    break
        else:
            raise ValueError(f"Unbalanced div structure after card marker: {start}")


def _clean_html_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


def _resolve_vlm_api_key() -> str:
    for env_name in VLM_API_KEY_ENV_NAMES:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key
    logger.warning(
        "Using placeholder API key %r because neither %s nor %s is set",
        PLACEHOLDER_VLM_API_KEY,
        VLM_API_KEY_ENV_NAMES[0],
        VLM_API_KEY_ENV_NAMES[1],
    )
    return PLACEHOLDER_VLM_API_KEY


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "concurrent_request_limit_exceeded" in text
        or "too many concurrent requests" in text
        or "rate limit" in text
    )


def _is_authentication_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "401" in text
        or "unauthorized" in text
        or "authentication" in text
        or "invalid api key" in text
    )


def _call_with_retries(create_fn, *, context: str):
    last_exc: Exception | None = None
    for attempt in range(1, QUESTION_REVIEW_MAX_RETRIES + 1):
        try:
            return create_fn()
        except Exception as exc:
            last_exc = exc
            if _is_authentication_error(exc):
                raise RuntimeError(
                    f"{context} failed with an authentication error: {exc}"
                ) from exc
            if not _is_retryable_error(exc) or attempt >= QUESTION_REVIEW_MAX_RETRIES:
                raise
            delay_seconds = QUESTION_REVIEW_RETRY_DELAY_SECONDS * attempt
            logger.warning(
                "%s hit a transient limit (%d/%d). Retrying in %.1fs: %s",
                context,
                attempt,
                QUESTION_REVIEW_MAX_RETRIES,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)
    if last_exc is None:
        raise RuntimeError(f"{context} failed without raising an error")
    raise last_exc


def _parse_mcq_answer(raw: str) -> str | None:
    if not raw:
        return None
    stripped = raw.strip()
    if not stripped:
        return None

    upper = stripped.upper()
    if re.fullmatch(r"[ABCD]", upper):
        return upper

    match = re.match(r"^[\(\[]?([ABCD])(?:[\)\].:\s-]*)?$", upper)
    if match:
        return match.group(1)

    explicit_patterns = [
        r"\bANSWER(?:\s+IS)?\s*[:\-]?\s*[\(\[]?([ABCD])(?:[\)\]]|\b)",
        r"\bOPTION\s+([ABCD])\b",
        r"\bI\s+CHOOSE\s+([ABCD])\b",
        r"\bMY\s+ANSWER\s+IS\s+([ABCD])\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    standalone_letters = re.findall(r"\b([ABCD])\b", upper)
    unique_letters: list[str] = []
    for letter in standalone_letters:
        if letter not in unique_letters:
            unique_letters.append(letter)
    return unique_letters[0] if len(unique_letters) == 1 else None


def _option_for_answer(options: list[dict[str, str]], answer: str | None) -> str | None:
    if answer not in {"A", "B", "C", "D"}:
        return None
    for option in options:
        if option["letter"] == answer:
            return option["text"]
    return None


def parse_viewer_html(
    html_text: str,
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for start, end, card_start_html in _iter_card_div_ranges(html_text):
        deleted = DATA_DELETED_RE.search(card_start_html) is not None
        if deleted and not include_deleted:
            continue

        card_html = html_text[start:end]

        image_match = IMAGE_RE.search(card_html)
        qtext_match = QTEXT_RE.search(card_html)
        footer_match = FOOTER_RE.search(card_html)
        idx_match = IDX_RE.search(card_html)
        meta_match = META_RE.search(card_html)
        option_matches = list(OPTION_RE.finditer(card_html))

        if image_match is None or qtext_match is None or footer_match is None or not option_matches:
            continue

        options: list[dict[str, str]] = []
        gold_answer = None
        for match in option_matches:
            letter = str(match.group("letter")).upper()
            text = _clean_html_text(str(match.group("text")))
            options.append({"letter": letter, "text": text})
            if match.group("correct"):
                gold_answer = letter

        badges: list[str] = []
        if meta_match is not None:
            for badge_match in BADGE_RE.finditer(meta_match.group("meta")):
                badge_text = _clean_html_text(badge_match.group("text"))
                if badge_text and not badge_text.startswith("#"):
                    badges.append(badge_text)

        scene_id = _clean_html_text(footer_match.group("scene_id"))
        image_name = _clean_html_text(footer_match.group("image_name"))
        question_text = _clean_html_text(qtext_match.group("text"))

        cards.append(
            {
                "viewer_index": int(idx_match.group("idx")) if idx_match is not None else len(cards) + 1,
                "scene_id": scene_id,
                "image_name": image_name,
                "question": question_text,
                "options": options,
                "gold_answer": gold_answer,
                "gold_option": _option_for_answer(options, gold_answer),
                "image_data_url": f"data:{image_match.group('mime')};base64,{image_match.group('b64')}",
                "mime": image_match.group("mime"),
                "badges": badges,
                "deleted": deleted,
            }
        )

    cards.sort(key=lambda item: int(item["viewer_index"]))
    return cards


def build_prompt(question_text: str, options: list[dict[str, str]]) -> str:
    parts = [question_text, ""]
    for option in options:
        parts.append(f"{option['letter']}) {option['text']}")
    parts.append("")
    parts.append("Answer with a single letter only (A, B, C, or D). Do not explain.")
    return "\n".join(parts)


def make_openai_local_caller(
    base_url: str,
    model_name: str,
    *,
    api_key: str,
):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(image_data_url: str, prompt: str, *, context: str) -> str:
        resp = _call_with_retries(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=32,
                temperature=0,
            ),
            context=context,
        )
        return str(resp.choices[0].message.content or "").strip()

    return client, call


def resolve_model_name(base_url: str, model_name: str | None, *, api_key: str) -> str:
    if model_name:
        return model_name
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    models = client.models.list()
    available = [str(model.id) for model in models.data]
    if not available:
        raise RuntimeError(f"No models available at {base_url}")
    return available[0]


def review_question(
    question: dict[str, Any],
    *,
    call_model,
) -> dict[str, Any]:
    prompt = build_prompt(question["question"], question["options"])
    raw_response = call_model(
        question["image_data_url"],
        prompt,
        context=f"viewer question #{question['viewer_index']}",
    )
    predicted_answer = _parse_mcq_answer(raw_response)
    predicted_option = _option_for_answer(question["options"], predicted_answer)
    gold_answer = question.get("gold_answer")
    gold_option = question.get("gold_option")
    is_correct = (
        predicted_answer in {"A", "B", "C", "D"}
        and gold_answer in {"A", "B", "C", "D"}
        and predicted_answer == gold_answer
    )
    return {
        **question,
        "predicted_answer": predicted_answer,
        "predicted_option": predicted_option,
        "raw_response": raw_response,
        "is_correct": is_correct,
        "flagged": not is_correct,
    }


def _render_options(options: list[dict[str, str]], gold_answer: str | None, predicted_answer: str | None) -> str:
    blocks: list[str] = []
    for option in options:
        letter = option["letter"]
        classes = ["opt"]
        if letter == gold_answer:
            classes.append("gold")
        if letter == predicted_answer:
            classes.append("predicted")
        blocks.append(
            f'<div class="{" ".join(classes)}"><span class="letter">{html.escape(letter)}</span>'
            f'<span class="text">{html.escape(option["text"])}</span></div>'
        )
    return "".join(blocks)


def build_flagged_html(report: dict[str, Any]) -> str:
    flagged = report.get("questions", [])
    cards: list[str] = []
    for item in flagged:
        badges = "".join(
            f'<span class="badge">{html.escape(str(text))}</span>'
            for text in item.get("badges", [])
        )
        cards.append(
            f"""
<div class="card">
  <div class="img-wrap"><img src="{item['image_data_url']}" alt="question image"></div>
  <div class="body">
    <div class="meta">{badges}<span class="idx">#{item['viewer_index']}</span></div>
    <div class="footer-mini">{html.escape(item['scene_id'])} / {html.escape(item['image_name'])}</div>
    <p class="qtext">{html.escape(item['question'])}</p>
    <div class="opts">{_render_options(item['options'], item.get('gold_answer'), item.get('predicted_answer'))}</div>
    <div class="review">
      <div><strong>Predicted:</strong> {html.escape(str(item.get('predicted_answer') or '-'))}
      {html.escape(f"({item.get('predicted_option')})" if item.get('predicted_option') else '')}</div>
      <div><strong>Gold:</strong> {html.escape(str(item.get('gold_answer') or '-'))}
      {html.escape(f"({item.get('gold_option')})" if item.get('gold_option') else '')}</div>
      <div><strong>Raw:</strong> {html.escape(str(item.get('raw_response') or ''))}</div>
    </div>
  </div>
</div>
"""
        )

    summary = (
        f"{report.get('flagged_question_count', 0)} flagged / "
        f"{report.get('reviewed_question_count', 0)} reviewed"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Viewer QA Review Flagged</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:20px}}
h1{{margin:0 0 8px}}
.stats{{color:#555;margin-bottom:20px}}
.card{{display:flex;background:#fff;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.12);margin-bottom:18px;overflow:hidden}}
.img-wrap{{flex:0 0 auto;width:480px;background:#222;display:flex;align-items:center;justify-content:center}}
.img-wrap img{{width:480px;display:block}}
.body{{padding:18px 20px;flex:1}}
.meta{{font-size:12px;color:#666;margin-bottom:6px}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:11px;font-weight:700;margin-right:6px}}
.idx{{float:right;color:#9ca3af}}
.footer-mini{{font-size:12px;color:#666;margin-bottom:10px}}
.qtext{{font-size:15px;font-weight:600;color:#111;margin:0 0 14px}}
.opt{{padding:7px 12px;margin:4px 0;border-radius:6px;font-size:14px;background:#f8f9fa;border:1px solid #e5e7eb}}
.opt.gold{{background:#dcfce7;border-color:#86efac}}
.opt.predicted{{outline:2px solid #fb923c}}
.letter{{display:inline-block;width:24px;font-weight:700}}
.review{{margin-top:14px;padding:12px 14px;border-radius:8px;background:#fff7ed;border:1px solid #fed7aa;font-size:13px;line-height:1.6}}
</style>
</head>
<body>
<h1>Viewer QA Review Flagged</h1>
<div class="stats">{html.escape(summary)}</div>
{''.join(cards)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a self-contained benchmark viewer HTML with a VLM")
    parser.add_argument("--html", type=Path, required=True, help="Input viewer HTML path")
    parser.add_argument("--base_url", type=str, default=DEFAULT_VLM_URL, help="OpenAI-compatible VLM API base URL")
    parser.add_argument("--model_name", type=str, default=None, help="Model name; auto-detect if omitted")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency for VLM requests")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of questions to review")
    parser.add_argument("--output_json", type=Path, default=None, help="Full review JSON output")
    parser.add_argument("--flagged_json", type=Path, default=None, help="Flagged-only JSON output")
    parser.add_argument("--flagged_html", type=Path, default=None, help="Flagged-only HTML output")
    args = parser.parse_args()

    html_path = args.html.resolve()
    html_text = html_path.read_text(encoding="utf-8")
    questions = parse_viewer_html(html_text)
    if args.limit is not None and args.limit >= 0:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError(f"No questions parsed from {html_path}")

    output_json = args.output_json or html_path.with_name(f"{html_path.stem}_qa_review.json")
    flagged_json = args.flagged_json or html_path.with_name(f"{html_path.stem}_qa_review_flagged.json")
    flagged_html = args.flagged_html or html_path.with_name(f"{html_path.stem}_qa_review_flagged.html")

    api_key = _resolve_vlm_api_key()
    model_name = resolve_model_name(args.base_url, args.model_name, api_key=api_key)
    _, call_model = make_openai_local_caller(args.base_url, model_name, api_key=api_key)

    logger.info("Parsed %d questions from %s", len(questions), html_path)
    logger.info("Using VLM model: %s", model_name)

    reviewed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(review_question, question, call_model=call_model)
            for question in questions
        ]
        for future in as_completed(futures):
            reviewed.append(future.result())

    reviewed.sort(key=lambda item: int(item["viewer_index"]))
    flagged = [item for item in reviewed if bool(item.get("flagged"))]
    parse_failures = sum(1 for item in reviewed if item.get("predicted_answer") is None)
    mismatches = sum(
        1
        for item in reviewed
        if item.get("predicted_answer") is not None
        and item.get("gold_answer") in {"A", "B", "C", "D"}
        and item.get("predicted_answer") != item.get("gold_answer")
    )

    full_report = {
        "name": "Viewer QA Review",
        "source_html": str(html_path),
        "model": model_name,
        "reviewed_question_count": len(reviewed),
        "flagged_question_count": len(flagged),
        "parse_failure_count": parse_failures,
        "answer_mismatch_count": mismatches,
        "questions": reviewed,
    }
    flagged_report = {
        "name": "Viewer QA Review (flagged)",
        "source_html": str(html_path),
        "model": model_name,
        "reviewed_question_count": len(reviewed),
        "flagged_question_count": len(flagged),
        "parse_failure_count": parse_failures,
        "answer_mismatch_count": mismatches,
        "questions": flagged,
    }

    output_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    flagged_json.write_text(json.dumps(flagged_report, indent=2, ensure_ascii=False), encoding="utf-8")
    flagged_html.write_text(build_flagged_html(flagged_report), encoding="utf-8")

    logger.info(
        "Viewer QA review complete: %d reviewed, %d flagged. JSON: %s Flagged JSON: %s HTML: %s",
        len(reviewed),
        len(flagged),
        output_json,
        flagged_json,
        flagged_html,
    )


if __name__ == "__main__":
    main()
