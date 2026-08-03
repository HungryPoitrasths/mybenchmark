#!/usr/bin/env python3
"""Generate independent teacher CoTs and retain only strictly correct responses."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import threading
import time
from io import BytesIO
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.evaluation import parse_strict_answer
from src.cot.facts import build_fact_record, question_uid
from src.cot.images import resolve_image_paths

try:
    from PIL import Image
except ImportError:  # pragma: no cover - resizing is optional
    Image = None


SYSTEM_PROMPT = (
    "You are a careful vision-language assistant. Independently solve the "
    "multiple-choice spatial reasoning problem from the supplied images."
)
REASONING_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
REPETITION_TOKEN_RE = re.compile(r"[a-z0-9]+")
ClientFactory = Callable[[], Any]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_questions(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    payload = load_json(path)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not all(isinstance(row, dict) for row in questions):
        raise ValueError(f"{path}: expected a list or an object containing questions")
    return payload, [dict(row) for row in questions]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid cache JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"invalid cache record at {path}:{line_number}")
            records.append(record)
    return records


def append_cache(handle: Any, records: list[dict[str, Any]]) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def question_prompt(question: dict[str, Any], image_count: int) -> str:
    options = question.get("options") or []
    option_text = "\n".join(
        f"{chr(ord('A') + index)}. {str(value).strip()}"
        for index, value in enumerate(options)
    )
    selection = (
        "One or more choices may be correct; list every correct letter."
        if bool(question.get("multi_select"))
        else "Exactly one choice is correct."
    )
    return (
        f"You are given {image_count} image(s), in the order supplied.\n"
        f"{str(question.get('question') or '').strip()}\n"
        f"Options:\n{option_text}\n\n"
        f"{selection}\n"
        "Reason independently from the visual evidence. Do not guess from dataset "
        "metadata. Write 2-5 brief reasoning sentences totaling 15-180 words, then "
        "write exactly one final line in this format:\n"
        "Answer: <letter(s)>\n"
        "Separate multiple letters with single spaces. The Answer line must be last."
    )


def encode_image(path: Path, max_px: int = 0) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if max_px > 0 and Image is not None:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_px, max_px))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode("ascii"), mime


def build_request_messages(
    prompt: str, encoded_images: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }
        for encoded, mime in encoded_images
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _get_field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = _get_field(part, "text") or _get_field(part, "content")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def extract_response_text(response: Any, *, include_reasoning: bool = True) -> str:
    if isinstance(response, str):
        return response.strip()
    choices = _get_field(response, "choices") or []
    parts: list[str] = []
    fields = ("reasoning_content", "content", "text") if include_reasoning else ("content", "text")
    for choice in choices:
        for container_name in ("delta", "message"):
            container = _get_field(choice, container_name)
            if container is None:
                continue
            for field in fields:
                text = _content_text(_get_field(container, field))
                if text:
                    parts.append(text)
        text = _content_text(_get_field(choice, "text"))
        if text:
            parts.append(text)
    if parts:
        return "".join(parts).strip()
    return _content_text(_get_field(response, "content")).strip()


def _is_reasoning_chat_model(model: str) -> bool:
    return str(model).lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _should_omit_temperature(model: str) -> bool:
    name = str(model).lower()
    return _is_reasoning_chat_model(model) or name.startswith(
        ("claude-opus-4", "claude-sonnet-4", "claude-4")
    )


def _supports_qwen_thinking_control(model: str) -> bool:
    normalized = str(model).strip().lower().replace("_", "-")
    return "qwen3.5" in normalized or "qwen3-5" in normalized


def _require_response_text(text: str, *, provider: str, model: str) -> str:
    text = text.strip()
    if not text:
        raise RuntimeError(
            f"{provider} returned an empty response for model {model!r}; check the "
            "base URL, model name, provider protocol, and streaming compatibility"
        )
    return text


def _responses_output_text(response: Any) -> str:
    output_text = _get_field(response, "output_text")
    if output_text:
        return str(output_text).strip()
    parts: list[str] = []
    for item in _get_field(response, "output") or []:
        for content in _get_field(item, "content") or []:
            text = _get_field(content, "text")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def call_model(
    client: Any,
    *,
    api_provider: str,
    model: str,
    encoded_images: list[tuple[str, str]],
    prompt: str,
    max_tokens: int,
    temperature: float,
    on_stream_start: Callable[[str], None] | None = None,
) -> str:
    omit_temperature = _should_omit_temperature(model)
    if api_provider == "openai_responses":
        user_content: list[dict[str, Any]] = [
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
            }
            for encoded, mime in encoded_images
        ]
        user_content.append({"type": "input_text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {"role": "user", "content": user_content},
            ],
            "max_output_tokens": max_tokens,
            "store": False,
            "stream": True,
        }
        if not omit_temperature:
            kwargs["temperature"] = temperature
        response_stream = client.responses.create(**kwargs)
        deltas: list[str] = []
        done_text = ""
        completed_response: Any = None
        stream_error = ""
        stream_started = False
        try:
            for event in response_stream:
                event_type = str(_get_field(event, "type") or "")
                if not stream_started:
                    stream_started = True
                    if on_stream_start is not None:
                        on_stream_start(event_type or "unknown")
                if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                    delta = _get_field(event, "delta")
                    if delta:
                        deltas.append(str(delta))
                elif event_type in {"response.output_text.done", "response.refusal.done"}:
                    text = _get_field(event, "text")
                    if text:
                        done_text = str(text)
                elif event_type == "response.completed":
                    completed_response = _get_field(event, "response")
                    break
                elif event_type in {"error", "response.failed", "response.incomplete"}:
                    error = _get_field(event, "error")
                    response = _get_field(event, "response")
                    stream_error = str(error or _get_field(response, "error") or event_type)
                    break
        finally:
            close = getattr(response_stream, "close", None)
            if callable(close):
                close()
        if stream_error:
            raise RuntimeError(f"Responses stream failed: {stream_error}")
        text = "".join(deltas).strip()
        if not text:
            text = done_text.strip()
        if not text and completed_response is not None:
            text = _responses_output_text(completed_response)
        return _require_response_text(text, provider=api_provider, model=model)

    if api_provider == "anthropic":
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": encoded,
                },
            }
            for encoded, mime in encoded_images
        ]
        user_content.append({"type": "text", "text": prompt})
        kwargs = {
            "model": model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
        }
        if not omit_temperature:
            kwargs["temperature"] = temperature
        response = client.messages.create(**kwargs)
        parts = [
            str(_get_field(block, "text"))
            for block in (_get_field(response, "content") or [])
            if _get_field(block, "type") == "text" and _get_field(block, "text")
        ]
        return _require_response_text("\n".join(parts), provider=api_provider, model=model)

    kwargs = {
        "model": model,
        "messages": build_request_messages(prompt, encoded_images),
        "stream": True,
    }
    if _is_reasoning_chat_model(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if not omit_temperature:
        kwargs["temperature"] = temperature
    if _supports_qwen_thinking_control(model):
        kwargs["extra_body"] = {"enable_thinking": True}
    response = client.chat.completions.create(**kwargs)
    if _get_field(response, "choices") is not None:
        return _require_response_text(
            extract_response_text(response), provider=api_provider, model=model
        )
    parts = [extract_response_text(chunk) for chunk in response]
    return _require_response_text("".join(parts), provider=api_provider, model=model)


def repeated_ngram(reasoning: str, *, width: int = 8, threshold: int = 3) -> bool:
    tokens = REPETITION_TOKEN_RE.findall(reasoning.lower())
    if len(tokens) < width * threshold:
        return False
    counts = Counter(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1))
    return any(count >= threshold for count in counts.values())


def expected_letters(question: dict[str, Any]) -> str:
    return " ".join(build_fact_record(question).answer_letters)


def validate_teacher_response(
    question: dict[str, Any], response: str
) -> tuple[str | None, str, dict[str, Any]]:
    options = question.get("options") or []
    multi_select = bool(question.get("multi_select"))
    parsed = parse_strict_answer(
        response,
        option_count=len(options),
        multi_select=multi_select,
    )
    expected = expected_letters(question)
    metadata = {"parsed_answer": parsed, "expected_answer": expected}
    if parsed is None:
        return None, "invalid_answer_format", metadata
    parsed_letters = parsed.split()
    expected_order = expected.split()
    correct = (
        set(parsed_letters) == set(expected_order)
        if multi_select
        else parsed_letters == expected_order
    )
    if not correct:
        return None, "wrong_answer", metadata

    lines = response.rstrip().splitlines()
    reasoning = "\n".join(lines[:-1]).strip()
    word_count = len(REASONING_WORD_RE.findall(reasoning))
    metadata["reasoning_word_count"] = word_count
    if not 15 <= word_count <= 180:
        return None, "reasoning_word_count", metadata
    if repeated_ngram(reasoning):
        return None, "repeated_8gram", metadata

    normalized = "\n".join([*lines[:-1], f"Answer: {expected}"]).strip()
    return normalized, "accepted", metadata


def selection_key(seed: int, uid: str) -> str:
    return hashlib.sha256(f"{seed}|{uid}".encode("utf-8")).hexdigest()


def select_questions(
    questions: list[dict[str, Any]],
    *,
    seed: int,
    limit: int | None,
    limit_per_type: int | None,
    requested_uids: list[str],
) -> list[dict[str, Any]]:
    if requested_uids:
        requested = set(requested_uids)
        questions = [row for row in questions if question_uid(row) in requested]
        found = {question_uid(row) for row in questions}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown --question-uid values: {missing[:5]}")
    ordered = sorted(questions, key=lambda row: selection_key(seed, question_uid(row)))
    if limit_per_type is not None:
        counts: Counter[str] = Counter()
        filtered: list[dict[str, Any]] = []
        for row in ordered:
            question_type = str(row.get("type") or "")
            if counts[question_type] >= limit_per_type:
                continue
            counts[question_type] += 1
            filtered.append(row)
        ordered = filtered
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


class ThreadLocalClientFactory:
    def __init__(
        self,
        *,
        api_provider: str,
        api_key: str,
        base_url: str,
        timeout: float,
        credential_kind: str = "api_key",
    ):
        self.api_provider = api_provider
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.credential_kind = credential_kind
        self.local = threading.local()

    def __call__(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is None:
            if self.api_provider == "anthropic":
                from anthropic import Anthropic

                credential = (
                    {"auth_token": self.api_key}
                    if self.credential_kind == "auth_token"
                    else {"api_key": self.api_key}
                )
                client = Anthropic(
                    **credential,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    max_retries=0,
                )
            else:
                from openai import OpenAI

                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    max_retries=0,
                )
            self.local.client = client
        return client


def resolve_api_credential(args: argparse.Namespace) -> tuple[str, str]:
    direct = str(getattr(args, "api_key", None) or "").strip()
    if direct:
        return direct, "api_key"
    env_name = str(getattr(args, "api_key_env", None) or "").strip()
    if env_name:
        value = os.getenv(env_name, "").strip()
        if value:
            kind = (
                "auth_token"
                if args.api_provider == "anthropic" and env_name.upper().endswith("AUTH_TOKEN")
                else "api_key"
            )
            return value, kind
    if args.api_provider == "anthropic":
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
        if auth_token:
            return auth_token, "auth_token"
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if anthropic_key:
            return anthropic_key, "api_key"
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            return openai_key, "api_key"
    return os.getenv("DASHSCOPE_API_KEY", "").strip() or "EMPTY", "api_key"


def input_fingerprint(
    *,
    uid: str,
    prompt: str,
    images: list[Path],
    model: str,
    base_url: str,
    api_provider: str,
    temperature: float,
    max_tokens: int,
    image_max_px: int,
) -> str:
    material = json.dumps(
        {
            "uid": uid,
            "system_prompt": SYSTEM_PROMPT,
            "prompt": prompt,
            "images": [str(path.resolve()) for path in images],
            "model": model,
            "base_url": base_url,
            "api_provider": api_provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "image_max_px": image_max_px,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def process_question(
    question: dict[str, Any],
    *,
    benchmark_path: Path,
    args: argparse.Namespace,
    client_factory: ClientFactory,
    cached: list[dict[str, Any]],
) -> tuple[str, str | None, list[dict[str, Any]], dict[str, Any]]:
    uid = question_uid(question)
    try:
        images, _ = resolve_image_paths(
            question,
            benchmark_path=benchmark_path,
            scannet_roots=[path.resolve() for path in args.scannet_image_root],
            scannetpp_roots=[path.resolve() for path in args.scannetpp_image_root],
            scannetpp_sensor=args.scannetpp_sensor,
            require_exists=True,
        )
        image_paths = [Path(path) for path in images]
        prompt = question_prompt(question, len(image_paths))
        image_max_px = int(getattr(args, "api_image_max_px", 0) or 0)
        encoded_images = [encode_image(path, image_max_px) for path in image_paths]
        fingerprint = input_fingerprint(
            uid=uid,
            prompt=prompt,
            images=image_paths,
            model=args.model,
            base_url=args.base_url,
            api_provider=getattr(args, "api_provider", "openai_chat"),
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
            image_max_px=image_max_px,
        )
        expected_letters(question)
    except Exception as exc:
        record = {
            "question_uid": uid,
            "kind": "terminal",
            "status": "preparation_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return uid, None, [record], {"status": "preparation_error"}

    relevant = [row for row in cached if row.get("input_fingerprint") == fingerprint]
    terminal = next((row for row in reversed(relevant) if row.get("kind") == "terminal"), None)
    if terminal is not None:
        response = str(terminal.get("teacher_cot") or "").strip() or None
        return uid, response, [], {
            "status": str(terminal.get("status") or "cached"),
            "semantic_attempts": terminal.get("semantic_attempts"),
        }

    successful_attempt = next(
        (
            row
            for row in reversed(relevant)
            if row.get("kind") == "semantic_attempt" and row.get("accepted")
        ),
        None,
    )
    if successful_attempt is not None:
        normalized = str(successful_attempt.get("normalized_response") or "").strip()
        terminal_record = {
            "question_uid": uid,
            "input_fingerprint": fingerprint,
            "kind": "terminal",
            "status": "accepted",
            "teacher_cot": normalized,
            "semantic_attempts": int(successful_attempt["semantic_attempt"]),
        }
        return uid, normalized, [terminal_record], {"status": "accepted"}

    completed_attempts = [
        int(row.get("semantic_attempt") or 0)
        for row in relevant
        if row.get("kind") == "semantic_attempt"
    ]
    start_attempt = max(completed_attempts, default=0) + 1
    new_records: list[dict[str, Any]] = []
    for semantic_attempt in range(start_attempt, args.max_attempts + 1):
        response_text: str | None = None
        transport_errors: list[str] = []
        for transport_attempt in range(1, args.transport_retries + 1):
            try:
                request_started = time.monotonic()
                encoded_mib = sum(len(encoded) for encoded, _ in encoded_images) / (1024 * 1024)
                print(
                    f"teacher API request: question_uid={uid} "
                    f"semantic_attempt={semantic_attempt}/{args.max_attempts} "
                    f"transport_attempt={transport_attempt}/{args.transport_retries} "
                    f"provider={getattr(args, 'api_provider', 'openai_chat')} "
                    f"images={len(encoded_images)} encoded_mib={encoded_mib:.2f}",
                    flush=True,
                )

                def report_stream_start(event_type: str) -> None:
                    elapsed = time.monotonic() - request_started
                    print(
                        f"teacher API stream started: question_uid={uid} "
                        f"event={event_type} elapsed_seconds={elapsed:.1f}",
                        flush=True,
                    )

                response_text = call_model(
                    client_factory(),
                    api_provider=getattr(args, "api_provider", "openai_chat"),
                    model=args.model,
                    encoded_images=encoded_images,
                    prompt=prompt,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    on_stream_start=report_stream_start,
                )
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                transport_errors.append(error)
                print(
                    f"teacher API failed: question_uid={uid} "
                    f"semantic_attempt={semantic_attempt}/{args.max_attempts} "
                    f"transport_attempt={transport_attempt}/{args.transport_retries} "
                    f"error={' '.join(error.splitlines())}",
                    flush=True,
                )
                new_records.append(
                    {
                        "question_uid": uid,
                        "input_fingerprint": fingerprint,
                        "kind": "transport_error",
                        "semantic_attempt": semantic_attempt,
                        "transport_attempt": transport_attempt,
                        "error": error,
                    }
                )
                if transport_attempt < args.transport_retries and args.retry_delay > 0:
                    time.sleep(args.retry_delay * transport_attempt)
        if response_text is None:
            return uid, None, new_records, {
                "status": "transport_failed",
                "transport_errors": transport_errors,
            }

        normalized, reason, validation = validate_teacher_response(question, response_text)
        print(
            f"teacher API success: question_uid={uid} "
            f"semantic_attempt={semantic_attempt}/{args.max_attempts} "
            f"transport_attempt={transport_attempt}/{args.transport_retries} "
            f"response_chars={len(response_text)} "
            f"validation={'accepted' if normalized is not None else 'rejected'} "
            f"reason={reason}",
            flush=True,
        )
        attempt_record = {
            "question_uid": uid,
            "input_fingerprint": fingerprint,
            "kind": "semantic_attempt",
            "semantic_attempt": semantic_attempt,
            "accepted": normalized is not None,
            "reason": reason,
            "response": response_text,
            "normalized_response": normalized,
            "validation": validation,
        }
        new_records.append(attempt_record)
        if normalized is not None:
            terminal_record = {
                "question_uid": uid,
                "input_fingerprint": fingerprint,
                "kind": "terminal",
                "status": "accepted",
                "teacher_cot": normalized,
                "semantic_attempts": semantic_attempt,
            }
            new_records.append(terminal_record)
            return uid, normalized, new_records, {
                "status": "accepted",
                "semantic_attempts": semantic_attempt,
            }

    terminal_record = {
        "question_uid": uid,
        "input_fingerprint": fingerprint,
        "kind": "terminal",
        "status": "rejected",
        "semantic_attempts": args.max_attempts,
    }
    new_records.append(terminal_record)
    return uid, None, new_records, {
        "status": "rejected",
        "semantic_attempts": args.max_attempts,
    }


def run_filter(
    args: argparse.Namespace,
    *,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any]:
    source_payload, all_questions = load_questions(args.benchmark)
    selected = select_questions(
        all_questions,
        seed=args.seed,
        limit=args.limit,
        limit_per_type=args.limit_per_type,
        requested_uids=args.question_uid,
    )
    if client_factory is None:
        api_key, credential_kind = resolve_api_credential(args)
        client_factory = ThreadLocalClientFactory(
            api_provider=args.api_provider,
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            credential_kind=credential_kind,
        )

    cached_records = load_cache(args.cache_jsonl)
    cache_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in cached_records:
        cache_by_uid[str(record.get("question_uid") or "")].append(record)

    args.cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[str | None, dict[str, Any]]] = {}
    with args.cache_jsonl.open("a", encoding="utf-8", newline="\n") as cache_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_question,
                    question,
                    benchmark_path=args.benchmark,
                    args=args,
                    client_factory=client_factory,
                    cached=cache_by_uid.get(question_uid(question), []),
                ): question
                for question in selected
            }
            completed = 0
            for future in as_completed(futures):
                uid, teacher_cot, new_records, metadata = future.result()
                append_cache(cache_handle, new_records)
                results[uid] = (teacher_cot, metadata)
                completed += 1
                if completed == 1 or completed % args.progress_every == 0 or completed == len(selected):
                    accepted_so_far = sum(value[0] is not None for value in results.values())
                    print(
                        f"teacher filter: {completed}/{len(selected)} complete, "
                        f"{accepted_so_far} accepted",
                        flush=True,
                    )

    accepted_questions: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    attempts: Counter[int] = Counter()
    selected_uids = {question_uid(question) for question in selected}
    for question in selected:
        uid = question_uid(question)
        teacher_cot, metadata = results.get(uid, (None, {"status": "missing_result"}))
        status = str(metadata.get("status") or "unknown")
        statuses[status] += 1
        if teacher_cot is not None and metadata.get("semantic_attempts") is not None:
            attempts[int(metadata["semantic_attempts"])] += 1
        if teacher_cot is None:
            continue
        output_question = dict(question)
        output_question["question_uid"] = uid
        output_question["teacher_cot"] = teacher_cot
        output_question["teacher_cot_metadata"] = {
            "model": args.model,
            "api_provider": args.api_provider,
            "status": status,
            "semantic_attempts": metadata.get("semantic_attempts"),
            "independent_from_gold": True,
        }
        accepted_questions.append(output_question)

    by_type = Counter(str(row.get("type") or "") for row in accepted_questions)
    statistics = {
        "input_count": len(all_questions),
        "selected_count": len(selected_uids),
        "accepted_count": len(accepted_questions),
        "rejected_count": len(selected_uids) - len(accepted_questions),
        "acceptance_rate": (
            len(accepted_questions) / len(selected_uids) if selected_uids else 0.0
        ),
        "statuses": dict(sorted(statuses.items())),
        "accepted_by_type": dict(sorted(by_type.items())),
        "accepted_by_semantic_attempt": {
            str(key): value for key, value in sorted(attempts.items())
        },
    }
    source_metadata = (
        {key: value for key, value in source_payload.items() if key != "questions"}
        if isinstance(source_payload, dict)
        else {}
    )
    output = {
        "schema_version": "predictive-spatial-teacher-cot-filter-v1",
        "name": "Correct independently solved teacher CoT questions",
        "source": {
            "benchmark": str(args.benchmark.resolve()),
            "source_metadata": source_metadata,
        },
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "api_provider": args.api_provider,
            "max_semantic_attempts": args.max_attempts,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "gold_or_oracle_in_request": False,
        },
        "statistics": statistics,
        "questions": accepted_questions,
    }
    write_json_atomic(args.output, output)
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", "--output-json", "--output_json", dest="output", type=Path, required=True)
    parser.add_argument("--cache-jsonl", "--cache_jsonl", dest="cache_jsonl", type=Path)
    parser.add_argument(
        "--base-url",
        "--base_url",
        "--vlm-url",
        "--vlm_url",
        dest="base_url",
        default="https://www.packyapi.com/v1",
    )
    parser.add_argument("--model", "--vlm-model", "--vlm_model", dest="model", required=True)
    parser.add_argument(
        "--api-provider",
        "--api_provider",
        dest="api_provider",
        choices=("openai_chat", "openai_responses", "anthropic"),
        default="openai_chat",
        help="Wire protocol used by the VLM endpoint.",
    )
    parser.add_argument("--api-key", "--api_key", dest="api_key")
    parser.add_argument("--api-key-env", "--api_key_env", dest="api_key_env")
    parser.add_argument("--max-attempts", "--max_attempts", dest="max_attempts", type=int, default=2)
    parser.add_argument(
        "--transport-retries",
        "--transport_retries",
        "--retries",
        dest="transport_retries",
        type=int,
        default=3,
    )
    parser.add_argument("--retry-delay", "--retry_delay", dest="retry_delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", "--vlm-workers", "--vlm_workers", dest="workers", type=int, default=4)
    parser.add_argument(
        "--max-output-tokens",
        "--max-tokens",
        "--max_tokens",
        dest="max_output_tokens",
        type=int,
        default=384,
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--api-image-max-px",
        "--api_image_max_px",
        dest="api_image_max_px",
        type=int,
        default=1280,
        help="Resize the longest image side before API calls; 0 disables resizing.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-type", "--limit_per_type", dest="limit_per_type", type=int)
    parser.add_argument("--question-uid", "--question_uid", dest="question_uid", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", "--progress_every", dest="progress_every", type=int, default=25)
    parser.add_argument(
        "--scannet-image-root",
        "--scannet_image_root",
        dest="scannet_image_root",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--scannetpp-image-root",
        "--scannetpp_image_root",
        dest="scannetpp_image_root",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--scannetpp-sensor",
        "--scannetpp_sensor",
        dest="scannetpp_sensor",
        choices=("iphone", "dslr"),
        default="iphone",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cache_jsonl is None:
        args.cache_jsonl = args.output.with_suffix(".cache.jsonl")
    positive = {
        "max_attempts": args.max_attempts,
        "transport_retries": args.transport_retries,
        "workers": args.workers,
        "max_output_tokens": args.max_output_tokens,
        "progress_every": args.progress_every,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_attempts > 2:
        raise ValueError("--max-attempts cannot exceed 2 semantic attempts")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.limit_per_type is not None and args.limit_per_type <= 0:
        raise ValueError("--limit-per-type must be positive")
    if args.api_image_max_px < 0:
        raise ValueError("--api-image-max-px cannot be negative")
    statistics = run_filter(args)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
