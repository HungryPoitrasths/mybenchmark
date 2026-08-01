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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.evaluation import parse_strict_answer
from src.cot.facts import build_fact_record, question_uid
from src.cot.images import resolve_image_paths


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


def encode_image(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_request_messages(prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": encode_image(path)}}
        for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def extract_response_text(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
    else:
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(getattr(part, "text", None), str):
                parts.append(part.text)
        return "".join(parts).strip()
    return ""


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
    def __init__(self, *, api_key: str, base_url: str, timeout: float):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.local = threading.local()

    def __call__(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self.local.client = client
        return client


def input_fingerprint(
    *,
    uid: str,
    prompt: str,
    images: list[Path],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> str:
    material = json.dumps(
        {
            "uid": uid,
            "system_prompt": SYSTEM_PROMPT,
            "prompt": prompt,
            "images": [str(path.resolve()) for path in images],
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
        fingerprint = input_fingerprint(
            uid=uid,
            prompt=prompt,
            images=image_paths,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
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
    messages: list[dict[str, Any]] | None = None
    for semantic_attempt in range(start_attempt, args.max_attempts + 1):
        if messages is None:
            messages = build_request_messages(prompt, image_paths)
        response_text: str | None = None
        transport_errors: list[str] = []
        for transport_attempt in range(1, args.transport_retries + 1):
            try:
                response = client_factory().chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    stream=False,
                )
                response_text = extract_response_text(response)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                transport_errors.append(error)
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
        api_key = os.getenv(args.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"environment variable {args.api_key_env} is empty")
        client_factory = ThreadLocalClientFactory(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-jsonl", type=Path)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-type", type=int)
    parser.add_argument("--question-uid", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--scannet-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
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
    statistics = run_filter(args)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
