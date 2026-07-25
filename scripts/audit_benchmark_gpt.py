#!/usr/bin/env python3
"""Audit benchmark questions with OpenAI vision models.

The primary model reviews every question. Questions with any failed,
uncertain, malformed, or errored primary check are independently reviewed by
the secondary model. The secondary decision is authoritative when present.

All images needed by a multi-view question are sent in their benchmark order:
the first main frame, bridge/auxiliary frames, and the final reasoning frame.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_viewer import _collect_aux_image_names, _resolve_image_path

logger = logging.getLogger("gpt_benchmark_audit")

AUDIT_SCHEMA_VERSION = "gpt-benchmark-audit-v1"
PROMPT_VERSION = "2026-07-24-v1"
DEFAULT_PRIMARY_MODEL = "gpt-4.1-mini"
DEFAULT_REVIEW_MODEL = "gpt-5.2"
DEFAULT_MAX_OUTPUT_TOKENS = 2400
DEFAULT_MAX_IMAGE_EDGE = 2048
DEFAULT_MAX_WORKERS = 4
DEFAULT_FAILED_API_RETRIES = 1
DEFAULT_FAILED_API_RETRY_DELAY = 5.0
DEFAULT_REQUEST_INTERVAL = 0.0
MAX_API_ATTEMPTS = 4
DEFAULT_API_KEY_ENV_NAMES = ("OPENAI_API_KEY", "API_KEY")

CHECK_REFERABILITY = "referability"
CHECK_OCCLUSION = "occlusion_visibility"
CHECK_ATTACHMENT = "attachment_pair"
CHECK_CONTINUITY = "continuity"
CHECK_FAIRNESS = "fairness"
CHECK_NAMES = (
    CHECK_REFERABILITY,
    CHECK_OCCLUSION,
    CHECK_ATTACHMENT,
    CHECK_CONTINUITY,
    CHECK_FAIRNESS,
)
VERDICTS = {"pass", "fail", "uncertain", "not_applicable"}

ISSUE_CODES = (
    "object_not_visible",
    "ambiguous_instance",
    "wrong_or_unclear_label",
    "insufficient_object_evidence",
    "gt_visibility_mismatch",
    "uncertain_visibility",
    "no_physical_support",
    "mere_proximity",
    "attachment_pair_not_visible",
    "scene_mismatch",
    "large_viewpoint_jump",
    "wrong_image_order",
    "duplicate_or_reversed_view",
    "insufficient_endpoint_evidence",
    "unequal_object_evidence",
    "answer_leakage",
    "poor_image_quality",
    "missing_input",
    "model_output_invalid",
    "api_error",
    "other",
)

_PAIR_RE = re.compile(r"^\s*(-?\d+)\s*->\s*(-?\d+)\s*$")
_CLIENT_LOCAL = threading.local()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_api_key(preferred_env: str | None = None) -> tuple[str, str] | None:
    """Return the first configured API key and its environment variable name."""
    env_names = (preferred_env,) if preferred_env else DEFAULT_API_KEY_ENV_NAMES
    for env_name in env_names:
        value = str(os.getenv(env_name, "")).strip()
        if value:
            return value, env_name
    return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_benchmark(path: Path) -> tuple[dict[str, Any] | list[Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        questions = payload["questions"]
    elif isinstance(payload, list):
        questions = payload
    else:
        raise RuntimeError(f"Unsupported benchmark structure in {path}")
    if not all(isinstance(item, dict) for item in questions):
        raise RuntimeError(f"Every benchmark question must be a JSON object: {path}")
    return payload, [dict(item) for item in questions]


def ordered_image_names(question: dict[str, Any]) -> list[str]:
    first = str(question.get("image_name") or "").strip()
    names = [first] if first else []
    for name in _collect_aux_image_names(question):
        normalized = str(name).strip()
        if normalized and normalized not in names:
            names.append(normalized)
    return names


def image_roles(question: dict[str, Any], names: list[str]) -> list[str]:
    if not names:
        return []
    final_name = str(question.get("reasoning_frame_2") or "").strip()
    roles: list[str] = []
    for index, name in enumerate(names):
        if index == 0:
            roles.append("first_main_view")
        elif final_name and name == final_name:
            roles.append("last_main_view")
        else:
            roles.append("bridge_view")
    return roles


def question_fingerprint(question: dict[str, Any], source_index: int) -> str:
    identity = {
        "source_index": source_index,
        "scene_id": question.get("scene_id"),
        "image_names": ordered_image_names(question),
        "question": question.get("question"),
        "options": question.get("options", []),
        "answer": question.get("answer"),
        "correct_value": question.get("correct_value"),
        "attachment_pair_id": question.get("attachment_pair_id"),
        "attachment_parent_id": question.get("attachment_parent_id"),
        "attachment_child_id": question.get("attachment_child_id"),
    }
    return sha256_json(identity)


def is_occlusion_question(question: dict[str, Any]) -> bool:
    return question.get("level") == "L1" and question.get("type") == "occlusion"


def is_attachment_question(question: dict[str, Any]) -> bool:
    return bool(
        question.get("attachment_remapped") is True
        or question.get("attachment_pair_id")
        or question.get("attachment_parent_id") is not None
        or question.get("attachment_child_id") is not None
        or question.get("type") in {"attachment_chain", "attachment_move"}
    )


def applicable_checks(question: dict[str, Any], image_names: list[str]) -> list[str]:
    checks = [CHECK_REFERABILITY]
    if is_occlusion_question(question):
        checks.append(CHECK_OCCLUSION)
    if is_attachment_question(question):
        checks.append(CHECK_ATTACHMENT)
    if len(image_names) > 1:
        checks.extend((CHECK_CONTINUITY, CHECK_FAIRNESS))
    return checks


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class SceneMetadataResolver:
    def __init__(self, root: Path | None) -> None:
        self.root = root
        self._cache: dict[str, dict[int, str] | None] = {}
        self._lock = threading.Lock()

    def object_labels(self, scene_id: str) -> dict[int, str] | None:
        with self._lock:
            if scene_id in self._cache:
                return self._cache[scene_id]
        if self.root is None:
            with self._lock:
                self._cache[scene_id] = None
            return None
        path = self.root / f"{scene_id}.json"
        if not path.exists():
            with self._lock:
                self._cache[scene_id] = None
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            objects = payload.get("objects", []) if isinstance(payload, dict) else []
            labels = {
                int(item["id"]): str(item.get("label") or item.get("canonical_label") or "").strip()
                for item in objects
                if isinstance(item, dict)
                and _coerce_int(item.get("id")) is not None
                and str(item.get("label") or item.get("canonical_label") or "").strip()
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Could not read scene metadata %s: %s", path, exc)
            labels = None
        with self._lock:
            self._cache[scene_id] = labels
        return labels


def _pair_from_question(question: dict[str, Any]) -> tuple[int | None, int | None]:
    parent_id = _coerce_int(question.get("attachment_parent_id"))
    child_id = _coerce_int(question.get("attachment_child_id"))
    pair_id = str(question.get("attachment_pair_id") or "")
    match = _PAIR_RE.match(pair_id)
    if match:
        parent_id = parent_id if parent_id is not None else int(match.group(1))
        child_id = child_id if child_id is not None else int(match.group(2))
    return parent_id, child_id


def attachment_pairs(
    question: dict[str, Any],
    metadata: SceneMetadataResolver,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not is_attachment_question(question):
        return [], []

    scene_id = str(question.get("scene_id") or "").strip()
    labels_by_id = metadata.object_labels(scene_id) or {}
    parent_id, child_id = _pair_from_question(question)
    parent_label = str(
        question.get("attachment_parent_label")
        or (labels_by_id.get(parent_id) if parent_id is not None else "")
        or question.get("parent_label")
        or ""
    ).strip()
    child_label = str(
        question.get("attachment_child_label")
        or (labels_by_id.get(child_id) if child_id is not None else "")
        or question.get("child_label")
        or ""
    ).strip()

    pairs: list[dict[str, Any]] = []
    errors: list[str] = []
    if parent_label and child_label:
        pairs.append(
            {
                "parent_id": parent_id,
                "parent_label": parent_label,
                "child_id": child_id,
                "child_label": child_label,
            }
        )
    elif question.get("type") == "attachment_chain":
        chain_labels = [
            str(question.get("grandparent_label") or "").strip(),
            str(question.get("parent_label") or "").strip(),
            str(question.get("grandchild_label") or "").strip(),
        ]
        if all(chain_labels):
            pairs.extend(
                [
                    {"parent_id": None, "parent_label": chain_labels[0], "child_id": None, "child_label": chain_labels[1]},
                    {"parent_id": None, "parent_label": chain_labels[1], "child_id": None, "child_label": chain_labels[2]},
                ]
            )
        else:
            errors.append("attachment chain labels are incomplete")
    else:
        missing = []
        if not parent_label:
            missing.append("parent label")
        if not child_label:
            missing.append("child label")
        errors.append("attachment pair cannot be identified: missing " + " and ".join(missing))
    return pairs, errors


def mentioned_object_context(question: dict[str, Any]) -> list[dict[str, Any]]:
    groups = question.get("object_frame_groups")
    frame_1_ids: set[int] = set()
    frame_2_ids: set[int] = set()
    if isinstance(groups, dict):
        frame_1_ids = {_coerce_int(value) for value in groups.get("frame_1", [])}
        frame_2_ids = {_coerce_int(value) for value in groups.get("frame_2", [])}
        frame_1_ids.discard(None)
        frame_2_ids.discard(None)

    result: list[dict[str, Any]] = []
    raw_objects = question.get("mentioned_objects")
    if isinstance(raw_objects, list):
        for item in raw_objects:
            if not isinstance(item, dict):
                continue
            obj_id = _coerce_int(item.get("obj_id"))
            if obj_id in frame_1_ids:
                assigned_view = "first_main_view"
            elif obj_id in frame_2_ids:
                assigned_view = "last_main_view"
            else:
                assigned_view = "relevant_main_view"
            result.append(
                {
                    "role": str(item.get("role") or "object"),
                    "label": str(item.get("label") or "unknown"),
                    "obj_id": obj_id,
                    "assigned_view": assigned_view,
                }
            )
    if result:
        return result

    fallback_fields = (
        ("obj_a", "obj_a_id", "obj_a_label"),
        ("obj_b", "obj_b_id", "obj_b_label"),
        ("moved_object", "moved_obj_id", "moved_obj_label"),
        ("query_object", "query_obj_id", "query_obj_label"),
        ("reference_object", "obj_ref_id", "obj_ref_label"),
        ("facing_object", "obj_face_id", "obj_face_label"),
    )
    seen: set[tuple[int | None, str]] = set()
    for role, id_field, label_field in fallback_fields:
        label = str(question.get(label_field) or "").strip()
        if not label:
            continue
        obj_id = _coerce_int(question.get(id_field))
        key = (obj_id, label.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {"role": role, "label": label, "obj_id": obj_id, "assigned_view": "relevant_main_view"}
        )
    return result


def resolve_question_images(
    question: dict[str, Any],
    *,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    names = ordered_image_names(question)
    roles = image_roles(question, names)
    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    if not names:
        return [], ["question does not define image_name"]
    for index, (name, role) in enumerate(zip(names, roles), start=1):
        image_question = {**question, "image_name": name}
        path = _resolve_image_path(
            image_question,
            scannet_roots,
            scannetpp_roots,
            scannetpp_sensor,
        )
        exists = path.is_file()
        resolved.append(
            {
                "index": index,
                "name": name,
                "role": role,
                "path": str(path),
                "exists": exists,
            }
        )
        if not exists:
            errors.append(f"image {index} not found: {path}")
    return resolved, errors


def image_to_data_url(path: Path, max_edge: int) -> str:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        if max_edge > 0 and max(image.size) > max_edge:
            scale = max_edge / max(image.size)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _issue_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "code": {"type": "string", "enum": list(ISSUE_CODES)},
            "message_zh": {"type": "string"},
            "image_indices": {"type": "array", "items": {"type": "integer"}},
            "object_labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["code", "message_zh", "image_indices", "object_labels"],
    }


def response_json_schema() -> dict[str, Any]:
    verdict = {"type": "string", "enum": sorted(VERDICTS)}
    issue_array = {"type": "array", "items": _issue_schema()}
    object_check = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string"},
            "label": {"type": "string"},
            "assigned_view": {"type": "string"},
            "verdict": verdict,
            "notes_zh": {"type": "string"},
        },
        "required": ["role", "label", "assigned_view", "verdict", "notes_zh"],
    }
    pair_check = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "parent_label": {"type": "string"},
            "child_label": {"type": "string"},
            "supported": {"type": ["boolean", "null"]},
            "notes_zh": {"type": "string"},
        },
        "required": ["parent_label", "child_label", "supported", "notes_zh"],
    }

    def check_schema(extra: dict[str, Any]) -> dict[str, Any]:
        properties = {"verdict": verdict, "summary_zh": {"type": "string"}, "issues": issue_array, **extra}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        }

    checks = {
        CHECK_REFERABILITY: check_schema(
            {"object_checks": {"type": "array", "items": object_check}}
        ),
        CHECK_OCCLUSION: check_schema(
            {
                "observed_status": {
                    "type": ["string", "null"],
                    "enum": ["not occluded", "occluded", "not visible", "uncertain", None],
                },
                "gt_correct": {"type": ["boolean", "null"]},
            }
        ),
        CHECK_ATTACHMENT: check_schema(
            {"pair_checks": {"type": "array", "items": pair_check}}
        ),
        CHECK_CONTINUITY: check_schema(
            {"sequence_continuous": {"type": ["boolean", "null"]}}
        ),
        CHECK_FAIRNESS: check_schema(
            {
                "evidence_sufficient": {"type": ["boolean", "null"]},
                "answer_leakage": {"type": ["boolean", "null"]},
            }
        ),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "checks": {
                "type": "object",
                "additionalProperties": False,
                "properties": checks,
                "required": list(CHECK_NAMES),
            }
        },
        "required": ["checks"],
    }


SYSTEM_PROMPT = """You audit visual spatial-reasoning benchmark questions.
Use only the supplied ordered images and question metadata. Be conservative:
when the visual evidence cannot support a reliable decision, return uncertain.
Return the requested strict JSON. Write concise issue summaries in Chinese while
preserving object labels exactly as provided."""


def build_audit_prompt(context: dict[str, Any]) -> str:
    return f"""Audit this benchmark question on every applicable dimension.

Applicability and rules:
1. referability (always applicable): Each mentioned object must be visually
   identifiable, uniquely referable, and correctly named in its assigned main
   view. It does not need to appear in every bridge view. Special exception:
   for an L1 occlusion target whose ground truth and observed status are
   'not visible', do not fail referability merely because it is absent; judge
   whether the label itself is clear and let occlusion_visibility judge absence.
2. occlusion_visibility (only when applicable): Use exactly 'not occluded',
   'occluded', 'not visible', or 'uncertain'. Partial clipping by the image
   boundary alone is not occlusion. Compare the observation to correct_value.
3. attachment_pair (only when applicable): Check every supplied parent->child
   pair. Accept visible, physically plausible support or fixed attachment.
   Mere proximity without support/attachment is a failure. If the exact pair
   cannot be seen reliably, return uncertain.
4. continuity (only for multiple images): The ordered images must depict the
   same scene along a visually continuous route without a major unexplained
   jump, wrong order, reversal, or redundant duplicate.
5. fairness (only for multiple images): The assigned endpoint objects must have
   sufficient and reasonably balanced visual evidence, image quality must allow
   the intended reasoning, and bridge views must not reveal the answer through
   an unintended asymmetric shortcut.

For non-occlusion questions, do not audit whether the spatial answer itself is
correct. For checks listed as not applicable, use verdict='not_applicable', an
empty issues list, and null/empty specialized fields. A fail or uncertain check
must contain a concrete Chinese issue explaining what a human should inspect.

Question context:
{json.dumps(context, ensure_ascii=False, indent=2)}"""


def build_model_content(
    prompt: str,
    images: list[dict[str, Any]],
    *,
    max_image_edge: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for item in images:
        label = f"IMAGE {item['index']} | {item['role']} | {item['name']}"
        content.append({"type": "input_text", "text": label})
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(Path(item["path"]), max_image_edge),
                "detail": "high",
            }
        )
    return content


def _response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    return {"value": str(usage)}


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("rate limit", "timeout", "timed out", "overloaded", "connection reset"))


class ApiRequestLimiter:
    """Enforce a process-wide minimum interval between API request starts."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_allowed_at - now)
            request_start = now + delay
            self._next_allowed_at = request_start + self.interval_seconds
        if delay > 0:
            logger.info("API rate limit: waiting %.1fs before the next request", delay)
            self._sleeper(delay)


def _get_openai_client(api_key: str, base_url: str | None):
    key = (api_key, base_url)
    cache = getattr(_CLIENT_LOCAL, "clients", None)
    if not isinstance(cache, dict):
        cache = {}
        _CLIENT_LOCAL.clients = cache
    if key not in cache:
        from openai import OpenAI

        # Retries are handled below so every HTTP attempt passes through the
        # shared request limiter instead of the SDK retrying immediately.
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        cache[key] = OpenAI(**kwargs)
    return cache[key]


def call_openai_responses(
    *,
    model: str,
    content: list[dict[str, Any]],
    api_key: str,
    base_url: str | None,
    max_output_tokens: int,
    timeout: float,
    request_limiter: ApiRequestLimiter | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None, str | None]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            if request_limiter is not None:
                request_limiter.wait()
            client = _get_openai_client(api_key, base_url)
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": content}],
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "benchmark_question_audit",
                        "description": "Structured visual benchmark audit result",
                        "strict": True,
                        "schema": response_json_schema(),
                    }
                },
                store=False,
                timeout=timeout,
            )
            raw_text = str(getattr(response, "output_text", "") or "").strip()
            if not raw_text:
                raise ValueError("model returned empty output_text")
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("model response is not a JSON object")
            return parsed, raw_text, _response_usage(response), getattr(response, "id", None)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= 2:
                break
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_API_ATTEMPTS or not _is_retryable(exc):
                break
        delay = min(16.0, 2.0 ** (attempt - 1))
        logger.warning("%s attempt %d failed; retrying in %.1fs: %s", model, attempt, delay, last_error)
        time.sleep(delay)
    raise RuntimeError(f"{model} request failed after retries: {last_error}") from last_error


def validate_model_result(
    parsed: dict[str, Any],
    expected_checks: list[str],
) -> list[str]:
    errors: list[str] = []
    checks = parsed.get("checks")
    if not isinstance(checks, dict):
        return ["response is missing checks object"]
    expected = set(expected_checks)
    for name in CHECK_NAMES:
        item = checks.get(name)
        if not isinstance(item, dict):
            errors.append(f"missing check: {name}")
            continue
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            errors.append(f"invalid verdict for {name}: {verdict!r}")
            continue
        if name in expected and verdict == "not_applicable":
            errors.append(f"applicable check marked not_applicable: {name}")
        if name not in expected and verdict != "not_applicable":
            errors.append(f"non-applicable check has verdict {verdict}: {name}")
        issues = item.get("issues")
        if not isinstance(issues, list):
            errors.append(f"issues is not a list for {name}")
        elif verdict in {"fail", "uncertain"} and not issues:
            errors.append(f"{name} has {verdict} verdict without an issue")
    return errors


def stage_passed(stage: dict[str, Any], expected_checks: list[str]) -> bool:
    if stage.get("status") != "ok":
        return False
    checks = (stage.get("result") or {}).get("checks", {})
    return all((checks.get(name) or {}).get("verdict") == "pass" for name in expected_checks)


def run_model_stage(
    *,
    model: str,
    prompt: str,
    images: list[dict[str, Any]],
    expected_checks: list[str],
    max_image_edge: int,
    max_output_tokens: int,
    timeout: float,
    api_key: str,
    base_url: str | None,
    caller: Callable[..., tuple[dict[str, Any], str, dict[str, Any] | None, str | None]] = call_openai_responses,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    try:
        content = build_model_content(prompt, images, max_image_edge=max_image_edge)
        parsed, raw_text, usage, response_id = caller(
            model=model,
            content=content,
            api_key=api_key,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        validation_errors = validate_model_result(parsed, expected_checks)
        status = "ok" if not validation_errors else "invalid"
        return {
            "model": model,
            "status": status,
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "response_id": response_id,
            "usage": usage,
            "validation_errors": validation_errors,
            "result": parsed,
            "raw_response": raw_text,
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model,
            "status": "error",
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "response_id": None,
            "usage": None,
            "validation_errors": [],
            "result": None,
            "raw_response": "",
            "error": str(exc),
        }


def input_error_stage(errors: list[str]) -> dict[str, Any]:
    issues = [
        {
            "code": "missing_input",
            "message_zh": error,
            "image_indices": [],
            "object_labels": [],
        }
        for error in errors
    ]
    return {
        "model": None,
        "status": "input_error",
        "started_at": utc_now_iso(),
        "completed_at": utc_now_iso(),
        "response_id": None,
        "usage": None,
        "validation_errors": [],
        "result": {
            "checks": {
                name: {
                    "verdict": "uncertain" if name == CHECK_REFERABILITY else "not_applicable",
                    "summary_zh": "输入材料不完整，需人工检查" if name == CHECK_REFERABILITY else "",
                    "issues": issues if name == CHECK_REFERABILITY else [],
                }
                for name in CHECK_NAMES
            }
        },
        "raw_response": "",
        "error": "; ".join(errors),
    }


def _final_problem_checks(stage: dict[str, Any], expected_checks: list[str]) -> list[str]:
    if stage.get("status") != "ok":
        return list(expected_checks)
    checks = (stage.get("result") or {}).get("checks", {})
    return [name for name in expected_checks if (checks.get(name) or {}).get("verdict") != "pass"]


def result_has_unresolved_api_failure(result: dict[str, Any]) -> bool:
    """Return True when the authoritative model stage ended in an API error."""
    if result.get("final_source") == "input_validation":
        return False
    final_stage = result.get("final_result")
    return isinstance(final_stage, dict) and final_stage.get("status") == "error"


def reusable_progress_result(result: dict[str, Any]) -> bool:
    """Only reuse deterministic input errors or a valid authoritative model result."""
    if result.get("final_source") == "input_validation":
        return True
    final_stage = result.get("final_result")
    return isinstance(final_stage, dict) and final_stage.get("status") == "ok"


def _api_failure_snapshot(result: dict[str, Any], attempt: int) -> dict[str, Any]:
    def stage_summary(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        return {
            "model": value.get("model"),
            "status": value.get("status"),
            "error": value.get("error"),
        }

    return {
        "attempt": attempt,
        "completed_at": result.get("completed_at"),
        "primary": stage_summary(result.get("primary_result")),
        "review": stage_summary(result.get("review_result")),
        "final": stage_summary(result.get("final_result")),
    }


def run_with_api_failure_retries(
    audit_once: Callable[[], dict[str, Any]],
    *,
    source_index: int,
    failed_api_retries: int,
    failed_api_retry_delay: float,
) -> dict[str, Any]:
    """Retry a whole question when its authoritative API stage failed."""
    retry_count = 0
    failure_history: list[dict[str, Any]] = []
    while True:
        result = audit_once()
        if not result_has_unresolved_api_failure(result):
            break
        failure_history.append(_api_failure_snapshot(result, retry_count + 1))
        if retry_count >= failed_api_retries:
            break
        delay = failed_api_retry_delay * (2**retry_count)
        retry_count += 1
        logger.warning(
            "Question %d ended with an API error; whole-question retry %d/%d in %.1fs",
            source_index,
            retry_count,
            failed_api_retries,
            delay,
        )
        if delay > 0:
            time.sleep(delay)

    if failure_history:
        result = {
            **result,
            "api_retry_count": retry_count,
            "api_failure_history": failure_history,
        }
    return result


def audit_question(
    question: dict[str, Any],
    source_index: int,
    *,
    metadata: SceneMetadataResolver,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
    primary_model: str,
    review_model: str,
    max_image_edge: int,
    max_output_tokens: int,
    timeout: float,
    api_key: str,
    base_url: str | None,
    caller: Callable[..., tuple[dict[str, Any], str, dict[str, Any] | None, str | None]] = call_openai_responses,
) -> dict[str, Any]:
    fingerprint = question_fingerprint(question, source_index)
    images, image_errors = resolve_question_images(
        question,
        scannet_roots=scannet_roots,
        scannetpp_roots=scannetpp_roots,
        scannetpp_sensor=scannetpp_sensor,
    )
    pairs, pair_errors = attachment_pairs(question, metadata)
    expected_checks = applicable_checks(question, [item["name"] for item in images])
    input_errors = list(image_errors)
    if CHECK_ATTACHMENT in expected_checks:
        input_errors.extend(pair_errors)

    context = {
        "source_index": source_index,
        "trace_question_id": question.get("trace_question_id"),
        "level": question.get("level"),
        "type": question.get("type"),
        "scene_id": question.get("scene_id"),
        "question": question.get("question"),
        "options": question.get("options", []),
        "answer": question.get("answer"),
        "correct_value": question.get("correct_value"),
        "applicable_checks": expected_checks,
        "images": [{key: item[key] for key in ("index", "name", "role")} for item in images],
        "mentioned_objects": mentioned_object_context(question),
        "attachment_pairs": pairs,
        "occlusion_not_visible_referability_exemption": (
            is_occlusion_question(question)
            and str(question.get("correct_value") or "").strip().lower() == "not visible"
        ),
    }
    prompt = build_audit_prompt(context)

    if input_errors:
        primary = input_error_stage(input_errors)
        review = None
        final_stage = primary
        final_source = "input_validation"
    else:
        primary = run_model_stage(
            model=primary_model,
            prompt=prompt,
            images=images,
            expected_checks=expected_checks,
            max_image_edge=max_image_edge,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            caller=caller,
        )
        review = None
        final_stage = primary
        final_source = "primary"
        if not stage_passed(primary, expected_checks):
            review = run_model_stage(
                model=review_model,
                prompt=prompt,
                images=images,
                expected_checks=expected_checks,
                max_image_edge=max_image_edge,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                api_key=api_key,
                base_url=base_url,
                caller=caller,
            )
            final_stage = review
            final_source = "review"

    final_status = "passed" if stage_passed(final_stage, expected_checks) else "flagged"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_index": source_index,
        "question_fingerprint": fingerprint,
        "trace_question_id": question.get("trace_question_id"),
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "image_names": [item["name"] for item in images],
        "image_roles": [item["role"] for item in images],
        "level": question.get("level"),
        "type": question.get("type"),
        "applicable_checks": expected_checks,
        "attachment_pairs": pairs,
        "input_errors": input_errors,
        "primary_result": primary,
        "review_result": review,
        "final_source": final_source,
        "final_status": final_status,
        "problem_checks": _final_problem_checks(final_stage, expected_checks),
        "final_result": final_stage,
        "completed_at": utc_now_iso(),
    }


def cache_key(
    question: dict[str, Any],
    source_index: int,
    *,
    primary_model: str,
    review_model: str,
    max_image_edge: int,
    max_output_tokens: int,
    scannetpp_sensor: str,
    base_url: str | None,
) -> str:
    return sha256_json(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "question_fingerprint": question_fingerprint(question, source_index),
            "primary_model": primary_model,
            "review_model": review_model,
            "max_image_edge": max_image_edge,
            "max_output_tokens": max_output_tokens,
            "scannetpp_sensor": scannetpp_sensor,
            "base_url": base_url,
        }
    )


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed progress line %d in %s", line_number, path)
            continue
        if isinstance(item, dict) and isinstance(item.get("cache_key"), str):
            cached[item["cache_key"]] = item
    return cached


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(item.get("final_status")) for item in results)
    problem_counts = Counter(
        check for item in results for check in item.get("problem_checks", [])
    )
    primary_referred = sum(1 for item in results if item.get("review_result") is not None)
    review_passed = sum(
        1
        for item in results
        if item.get("review_result") is not None and item.get("final_status") == "passed"
    )
    return {
        "total": len(results),
        "passed": status_counts.get("passed", 0),
        "flagged": status_counts.get("flagged", 0),
        "sent_to_review_model": primary_referred,
        "review_model_overturned_to_pass": review_passed,
        "api_retried_questions": sum(
            1 for item in results if int(item.get("api_retry_count") or 0) > 0
        ),
        "unresolved_api_failures": sum(
            1 for item in results if result_has_unresolved_api_failure(item)
        ),
        "problem_checks": dict(sorted(problem_counts.items())),
    }


def compile_report(
    results: list[dict[str, Any]],
    *,
    benchmark_path: Path,
    primary_model: str,
    review_model: str,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now_iso(),
        "source_benchmark": str(benchmark_path.resolve()),
        "primary_model": primary_model,
        "review_model": review_model,
        "summary": result_summary(results),
        "results": results,
    }


def run_audit(
    questions: list[dict[str, Any]],
    *,
    benchmark_path: Path,
    output_dir: Path,
    metadata_root: Path | None,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
    primary_model: str,
    review_model: str,
    max_image_edge: int,
    max_output_tokens: int,
    timeout: float,
    max_workers: int,
    api_key: str,
    base_url: str | None,
    resume: bool,
    failed_api_retries: int,
    failed_api_retry_delay: float,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    caller: Callable[..., tuple[dict[str, Any], str, dict[str, Any] | None, str | None]] = call_openai_responses,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    cached = load_progress(progress_path) if resume else {}
    progress_mode = "a" if resume else "w"
    metadata = SceneMetadataResolver(metadata_root)
    results_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any], str]] = []
    rejected_cached_results = 0
    effective_caller = caller
    if request_interval > 0:
        request_limiter = ApiRequestLimiter(request_interval)
        if caller is call_openai_responses:
            effective_caller = partial(caller, request_limiter=request_limiter)
        else:
            def rate_limited_caller(**kwargs: Any):
                request_limiter.wait()
                return caller(**kwargs)

            effective_caller = rate_limited_caller
        logger.info(
            "Global API request interval enabled: %.1fs (effective maximum %.2f RPM)",
            request_interval,
            60.0 / request_interval,
        )

    for source_index, question in enumerate(questions):
        key = cache_key(
            question,
            source_index,
            primary_model=primary_model,
            review_model=review_model,
            max_image_edge=max_image_edge,
            max_output_tokens=max_output_tokens,
            scannetpp_sensor=scannetpp_sensor,
            base_url=base_url,
        )
        cached_item = cached.get(key)
        cached_result = cached_item.get("result") if isinstance(cached_item, dict) else None
        if isinstance(cached_result, dict) and reusable_progress_result(cached_result):
            results_by_index[source_index] = cached_result
        else:
            if isinstance(cached_result, dict):
                rejected_cached_results += 1
            pending.append((source_index, question, key))

    logger.info(
        "Audit selection: total=%d cached=%d retry_failed_cache=%d pending=%d",
        len(questions),
        len(results_by_index),
        rejected_cached_results,
        len(pending),
    )

    with progress_path.open(progress_mode, encoding="utf-8") as progress_file:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(
                    run_with_api_failure_retries,
                    partial(
                        audit_question,
                        question,
                        source_index,
                        metadata=metadata,
                        scannet_roots=scannet_roots,
                        scannetpp_roots=scannetpp_roots,
                        scannetpp_sensor=scannetpp_sensor,
                        primary_model=primary_model,
                        review_model=review_model,
                        max_image_edge=max_image_edge,
                        max_output_tokens=max_output_tokens,
                        timeout=timeout,
                        api_key=api_key,
                        base_url=base_url,
                        caller=effective_caller,
                    ),
                    source_index=source_index,
                    failed_api_retries=failed_api_retries,
                    failed_api_retry_delay=failed_api_retry_delay,
                ): (source_index, key)
                for source_index, question, key in pending
            }
            completed = 0
            for future in as_completed(futures):
                source_index, key = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.exception("Unexpected audit worker failure for question %d", source_index)
                    result = {
                        "schema_version": AUDIT_SCHEMA_VERSION,
                        "prompt_version": PROMPT_VERSION,
                        "source_index": source_index,
                        "question_fingerprint": question_fingerprint(questions[source_index], source_index),
                        "scene_id": questions[source_index].get("scene_id"),
                        "image_name": questions[source_index].get("image_name"),
                        "level": questions[source_index].get("level"),
                        "type": questions[source_index].get("type"),
                        "applicable_checks": [CHECK_REFERABILITY],
                        "input_errors": [],
                        "primary_result": None,
                        "review_result": None,
                        "final_source": "worker_error",
                        "final_status": "flagged",
                        "problem_checks": [CHECK_REFERABILITY],
                        "final_result": {"status": "error", "error": str(exc), "result": None},
                        "completed_at": utc_now_iso(),
                    }
                results_by_index[source_index] = result
                progress_file.write(canonical_json({"cache_key": key, "result": result}) + "\n")
                progress_file.flush()
                completed += 1
                logger.info(
                    "Completed %d/%d new questions: index=%d final=%s",
                    completed,
                    len(pending),
                    source_index,
                    result.get("final_status"),
                )

    return [results_by_index[index] for index in sorted(results_by_index)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit multi-image benchmark questions with GPT vision models")
    parser.add_argument("--benchmark", required=True, type=Path, help="Input benchmark JSON")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for audit artifacts")
    parser.add_argument("--scannet_image_root", action="append", default=[], help="ScanNet scans root; repeatable")
    parser.add_argument("--scannetpp_image_root", action="append", default=[], help="ScanNet++ image root; repeatable")
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--scene_metadata_root", type=Path, default=None, help="Scene metadata directory; defaults to <benchmark_dir>/scene_metadata")
    parser.add_argument("--primary_model", default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--review_model", default=DEFAULT_REVIEW_MODEL)
    parser.add_argument(
        "--api_key_env",
        default=None,
        help="API-key environment variable; by default tries OPENAI_API_KEY then API_KEY",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Optional OpenAI-compatible API base URL (OPENAI_BASE_URL is also supported by the SDK)",
    )
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max_image_edge", type=int, default=DEFAULT_MAX_IMAGE_EDGE)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--failed_api_retries",
        type=int,
        default=DEFAULT_FAILED_API_RETRIES,
        help="Whole-question retries after the authoritative API stage fails",
    )
    parser.add_argument(
        "--failed_api_retry_delay",
        type=float,
        default=DEFAULT_FAILED_API_RETRY_DELAY,
        help="Initial delay in seconds between whole-question API retries",
    )
    parser.add_argument(
        "--request_interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL,
        help="Minimum seconds between all API request starts; use 65 for an RPM=1 key",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N questions")
    parser.add_argument("--viewer_output", type=Path, default=None, help="Viewer path; defaults to <output_dir>/review.html")
    parser.add_argument("--no_viewer", action="store_true", help="Skip automatic flagged-question viewer generation")
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    benchmark_path = args.benchmark.resolve()
    output_dir = args.output_dir.resolve()
    _, questions = load_benchmark(benchmark_path)
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        questions = questions[: args.limit]

    scannet_roots = [Path(path).resolve() for path in args.scannet_image_root]
    scannetpp_roots = [Path(path).resolve() for path in args.scannetpp_image_root]
    if not scannet_roots and not scannetpp_roots:
        parser.error("At least one --scannet_image_root or --scannetpp_image_root is required")
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.failed_api_retries < 0:
        parser.error("--failed_api_retries must be non-negative")
    if args.failed_api_retry_delay < 0:
        parser.error("--failed_api_retry_delay must be non-negative")
    if args.request_interval < 0:
        parser.error("--request_interval must be non-negative")

    metadata_root = (
        args.scene_metadata_root.resolve()
        if args.scene_metadata_root is not None
        else benchmark_path.parent / "scene_metadata"
    )
    if not metadata_root.is_dir():
        logger.warning("Scene metadata root is unavailable: %s", metadata_root)
        metadata_root = None

    resolved_api_key = resolve_api_key(args.api_key_env)
    if resolved_api_key is None:
        if args.api_key_env:
            parser.error(f"Environment variable {args.api_key_env} is not set")
        parser.error(
            "None of the API-key environment variables are set: "
            + ", ".join(DEFAULT_API_KEY_ENV_NAMES)
        )
    api_key, api_key_env = resolved_api_key
    logger.info("Using API key from environment variable %s", api_key_env)
    base_url = str(args.base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    if base_url:
        logger.info("Using OpenAI-compatible API base URL %s", base_url)

    results = run_audit(
        questions,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        metadata_root=metadata_root,
        scannet_roots=scannet_roots,
        scannetpp_roots=scannetpp_roots,
        scannetpp_sensor=args.scannetpp_sensor,
        primary_model=args.primary_model,
        review_model=args.review_model,
        max_image_edge=args.max_image_edge,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        max_workers=args.max_workers,
        api_key=api_key,
        base_url=base_url,
        resume=args.resume,
        failed_api_retries=args.failed_api_retries,
        failed_api_retry_delay=args.failed_api_retry_delay,
        request_interval=args.request_interval,
    )
    full_report = compile_report(
        results,
        benchmark_path=benchmark_path,
        primary_model=args.primary_model,
        review_model=args.review_model,
    )
    flagged_report = {
        **{key: value for key, value in full_report.items() if key != "results"},
        "results": [item for item in results if item.get("final_status") == "flagged"],
    }
    full_path = output_dir / "full_results.json"
    flagged_path = output_dir / "flagged_questions.json"
    write_json(full_path, full_report)
    write_json(flagged_path, flagged_report)
    logger.info("Wrote %s", full_path)
    logger.info("Wrote %s", flagged_path)

    if not args.no_viewer:
        from scripts.make_gpt_audit_viewer import generate_viewer

        viewer_path = (args.viewer_output or (output_dir / "review.html")).resolve()
        generate_viewer(
            benchmark_path=benchmark_path,
            audit_path=flagged_path,
            output_path=viewer_path,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=args.scannetpp_sensor,
        )
        logger.info("Wrote %s", viewer_path)

    print(json.dumps(full_report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
