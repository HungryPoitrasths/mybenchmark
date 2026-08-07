from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# MS-SWIFT loads external plugin files as top-level modules, outside the src.cot
# package. Add the repository root so package imports work in both contexts.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot.facts import question_uid
from src.cot.images import resolve_image_paths
from src.cot.pipeline import format_user_prompt
from src.cot.sampling import SUPPORTED_TYPE_ORDER, TYPES_BY_LEVEL


def _cast_adam_moments(optimizer: Any, target_dtype: Any) -> int:
    """Restore Adam moment dtype after a cross-version checkpoint load."""
    import torch

    converted = 0
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.is_floating_point() and value.dtype != target_dtype:
                state[key] = value.to(dtype=target_dtype)
                converted += 1
    return converted


def _patch_adamw_resume_state_dtype() -> None:
    dtype_name = os.environ.get("PSR_RESUME_ADAM_STATE_DTYPE")
    if not dtype_name:
        return

    import torch

    target_dtype = getattr(torch, dtype_name)
    original = torch.optim.AdamW.load_state_dict
    if getattr(original, "_psr_state_dtype_patch", False):
        return

    def load_state_dict(optimizer: Any, state_dict: dict[str, Any]) -> Any:
        result = original(optimizer, state_dict)
        converted = _cast_adam_moments(optimizer, target_dtype)
        print(
            f"[PSR] Restored {converted} Adam moment tensors to {dtype_name}",
            flush=True,
        )
        return result

    load_state_dict._psr_state_dtype_patch = True  # type: ignore[attr-defined]
    torch.optim.AdamW.load_state_dict = load_state_dict


_patch_adamw_resume_state_dtype()


R1_COMPLETION_RE = re.compile(
    r"\A<think>(?P<thinking>.*?)</think>\s*"
    r"<answer>\s*(?P<answer>[A-Z](?:\s+[A-Z])*)\s*</answer>\s*\Z",
    re.DOTALL,
)
ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(?P<answer>[A-Z](?:[\s,]+[A-Z])*)\s*</answer>",
    re.IGNORECASE,
)
FINAL_ANSWER_RE = re.compile(
    r"(?:\A|\n)(?:Answer:\s*)?(?P<answer>[A-Z](?:[\s,]+[A-Z])*)\s*\Z",
    re.IGNORECASE,
)
LEGACY_ANSWER_RE = re.compile(r"^Answer: ([A-Z](?: [A-Z])*)$")


def normalize_letters(
    value: str,
    *,
    option_count: int,
    multi_select: bool,
) -> str | None:
    letters = re.findall(r"[A-Z]", value.upper())
    if not letters or option_count <= 0:
        return None
    if not multi_select and len(letters) != 1:
        return None
    if len(set(letters)) != len(letters):
        return None
    if any(ord(letter) - ord("A") >= option_count for letter in letters):
        return None
    return " ".join(letters)


def extract_reward_answer(
    completion: str,
    *,
    option_count: int,
    multi_select: bool,
) -> str | None:
    """Extract only an explicit answer block or the completion's final answer line."""
    tagged = list(ANSWER_TAG_RE.finditer(completion))
    if tagged:
        raw = tagged[-1].group("answer")
    else:
        match = FINAL_ANSWER_RE.search(completion.rstrip())
        if match is None:
            return None
        raw = match.group("answer")
    return normalize_letters(
        raw,
        option_count=option_count,
        multi_select=multi_select,
    )


def has_valid_r1_format(
    completion: str,
    *,
    option_count: int,
    multi_select: bool,
) -> bool:
    match = R1_COMPLETION_RE.fullmatch(completion)
    if match is None or not match.group("thinking").strip():
        return False
    answer = match.group("answer")
    normalized = normalize_letters(
        answer,
        option_count=option_count,
        multi_select=multi_select,
    )
    return normalized is not None and answer.strip() == normalized


def answer_rewards(
    completions: Iterable[str],
    solutions: Iterable[str],
    option_counts: Iterable[int],
    multi_select_values: Iterable[bool],
) -> list[float]:
    rewards: list[float] = []
    for completion, solution, option_count, multi_select in zip(
        completions,
        solutions,
        option_counts,
        multi_select_values,
        strict=True,
    ):
        count = int(option_count)
        is_multi = bool(multi_select)
        predicted = extract_reward_answer(
            str(completion),
            option_count=count,
            multi_select=is_multi,
        )
        expected = normalize_letters(
            str(solution),
            option_count=count,
            multi_select=is_multi,
        )
        rewards.append(float(predicted is not None and predicted == expected))
    return rewards


def format_rewards(
    completions: Iterable[str],
    option_counts: Iterable[int],
    multi_select_values: Iterable[bool],
) -> list[float]:
    return [
        float(
            has_valid_r1_format(
                str(completion),
                option_count=int(option_count),
                multi_select=bool(multi_select),
            )
        )
        for completion, option_count, multi_select in zip(
            completions,
            option_counts,
            multi_select_values,
            strict=True,
        )
    ]


try:
    from swift.rewards import ORM, orms
except ImportError:  # Unit tests and dry-runs do not require MS-SWIFT.
    class ORM:  # type: ignore[no-redef]
        def __init__(self, args: Any = None, **kwargs: Any) -> None:
            self.args = args

    orms: dict[str, type[ORM]] = {}


class PSRAnswerReward(ORM):
    def __call__(
        self,
        completions: list[str],
        solution: list[str],
        option_count: list[int],
        multi_select: list[bool],
        **kwargs: Any,
    ) -> list[float]:
        return answer_rewards(completions, solution, option_count, multi_select)


class PSRFormatReward(ORM):
    def __call__(
        self,
        completions: list[str],
        option_count: list[int],
        multi_select: list[bool],
        **kwargs: Any,
    ) -> list[float]:
        return format_rewards(completions, option_count, multi_select)


orms["psr_answer"] = PSRAnswerReward
orms["psr_format"] = PSRFormatReward


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_benchmark_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not all(isinstance(row, dict) for row in questions):
        raise ValueError(f"{path}: expected a list of question objects")
    return questions


def _index_sidecar(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        uid = str(row.get("question_uid") or "").strip()
        if not uid:
            raise ValueError(f"{path}:{index}: missing question_uid")
        if uid in result:
            raise ValueError(f"{path}:{index}: duplicate question_uid {uid!r}")
        result[uid] = row
    return result


def _legacy_assistant_answer(messages: list[dict[str, Any]]) -> str | None:
    assistant_messages = [
        message
        for message in messages
        if str(message.get("role") or "").lower() == "assistant"
    ]
    if len(assistant_messages) != 1:
        return None
    content = assistant_messages[0].get("content")
    if not isinstance(content, str):
        return None
    lines = content.rstrip().splitlines()
    if not lines:
        return None
    match = LEGACY_ANSWER_RE.fullmatch(lines[-1])
    return match.group(1) if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _raw_question_level(question_type: str) -> str:
    for level, question_types in TYPES_BY_LEVEL.items():
        if question_type in question_types:
            return level
    return ""


def _stable_question_rank(seed: int, question_type: str, uid: str) -> str:
    return hashlib.sha256(
        f"{seed}|{question_type}|{uid}".encode("utf-8")
    ).hexdigest()


def _raw_solution(question: dict[str, Any], *, question_type: str) -> tuple[str, int, bool]:
    options = question.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"{question_type}: options must be a non-empty list")
    raw_answer = question.get("answer")
    if isinstance(raw_answer, list):
        answer_text = " ".join(str(value) for value in raw_answer)
    else:
        answer_text = str(raw_answer or "")
    multi_select = bool(question.get("multi_select"))
    solution = normalize_letters(
        answer_text,
        option_count=len(options),
        multi_select=multi_select,
    )
    if solution is None:
        raise ValueError(f"{question_type}: invalid answer {raw_answer!r}")
    return solution, len(options), multi_select


def prepare_balanced_benchmark_grpo_dataset(
    benchmark_path: Path,
    output_path: Path,
    *,
    samples_per_type: int | None,
    seed: int,
    preserve_input_order: bool = False,
    scannet_roots: list[Path] | None = None,
    scannetpp_roots: list[Path] | None = None,
    scannetpp_sensor: str = "iphone",
    check_images: bool = True,
) -> dict[str, Any]:
    """Build GRPO data from all benchmark rows or an equal per-type sample."""
    benchmark_path = benchmark_path.resolve()
    output_path = output_path.resolve()
    questions = load_benchmark_questions(benchmark_path)
    candidates_by_type: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    candidates_in_order: list[tuple[str, dict[str, Any]]] = []
    seen_uids: set[str] = set()

    for index, question in enumerate(questions, start=1):
        question_type = str(question.get("type") or "").strip()
        if not question_type:
            raise ValueError(f"{benchmark_path}:{index}: missing question type")
        expected_level = _raw_question_level(question_type)
        if expected_level and str(question.get("level") or "").upper() != expected_level:
            raise ValueError(
                f"{benchmark_path}:{index}: {question_type} must have level {expected_level}"
            )
        _raw_solution(question, question_type=question_type)
        uid = question_uid(question)
        if uid in seen_uids:
            raise ValueError(f"{benchmark_path}:{index}: duplicate question_uid {uid!r}")
        seen_uids.add(uid)
        candidate = (uid, question)
        candidates_by_type[question_type].append(candidate)
        candidates_in_order.append(candidate)

    available_by_type = {
        question_type: len(candidates)
        for question_type, candidates in sorted(candidates_by_type.items())
    }
    if not available_by_type:
        raise ValueError(f"{benchmark_path}: no supported question types were found")
    selected: list[tuple[str, dict[str, Any]]] = []
    type_order = [
        *[
            question_type
            for question_type in SUPPORTED_TYPE_ORDER
            if question_type in candidates_by_type
        ],
        *sorted(set(candidates_by_type) - set(SUPPORTED_TYPE_ORDER)),
    ]
    if samples_per_type is not None and samples_per_type <= 0:
        raise ValueError("samples_per_type must be positive")
    if preserve_input_order and samples_per_type is not None:
        raise ValueError(
            "preserve_input_order cannot be combined with samples_per_type"
        )
    if samples_per_type is not None:
        undersized = {
            question_type: count
            for question_type, count in available_by_type.items()
            if count < samples_per_type
        }
        if undersized:
            raise ValueError(
                f"cannot sample {samples_per_type} records per type; "
                f"insufficient capacities: {undersized}"
            )
    if preserve_input_order:
        selected = list(candidates_in_order)
    else:
        for question_type in type_order:
            candidates = candidates_by_type[question_type]
            ordered = sorted(
                candidates,
                key=lambda item: _stable_question_rank(seed, question_type, item[0]),
            )
            selected.extend(
                ordered if samples_per_type is None else ordered[:samples_per_type]
            )
        selected.sort(key=lambda item: _stable_question_rank(seed, "global", item[0]))

    prepared: list[dict[str, Any]] = []
    image_counts: list[int] = []
    selected_by_type: Counter[str] = Counter()
    source_uid_counts: Counter[str] = Counter()
    for uid, question in selected:
        question_type = str(question["type"])
        solution, option_count, multi_select = _raw_solution(
            question,
            question_type=question_type,
        )
        images, _diagnostics = resolve_image_paths(
            question,
            benchmark_path=benchmark_path,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=scannetpp_sensor,
            require_exists=check_images,
        )
        prepared.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": format_user_prompt(question, len(images)),
                    }
                ],
                "images": images,
                "solution": solution,
                "option_count": option_count,
                "multi_select": multi_select,
                "question_uid": uid,
                "source_question_uid": str(
                    question.get("source_question_uid") or uid
                ),
                "sampling_repeat_index": int(
                    question.get("sampling_repeat_index") or 0
                ),
                "question_type": question_type,
                "signature_id": str(question.get("signature_id") or question_type),
            }
        )
        image_counts.append(len(images))
        selected_by_type[question_type] += 1
        source_uid_counts[str(question.get("source_question_uid") or uid)] += 1

    _write_jsonl(output_path, prepared)
    return {
        "schema_version": "predictive-spatial-grpo-balanced-benchmark-v1",
        "benchmark": str(benchmark_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "prepared_dataset": str(output_path),
        "input_question_count": len(questions),
        "selected_count": len(prepared),
        "sampling_mode": (
            "all_questions_in_source_order"
            if preserve_input_order
            else "all_questions"
            if samples_per_type is None
            else "equal_per_type"
        ),
        "samples_per_type": samples_per_type,
        "input_order_preserved": preserve_input_order,
        "available_by_type": dict(sorted(available_by_type.items())),
        "selected_by_type": dict(sorted(selected_by_type.items())),
        "unique_source_question_count": len(source_uid_counts),
        "repeated_instance_count": len(prepared) - len(source_uid_counts),
        "repeated_source_question_count": sum(
            count > 1 for count in source_uid_counts.values()
        ),
        "max_source_instances": max(source_uid_counts.values()),
        "missing_supported_types": [
            question_type
            for question_type in SUPPORTED_TYPE_ORDER
            if question_type not in available_by_type
        ],
        "min_images": min(image_counts),
        "max_images": max(image_counts),
        "checked_images": check_images,
        "seed": seed,
    }


def prepare_grpo_dataset(
    dataset_path: Path,
    sidecar_path: Path,
    output_path: Path,
    *,
    check_images: bool = True,
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    sidecar_path = sidecar_path.resolve()
    output_path = output_path.resolve()
    dataset_rows = load_jsonl(dataset_path)
    sidecar_rows = load_jsonl(sidecar_path)
    sidecar_by_uid = _index_sidecar(sidecar_rows, sidecar_path)

    prepared: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    image_counts: list[int] = []
    answer_counts: Counter[str] = Counter()
    multi_select_count = 0

    for index, row in enumerate(dataset_rows, start=1):
        uid = str(row.get("question_uid") or "").strip()
        if not uid:
            raise ValueError(f"{dataset_path}:{index}: missing question_uid")
        if uid in seen_uids:
            raise ValueError(f"{dataset_path}:{index}: duplicate question_uid {uid!r}")
        seen_uids.add(uid)
        gold = sidecar_by_uid.get(uid)
        if gold is None:
            raise ValueError(f"{dataset_path}:{index}: UID is missing from sidecar: {uid!r}")

        messages = row.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError(f"{dataset_path}:{index}: messages must be a list of objects")
        prompt_messages = [
            dict(message)
            for message in messages
            if str(message.get("role") or "").lower() != "assistant"
        ]
        if not any(
            str(message.get("role") or "").lower() == "user"
            for message in prompt_messages
        ):
            raise ValueError(f"{dataset_path}:{index}: prompt has no user message")

        option_count = int(gold.get("option_count") or 0)
        multi_select = bool(gold.get("multi_select"))
        answer_values = gold.get("answer_letters")
        if not isinstance(answer_values, list):
            raise ValueError(f"{sidecar_path}:{index}: answer_letters must be a list")
        solution = normalize_letters(
            " ".join(str(value) for value in answer_values),
            option_count=option_count,
            multi_select=multi_select,
        )
        if solution is None:
            raise ValueError(f"{sidecar_path}:{index}: invalid answer_letters")
        legacy_answer = _legacy_assistant_answer(messages)
        if legacy_answer is None or legacy_answer != solution:
            raise ValueError(
                f"{dataset_path}:{index}: assistant answer does not match sidecar "
                f"({legacy_answer!r} != {solution!r})"
            )

        images = row.get("images")
        if not isinstance(images, list) or not all(isinstance(path, str) for path in images):
            raise ValueError(f"{dataset_path}:{index}: images must be a list of paths")
        placeholder_count = sum(
            str(message.get("content") or "").count("<image>")
            for message in prompt_messages
        )
        if placeholder_count != len(images):
            raise ValueError(
                f"{dataset_path}:{index}: {placeholder_count} image placeholders for "
                f"{len(images)} image paths"
            )
        if check_images:
            missing = [path for path in images if not Path(path).is_file()]
            if missing:
                raise ValueError(f"{dataset_path}:{index}: image does not exist: {missing[0]}")

        prepared.append(
            {
                "messages": prompt_messages,
                "images": images,
                "solution": solution,
                "option_count": option_count,
                "multi_select": multi_select,
                "question_uid": uid,
                "question_type": str(gold.get("question_type") or ""),
                "signature_id": str(gold.get("signature_id") or ""),
            }
        )
        image_counts.append(len(images))
        answer_counts[solution] += 1
        multi_select_count += int(multi_select)

    extra_uids = set(sidecar_by_uid) - seen_uids
    if extra_uids:
        example = min(extra_uids)
        raise ValueError(
            f"sidecar contains {len(extra_uids)} UID(s) absent from the dataset; "
            f"example: {example!r}"
        )
    if not prepared:
        raise ValueError("training dataset is empty")

    _write_jsonl(output_path, prepared)
    return {
        "schema_version": "predictive-spatial-grpo-v1",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "prepared_dataset": str(output_path),
        "row_count": len(prepared),
        "min_images": min(image_counts),
        "max_images": max(image_counts),
        "multi_select_count": multi_select_count,
        "answer_counts": dict(sorted(answer_counts.items())),
        "checked_images": check_images,
    }
