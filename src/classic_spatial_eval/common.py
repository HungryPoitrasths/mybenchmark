"""Shared schemas, I/O helpers, and answer parsing for classic evals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "classic-spatial-eval-manifest-v1"
LOCK_SCHEMA_VERSION = "classic-spatial-eval-lock-v1"
RESULT_SCHEMA_VERSION = "classic-spatial-eval-result-v1"

TARGET_COUNTS: dict[str, dict[str, int]] = {
    "mmsi": {"object_motion": 76, "multi_step_reasoning": 198},
    "spar": {
        "ViewChg": 50,
        "SpImag_OC": 50,
        "SpImag_OC_MV": 50,
        "SpImag_OO": 50,
        "SpImag_OO_MV": 50,
    },
    "mindcube": {"dynamics": 292},
    "vsi": {
        "relative_direction": 50,
        "relative_distance": 50,
        "route_planning": 50,
    },
    "mvbench": {
        "object_shuffle": 200,
        "moving_direction": 200,
        "egocentric_navigation": 200,
    },
    "clevrer": {
        "explanatory": 200,
        "predictive": 200,
        "counterfactual": 200,
    },
    "blink": {
        "Multi-view_Reasoning": 133,
        "Relative_Depth": 124,
        "Spatial_Relation": 143,
        "Visual_Correspondence": 172,
    },
    "vsr": {"zero_shot_test": 731},
}

EXPECTED_TOTAL = sum(
    count for subsets in TARGET_COUNTS.values() for count in subsets.values()
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(*parts: object, seed: int = 42) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return sha256_bytes(payload.encode("utf-8"))


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    _atomic_write(path, content)


def append_jsonl(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(canonical_json(dict(row)) + "\n")
    handle.flush()


def option_letters(option_count: int) -> str:
    if option_count <= 0 or option_count > 26:
        raise ValueError(f"option count must be in [1, 26], got {option_count}")
    return "".join(chr(ord("A") + index) for index in range(option_count))


def canonical_letters(values: Iterable[str], allowed: str) -> tuple[str, ...]:
    selected = {str(value).strip().upper() for value in values}
    return tuple(letter for letter in allowed if letter in selected)


def _letters_from_fragment(fragment: str, allowed: str) -> tuple[str, ...]:
    escaped = re.escape(allowed)
    tokens = re.findall(
        rf"(?<![A-Z0-9])([{escaped}])(?![A-Z0-9])", fragment.upper()
    )
    if tokens:
        return canonical_letters(tokens, allowed)
    compact = re.sub(r"[\s,;/&+|，、-]+", "", fragment.upper())
    if compact and re.fullmatch(rf"[{escaped}]+", compact):
        return canonical_letters(compact, allowed)
    return ()


@dataclass(frozen=True)
class AnswerParseResult:
    letters: tuple[str, ...]
    status: str
    source: str | None = None

    @property
    def text(self) -> str | None:
        return " ".join(self.letters) if self.letters else None


def parse_answer(
    raw: str | None,
    options: Sequence[str],
    *,
    multi_select: bool = False,
) -> AnswerParseResult:
    """Parse a model answer conservatively without guessing through conflicts."""

    if not raw or not raw.strip():
        return AnswerParseResult((), "empty")
    allowed = option_letters(len(options))
    text = raw.strip()

    tagged = re.findall(r"<answer\b[^>]*>(.*?)</answer>", text, flags=re.I | re.S)
    if tagged:
        parsed = [_letters_from_fragment(fragment, allowed) for fragment in tagged]
        parsed = [candidate for candidate in parsed if candidate]
        unique = {candidate for candidate in parsed}
        if len(unique) == 1:
            letters = next(iter(unique))
            if not multi_select and len(letters) != 1:
                return AnswerParseResult((), "invalid_multiple", "answer_tag")
            return AnswerParseResult(letters, "ok", "answer_tag")
        if len(unique) > 1:
            return AnswerParseResult((), "conflict", "answer_tag")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = re.sub(
            r"^(?:final\s+)?(?:answer|choice|option)\s*[:：]?\s*",
            "",
            lines[-1],
            flags=re.I,
        )
        letters = _letters_from_fragment(last, allowed)
        if letters and (multi_select or len(letters) == 1):
            return AnswerParseResult(letters, "ok", "last_line")

    explicit = re.findall(
        r"(?:final\s+)?(?:answer|choice|option)\s*(?:is|:|：)\s*([^\r\n]+)",
        text,
        flags=re.I,
    )
    explicit_candidates = {
        candidate
        for fragment in explicit
        if (candidate := _letters_from_fragment(fragment, allowed))
    }
    if len(explicit_candidates) == 1:
        letters = next(iter(explicit_candidates))
        if multi_select or len(letters) == 1:
            return AnswerParseResult(letters, "ok", "explicit_answer")
    if len(explicit_candidates) > 1:
        return AnswerParseResult((), "conflict", "explicit_answer")

    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    text_matches = [
        allowed[index]
        for index, option in enumerate(options)
        if normalized == re.sub(r"\s+", " ", str(option)).strip().casefold()
    ]
    if len(text_matches) == 1:
        return AnswerParseResult(tuple(text_matches), "ok", "option_text")

    return AnswerParseResult((), "invalid")


@dataclass(frozen=True)
class ExactAnswerParseResult:
    text: str | None
    status: str
    source: str | None = None


def canonical_exact_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def parse_exact_answer(raw: str | None) -> ExactAnswerParseResult:
    if not raw or not raw.strip():
        return ExactAnswerParseResult(None, "empty")
    text = raw.strip()
    tagged = [
        fragment.strip()
        for fragment in re.findall(
            r"<answer\b[^>]*>(.*?)</answer>", text, flags=re.I | re.S
        )
        if fragment.strip()
    ]
    if tagged:
        canonical = {canonical_exact_text(value) for value in tagged}
        if len(canonical) == 1:
            return ExactAnswerParseResult(tagged[0], "ok", "answer_tag")
        return ExactAnswerParseResult(None, "conflict", "answer_tag")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ExactAnswerParseResult(None, "empty")
    answer = re.sub(
        r"^(?:final\s+)?answer\s*[:：]?\s*",
        "",
        lines[-1],
        flags=re.I,
    ).strip(" `")
    if not answer:
        return ExactAnswerParseResult(None, "invalid", "last_line")
    return ExactAnswerParseResult(answer, "ok", "last_line")


def answer_to_letters(
    answer: Any,
    options: Sequence[str],
    *,
    multi_select: bool | None = None,
) -> tuple[str, ...]:
    """Normalize official answer fields to canonical option letters."""

    allowed = option_letters(len(options))
    if isinstance(answer, (list, tuple, set)):
        raw_values = list(answer)
    else:
        raw_values = [answer]

    selected: list[str] = []
    normalized_options = [re.sub(r"\s+", " ", str(item)).strip().casefold() for item in options]
    for value in raw_values:
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            if 0 <= value < len(options):
                selected.append(allowed[value])
                continue
        text = str(value).strip()
        letter_match = re.fullmatch(r"[\(\[]?\s*([A-Z])\s*[\)\].]?", text.upper())
        if letter_match and letter_match.group(1) in allowed:
            selected.append(letter_match.group(1))
            continue
        normalized = re.sub(r"\s+", " ", text).strip().casefold()
        if normalized in normalized_options:
            selected.append(allowed[normalized_options.index(normalized)])
            continue
        parsed = _letters_from_fragment(text, allowed)
        if parsed:
            selected.extend(parsed)
            continue
        raise ValueError(f"cannot map answer {value!r} to options {list(options)!r}")

    letters = canonical_letters(selected, allowed)
    if not letters:
        raise ValueError("answer resolves to no options")
    if multi_select is False and len(letters) != 1:
        raise ValueError(f"single-select answer resolves to {letters}")
    return letters


_INLINE_OPTION = re.compile(
    r"(?<![A-Za-z0-9])([A-Z])[\).:]\s*(.+?)(?=(?<![A-Za-z0-9])[A-Z][\).:]\s|$)",
    flags=re.S,
)


def split_inline_options(question: str) -> tuple[str, list[str]]:
    matches = list(_INLINE_OPTION.finditer(question))
    if len(matches) < 2:
        return question.strip(), []
    expected = [chr(ord("A") + index) for index in range(len(matches))]
    if [match.group(1) for match in matches] != expected:
        return question.strip(), []
    stem = question[: matches[0].start()].strip()
    return stem, [match.group(2).strip() for match in matches]


def resolve_media_path(path: str, manifest_path: Path) -> Path:
    media_path = Path(path)
    if not media_path.is_absolute():
        media_path = manifest_path.parent / media_path
    return media_path.resolve()


def validate_manifest_row(row: Mapping[str, Any], *, row_number: int) -> None:
    required = {
        "schema_version",
        "benchmark",
        "subset",
        "split",
        "sample_id",
        "source_id",
        "media",
        "question",
        "options",
        "gold",
        "multi_select",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"manifest row {row_number} missing fields: {missing}")
    if row["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest row {row_number} has unsupported schema {row['schema_version']!r}"
        )
    options = row["options"]
    gold = row["gold"]
    answer_type = str(row.get("answer_type", "choice"))
    if answer_type == "exact_text":
        if not isinstance(options, list):
            raise ValueError(f"manifest row {row_number} options must be a list")
        if gold not in ([], None):
            raise ValueError(f"manifest row {row_number} exact-text gold must be empty")
        if not isinstance(row.get("gold_text"), str) or not row["gold_text"].strip():
            raise ValueError(f"manifest row {row_number} has no exact-text gold answer")
        if row["multi_select"]:
            raise ValueError(f"manifest row {row_number} exact text cannot be multi-select")
    elif answer_type == "choice":
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(
                f"manifest row {row_number} must contain at least two options"
            )
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"manifest row {row_number} has no gold answer")
        allowed = option_letters(len(options))
        if any(letter not in allowed for letter in gold):
            raise ValueError(f"manifest row {row_number} has invalid gold answer {gold}")
        if not row["multi_select"] and len(gold) != 1:
            raise ValueError(
                f"manifest row {row_number} has multiple single-select answers"
            )
    else:
        raise ValueError(
            f"manifest row {row_number} has unsupported answer_type {answer_type!r}"
        )
    if not isinstance(row["media"], list) or not row["media"]:
        raise ValueError(f"manifest row {row_number} has no visual media")


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        validate_manifest_row(row, row_number=index)
        key = (str(row["benchmark"]), str(row["sample_id"]))
        if key in seen:
            raise ValueError(f"duplicate manifest key: {key}")
        seen.add(key)
