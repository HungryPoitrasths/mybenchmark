#!/usr/bin/env python3
"""Find strong duplicate questions in the merged review HTML."""

from __future__ import annotations

import argparse
import html
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "edited" / "merged_1-4081.html"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "edited" / "strong_duplicate_questions.txt"

DIV_TOKEN_RE = re.compile(r"<div\b[^>]*>|</div>", flags=re.IGNORECASE)
CARD_START_RE = re.compile(
    r'<div\b(?=[^>]*\bclass=(["\'])[^"\']*\bcard\b[^"\']*\1)[^>]*>',
    flags=re.IGNORECASE,
)
IDX_RE = re.compile(r'<span class="idx">#(?P<idx>\d+)</span>', flags=re.IGNORECASE)
LEVEL_RE = re.compile(r'<span class="badge (?P<level>L\d)">(?P=level)</span>', flags=re.IGNORECASE)
TYPE_RE = re.compile(r'<span class="badge extra">(?P<type>[^<]+)</span>', flags=re.IGNORECASE)
FOOTER_RE = re.compile(r'<div class="footer">\s*(?P<scene>[^&<]+)', flags=re.IGNORECASE)
SIMPLE_ITEM_RE = re.compile(
    r'<div class="simple-item">\s*'
    r'<div class="simple-key">(?P<key>.*?)</div>\s*'
    r'<div class="simple-value">(?P<value>.*?)</div>\s*'
    r"</div>",
    flags=re.IGNORECASE | re.DOTALL,
)

ROLE_KEYS = (
    "query",
    "reference",
    "moved",
    "obj_a",
    "obj_b",
    "obj_c",
    "facing",
    "root",
    "parent",
    "child",
    "grandchild",
    "displaced",
    "camera",
    "rotation_angle",
)
ANSWER_KEYS = (
    "direction",
    "old_direction",
    "new_direction",
    "distance_bin",
    "old_distance_bin",
    "new_distance_bin",
    "distance_m",
    "old_distance_m",
    "new_distance_m",
    "occlusion_status",
    "old_occlusion",
    "new_occlusion",
    "gt",
)


@dataclass(frozen=True)
class CardInfo:
    idx: int
    level: str
    question_type: str
    scene: str
    scene_base: str
    roles: tuple[tuple[str, str], ...]
    answers: tuple[tuple[str, str], ...]


def _iter_card_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    search_from = 0
    while True:
        start_match = CARD_START_RE.search(text, search_from)
        if start_match is None:
            return ranges

        depth = 0
        for match in DIV_TOKEN_RE.finditer(text, start_match.start()):
            token = match.group(0).lower()
            if token.startswith("<div"):
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    ranges.append((start_match.start(), match.end()))
                    search_from = match.end()
                    break
        else:
            raise ValueError(f"Unbalanced card div at byte offset {start_match.start()}")


def _clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def _card_index(card_html: str) -> int:
    match = IDX_RE.search(card_html)
    if not match:
        raise ValueError("Card is missing .idx metadata")
    return int(match.group("idx"))


def _card_level(card_html: str) -> str:
    match = LEVEL_RE.search(card_html)
    if not match:
        raise ValueError(f"Card #{_card_index(card_html)} is missing L1/L2/L3 badge")
    return match.group("level").upper()


def _card_type(card_html: str) -> str:
    match = TYPE_RE.search(card_html)
    if not match:
        raise ValueError(f"Card #{_card_index(card_html)} is missing type badge")
    return _clean(match.group("type"))


def _card_scene(card_html: str) -> str:
    match = FOOTER_RE.search(card_html)
    if not match:
        raise ValueError(f"Card #{_card_index(card_html)} is missing footer scene")
    return html.unescape(match.group("scene")).strip()


def _scene_base(scene: str) -> str:
    match = re.fullmatch(r"(scene\d+)_\d+", scene)
    if match:
        return match.group(1)
    return scene


def _simple_fields(card_html: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in SIMPLE_ITEM_RE.finditer(card_html):
        key = _clean(match.group("key")).lower()
        value = _clean(match.group("value")).lower()
        fields[key] = value
    return fields


def _card_info(card_html: str) -> CardInfo:
    fields = _simple_fields(card_html)
    roles = tuple((key, fields[key]) for key in ROLE_KEYS if key in fields)
    answers = tuple((key, fields[key]) for key in ANSWER_KEYS if key in fields)
    scene = _card_scene(card_html)
    return CardInfo(
        idx=_card_index(card_html),
        level=_card_level(card_html),
        question_type=_card_type(card_html),
        scene=scene,
        scene_base=_scene_base(scene),
        roles=roles,
        answers=answers,
    )


def _format_pairs(pairs: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"{key}={value}" for key, value in pairs) if pairs else "-"


def _format_summary(info: CardInfo) -> str:
    fields = dict(info.roles)
    if "query" in fields and "reference" in fields:
        core = f"{fields['query']} -> {fields['reference']}"
        extras = [(key, value) for key, value in info.roles if key not in {"query", "reference"}]
        if extras:
            core += f" ({_format_pairs(tuple(extras))})"
    elif "obj_a" in fields and "obj_b" in fields:
        core = f"{fields['obj_a']} <-> {fields['obj_b']}"
        extras = [(key, value) for key, value in info.roles if key not in {"obj_a", "obj_b"}]
        if extras:
            core += f" ({_format_pairs(tuple(extras))})"
    elif "obj_b" in fields and "obj_c" in fields:
        core = f"{fields['obj_b']} <-> {fields['obj_c']}"
        extras = [(key, value) for key, value in info.roles if key not in {"obj_b", "obj_c"}]
        if extras:
            core += f" ({_format_pairs(tuple(extras))})"
    else:
        core = _format_pairs(info.roles)
    return f"{info.scene_base} | {info.question_type} | {core} | {_format_pairs(info.answers)}"


def find_strong_duplicates(input_path: Path) -> tuple[list[CardInfo], list[tuple[CardInfo, list[int]]]]:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    infos: list[CardInfo] = []
    groups: defaultdict[tuple[object, ...], list[CardInfo]] = defaultdict(list)

    for start, end in _iter_card_ranges(text):
        info = _card_info(text[start:end])
        infos.append(info)
        if not info.roles or not info.answers:
            continue
        key = (info.scene_base, info.question_type, info.roles, info.answers)
        groups[key].append(info)

    duplicates: list[tuple[CardInfo, list[int]]] = []
    for group_infos in groups.values():
        if len(group_infos) < 2:
            continue
        ordered = sorted(group_infos, key=lambda item: item.idx)
        duplicates.append((ordered[0], [item.idx for item in ordered]))
    duplicates.sort(key=lambda item: item[1][0])
    return infos, duplicates


def write_report(input_path: Path, output_path: Path) -> dict[str, object]:
    infos, duplicates = find_strong_duplicates(input_path)
    level_counts = Counter(info.level for info in infos)
    duplicate_question_count = sum(len(indices) for _, indices in duplicates)

    lines: list[str] = [
        "Strong duplicate questions",
        f"Input: {input_path}",
        f"Cards scanned: {len(infos)}",
        f"Levels: L1={level_counts['L1']}, L2={level_counts['L2']}, L3={level_counts['L3']}",
        f"Duplicate groups: {len(duplicates)}",
        f"Question placements in duplicate groups: {duplicate_question_count}",
        "",
    ]
    for info, indices in duplicates:
        ids = ", ".join(f"#{idx}" for idx in indices)
        lines.append(f"- {_format_summary(info)}: {ids}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "cards_scanned": len(infos),
        "levels": dict(sorted(level_counts.items())),
        "duplicate_groups": len(duplicates),
        "duplicate_question_count": duplicate_question_count,
        "first_groups": [indices for _, indices in duplicates[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = write_report(args.input, args.output)
    print(f"Wrote {result['output']}")
    print(f"Cards scanned: {result['cards_scanned']}")
    print(f"Levels: {result['levels']}")
    print(f"Duplicate groups: {result['duplicate_groups']}")
    print(f"Question placements in duplicate groups: {result['duplicate_question_count']}")
    print(f"First groups: {result['first_groups']}")


if __name__ == "__main__":
    main()
