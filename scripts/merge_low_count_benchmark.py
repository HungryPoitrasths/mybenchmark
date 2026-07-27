"""Merge selected low-count question types from validation benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality_control import compute_statistics
from scripts.run_pipeline import _balance_global_object_move_occlusion_three_way


INPUT_ROOTS = (
    PROJECT_ROOT / "output_val" / "scannet_polit",
    PROJECT_ROOT / "output_val" / "scannetpp_polit",
)
OUTPUT_PATH = PROJECT_ROOT / "output_val" / "polit_low_count_types" / "benchmark.json"
SELECTED_TYPES = {"attachment_move", "attachment_chain", "object_move_occlusion"}


def load_questions() -> list[dict]:
    questions: list[dict] = []
    for root in INPUT_ROOTS:
        for path in sorted(root.rglob("benchmark.json")):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            for question in payload.get("questions", []):
                if isinstance(question, dict) and question.get("type") in SELECTED_TYPES:
                    questions.append(question)
    return questions


def main() -> None:
    questions = load_questions()
    questions, occlusion_balance = _balance_global_object_move_occlusion_three_way(
        questions
    )
    statistics = compute_statistics(questions)
    output = {
        "name": "PSR-Bench",
        "version": "1.0",
        "statistics": statistics,
        "object_move_occlusion_balance": occlusion_balance,
        "questions": questions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
