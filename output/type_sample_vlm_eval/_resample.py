"""Resample per-type question lists from benchmark_subset.json.

For each question type:
- Sample 50 questions (or all if <50 available)
- Maximize scene diversity: prefer 1 per scene, then relax
- Write per-type JSON files in the same format as existing files.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent.parent / "benchmark_subset.json"
OUT_DIR = Path(__file__).resolve().parent

QTYPE_DISPLAY = {
    "direction_agent": "L1_direction_agent",
    "occlusion": "L1_occlusion",
    "distance": "L1_distance",
    "direction_object_centric": "L1_direction_object_centric",
    "direction_allocentric": "L1_direction_allocentric",
    "object_move_agent": "L2_object_move_agent",
    "object_move_distance": "L2_object_move_distance",
    "object_move_object_centric": "L2_object_move_object_centric",
    "object_rotate_object_centric": "L2_object_rotate_object_centric",
    "object_move_allocentric": "L2_object_move_allocentric",
    "object_remove": "L2_object_remove",
    "attachment_chain": "L3_attachment_chain",
    "attachment_move": "L3_attachment_move",
    "coordinate_rotation_agent": "L3_coordinate_rotation_agent",
    "coordinate_rotation_object_centric": "L3_coordinate_rotation_object_centric",
    "coordinate_rotation_allocentric": "L3_coordinate_rotation_allocentric",
}

PER_TYPE = 50
SEED = 20260602

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sample_per_type(questions: list[dict], per_type: int, rng: random.Random):
    """Sample up to per_type questions per type, maximizing scene diversity."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q)

    sampled: dict[str, list[dict]] = {}
    stats: dict[str, dict] = {}

    for qtype in sorted(by_type):
        group = list(by_type[qtype])
        rng.shuffle(group)

        chosen: list[dict] = []
        chosen_uids: set[str] = set()
        per_scene: dict[str, int] = defaultdict(int)

        # Phase 1: scene_cap=1, max diversity
        for q in group:
            if len(chosen) >= per_type:
                break
            scene = q["scene_id"]
            if per_scene[scene] >= 1:
                continue
            # Build unique ID
            uid = f'{q["scene_id"]}|{q["image_name"]}|{q["type"]}|{q["question"]}'
            if uid in chosen_uids:
                continue
            chosen.append(q)
            chosen_uids.add(uid)
            per_scene[scene] += 1

        # Phase 2: relax scene cap, fill remaining
        relaxed = 0
        if len(chosen) < per_type:
            for q in group:
                if len(chosen) >= per_type:
                    break
                uid = f'{q["scene_id"]}|{q["image_name"]}|{q["type"]}|{q["question"]}'
                if uid in chosen_uids:
                    continue
                chosen.append(q)
                chosen_uids.add(uid)
                relaxed += 1

        sampled[qtype] = chosen
        scenes_used = len({q["scene_id"] for q in chosen})
        avg_per_scene = len(chosen) / max(scenes_used, 1)
        stats[qtype] = {
            "available": len(group),
            "total_scenes_available": len({q["scene_id"] for q in group}),
            "sampled": len(chosen),
            "scenes_used": scenes_used,
            "relaxed": relaxed,
            "avg_per_scene": round(avg_per_scene, 2),
        }
        print(
            f"  {qtype:40s}: {len(chosen):3d}/{len(group):4d} from {scenes_used:3d} scenes"
            f" (available: {stats[qtype]['total_scenes_available']} scenes)"
            f"  relaxed={relaxed}"
        )

    return sampled, stats


def benchmark_to_display(q: dict, idx: int, display_type: str) -> dict:
    """Convert a benchmark question to the per-type display format."""
    options_raw = q.get("options", [])
    answer = str(q.get("answer", "")).strip().upper()

    # Build options with letter, text, is_gold
    options_display = []
    for i, opt_text in enumerate(options_raw):
        letter = ALPHABET[i]
        options_display.append({
            "letter": letter,
            "text": str(opt_text),
            "is_gold": letter == answer,
        })

    # Determine dataset from scene_id prefix
    scene_id = q.get("scene_id", "")
    dataset = "scannetpp" if len(scene_id) < 12 and "_" not in scene_id else "scannet"

    return {
        "result": None,
        "image_base64": "",
        "id": idx,
        "question_type": display_type,
        "dataset": dataset,
        "scene": scene_id,
        "frame": q.get("image_name", ""),
        "question": q.get("question", ""),
        "gold_answer": answer,
        "options": options_display,
        "gt_letter": answer,
        "model_letter": None,
        "correct_value": q.get("correct_value", ""),
        "model_reasoning": None,
        "model_raw_answer": None,
    }


def main():
    print(f"Loading benchmark from {BENCHMARK}")
    with open(BENCHMARK, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"Total questions: {len(questions)}")

    type_dist = defaultdict(int)
    for q in questions:
        type_dist[q["type"]] += 1
    print(f"Types: {len(type_dist)}")

    rng = random.Random(SEED)
    sampled, stats = sample_per_type(questions, PER_TYPE, rng)

    # Write per-type files
    total_written = 0
    for qtype, group in sampled.items():
        display_type = QTYPE_DISPLAY.get(qtype, qtype)
        items = []
        for idx, q in enumerate(group, 1):
            items.append(benchmark_to_display(q, idx, display_type))

        out_path = OUT_DIR / f"{display_type}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(items)} items to {out_path.name}")
        total_written += len(items)

    print(f"\nTotal written: {total_written} questions across {len(sampled)} types")

    # Print stats summary
    print("\n=== Sampling Summary ===")
    print(f"{'Type':40s} {'Sampled':>7s} {'Scenes':>7s} {'Avail.Scn':>10s} {'Avg/Scn':>8s}")
    print("-" * 75)
    for qtype, s in sorted(stats.items()):
        print(
            f"{qtype:40s} {s['sampled']:>7d} {s['scenes_used']:>7d}"
            f" {s['total_scenes_available']:>10d} {s['avg_per_scene']:>8.2f}"
        )

    # Highlight types with <50 available
    print("\nTypes with <50 available:")
    for qtype, s in sorted(stats.items()):
        if s["available"] < PER_TYPE:
            print(f"  {qtype}: only {s['available']} available (sampled {s['sampled']})")


if __name__ == "__main__":
    main()
