#!/usr/bin/env python3
"""Check for duplicate object labels in benchmark questions."""
import json
from collections import Counter, defaultdict

data = json.load(open("output/pilot/benchmark.json"))
qs = data["questions"] if isinstance(data, dict) else data
per_scene = defaultdict(list)
for q in qs:
    per_scene[(q["scene_id"], q["image_name"])].append(q)
for k, group in per_scene.items():
    all_labels = [q.get("obj_a_label", "") for q in group] + [q.get("obj_b_label", "") for q in group]
    c = Counter(all_labels)
    dups = {l: n for l, n in c.items() if n > 1 and l}
    if dups:
        print(f"{k}: {dups}")
if not any(True for k, group in per_scene.items()
           for _ in [1]
           if any(n > 1 and l for l, n in Counter(
               [q.get("obj_a_label", "") for q in group] + [q.get("obj_b_label", "") for q in group]
           ).items())):
    print("No duplicate labels found!")
