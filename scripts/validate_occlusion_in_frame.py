from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import make_data_source
from src.qa_generator import _bbox_fully_in_frame

SCANNET_RE = re.compile(r"scene\d{4}_\d{2}")
SCANNETPP_RE = re.compile(r"[0-9a-f]{8,}$")


def classify(scene_id: str) -> tuple[str, str | None]:
    if SCANNET_RE.fullmatch(scene_id):
        return "scannet", None
    if SCANNETPP_RE.fullmatch(scene_id):
        return "scannetpp", "iphone"
    raise ValueError(f"unrecognized scene_id style: {scene_id}")


def _load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("subset JSON must contain a list under key 'questions'")
    return [q for q in questions if isinstance(q, dict)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="output/benchmark_subset.json")
    parser.add_argument("--scannet_root", required=True, help="dir of ScanNet scene subdirs")
    parser.add_argument("--scannetpp_root", required=True, help="dir of ScanNet++ scene subdirs")
    parser.add_argument("--dropped_out", default="output/occlusion_dropped_ids.json")
    args = parser.parse_args()

    questions = _load_questions(Path(args.subset))
    occlusion_questions = [
        q for q in questions if q.get("type") == "object_move_occlusion"
    ]

    cache: dict[str, tuple[dict[int, dict[str, Any]], Any, dict[str, Any]]] = {}
    kept: list[str] = []
    dropped: list[str] = []
    errors: list[tuple[str | None, str, str, str]] = []
    proxy_partial_ids: set[str] = set()

    for q in occlusion_questions:
        scene_id = str(q.get("scene_id"))
        image_name = str(q.get("image_name"))
        trace_qid = q.get("trace_question_id")
        target_obj_id = q.get("target_obj_id")
        if not isinstance(trace_qid, str):
            trace_qid = None
        try:
            target_obj_id_int = int(target_obj_id)
        except (TypeError, ValueError):
            errors.append((trace_qid, scene_id, image_name, "invalid target_obj_id"))
            continue

        metrics = q.get("old_visibility_metrics")
        if isinstance(metrics, dict):
            try:
                in_frame_ratio = float(metrics.get("in_frame_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                in_frame_ratio = 0.0
        else:
            in_frame_ratio = 0.0
        if in_frame_ratio < 0.999 and trace_qid is not None:
            proxy_partial_ids.add(trace_qid)

        try:
            dataset, sensor = classify(scene_id)
            root = Path(args.scannet_root if dataset == "scannet" else args.scannetpp_root)
            if scene_id not in cache:
                if dataset == "scannet":
                    data_source = make_data_source(dataset, root / scene_id)
                else:
                    data_source = make_data_source(dataset, root / scene_id, sensor=sensor or "iphone")
                scene = data_source.load_scene()
                objects = scene.get("objects", [])
                obj_by_id = {
                    int(obj["id"]): obj
                    for obj in objects
                    if isinstance(obj, dict) and "id" in obj
                }
                cache[scene_id] = (obj_by_id, data_source.load_intrinsics(), data_source.load_poses())

            obj_by_id, intrinsics, poses = cache[scene_id]
            target_obj = obj_by_id.get(target_obj_id_int)
            camera_pose = poses.get(image_name)
            if target_obj is None or camera_pose is None:
                errors.append((
                    trace_qid,
                    scene_id,
                    image_name,
                    f"missing obj={target_obj is None} pose={camera_pose is None}",
                ))
                continue
            in_frame = _bbox_fully_in_frame(target_obj, camera_pose, intrinsics)
            if in_frame:
                if trace_qid is not None:
                    kept.append(trace_qid)
            elif trace_qid is not None:
                dropped.append(trace_qid)
        except Exception as exc:  # pragma: no cover - defensive reporting path
            errors.append((trace_qid, scene_id, image_name, repr(exc)))

    dropped_set = set(dropped)
    error_qids = {qid for (qid, _, _, _) in errors if qid is not None}
    missing = proxy_partial_ids - dropped_set - error_qids

    print(f"object_move_occlusion: {len(occlusion_questions)}")
    print(f"  KEPT  (8 corners in-frame): {len(kept)}")
    print(f"  DROPPED (some corner out):  {len(dropped_set)}")
    print(f"  ERRORS (unloadable):        {len(errors)}")
    print(
        f"  proxy(<1.0 surface ratio) count: {len(proxy_partial_ids)}; "
        f"NOT in dropped (expect 0 ideally): {len(missing)}"
    )
    if missing:
        print("  WARN proxy-partial-but-kept (investigate):", sorted(missing)[:10])
    for error in errors[:15]:
        print("  ERR", error)

    output_payload = {
        "kept": sorted(set(kept)),
        "dropped": sorted(dropped_set),
        "errors": errors,
        "proxy_partial": sorted(proxy_partial_ids),
    }
    dropped_out = Path(args.dropped_out)
    dropped_out.parent.mkdir(parents=True, exist_ok=True)
    dropped_out.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {dropped_out}")


if __name__ == "__main__":
    main()
