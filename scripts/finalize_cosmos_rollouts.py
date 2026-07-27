#!/usr/bin/env python3
"""Collect Cosmos videos, extract eight deterministic frames, and finalize a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_future_rollout_jobs import JOB_SCHEMA_VERSION  # noqa: E402
from scripts.run_sampled_type_vlm_eval import _sha256_file  # noqa: E402


def eight_frame_indices(frame_count: int) -> list[int]:
    if frame_count < 8:
        raise ValueError(f"Cosmos output needs at least 8 frames, found {frame_count}")
    last = frame_count - 1
    indices = [(index * last + 3) // 7 for index in range(8)]
    if indices[0] != 0 or indices[-1] != last or len(set(indices)) != 8:
        raise AssertionError("eight-frame sampling failed to preserve unique endpoints")
    return indices


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def _find_video(job: dict[str, Any], video_root: Path | None) -> Path:
    expected = Path(str(job["output_path"]))
    if expected.is_file():
        return expected
    if video_root is None:
        raise FileNotFoundError(expected)
    uid = str(job["question_uid"])
    direct_candidates = [video_root / f"{uid}.mp4", video_root / uid / f"{uid}.mp4"]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(video_root.rglob(f"{uid}*.mp4"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no Cosmos video found for {uid} below {video_root}")
    raise RuntimeError(f"multiple Cosmos videos found for {uid}: {matches}")


def extract_eight_frames(video_path: Path, output_dir: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open Cosmos video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = eight_frame_indices(frame_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for output_index, source_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"failed reading source frame {source_index} from {video_path}"
                )
            output_path = output_dir / f"frame_{output_index:02d}.jpg"
            temp_path = output_dir / f"frame_{output_index:02d}.tmp.jpg"
            if not cv2.imwrite(str(temp_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed writing {temp_path}")
            temp_path.replace(output_path)
            written.append(output_path)
    finally:
        capture.release()
    return {
        "source_frame_count": frame_count,
        "source_fps": fps,
        "actual_duration_seconds": round(frame_count / fps, 3) if fps > 0 else None,
        "sampled_source_indices": indices,
        "frame_paths": written,
    }


def _finalize_manifest_entry(
    manifest: dict[str, Any],
    *,
    job: dict[str, Any],
    video_path: Path,
    extraction: dict[str, Any],
) -> None:
    uid = str(job["question_uid"])
    for entry in manifest.get("entries", []):
        if str(entry.get("question_uid")) != uid:
            continue
        branch = entry.get("video")
        if not isinstance(branch, dict):
            raise ValueError(f"{uid}: public manifest has no video branch")
        generation = branch.get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"{uid}: public manifest has no generation provenance")
        if generation.get("request_sha256") != job.get("request_sha256"):
            raise ValueError(f"{uid}: request hash differs between job and manifest")
        generation.update(
            {
                "status": "succeeded",
                "response_id": None,
                "elapsed_seconds": None,
                "retries": 0,
                "error": None,
                "source_video_path": str(video_path.resolve()),
                "source_video_sha256": _sha256_file(video_path),
                "source_frame_count": extraction["source_frame_count"],
                "source_fps": extraction["source_fps"],
                "actual_duration_seconds": extraction["actual_duration_seconds"],
                "sampled_source_indices": extraction["sampled_source_indices"],
            }
        )
        prediction_items = [
            item for item in branch.get("media", []) if item.get("kind") == "prediction"
        ]
        if len(prediction_items) != 8:
            raise ValueError(f"{uid}: public manifest does not contain eight prediction frames")
        for index, (item, frame_path) in enumerate(
            zip(prediction_items, extraction["frame_paths"])
        ):
            if item.get("frame_index") != index:
                raise ValueError(f"{uid}: prediction frame indices are not 0..7")
            item["path"] = str(frame_path.resolve())
            item["sha256"] = _sha256_file(frame_path)
        return
    raise ValueError(f"{uid}: job is absent from public manifest")


def finalize_jobs(
    *, jobs_path: Path, manifest_path: Path, video_root: Path | None
) -> dict[str, int]:
    jobs_payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    if jobs_payload.get("schema_version") != JOB_SCHEMA_VERSION:
        raise ValueError(f"unsupported job schema in {jobs_path}")
    jobs = jobs_payload.get("entries")
    if not isinstance(jobs, list):
        raise ValueError("job entries must be an array")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = {"finalized": 0, "failed": 0}
    for job in jobs:
        uid = str(job["question_uid"])
        try:
            video_path = _find_video(job, video_root)
            extraction = extract_eight_frames(
                video_path, Path(str(job["frame_output_dir"]))
            )
            _finalize_manifest_entry(
                manifest,
                job=job,
                video_path=video_path,
                extraction=extraction,
            )
            stats["finalized"] += 1
        except Exception as exc:
            stats["failed"] += 1
            for entry in manifest.get("entries", []):
                if str(entry.get("question_uid")) == uid:
                    generation = (entry.get("video") or {}).get("generation")
                    if isinstance(generation, dict):
                        generation.update({"status": "failed", "error": str(exc)})
                    break
        _atomic_write_json(manifest_path, manifest)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video_root", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.jobs.is_file():
        parser.error(f"--jobs not found: {args.jobs}")
    if not args.manifest.is_file():
        parser.error(f"--manifest not found: {args.manifest}")
    if args.video_root is not None and not args.video_root.is_dir():
        parser.error(f"--video_root not found: {args.video_root}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats = finalize_jobs(
        jobs_path=args.jobs,
        manifest_path=args.manifest,
        video_root=args.video_root,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
