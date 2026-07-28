#!/usr/bin/env python3
"""Automatically select frames and prepare leakage-safe future-rollout jobs.

This script derives a private selection spec from dataset geometry, then uses
the selected camera rotation only to convert the benchmark's world-space
movement delta into camera-relative action components for GPT/Qwen image edits
and Cosmos Image2World. It never copies the question, options, answer, GT
future coordinates, pose, or projection data into a model request or public
evaluation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sampled_type_vlm_eval import (  # noqa: E402
    ROLLOUT_SCHEMA_VERSION,
    _sha256_file,
    load_fixed_questions,
)
from scripts.generate_rollout_selection_spec import (  # noqa: E402
    DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
    DEFAULT_MESH_RAY_SHORTLIST_SIZE,
    DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    generate_selection_spec,
)
from scripts.validate_rollout_manifest import L2_ROLLOUT_TYPES  # noqa: E402
from src.frame_selector import (  # noqa: E402
    FRAME_STRIDE_SCANNET,
    FRAME_STRIDE_SCANNETPP,
)


PICTURE_PROMPT_VERSION = "agent-motion-picture-v1"
VIDEO_PROMPT_VERSION = "agent-motion-video-v1"
JOB_SCHEMA_VERSION = "predictive-spatial-generation-jobs-v1"
DEFAULT_GPT_MODEL = "gpt-image-1.5"
DEFAULT_QWEN_CHECKPOINT = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_COSMOS_CHECKPOINT = "Cosmos-Predict2.5-14B/post-trained"
SAFE_CONTEXT_ROLES = {"destination_to_query_bridge", "query_reference_view"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_label(value: Any, *, field: str) -> str:
    label = " ".join(str(value or "").split())
    if not label:
        raise ValueError(f"{field} must be a non-empty object label")
    if any(char in label for char in "\r\n\t"):
        raise ValueError(f"{field} must be a single-line object label")
    return label


def _rotation_matrix(value: Any) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("camera_rotation_world_to_camera must be a finite 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-2):
        raise ValueError("camera_rotation_world_to_camera is not approximately orthonormal")
    return rotation


def world_delta_to_agent_components(
    delta_world: Any,
    camera_rotation_world_to_camera: Any,
) -> dict[str, float]:
    """Return floor-plane camera-relative movement without exposing an endpoint.

    OpenCV camera +x points image-right and +z points away from the camera.
    Camera axes are projected onto the world XY floor plane to avoid mixing
    camera pitch into horizontal motion, matching the benchmark's semantics.
    """
    delta = np.asarray(delta_world, dtype=np.float64)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("question.delta must be a finite 3-vector")
    rotation = _rotation_matrix(camera_rotation_world_to_camera)
    right_world = rotation.T[:, 0].copy()
    away_world = rotation.T[:, 2].copy()
    right_world[2] = 0.0
    away_world[2] = 0.0
    right_norm = float(np.linalg.norm(right_world))
    away_norm = float(np.linalg.norm(away_world))
    if right_norm < 1e-6 or away_norm < 1e-6:
        raise ValueError("camera ground-plane axes are degenerate")
    right_world /= right_norm
    away_world /= away_norm
    return {
        "right_m": float(np.dot(delta[:2], right_world[:2])),
        "away_m": float(np.dot(delta[:2], away_world[:2])),
        "up_m": float(delta[2]),
        "path_length_m": float(np.linalg.norm(delta)),
    }


def action_duration_seconds(
    path_length_m: float,
    *,
    speed_mps: float = 0.8,
    settle_seconds: float = 0.75,
) -> float:
    if not math.isfinite(path_length_m) or path_length_m < 0:
        raise ValueError("path_length_m must be finite and non-negative")
    if speed_mps <= 0 or settle_seconds < 0:
        raise ValueError("speed_mps must be positive and settle_seconds non-negative")
    duration = settle_seconds + path_length_m / speed_mps
    return round(min(5.0, max(2.0, duration)), 3)


def describe_agent_motion(components: dict[str, float], *, threshold_m: float = 0.03) -> str:
    phrases: list[str] = []
    right = float(components["right_m"])
    away = float(components["away_m"])
    up = float(components["up_m"])
    if abs(away) >= threshold_m:
        phrases.append(
            f"{abs(away):.2f} meters "
            + ("away from the camera" if away > 0 else "toward the camera")
        )
    if abs(right) >= threshold_m:
        phrases.append(
            f"{abs(right):.2f} meters "
            + ("toward image-right" if right > 0 else "toward image-left")
        )
    if abs(up) >= threshold_m:
        phrases.append(
            f"{abs(up):.2f} meters " + ("upward" if up > 0 else "downward")
        )
    if not phrases:
        raise ValueError("movement is too small to describe safely")
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _moving_group_labels(question: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    raw_group = spec.get("moving_group")
    has_attachments = bool(question.get("has_attachment_chain"))
    if raw_group is None:
        if has_attachments:
            raise ValueError(
                "attachment questions require selection_spec.moving_group with the full transitive chain"
            )
        raw_group = [{"label": question.get("moved_obj_label")}]
    if not isinstance(raw_group, list) or not raw_group:
        raise ValueError("moving_group must be a non-empty array")
    labels: list[str] = []
    object_ids: list[int] = []
    for index, item in enumerate(raw_group):
        if has_attachments and not isinstance(item, dict):
            raise ValueError(
                f"moving_group[{index}] must include obj_id and label for attachment questions"
            )
        label_value = item.get("label") if isinstance(item, dict) else item
        label = _safe_label(label_value, field=f"moving_group[{index}].label")
        labels.append(label)
        if has_attachments:
            raw_id = item.get("obj_id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise ValueError(f"moving_group[{index}].obj_id must be an integer")
            object_ids.append(raw_id)
    moved_label = _safe_label(question.get("moved_obj_label"), field="moved_obj_label")
    if labels[0] != moved_label:
        raise ValueError(
            f"moving_group must start with moved object {moved_label!r}, found {labels[0]!r}"
        )
    if has_attachments:
        moved_id = question.get("moved_obj_id")
        if isinstance(moved_id, bool) or not isinstance(moved_id, int):
            raise ValueError("attachment question moved_obj_id must be an integer")
        if object_ids[0] != moved_id:
            raise ValueError(
                f"moving_group must start with moved object id {moved_id}, found {object_ids[0]}"
            )
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("moving_group obj_id values must be unique")
        known_attachment_ids = {
            value
            for field in ("attachment_parent_id", "attachment_child_id")
            if isinstance((value := question.get(field)), int)
            and not isinstance(value, bool)
        }
        missing_known_ids = sorted(known_attachment_ids - set(object_ids))
        if missing_known_ids:
            raise ValueError(
                "moving_group omits benchmark-known attachment ids: "
                + ", ".join(str(value) for value in missing_known_ids)
            )
    return labels


def build_safe_action(question: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    qtype = str(question.get("type") or "")
    if qtype not in L2_ROLLOUT_TYPES:
        raise ValueError(f"unsupported rollout question type: {qtype!r}")
    moving_group_labels = _moving_group_labels(question, spec)
    components = world_delta_to_agent_components(
        question.get("delta"),
        spec.get("camera_rotation_world_to_camera"),
    )
    action: dict[str, Any] = {
        "action_type": "orbit" if qtype == "object_rotate_object_centric" else "translate",
        "moving_group_labels": moving_group_labels,
        "agent_motion_text": describe_agent_motion(components),
        "path_length_m": round(float(components["path_length_m"]), 4),
        "duration_seconds": action_duration_seconds(float(components["path_length_m"])),
        "preserve_group_geometry": True,
        "preserve_object_orientation": True,
    }
    if action["action_type"] == "orbit":
        action["orbit_anchor_label"] = _safe_label(
            spec.get("orbit_anchor_label") or question.get("obj_face_label"),
            field="orbit_anchor_label",
        )
        angle = float(question.get("rotation_angle"))
        direction = str(question.get("rotation_direction") or "").strip().lower()
        if not math.isfinite(angle) or angle <= 0 or direction not in {"clockwise", "counterclockwise"}:
            raise ValueError("orbit requires a positive rotation_angle and a valid rotation_direction")
        action["orbit_angle_degrees"] = round(angle, 3)
        action["orbit_direction"] = direction
    return action


def _group_phrase(labels: list[str]) -> str:
    return ", ".join(json.dumps(label, ensure_ascii=False) for label in labels)


def build_picture_prompt(action: dict[str, Any]) -> str:
    group = _group_phrase(action["moving_group_labels"])
    lines = [
        "Edit only the supplied image while preserving its exact camera viewpoint and composition.",
        f"The rigid moving group, in parent-to-descendant order, is: {group}.",
    ]
    if action["action_type"] == "orbit":
        lines.append(
            f"Move the group along a {action['orbit_angle_degrees']:g}-degree "
            f"{action['orbit_direction']} floor-plane orbit around "
            f'"{action["orbit_anchor_label"]}".'
        )
        lines.append(
            f"From the current camera viewpoint, its endpoint displacement is "
            f"{action['agent_motion_text']}."
        )
    else:
        lines.append(
            f"Translate the group on the floor plane {action['agent_motion_text']}."
        )
    lines.extend(
        [
            "Move every listed group member together as one rigid assembly; preserve all relative positions and orientations inside the group.",
            "Preserve the facing direction of every moved object; do not rotate objects in place.",
            "Remove the group from its old location if that location is visible; never duplicate it.",
            "Keep every unlisted object, wall, floor, furniture item, light, texture, and camera parameter unchanged.",
            "Do not add text, arrows, paths, boxes, markers, overlays, or new objects.",
            "Return only one photorealistic edited image.",
        ]
    )
    return "\n".join(lines)


def build_cosmos_prompt(action: dict[str, Any]) -> str:
    group = _group_phrase(action["moving_group_labels"])
    lines = [
        "Generate one continuous physically realistic video from the supplied condition frame.",
        "Keep the camera completely fixed: no cuts, pans, tilts, zooms, reframing, or viewpoint changes.",
        f"The rigid moving group, in parent-to-descendant order, is: {group}.",
    ]
    if action["action_type"] == "orbit":
        lines.append(
            f"Move the group smoothly along a {action['orbit_angle_degrees']:g}-degree "
            f"{action['orbit_direction']} floor-plane orbit around "
            f'"{action["orbit_anchor_label"]}".'
        )
        lines.append(
            f"From the fixed camera viewpoint, its endpoint displacement is "
            f"{action['agent_motion_text']}."
        )
    else:
        lines.append(
            f"Move the group smoothly on the floor plane {action['agent_motion_text']}."
        )
    lines.extend(
        [
            "Move every listed member together as one rigid assembly and preserve all internal relative positions and orientations.",
            "Preserve each moved object's facing direction throughout the motion.",
            f"Complete the motion naturally within about {action['duration_seconds']:g} seconds and remain still at the endpoint during the final part.",
            "All unlisted objects and the entire static scene must remain stationary and unchanged.",
            "Do not add text, arrows, trajectories, boxes, markers, overlays, or new objects.",
        ]
    )
    return "\n".join(lines)


def assert_safe_generation_job(job: dict[str, Any]) -> None:
    forbidden_keys = {
        "question",
        "options",
        "answer",
        "correct_value",
        "correct_values",
        "future_position",
        "future_coordinates",
        "future_bbox",
        "future_projection",
    }

    def _walk(value: Any, location: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    raise ValueError(f"generation job contains forbidden field {location}.{key}")
                _walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _walk(child, f"{location}[{index}]")

    _walk(job)


def _resolve_spec_path(raw_path: Any, spec_path: Path) -> Path:
    path = Path(str(raw_path or ""))
    if not str(path):
        raise ValueError("motion_frame_path is required")
    return path if path.is_absolute() else (spec_path.parent / path).resolve()


def _context_media(spec: dict[str, Any], spec_path: Path) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for index, item in enumerate(spec.get("answer_context_media") or []):
        if not isinstance(item, dict):
            raise ValueError(f"answer_context_media[{index}] must be an object")
        role = str(item.get("role") or "")
        if role not in SAFE_CONTEXT_ROLES:
            raise ValueError(
                f"answer_context_media[{index}].role must be one of {sorted(SAFE_CONTEXT_ROLES)}"
            )
        path = _resolve_spec_path(item.get("path"), spec_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        media.append(
            {
                "path": str(path),
                "role": role,
                "kind": "context",
                "sha256": _sha256_file(path),
            }
        )
    role_order = {"destination_to_query_bridge": 0, "query_reference_view": 1}
    if [role_order[item["role"]] for item in media] != sorted(
        role_order[item["role"]] for item in media
    ):
        raise ValueError("answer_context_media roles are out of order")
    return media


def _generation_provenance(
    *, model: str, checkpoint: str, seed: int, prompt_version: str, request_sha256: str
) -> dict[str, Any]:
    return {
        "model": model,
        "checkpoint": checkpoint,
        "seed": seed,
        "prompt_version": prompt_version,
        "request_sha256": request_sha256,
        "response_id": None,
        "elapsed_seconds": None,
        "retries": 0,
        "status": "pending",
    }


def prepare_jobs(
    *,
    benchmark_path: Path,
    selection_spec_path: Path,
    output_dir: Path,
    seed: int,
    expected_picture_per_type: int = 50,
) -> dict[str, Path]:
    questions, _, _ = load_fixed_questions(benchmark_path)
    selection_payload = json.loads(selection_spec_path.read_text(encoding="utf-8"))
    raw_specs = selection_payload.get("entries") if isinstance(selection_payload, dict) else None
    if not isinstance(raw_specs, list):
        raise ValueError("selection spec must contain an entries array")
    questions_by_uid = {str(question["question_uid"]): question for question in questions}

    private_dir = output_dir / "private_jobs"
    manifest_dir = output_dir / "manifests"
    gpt_media_dir = output_dir / "media" / "gpt"
    qwen_media_dir = output_dir / "media" / "qwen"
    cosmos_media_dir = output_dir / "media" / "cosmos"
    cosmos_input_dir = private_dir / "cosmos_inputs"
    for directory in (
        private_dir,
        manifest_dir,
        gpt_media_dir,
        qwen_media_dir,
        cosmos_media_dir,
        cosmos_input_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    gpt_jobs: list[dict[str, Any]] = []
    qwen_jobs: list[dict[str, Any]] = []
    cosmos_jobs: list[dict[str, Any]] = []
    picture_entries: dict[str, list[dict[str, Any]]] = {"gpt": [], "qwen": []}
    video_entries: list[dict[str, Any]] = []
    picture_counts: Counter[str] = Counter()

    for spec_index, spec in enumerate(raw_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"selection entries[{spec_index}] must be an object")
        uid = str(spec.get("question_uid") or "")
        question = questions_by_uid.get(uid)
        if question is None:
            source_index = spec.get("source_index")
            if isinstance(source_index, int) and 0 <= source_index < len(questions):
                question = questions[source_index]
                uid = str(question["question_uid"])
            else:
                raise ValueError(f"selection entry {spec_index} does not match a benchmark question")
        qtype = str(question.get("type") or "")
        if qtype not in L2_ROLLOUT_TYPES:
            raise ValueError(f"{uid}: unsupported question type {qtype!r}")

        motion_path = _resolve_spec_path(spec.get("motion_frame_path"), selection_spec_path)
        if not motion_path.is_file():
            raise FileNotFoundError(motion_path)
        motion_sha = _sha256_file(motion_path)
        action = build_safe_action(question, spec)
        picture_prompt = build_picture_prompt(action)
        cosmos_prompt = build_cosmos_prompt(action)
        context_tail = _context_media(spec, selection_spec_path)
        common_identity = {
            "question_uid": uid,
            "question_type": qtype,
            "scene_id": question.get("scene_id"),
        }

        picture_eligible = bool(spec.get("picture_eligible", True))
        picture_reasons = list(spec.get("picture_rejection_reasons") or [])
        video_eligible = bool(spec.get("video_eligible", True))
        video_reasons = list(spec.get("video_rejection_reasons") or [])

        for backend, model, checkpoint, media_dir, jobs in (
            ("gpt", DEFAULT_GPT_MODEL, DEFAULT_GPT_MODEL, gpt_media_dir, gpt_jobs),
            ("qwen", "qwen-image-edit", DEFAULT_QWEN_CHECKPOINT, qwen_media_dir, qwen_jobs),
        ):
            output_path = (media_dir / f"{uid}.png").resolve()
            request_core = {
                "model": model,
                "checkpoint": checkpoint,
                "seed": seed,
                "prompt_version": PICTURE_PROMPT_VERSION,
                "input_image_sha256": motion_sha,
                "prompt": picture_prompt,
            }
            request_sha = sha256_json(request_core)
            job = {
                **common_identity,
                "backend": backend,
                "model": model,
                "checkpoint": checkpoint,
                "seed": seed,
                "prompt_version": PICTURE_PROMPT_VERSION,
                "input_image_path": str(motion_path.resolve()),
                "input_image_sha256": motion_sha,
                "output_path": str(output_path),
                "prompt": picture_prompt,
                "request_sha256": request_sha,
            }
            assert_safe_generation_job(job)
            if picture_eligible:
                jobs.append(job)
                picture_counts[qtype] += 1 if backend == "gpt" else 0
                media = [
                    {
                        "path": str(motion_path.resolve()),
                        "role": "motion_reference_view",
                        "kind": "context",
                        "sha256": motion_sha,
                    },
                    {
                        "path": str(output_path),
                        "role": "predicted_future_view",
                        "kind": "prediction",
                    },
                    *context_tail,
                ]
            else:
                media = []
            picture_entries[backend].append(
                {
                    **common_identity,
                    "picture": {
                        "eligible": picture_eligible,
                        "rejection_reasons": picture_reasons,
                        "media": media,
                        "generation": _generation_provenance(
                            model=model,
                            checkpoint=checkpoint,
                            seed=seed,
                            prompt_version=PICTURE_PROMPT_VERSION,
                            request_sha256=request_sha,
                        ),
                    },
                }
            )

        video_output = (cosmos_media_dir / f"{uid}.mp4").resolve()
        frame_dir = (cosmos_media_dir / uid / "frames").resolve()
        cosmos_request_core = {
            "model": "cosmos-predict2.5",
            "checkpoint": DEFAULT_COSMOS_CHECKPOINT,
            "seed": seed,
            "prompt_version": VIDEO_PROMPT_VERSION,
            "input_image_sha256": motion_sha,
            "prompt": cosmos_prompt,
            "duration_seconds": action["duration_seconds"],
        }
        cosmos_request_sha = sha256_json(cosmos_request_core)
        cosmos_job = {
            **common_identity,
            "backend": "cosmos",
            "model": "cosmos-predict2.5",
            "checkpoint": DEFAULT_COSMOS_CHECKPOINT,
            "seed": seed,
            "prompt_version": VIDEO_PROMPT_VERSION,
            "input_image_path": str(motion_path.resolve()),
            "input_image_sha256": motion_sha,
            "output_path": str(video_output),
            "frame_output_dir": str(frame_dir),
            "duration_seconds": action["duration_seconds"],
            "prompt": cosmos_prompt,
            "request_sha256": cosmos_request_sha,
        }
        assert_safe_generation_job(cosmos_job)
        if video_eligible:
            cosmos_jobs.append(cosmos_job)
            official_input = {
                "inference_type": "image2world",
                "name": uid,
                "prompt": cosmos_prompt,
                "input_path": str(motion_path.resolve()),
            }
            (cosmos_input_dir / f"{uid}.json").write_text(
                json.dumps(official_input, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            video_media = [
                {
                    "path": str(motion_path.resolve()),
                    "role": "motion_reference_view",
                    "kind": "context",
                    "sha256": motion_sha,
                },
                *[
                    {
                        "path": str(frame_dir / f"frame_{index:02d}.jpg"),
                        "role": "predicted_video_frame",
                        "kind": "prediction",
                        "frame_index": index,
                    }
                    for index in range(8)
                ],
                *context_tail,
            ]
        else:
            video_media = []
        video_entries.append(
            {
                **common_identity,
                "video": {
                    "eligible": video_eligible,
                    "rejection_reasons": video_reasons,
                    "media": video_media,
                    "target_duration_seconds": action["duration_seconds"],
                    "generation": _generation_provenance(
                        model="cosmos-predict2.5",
                        checkpoint=DEFAULT_COSMOS_CHECKPOINT,
                        seed=seed,
                        prompt_version=VIDEO_PROMPT_VERSION,
                        request_sha256=cosmos_request_sha,
                    ),
                },
            }
        )

    if expected_picture_per_type > 0:
        excess_counts = {
            qtype: picture_counts[qtype]
            for qtype in L2_ROLLOUT_TYPES
            if picture_counts[qtype] > expected_picture_per_type
        }
        if excess_counts:
            raise ValueError(
                f"picture sample exceeds the {expected_picture_per_type} per-type cap; "
                f"excess={excess_counts}"
            )

    job_metadata = {
        "schema_version": JOB_SCHEMA_VERSION,
        "benchmark_sha256": _sha256_file(benchmark_path),
        "selection_spec_sha256": _sha256_file(selection_spec_path),
        "seed": seed,
    }
    job_paths: dict[str, Path] = {}
    for backend, jobs in (("gpt", gpt_jobs), ("qwen", qwen_jobs), ("cosmos", cosmos_jobs)):
        path = private_dir / f"{backend}_jobs.json"
        path.write_text(
            json.dumps({**job_metadata, "backend": backend, "entries": jobs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        job_paths[f"{backend}_jobs"] = path

    public_metadata = {
        "prompt_version_picture": PICTURE_PROMPT_VERSION,
        "prompt_version_video": VIDEO_PROMPT_VERSION,
        "seed": seed,
        "benchmark_sha256": _sha256_file(benchmark_path),
    }
    for backend in ("gpt", "qwen"):
        path = manifest_dir / f"{backend}_picture.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": ROLLOUT_SCHEMA_VERSION,
                    "metadata": {**public_metadata, "generator": backend},
                    "entries": picture_entries[backend],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        job_paths[f"{backend}_manifest"] = path
    cosmos_manifest_path = manifest_dir / "cosmos_video.json"
    cosmos_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": ROLLOUT_SCHEMA_VERSION,
                "metadata": {**public_metadata, "generator": "cosmos-predict2.5"},
                "entries": video_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    job_paths["cosmos_manifest"] = cosmos_manifest_path
    job_paths["cosmos_inputs"] = cosmos_input_dir
    return job_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scannet_root", type=Path, default=None)
    parser.add_argument("--scannetpp_root", type=Path, default=None)
    parser.add_argument("--scannetpp_frame_root", type=Path, default=None)
    parser.add_argument("--scannetpp_sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--frame_stride_scannet", type=int, default=FRAME_STRIDE_SCANNET)
    parser.add_argument("--frame_stride_scannetpp", type=int, default=FRAME_STRIDE_SCANNETPP)
    parser.add_argument(
        "--mesh_ray_shortlist_size",
        type=int,
        default=DEFAULT_MESH_RAY_SHORTLIST_SIZE,
    )
    parser.add_argument(
        "--mesh_ray_surface_samples",
        type=int,
        default=DEFAULT_MESH_RAY_SURFACE_SAMPLES,
    )
    parser.add_argument(
        "--mesh_ray_local_resamples",
        type=int,
        default=DEFAULT_MESH_RAY_LOCAL_RESAMPLES,
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--expected_picture_per_type",
        type=int,
        default=50,
        help="maximum number of automatically selected questions per supported type",
    )
    args = parser.parse_args(argv)
    if not args.benchmark_file.is_file():
        parser.error(f"--benchmark_file not found: {args.benchmark_file}")
    if args.expected_picture_per_type <= 0:
        parser.error("--expected_picture_per_type must be positive")
    if args.frame_stride_scannet <= 0 or args.frame_stride_scannetpp <= 0:
        parser.error("frame strides must be positive")
    if args.mesh_ray_shortlist_size <= 0:
        parser.error("--mesh_ray_shortlist_size must be positive")
    if args.mesh_ray_surface_samples <= 0:
        parser.error("--mesh_ray_surface_samples must be positive")
    if args.mesh_ray_local_resamples < 0:
        parser.error("--mesh_ray_local_resamples must be non-negative")
    for field in ("scannet_root", "scannetpp_root", "scannetpp_frame_root"):
        path = getattr(args, field)
        if path is not None and not path.is_dir():
            parser.error(f"--{field} is not a directory: {path}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selection_paths = generate_selection_spec(
        benchmark_path=args.benchmark_file,
        output_dir=args.output_dir,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_frame_root=args.scannetpp_frame_root,
        scannetpp_sensor=args.scannetpp_sensor,
        expected_per_type=args.expected_picture_per_type,
        frame_stride_scannet=args.frame_stride_scannet,
        frame_stride_scannetpp=args.frame_stride_scannetpp,
        mesh_ray_shortlist_size=args.mesh_ray_shortlist_size,
        mesh_ray_surface_samples=args.mesh_ray_surface_samples,
        mesh_ray_local_resamples=args.mesh_ray_local_resamples,
    )
    outputs = prepare_jobs(
        benchmark_path=args.benchmark_file,
        selection_spec_path=selection_paths.spec,
        output_dir=args.output_dir,
        seed=args.seed,
        expected_picture_per_type=args.expected_picture_per_type,
    )
    print(f"{'selection_spec':20s}: {selection_paths.spec}")
    print(f"{'selection_audit':20s}: {selection_paths.audit}")
    for name, path in outputs.items():
        print(f"{name:20s}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
