#!/usr/bin/env python3
"""Run resumable GPT Image or Qwen Image Edit future-state generation jobs."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib.request import urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_future_rollout_jobs import (  # noqa: E402
    DEFAULT_GPT_MODEL,
    DEFAULT_QWEN_CHECKPOINT,
    JOB_SCHEMA_VERSION,
    assert_safe_generation_job,
)
from scripts.run_sampled_type_vlm_eval import _sha256_file  # noqa: E402


GenerateFn = Callable[[dict[str, Any]], tuple[Image.Image, str | None]]


class RetryableGenerationError(RuntimeError):
    """A transport, empty-response, or corrupt-media failure safe to retry."""


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RetryableGenerationError):
        return True
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "timeout",
            "timed out",
            "overloaded",
            "connection reset",
            "connection error",
        )
    )


def _atomic_save_image(image: Image.Image, output_path: Path, target_size: tuple[int, int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = image.convert("RGB")
    if normalized.size != target_size:
        normalized = normalized.resize(target_size, Image.Resampling.LANCZOS)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    normalized.save(temp_path, format="PNG")
    temp_path.replace(output_path)


def _valid_image(path: Path, target_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == target_size
    except (OSError, ValueError):
        return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def _cached_output_matches(
    manifest: dict[str, Any],
    *,
    uid: str,
    request_sha256: str,
    output_path: Path,
    target_size: tuple[int, int],
) -> bool:
    if not _valid_image(output_path, target_size):
        return False
    for entry in manifest.get("entries", []):
        if str(entry.get("question_uid")) != uid:
            continue
        branch = entry.get("picture")
        generation = branch.get("generation") if isinstance(branch, dict) else None
        if not isinstance(generation, dict):
            return False
        if generation.get("request_sha256") != request_sha256:
            return False
        if generation.get("status") not in {"succeeded", "cached"}:
            return False
        for item in branch.get("media", []):
            if item.get("kind") != "prediction":
                continue
            expected_sha = str(item.get("sha256") or "").lower()
            return len(expected_sha) == 64 and _sha256_file(output_path) == expected_sha
        return False
    return False


def _update_manifest_result(
    manifest: dict[str, Any],
    *,
    uid: str,
    request_sha256: str,
    status: str,
    output_path: Path,
    elapsed_seconds: float,
    retries: int,
    response_id: str | None,
    error: str | None,
) -> None:
    for entry in manifest.get("entries", []):
        if str(entry.get("question_uid")) != uid:
            continue
        branch = entry.get("picture")
        if not isinstance(branch, dict):
            raise ValueError(f"{uid}: public manifest has no picture branch")
        generation = branch.get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"{uid}: public manifest has no generation provenance")
        if generation.get("request_sha256") != request_sha256:
            raise ValueError(f"{uid}: request hash differs between job and manifest")
        generation.update(
            {
                "status": status,
                "response_id": response_id,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "retries": retries,
                "error": error,
            }
        )
        for item in branch.get("media", []):
            if item.get("kind") == "prediction":
                item["path"] = str(output_path.resolve())
                if output_path.is_file():
                    item["sha256"] = _sha256_file(output_path)
                else:
                    item.pop("sha256", None)
        return
    raise ValueError(f"{uid}: job is absent from public manifest")


def run_jobs(
    *,
    jobs_path: Path,
    manifest_path: Path,
    generate: GenerateFn,
    retries: int,
    retry_delay: float,
    request_interval: float = 0.0,
    limit: int | None = None,
) -> dict[str, int]:
    jobs_payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    if jobs_payload.get("schema_version") != JOB_SCHEMA_VERSION:
        raise ValueError(f"unsupported job schema in {jobs_path}")
    jobs = jobs_payload.get("entries")
    if not isinstance(jobs, list):
        raise ValueError("job entries must be an array")
    if limit is not None:
        jobs = jobs[:limit]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = {"generated": 0, "cached": 0, "failed": 0}
    next_request_at = 0.0

    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("every job must be an object")
        assert_safe_generation_job(job)
        uid = str(job["question_uid"])
        input_path = Path(str(job["input_image_path"]))
        output_path = Path(str(job["output_path"]))
        if _sha256_file(input_path) != job["input_image_sha256"]:
            raise ValueError(f"{uid}: input image hash changed after job preparation")
        with Image.open(input_path) as source:
            target_size = source.size

        started = time.monotonic()
        response_id: str | None = None
        attempts_used = 0
        error: str | None = None
        if _cached_output_matches(
            manifest,
            uid=uid,
            request_sha256=str(job["request_sha256"]),
            output_path=output_path,
            target_size=target_size,
        ):
            status = "cached"
            stats["cached"] += 1
        else:
            status = "failed"
            for attempt in range(retries + 1):
                attempts_used = attempt
                try:
                    wait_seconds = max(0.0, next_request_at - time.monotonic())
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    request_started = time.monotonic()
                    next_request_at = request_started + request_interval
                    image, response_id = generate(job)
                    if not isinstance(image, Image.Image):
                        raise RetryableGenerationError(
                            "generation backend returned no image"
                        )
                    _atomic_save_image(image, output_path, target_size)
                    if not _valid_image(output_path, target_size):
                        raise RetryableGenerationError(
                            "generated image is missing or corrupt after normalization"
                        )
                    status = "succeeded"
                    error = None
                    stats["generated"] += 1
                    break
                except Exception as exc:  # network/backend dependent
                    error = str(exc)
                    if attempt < retries and _is_retryable(exc):
                        time.sleep(retry_delay * (2**attempt))
                    else:
                        break
            if status == "failed":
                stats["failed"] += 1

        _update_manifest_result(
            manifest,
            uid=uid,
            request_sha256=str(job["request_sha256"]),
            status=status,
            output_path=output_path,
            elapsed_seconds=time.monotonic() - started,
            retries=attempts_used,
            response_id=response_id,
            error=error,
        )
        _atomic_write_json(manifest_path, manifest)
    return stats


def make_gpt_generator(*, api_key: str, base_url: str) -> GenerateFn:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def _generate(job: dict[str, Any]) -> tuple[Image.Image, str | None]:
        input_path = Path(str(job["input_image_path"]))
        with input_path.open("rb") as image_file:
            response = client.images.edit(
                model=str(job.get("model") or DEFAULT_GPT_MODEL),
                image=image_file,
                prompt=str(job["prompt"]),
                input_fidelity="high",
                quality="high",
            )
        data = getattr(response, "data", None) or []
        if not data:
            raise RetryableGenerationError("GPT Image returned an empty response")
        first = data[0]
        b64_data = getattr(first, "b64_json", None)
        if b64_data:
            raw = base64.b64decode(b64_data)
        else:
            url = getattr(first, "url", None)
            if not url:
                raise RetryableGenerationError(
                    "GPT Image response has neither b64_json nor URL"
                )
            with urlopen(str(url), timeout=120) as response_stream:  # noqa: S310
                raw = response_stream.read()
        try:
            image = Image.open(BytesIO(raw)).convert("RGB")
        except (OSError, ValueError) as exc:
            raise RetryableGenerationError(
                "GPT Image returned corrupt image bytes"
            ) from exc
        return image, str(getattr(response, "id", "") or "") or None

    return _generate


def make_qwen_generator(
    *, checkpoint: str, device: str, cpu_offload: bool = False
) -> GenerateFn:
    try:
        import torch
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Qwen generation requires torch and a diffusers version providing "
            "QwenImageEditPlusPipeline"
        ) from exc

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    pipeline = QwenImageEditPlusPipeline.from_pretrained(checkpoint, torch_dtype=dtype)
    if cpu_offload:
        if device != "cuda":
            raise ValueError("Qwen CPU offload requires --device cuda")
        pipeline.enable_model_cpu_offload()
        generator_device = "cpu"
    else:
        pipeline.to(device)
        generator_device = device
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    pipeline.set_progress_bar_config(disable=False)

    def _generate(job: dict[str, Any]) -> tuple[Image.Image, str | None]:
        with Image.open(job["input_image_path"]) as source:
            condition = source.convert("RGB")
        generator = torch.Generator(device=generator_device).manual_seed(int(job["seed"]))
        with torch.inference_mode():
            result = pipeline(
                image=[condition],
                prompt=str(job["prompt"]),
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt="",
                num_inference_steps=40,
            )
        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("Qwen Image Edit returned no image")
        return images[0], None

    return _generate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpt", "qwen"), required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=2, help="GPT retry count for retryable failures; Qwen always uses zero")
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--request_interval", type=float, default=0.0, help="Minimum seconds between GPT request starts")
    parser.add_argument("--base_url", default=None, help="GPT endpoint; defaults to OPENAI_BASE_URL or the OpenAI API")
    parser.add_argument("--api_key_env", default=None, help="GPT API-key environment variable; defaults to OPENAI_API_KEY then API_KEY")
    parser.add_argument("--qwen_checkpoint", default=DEFAULT_QWEN_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen_cpu_offload", action="store_true", help="Offload Qwen modules to CPU between calls to reduce single-GPU memory")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N jobs for preflight; later runs resume from the manifest")
    args = parser.parse_args(argv)
    if not args.jobs.is_file():
        parser.error(f"--jobs not found: {args.jobs}")
    if not args.manifest.is_file():
        parser.error(f"--manifest not found: {args.manifest}")
    if args.retries < 0 or args.retry_delay < 0 or args.request_interval < 0:
        parser.error("retry values must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.backend == "gpt":
        key_env_names = (args.api_key_env,) if args.api_key_env else ("OPENAI_API_KEY", "API_KEY")
        api_key = ""
        for env_name in key_env_names:
            api_key = os.getenv(env_name, "").strip()
            if api_key:
                break
        if not api_key:
            raise RuntimeError(
                "missing GPT API key in " + ", ".join(key_env_names)
            )
        base_url = str(args.base_url or os.getenv("OPENAI_BASE_URL") or "").strip()
        generate = make_gpt_generator(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )
    else:
        generate = make_qwen_generator(
            checkpoint=args.qwen_checkpoint,
            device=args.device,
            cpu_offload=args.qwen_cpu_offload,
        )
    effective_retries = args.retries if args.backend == "gpt" else 0
    stats = run_jobs(
        jobs_path=args.jobs,
        manifest_path=args.manifest,
        generate=generate,
        retries=effective_retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval if args.backend == "gpt" else 0.0,
        limit=args.limit,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
