#!/usr/bin/env python3
"""Minimal image + question chat helper for an OpenAI-compatible VLM.

Usage:
    python scripts/chat_with_vlm.py \
        --image /path/to/image.jpg \
        --question "What objects are clearly visible?" \
        --base_url https://www.packyapi.com/v1 \
        --model qwen3-vl-flash
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path


def _encode_image(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask an OpenAI-compatible VLM about one image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--question", required=True, help="User question about the image")
    parser.add_argument(
        "--base_url",
        default="https://www.packyapi.com/v1",
        help="OpenAI-compatible VLM API base URL",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-flash",
        help="Model name to use",
    )
    parser.add_argument(
        "--api_key_env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable that stores the API key",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful vision-language assistant. Answer the user's question about the image.",
        help="Optional system prompt",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum output tokens",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY") or "EMPTY"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    b64, mime = _encode_image(image_path)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": args.question},
                ],
            },
        ],
        max_tokens=args.max_tokens,
        temperature=0,
    )
    print((resp.choices[0].message.content or "").strip())


if __name__ == "__main__":
    main()
