#!/usr/bin/env python3
"""
Quick smoke test: verify the model can actually see images in both modes.

Usage:
  python scripts/test_vision.py \
      --model gpt-4o \
      --api-base https://api.openai.com/v1 \
      --api-key sk-xxx

Tests performed:
  1. LOCAL mode  – generates a solid-yellow PNG, sends as base64,
                   asks the model what color the image is.
  2. URL mode    – sends a public image URL, asks the model to describe it.
"""

import argparse
import asyncio
import base64
import io
import struct
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclin import EvalConfig
from liveclin.client import APIClient


# ------------------------------------------------------------------
# Minimal PNG generator (no Pillow dependency)
# ------------------------------------------------------------------

def _make_solid_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Create a minimal solid-color PNG in pure Python."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    raw_rows = b""
    row = bytes([r, g, b] * width)
    for _ in range(height):
        raw_rows += b"\x00" + row  # filter byte + pixel data

    idat = _chunk(b"IDAT", zlib.compress(raw_rows))
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------

SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"


async def test_local_mode(config: EvalConfig) -> bool:
    """Generate a yellow image, send as base64, check if model sees yellow."""
    print("=" * 60)
    print("TEST 1: LOCAL mode (base64 solid-yellow image)")
    print("=" * 60)

    png_bytes = _make_solid_png(64, 64, 255, 255, 0)

    tmp = Path(tempfile.mktemp(suffix=".png"))
    tmp.write_bytes(png_bytes)
    print(f"  Generated {tmp} ({len(png_bytes)} bytes)")

    config_local = EvalConfig(
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        image_mode="local",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
        timeout=config.timeout,
    )
    client = APIClient(config_local)

    prompt = (
        "What is the dominant color of this image? "
        "Reply with just the color name in one word."
    )
    resp = await client.send(prompt, [tmp])
    tmp.unlink(missing_ok=True)

    if resp["status"] != "success":
        print(f"  FAIL: API error — {resp.get('message')}")
        return False

    answer = resp["content"]
    print(f"  Model response: {answer}")

    if "yellow" in answer.lower():
        print("  PASS: model correctly identified yellow.")
        return True
    else:
        print("  WARN: expected 'yellow', got something else. "
              "Model may still see the image but interpreted differently.")
        return False


async def test_url_mode(config: EvalConfig) -> bool:
    """Send a public image URL, check if model can describe it."""
    print()
    print("=" * 60)
    print("TEST 2: URL mode (public Wikipedia PNG)")
    print("=" * 60)
    print(f"  URL: {SAMPLE_URL}")

    config_url = EvalConfig(
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        image_mode="url",
        temperature=0.0,
        max_tokens=256,
        max_retries=2,
        timeout=config.timeout,
    )
    client = APIClient(config_url)

    prompt = (
        "Briefly describe what you see in this image in 1-2 sentences."
    )
    resp = await client.send(prompt, [SAMPLE_URL])

    if resp["status"] != "success":
        print(f"  FAIL: API error — {resp.get('message')}")
        return False

    answer = resp["content"]
    print(f"  Model response: {answer}")

    if len(answer.strip()) > 10:
        print("  PASS: model returned a substantive description.")
        return True
    else:
        print("  WARN: response seems too short, model may not have seen the image.")
        return False


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

async def main() -> None:
    p = argparse.ArgumentParser(
        description="Smoke test: verify the model can see images (local & URL).",
    )
    p.add_argument("--model", required=True, help="Model name.")
    p.add_argument("--api-base", required=True, help="API base URL.")
    p.add_argument("--api-key", required=True, help="API key.")
    p.add_argument("--timeout", type=float, default=60.0, help="Timeout (s).")
    args = p.parse_args()

    config = EvalConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        image_mode="local",
        timeout=args.timeout,
    )

    print()
    print(f"Model:    {config.model}")
    print(f"API Base: {config.api_base}")
    print()

    r1 = await test_local_mode(config)
    r2 = await test_url_mode(config)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Local (base64): {'PASS' if r1 else 'FAIL'}")
    print(f"  URL:            {'PASS' if r2 else 'FAIL'}")
    print()

    if r1 and r2:
        print("Both modes working. You are ready to evaluate.")
    elif r1:
        print("Only local mode works. Use --image-mode local for evaluation.")
    elif r2:
        print("Only URL mode works. Use --image-mode url for evaluation.")
    else:
        print("Neither mode working. Check model, API key, or endpoint.")

    sys.exit(0 if (r1 and r2) else 1)


if __name__ == "__main__":
    asyncio.run(main())
