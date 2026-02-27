#!/usr/bin/env python3
"""
LiveClin Evaluation CLI
=======================
Single entry-point for evaluating a model on the LiveClin benchmark.

Usage examples:

  # Evaluate via a remote API (images sent as URLs)
  python evaluate.py \\
      --model gpt-4o \\
      --api-base https://api.openai.com/v1 \\
      --api-key sk-xxx \\
      --image-mode url

  # Evaluate a locally-served model (images sent as base64)
  python evaluate.py \\
      --model Qwen2.5-VL-7B-Instruct \\
      --api-base http://localhost:8000/v1 \\
      --api-key token-xxx \\
      --image-mode local

  # With all options
  python evaluate.py \\
      --model gpt-4o \\
      --api-base https://api.openai.com/v1 \\
      --api-key sk-xxx \\
      --image-mode url \\
      --dataset 2025_H1 \\
      --concurrency 100 \\
      --output results/gpt-4o.json \\
      --resume
"""

import argparse
import asyncio
import sys
from pathlib import Path

from liveclin import EvalConfig
from liveclin.analyzer import analyze, print_summary
from liveclin.data import ensure_dataset, get_image_root, load_cases
from liveclin.runner import run_evaluation, save_results


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(
        description="LiveClin: evaluate a model on the clinical benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --model gpt-4o --api-base https://api.openai.com/v1 "
            "--api-key sk-xxx --image-mode url\n"
            "  python evaluate.py --model Qwen2.5-VL-7B --api-base http://localhost:8000/v1 "
            "--api-key token --image-mode local\n"
        ),
    )

    # Required
    p.add_argument("--model", required=True,
                   help="Model name (must match the API's model identifier).")
    p.add_argument("--api-base", required=True,
                   help="API base URL (e.g. https://api.openai.com/v1).")
    p.add_argument("--api-key", default="token",
                   help="API key for authentication (default: 'token'; "
                        "not needed for local SGLang deployments).")
    p.add_argument("--image-mode", required=True, choices=["url", "local"],
                   help="How to send images: 'url' passes image URLs; "
                        "'local' reads files and sends base64.")

    # Optional
    p.add_argument("--dataset", default="2025_H1",
                   help="Dataset config name (default: 2025_H1).")
    p.add_argument("--concurrency", type=int, default=100,
                   help="Max concurrent case evaluations (default: 100).")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: results/<model>_<dataset>.json).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from an existing results file.")
    p.add_argument("--data-dir", default="data",
                   help="Local directory for dataset storage (default: data).")
    p.add_argument("--jsonl-path", default=None,
                   help="Override: path to a JSONL file (skips auto-download).")
    p.add_argument("--image-root", default=None,
                   help="Override: path to the image directory.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0).")
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="Max tokens per response (default: 16384).")
    p.add_argument("--max-retries", type=int, default=5,
                   help="Max retries per API call (default: 5).")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="API call timeout in seconds (default: 120).")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging.")

    args = p.parse_args()

    return EvalConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        image_mode=args.image_mode,
        dataset=args.dataset,
        concurrency=args.concurrency,
        output=args.output,
        resume=args.resume,
        data_dir=args.data_dir,
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout=args.timeout,
        verbose=args.verbose,
    )


async def main_async(config: EvalConfig) -> None:
    # 1. Resolve data paths (explicit overrides take priority)
    if config.jsonl_path:
        jsonl_path = Path(config.jsonl_path)
        if not jsonl_path.is_file():
            print(f"Error: --jsonl-path not found: {jsonl_path}")
            sys.exit(1)
    else:
        jsonl_path = ensure_dataset(config.data_dir, config.dataset)

    if config.image_root:
        image_root = Path(config.image_root)
    else:
        image_root = get_image_root(config.data_dir, config.dataset)

    if config.image_mode == "local" and not image_root.is_dir():
        print(f"Warning: image directory not found at {image_root}")
        print("Local images may not be available. Consider using --image-mode url.")

    # 2. Load cases
    cases = load_cases(jsonl_path)
    print(f"Loaded {len(cases)} cases from {jsonl_path}")

    # 3. Run evaluation
    results = await run_evaluation(cases, config, image_root)

    # 4. Analyze and print summary
    results = analyze(results)
    save_results(results, config.output_path)
    print_summary(results)

    print(f"Detailed results: {config.output_path}")


def main() -> None:
    config = parse_args()

    print()
    print("LiveClin Evaluation")
    print("=" * 50)
    print(f"  Model:      {config.model}")
    print(f"  API Base:   {config.api_base}")
    print(f"  Image Mode: {config.image_mode}")
    print(f"  Dataset:    {config.dataset}")
    print(f"  Concurrency:{config.concurrency}")
    print(f"  Output:     {config.output_path}")
    print(f"  Resume:     {config.resume}")
    print("=" * 50)
    print()

    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results have been saved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
