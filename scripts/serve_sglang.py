#!/usr/bin/env python3
"""
SGLang Model Server Launcher
=============================
Helper script to deploy a local model with SGLang for LiveClin evaluation.

Usage:
  python scripts/serve_sglang.py \\
      --model-path /path/to/Qwen2.5-VL-7B-Instruct \\
      --tp 2 --dp 4 --port 8000

Then evaluate with:
  python evaluate.py \\
      --model Qwen2.5-VL-7B-Instruct \\
      --api-base http://localhost:8000/v1 \\
      --api-key token \\
      --image-mode local
"""

import argparse
import subprocess
import sys
import time

import requests


def wait_for_ready(base_url: str, timeout: int = 300) -> bool:
    """Poll the server until it responds to /v1/models."""
    print(f"Waiting for server at {base_url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"Server ready ({elapsed:.0f}s)")
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    print(f"Server not ready after {timeout}s")
    return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="Launch an SGLang model server for LiveClin evaluation.",
    )
    p.add_argument("--model-path", required=True,
                   help="Path to the model (local directory or HuggingFace ID).")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallelism degree (default: 1).")
    p.add_argument("--dp", type=int, default=1,
                   help="Data parallelism degree (default: 1).")
    p.add_argument("--port", type=int, default=8000,
                   help="Server port (default: 8000).")
    p.add_argument("--host", default="127.0.0.1",
                   help="Server host (default: 127.0.0.1).")
    p.add_argument("--log-level", default="info",
                   help="Log level (default: info).")
    p.add_argument("--timeout", type=int, default=300,
                   help="Max seconds to wait for server startup (default: 300).")

    args = p.parse_args()

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--tp", str(args.tp),
        "--dp", str(args.dp),
        "--port", str(args.port),
        "--host", args.host,
        "--trust-remote-code",
        "--enable-multimodal",
        "--log-level", args.log_level,
    ]

    print(f"Launching: {' '.join(cmd)}")
    print()

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    try:
        base_url = f"http://{args.host}:{args.port}"
        if not wait_for_ready(base_url, args.timeout):
            print("Aborting: server failed to start.")
            process.terminate()
            sys.exit(1)

        print()
        print("Server is running. Evaluate with:")
        print(f"  python evaluate.py \\")
        print(f"      --model <model-name> \\")
        print(f"      --api-base {base_url}/v1 \\")
        print(f"      --api-key token \\")
        print(f"      --image-mode local")
        print()
        print("Press Ctrl+C to stop the server.")
        process.wait()

    except KeyboardInterrupt:
        print("\nShutting down server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped.")


if __name__ == "__main__":
    main()
