import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# =============================================================================
# 1. Global Configuration (edit for your environment)
# =============================================================================

EVALUATE_SCRIPT_PATH = os.getenv("EVALUATE_SCRIPT_PATH", "core.py")

# NOW: a single JSONL file (not a folder)
JSONL_PATH = os.getenv("JSONL_PATH", "demo/demo.jsonl")  # path to *.jsonl
IMAGE_ROOT_PATH = os.getenv("IMAGE_ROOT_PATH", "demo/images")  # set to "" if not needed

HOST = os.getenv("SGLANG_HOST", "127.0.0.1")
PORT = int(os.getenv("SGLANG_PORT", "30000"))
API_BASE_URL = f"http://{HOST}:{PORT}/v1"
LOG_LEVEL = os.getenv("SGLANG_LOG_LEVEL", "info")

# =============================================================================
# 2. Model configuration list
# =============================================================================

MODELS_TO_TEST = [
    {
        "name": "Qwen2.5-VL-7B",
        "path": "/path/to/Qwen2.5-VL-7B-Instruct",
        "tp": 2, "dp": 4,
        "api_id": "Qwen2.5-VL-7B-Instruct" # 这个ID将传递给测试脚本
    },
    # {
    #     "name": "GLM-4.1V-9B",
    #     "path": "/path/to/GLM-4.1V-9B",
    #     "tp": 2, "dp": 4,
    #     "api_id": "GLM-4.1V-9B"
    # },
    # {
    #     "name": "HuatuoGPT-Vision-7B",
    #     "path": "/path/to/HuatuoGPT-Vision-7B",
    #     "tp": 2, "dp": 4,
    #     "api_id": "HuatuoGPT-Vision-7B"
    # },
    # {
    #     "name": "InternVL3_5-8B",
    #     "path": "/path/to/InternVL3_5-8B",
    #     "tp": 2, "dp": 4,
    #     "api_id": "InternVL3_5-8B"
    # },
    # Add more models similarly, using env vars for paths and settings.
]

# =============================================================================
# 3. Helper functions
# =============================================================================

def wait_for_server_ready(api_base_url: str, timeout: int = 300) -> bool:
    print(f"\nWaiting for server to be ready at {api_base_url} ...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{api_base_url}/models", timeout=5)
            if resp.status_code == 200:
                print(f"Server is ready! (took {time.time() - start_time:.1f}s)")
                return True
        except requests.RequestException:
            pass
        time.sleep(5)

    print(f"Server failed to start within {timeout} seconds.")
    return False

# =============================================================================
# 4. Main execution flow
# =============================================================================

def main() -> None:
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL_PATH not found: {jsonl_path}")

    for model_config in MODELS_TO_TEST:
        model_name = model_config["name"]
        print("\n" + "=" * 80)
        print(f"STARTING TEST FOR MODEL: {model_name}")
        print("=" * 80 + "\n")

        # 1) Launch SGLang server
        launch_cmd = [
            sys.executable,
            "-m", "sglang.launch_server",
            "--model-path", model_config["path"],
            "--tp", str(model_config["tp"]),
            "--dp", str(model_config["dp"]),
            "--port", str(PORT),
            "--trust-remote-code",
            "--enable-multimodal",
            "--log-level", LOG_LEVEL,
        ]

        print(f"Launching SGLang server for {model_name}...")
        print(f"Command: {' '.join(launch_cmd)}")

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        log_file_path = log_dir / f"{safe_name}_{timestamp}.log"

        log_file = open(log_file_path, "w", encoding="utf-8")
        server_process = subprocess.Popen(
            launch_cmd,
            stdout=log_file,
            stderr=log_file,
            text=True,
        )
        print(f"Server log redirected to: {log_file_path}")

        server_ready = False

        try:
            # 2) Wait for server readiness
            if not wait_for_server_ready(API_BASE_URL):
                raise RuntimeError("Server readiness check failed.")
            server_ready = True

            # 3) Run evaluation script (JSONL input)
            print(f"\nRunning evaluation script for model API ID: {model_config['api_id']} ...")

            eval_cmd = [
                sys.executable, EVALUATE_SCRIPT_PATH,
                "--jsonl-path", str(jsonl_path),            
                "--model-api-id", model_config["api_id"],
                "--api-base-url", API_BASE_URL,
                "--resume",
                "--non-interactive",
            ]

            if IMAGE_ROOT_PATH:
                eval_cmd.extend(["--image-root-path", IMAGE_ROOT_PATH])

            eval_process = subprocess.Popen(
                eval_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            print("\n--- Evaluation Script Output ---")
            assert eval_process.stdout is not None
            for line in iter(eval_process.stdout.readline, ""):
                print(line, end="")
            eval_process.wait()
            print("\n--- End of Evaluation Script Output ---\n")

            if eval_process.returncode != 0:
                print(f"WARNING: evaluation exited with code {eval_process.returncode}")

        except Exception as e:
            print(f"\nERROR during process for {model_name}: {e}")
            if not server_ready:
                print("The server may have failed to start. Check the log file for details.")

        finally:
            # 4) Shutdown server
            if server_process and server_process.poll() is None:
                print(f"\nShutting down server for {model_name} (PID: {server_process.pid})...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                    print("Server shut down gracefully.")
                except subprocess.TimeoutExpired:
                    print("Server did not respond; forcing kill...")
                    server_process.kill()
                    server_process.wait()
                    print("Server killed.")

            if log_file:
                log_file.close()

            print("\nWaiting 10 seconds for GPU resources to be released...")
            time.sleep(10)

    print("\nAll model tests completed!")

if __name__ == "__main__":
    main()
