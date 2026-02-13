import os
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Set
import copy
import time
import argparse
from tqdm.asyncio import tqdm
import concurrent.futures

# --- Import the reusable API client ---
try:
    from api_client import CustomAPIClient
except ImportError:
    print(
        "Error: Could not import CustomAPIClient. Ensure api_client.py is in the same directory "
        "or available on your PYTHONPATH."
    )
    raise

# =============================================================================
# Default configuration (overridden by CLI args and/or environment variables)
# =============================================================================

# Clean defaults for open-source:
# - empty model id by default (must be provided via CLI or env)
# - empty API key by default (must be provided via CLI or env)
MODEL_API_IDS: List[str] = [os.getenv("MODEL_API_ID", "").strip()]
YOUR_API_KEY: str = os.getenv("YOUR_API_KEY", "").strip()
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000/v1").strip()

JSON_FOLDER_PATH_INPUT = ""
IMAGE_ROOT_PATH: Optional[Path] = None

# --- Retry configuration ---
MAX_API_CALL_ATTEMPTS = int(os.getenv("MAX_API_CALL_ATTEMPTS", "2"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1"))

# --- Concurrency configuration ---
MAX_CONCURRENT_MODELS = int(os.getenv("MAX_CONCURRENT_MODELS", "1"))  # models per file
MAX_CONCURRENT_FILES = int(os.getenv("MAX_CONCURRENT_FILES", "50"))   # client pool slots / file concurrency

# --- Logging configuration ---
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "true").lower() in ("1", "true", "yes", "y")
SHOW_API_DETAILS = os.getenv("SHOW_API_DETAILS", "true").lower() in ("1", "true", "yes", "y")

# If True, use URLs from JSON image_details['url']; else use local file paths from image_details['file']
USE_URL = os.getenv("USE_URL", "false").lower() in ("1", "true", "yes", "y")

# Global: track files to move to trash due to critical API errors (e.g., content_filter)
FILES_TO_TRASH: Set[Path] = set()

# Client pool globals
CLIENT_POOL: List[Dict[str, Any]] = []
CLIENT_POOL_LOCK: Optional[asyncio.Lock] = None
AVAILABLE_CLIENTS: List[int] = []


# =============================================================================
# Client pool
# =============================================================================

async def initialize_client_pool():
    """Initialize an API client pool (async + thread pool for faster init)."""
    global CLIENT_POOL, CLIENT_POOL_LOCK, AVAILABLE_CLIENTS

    CLIENT_POOL_LOCK = asyncio.Lock()
    print(f"Initializing client pool with {MAX_CONCURRENT_FILES} slots...")

    def init_client_sync(slot_idx: int, model_id: str) -> Tuple[int, str, Optional[Any]]:
        """Initialize one client synchronously (executed in a thread)."""
        try:
            client = CustomAPIClient(
                api_key=YOUR_API_KEY,
                model_name=model_id,
                api_base=API_BASE_URL,
            )
            if VERBOSE_LOGGING:
                print(f"  OK: Slot {slot_idx + 1}: Initialized client for {model_id}")
            return slot_idx, model_id, client
        except Exception as e:
            print(f"  FAIL: Slot {slot_idx + 1}: Could not init client for {model_id}: {e}")
            return slot_idx, model_id, None

    loop = asyncio.get_event_loop()

    init_tasks = []
    total_to_init = MAX_CONCURRENT_FILES * len(MODEL_API_IDS)
    print(f"Initializing {total_to_init} clients concurrently...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for slot_idx in range(MAX_CONCURRENT_FILES):
            for model_id in MODEL_API_IDS:
                init_tasks.append(loop.run_in_executor(executor, init_client_sync, slot_idx, model_id))
        results = await asyncio.gather(*init_tasks)

    # Group by slot index
    temp_pool: Dict[int, Dict[str, Any]] = {}
    for slot_idx, model_id, client in results:
        temp_pool.setdefault(slot_idx, {})
        if client is not None:
            temp_pool[slot_idx][model_id] = client

    successful_slots = 0
    total_clients = 0
    failed_clients = 0

    for slot_idx in range(MAX_CONCURRENT_FILES):
        if slot_idx in temp_pool and temp_pool[slot_idx]:
            CLIENT_POOL.append(temp_pool[slot_idx])
            AVAILABLE_CLIENTS.append(len(CLIENT_POOL) - 1)
            successful_slots += 1
            total_clients += len(temp_pool[slot_idx])
            failed_clients += len(MODEL_API_IDS) - len(temp_pool[slot_idx])
        else:
            failed_clients += len(MODEL_API_IDS)

    elapsed = time.time() - start_time
    print(f"Initialization completed in {elapsed:.2f}s")
    print(f"  Successful: {successful_slots} slots, {total_clients} clients")
    if failed_clients:
        print(f"  Failed clients: {failed_clients}")
    if successful_slots:
        print(f"  Avg models/slot: {total_clients / successful_slots:.1f}")
    print()


async def acquire_client_set() -> int:
    """Acquire one client-set index from the pool."""
    global AVAILABLE_CLIENTS
    assert CLIENT_POOL_LOCK is not None

    while True:
        async with CLIENT_POOL_LOCK:
            if AVAILABLE_CLIENTS:
                return AVAILABLE_CLIENTS.pop(0)
        await asyncio.sleep(0.1)


async def release_client_set(client_idx: int):
    """Return a client-set index to the pool and clear histories."""
    global AVAILABLE_CLIENTS
    assert CLIENT_POOL_LOCK is not None

    async with CLIENT_POOL_LOCK:
        client_set = CLIENT_POOL[client_idx]
        for _, client in client_set.items():
            if hasattr(client, "clear_conversation_history"):
                client.clear_conversation_history()
        AVAILABLE_CLIENTS.append(client_idx)


def cleanup_client_pool():
    """Close and cleanup all clients in the pool."""
    global CLIENT_POOL, AVAILABLE_CLIENTS

    for client_set in CLIENT_POOL:
        for _, client in client_set.items():
            if hasattr(client, "close"):
                try:
                    client.close()
                except Exception:
                    pass

    CLIENT_POOL.clear()
    AVAILABLE_CLIENTS.clear()
    print("Cleaned up client pool")


# =============================================================================
# Files & resume helpers
# =============================================================================

def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def prepare_json_folder_from_jsonl_to_parent_json_dir(jsonl_path: Path) -> Path:
    """
    Read <parent>/<name>.jsonl and write each line as a JSON file into:
      <parent>/json/PMCxxxxxx.json  (preferred)
    Fallback names:
      PMIDxxxx.json / DOI_xxx.json / LINE_000000123.json

    Resume behavior:
      If <parent>/json already contains *.json, do NOT overwrite (assume resume).
    Returns:
      The folder path <parent>/json
    """
    jsonl_path = jsonl_path.resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    json_dir = jsonl_path.parent / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted([p for p in json_dir.glob("*.json") if p.is_file()])
    if existing:
        print(f"[Resume] Found existing json folder: {json_dir} ({len(existing)} files). Not overwriting.")
        return json_dir

    print(f"Converting JSONL -> JSON folder: {jsonl_path} -> {json_dir}")

    used_names = set()
    written = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at line {i+1}: {e}")

            pmc = obj.get("pmc")
            doi = obj.get("doi")
            pmid = obj.get("pmid") or obj.get("PMID")

            if isinstance(pmc, str) and pmc.strip():
                base = safe_filename(pmc.strip())
                if not base.upper().startswith("PMC"):
                    base = "PMC" + base
            elif isinstance(pmid, (str, int)) and str(pmid).strip():
                base = "PMID" + safe_filename(str(pmid))
            elif isinstance(doi, str) and doi.strip():
                base = "DOI_" + safe_filename(doi)
            else:
                base = f"LINE_{i:09d}"

            filename = f"{base}.json"

            # handle collisions
            if filename in used_names or (json_dir / filename).exists():
                k = 2
                while True:
                    cand = f"{base}_{k}.json"
                    if cand not in used_names and not (json_dir / cand).exists():
                        filename = cand
                        break
                    k += 1

            used_names.add(filename)
            out_path = json_dir / filename

            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(obj, wf, ensure_ascii=False, indent=2)

            written += 1

    print(f"Done. Wrote {written} json files into {json_dir}")
    return json_dir



def move_file_to_trash(json_path: Path) -> bool:
    """Move JSON file into a sibling 'test_trash' folder."""
    try:
        trash_dir = json_path.parent / "test_trash"
        os.makedirs(trash_dir, exist_ok=True)

        trash_path = trash_dir / json_path.name
        if trash_path.exists():
            print(f"Warning: {json_path.name} already exists in test_trash. Skipping.")
            return False

        os.replace(json_path, trash_path)
        print(f"Moved {json_path.name} to test_trash due to critical API error")
        return True
    except OSError as e:
        print(f"Error moving {json_path.name} to test_trash: {e}")
        return False


def check_file_completion_status(json_path: Path) -> Tuple[bool, Set[str], List[str]]:
    """Check whether all models have completed all MCQs for a file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, set(), [f"Cannot read file: {e}"]

    final_policy = data.get("exam_creation", {}).get("final_policy")
    if not final_policy:
        return False, set(), ["Missing 'final_policy' in file"]

    mcqs = final_policy.get("mcqs", [])
    if not mcqs:
        return False, set(), ["No MCQs found in final_policy"]

    completed_models: Set[str] = set()
    errors: List[str] = []

    for model_id in MODEL_API_IDS:
        model_complete = True

        for mcq in mcqs:
            evaluate_list = mcq.get("evaluate", [])
            model_found = False

            for eval_result in evaluate_list:
                if eval_result.get("model_id") == model_id:
                    model_found = True
                    if eval_result.get("extracted_answer") is None:
                        model_complete = False
                        break

            if not model_found:
                model_complete = False
                break

        if model_complete:
            completed_models.add(model_id)

    all_complete = len(completed_models) == len(MODEL_API_IDS)
    return all_complete, completed_models, errors


def get_pending_models_for_file(json_path: Path) -> Tuple[List[str], Set[str], List[str]]:
    """Return pending models for a JSON file based on stored evaluation status."""
    is_complete, completed_models, errors = check_file_completion_status(json_path)

    if errors:
        return MODEL_API_IDS, set(), errors
    if is_complete:
        return [], completed_models, []

    pending_models = [m for m in MODEL_API_IDS if m not in completed_models]
    return pending_models, completed_models, []


def analyze_folder_status(json_folder_path: Path) -> Dict[str, Any]:
    """Analyze resume/completion status for a folder."""
    json_files = sorted([f for f in json_folder_path.glob("*.json") if f.is_file()])

    status_summary: Dict[str, Any] = {
        "total_files": len(json_files),
        "completed_files": 0,
        "partial_files": 0,
        "pending_files": 0,
        "error_files": 0,
        "file_details": {},
        "total_model_tasks": 0,
        "completed_model_tasks": 0,
    }

    for json_file in json_files:
        pending_models, completed_models, errors = get_pending_models_for_file(json_file)

        if errors:
            status_summary["error_files"] += 1
            file_status = "error"
        elif not pending_models:
            status_summary["completed_files"] += 1
            file_status = "completed"
        elif not completed_models:
            status_summary["pending_files"] += 1
            file_status = "pending"
        else:
            status_summary["partial_files"] += 1
            file_status = "partial"

        status_summary["file_details"][json_file.name] = {
            "status": file_status,
            "completed_models": list(completed_models),
            "pending_models": pending_models,
            "errors": errors,
        }

        status_summary["total_model_tasks"] += len(MODEL_API_IDS)
        status_summary["completed_model_tasks"] += len(completed_models)

    return status_summary


def print_status_summary(status_summary: Dict[str, Any]):
    """Pretty-print folder status summary."""
    print("\n" + "=" * 60)
    print("FOLDER STATUS ANALYSIS")
    print("=" * 60)
    print(f"Total files: {status_summary['total_files']}")
    print(f"Completed files: {status_summary['completed_files']}")
    print(f"Partially completed files: {status_summary['partial_files']}")
    print(f"Pending files: {status_summary['pending_files']}")
    print(f"Error files: {status_summary['error_files']}")

    total_tasks = status_summary["total_model_tasks"]
    completed_tasks = status_summary["completed_model_tasks"]
    pending_tasks = total_tasks - completed_tasks

    if total_tasks > 0:
        completion_rate = (completed_tasks / total_tasks) * 100
        print(f"\nProgress: {completed_tasks}/{total_tasks} tasks ({completion_rate:.1f}%)")
        print(f"Remaining: {pending_tasks} tasks")

    if status_summary["partial_files"] > 0 and VERBOSE_LOGGING:
        print("\nPartially completed files:")
        for filename, details in status_summary["file_details"].items():
            if details["status"] == "partial":
                completed = ", ".join(details["completed_models"])
                pending = ", ".join(details["pending_models"])
                print(f"  {filename}")
                print(f"    Completed: {completed}")
                print(f"    Pending:   {pending}")


# =============================================================================
# Prompt helpers
# =============================================================================

def get_multimodal_prompt_text(
    text_content: str,
    image_details: Optional[List[Dict[str, Any]]] = None,
    table_details: Optional[List[Dict[str, Any]]] = None,
    json_dir: Optional[Path] = None,
    image_root: Optional[Path] = None,
) -> Tuple[str, List[Union[Path, str]]]:
    """Build multimodal prompt text and a list of image sources (Path or URL string)."""
    prompt_parts = [text_content]
    image_sources: List[Union[Path, str]] = []

    if image_details:
        image_captions: List[str] = []
        for img_item in image_details:
            caption = img_item.get("caption_prefix", "")
            if not caption:
                continue

            if USE_URL:
                url = img_item.get("url")
                if url:
                    image_captions.append(caption)
                    image_sources.append(url)
                elif VERBOSE_LOGGING:
                    print("Warning: Image item has no 'url' in URL mode.")
            else:
                relative_path_str = img_item.get("file", "")
                if relative_path_str and (json_dir or image_root):
                    base_dir = image_root if image_root else json_dir
                    absolute_path = (base_dir / Path(str(relative_path_str))).resolve()
                    if absolute_path.is_file():
                        image_captions.append(caption)
                        image_sources.append(absolute_path)
                    elif VERBOSE_LOGGING:
                        print(f"Warning: Image file not found at {absolute_path}")

        if image_captions:
            prompt_parts.append("* Figures:")
            prompt_parts.extend(image_captions)

    if table_details:
        table_texts: List[str] = []
        for table_item in table_details:
            caption_prefix = table_item.get("caption_prefix", "")
            caption = table_item.get("caption", "")
            content = table_item.get("content", "")
            if content:
                table_texts.append(f"{caption_prefix} {caption}\n{content}".strip())

        if table_texts:
            prompt_parts.append("* Tables:")
            prompt_parts.extend(table_texts)

    return "\n".join(prompt_parts), image_sources


def format_options(options_dict: Dict[str, str]) -> str:
    """Format MCQ options as a human-readable string."""
    return "\n".join([f"{k}. {v}" for k, v in sorted(options_dict.items())])


def extract_answer(model_response: str) -> Optional[str]:
    """Extract answer letter (A-J) from model response."""
    if not model_response:
        return None

    boxed_match = re.search(r"\\boxed{\s*([A-J])\s*}", model_response, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()

    explicit_match = re.search(
        r"(?:answer|option|choice|selected option)\s*[:\-is\s]*\s*([A-Ja-j])\b",
        model_response,
        re.IGNORECASE,
    )
    if explicit_match:
        return explicit_match.group(1).upper()

    lines = model_response.strip().split("\n")
    for line in reversed(lines):
        trimmed = line.strip()
        match = re.match(r"^([A-Ja-j])[\s.)]*.*$", trimmed, re.IGNORECASE)
        if match:
            if len(trimmed) <= 3 and re.match(r"^[A-Ja-j][\s.)]*$", trimmed, re.IGNORECASE):
                return match.group(1).upper()

    single_letter_matches = re.findall(r"\b([A-J])\b", model_response)
    if single_letter_matches:
        return single_letter_matches[-1]

    if len(model_response.strip()) == 1 and model_response.strip().upper() in list("ABCDEFGHIJ"):
        return model_response.strip().upper()

    if VERBOSE_LOGGING:
        print(f"Warning: Could not extract answer from response: '{model_response[:100]}...'")
    return None


def check_correctness(extracted_answer: Optional[str], correct_answer: str) -> bool:
    """Check whether extracted answer matches the correct answer."""
    return extracted_answer is not None and extracted_answer == correct_answer.upper()


# =============================================================================
# Core evaluation logic
# =============================================================================

async def evaluate_mcq_series_for_model(
    model_id: str,
    api_client: CustomAPIClient,
    scenario_text: str,
    scenario_images: Optional[List[Dict[str, Any]]],
    scenario_tables: Optional[List[Dict[str, Any]]],
    mcqs: List[Dict[str, Any]],
    json_file_path: Path,
    image_root: Optional[Path] = None,
    progress_bar=None,
) -> List[Dict[str, Any]]:
    """Evaluate one model over all MCQs within one file."""
    evaluation_results: List[Dict[str, Any]] = []
    conversation_history_for_api: List[Dict[str, Any]] = []
    json_dir = json_file_path.parent

    if hasattr(api_client, "clear_conversation_history"):
        api_client.clear_conversation_history()

    scenario_prompt_text, scenario_image_sources = get_multimodal_prompt_text(
        f"* Scenario: {scenario_text}",
        scenario_images,
        scenario_tables,
        json_dir,
        image_root,
    )

    answer_format_instruction = (
        "Please provide the letter of the correct option, formatted as \\boxed{LETTER} (e.g., \\boxed{A})."
    )

    # Compose first turn using the first MCQ (so the scenario is included once)
    first_mcq = mcqs[0]
    q_text_fm = first_mcq.get("question", "")
    q_opts_fm = format_options(first_mcq.get("options", {}))

    first_q_prompt_section, first_q_image_sources = get_multimodal_prompt_text(
        f"* Question: {q_text_fm}\n* Options:\n{q_opts_fm}\n\n{answer_format_instruction}",
        first_mcq.get("image_details"),
        first_mcq.get("table_details"),
        json_dir,
        image_root,
    )

    initial_full_user_prompt = f"{scenario_prompt_text}\n\n{first_q_prompt_section}"
    initial_images_for_api = scenario_image_sources + first_q_image_sources

    conversation_history_for_api.append(
        {
            "role": "user",
            "content": initial_full_user_prompt,
            "image_paths_for_this_message": initial_images_for_api,
        }
    )

    for i, mcq_item in enumerate(mcqs):
        if i == 0:
            current_turn_text_prompt = initial_full_user_prompt
            current_turn_images = initial_images_for_api
        else:
            q_text = mcq_item.get("question", "")
            q_opts = format_options(mcq_item.get("options", {}))
            current_turn_text_prompt, current_turn_images = get_multimodal_prompt_text(
                f"* Question: {q_text}\n* Options:\n{q_opts}\n\n{answer_format_instruction}",
                mcq_item.get("image_details"),
                mcq_item.get("table_details"),
                json_dir,
                image_root,
            )
            conversation_history_for_api.append(
                {
                    "role": "user",
                    "content": current_turn_text_prompt,
                    "image_paths_for_this_message": current_turn_images,
                }
            )

        correct_answer_key = mcq_item.get("correct_answer")
        model_response_content = None
        api_error = None

        # Retry loop
        for attempt in range(MAX_API_CALL_ATTEMPTS):
            try:
                current_user_message_content = conversation_history_for_api[-1]["content"]
                current_user_message_images = conversation_history_for_api[-1].get(
                    "image_paths_for_this_message", []
                )

                api_result = await api_client.send_prompt_async(
                    text_prompt=current_user_message_content,
                    image_sources=current_user_message_images,
                )

                if api_result and api_result.get("status") == "success":
                    model_response_content = api_result.get("content")
                    if model_response_content:
                        api_error = None
                        break
                    api_error = "API returned success but empty content."
                elif api_result:
                    api_error = (
                        f"API Error (Status: {api_result.get('status', 'N/A')}): "
                        f"{api_result.get('message', 'Unknown API error')}"
                    )

                    # Critical errors: mark file for trash
                    error_msg = str(api_result.get("message", ""))
                    if "content_filter" in error_msg:
                        print(f"Critical API error for {model_id}: {error_msg}")
                        print(f"Marking file {json_file_path.name} for test_trash...")
                        FILES_TO_TRASH.add(json_file_path)

                        # Fill remaining MCQs as skipped
                        for j in range(i, len(mcqs)):
                            if j == i:
                                evaluation_results.append(
                                    {
                                        "model_id": model_id,
                                        "input_prompt_to_model": current_turn_text_prompt,
                                        "model_raw_response": None,
                                        "extracted_answer": None,
                                        "is_correct": False,
                                        "api_error": api_error,
                                        "image_paths_sent": [str(p) for p in current_turn_images],
                                    }
                                )
                            else:
                                q_text_skipped = mcqs[j].get("question", "SKIPPED_QUESTION")
                                q_opts_skipped = format_options(mcqs[j].get("options", {}))
                                skipped_prompt_text, _ = get_multimodal_prompt_text(
                                    f"* Question: {q_text_skipped}\n* Options:\n{q_opts_skipped}\n\n{answer_format_instruction}",
                                    mcqs[j].get("image_details"),
                                    mcqs[j].get("table_details"),
                                    json_dir,
                                    image_root,
                                )
                                evaluation_results.append(
                                    {
                                        "model_id": model_id,
                                        "input_prompt_to_model": skipped_prompt_text,
                                        "model_raw_response": None,
                                        "extracted_answer": None,
                                        "is_correct": False,
                                        "api_error": f"Skipped due to critical API error (content_filter) on question {i+1}",
                                        "image_paths_sent": [],
                                    }
                                )
                            if progress_bar:
                                progress_bar.update(1)
                        return evaluation_results
                else:
                    api_error = "API call did not return a result object."

            except Exception as e:
                api_error = f"API exception: {type(e).__name__} - {str(e)}"
                if "content_filter" in str(e):
                    print(f"Critical exception for {model_id}: {e}")
                    print(f"Marking file {json_file_path.name} for test_trash...")
                    FILES_TO_TRASH.add(json_file_path)

                    for j in range(i, len(mcqs)):
                        if j == i:
                            evaluation_results.append(
                                {
                                    "model_id": model_id,
                                    "input_prompt_to_model": current_turn_text_prompt,
                                    "model_raw_response": None,
                                    "extracted_answer": None,
                                    "is_correct": False,
                                    "api_error": api_error,
                                    "image_paths_sent": [str(p) for p in current_turn_images],
                                }
                            )
                        else:
                            q_text_skipped = mcqs[j].get("question", "SKIPPED_QUESTION")
                            q_opts_skipped = format_options(mcqs[j].get("options", {}))
                            skipped_prompt_text, _ = get_multimodal_prompt_text(
                                f"* Question: {q_text_skipped}\n* Options:\n{q_opts_skipped}\n\n{answer_format_instruction}",
                                mcqs[j].get("image_details"),
                                mcqs[j].get("table_details"),
                                json_dir,
                                image_root,
                            )
                            evaluation_results.append(
                                {
                                    "model_id": model_id,
                                    "input_prompt_to_model": skipped_prompt_text,
                                    "model_raw_response": None,
                                    "extracted_answer": None,
                                    "is_correct": False,
                                    "api_error": f"{model_id} skipped due to critical exception (content_filter) on question {i+1}",
                                    "image_paths_sent": [],
                                }
                            )
                        if progress_bar:
                            progress_bar.update(1)
                    return evaluation_results

            if attempt < MAX_API_CALL_ATTEMPTS - 1 and api_error:
                if VERBOSE_LOGGING:
                    print(f"Attempt {attempt + 1} failed for {model_id} on {json_file_path.name}, retrying...")
                await asyncio.sleep(RETRY_DELAY_SECONDS)

        if model_response_content:
            conversation_history_for_api.append({"role": "assistant", "content": model_response_content})

        extracted_ans = extract_answer(model_response_content) if model_response_content else None
        is_correct = check_correctness(extracted_ans, correct_answer_key) if correct_answer_key else False

        evaluation_results.append(
            {
                "model_id": model_id,
                "input_prompt_to_model": (
                    conversation_history_for_api[-2]["content"]
                    if len(conversation_history_for_api) > 1 and model_response_content
                    else current_turn_text_prompt
                ),
                "model_raw_response": model_response_content,
                "extracted_answer": extracted_ans,
                "is_correct": is_correct,
                "api_error": api_error,
                "image_paths_sent": [
                    str(p)
                    for p in (
                        conversation_history_for_api[-2].get("image_paths_for_this_message", [])
                        if len(conversation_history_for_api) > 1 and model_response_content
                        else current_turn_images
                    )
                ],
            }
        )

        if progress_bar:
            progress_bar.update(1)

        # If we failed to get content, skip remaining questions for this model
        if api_error and not model_response_content:
            if VERBOSE_LOGGING:
                print(f"API error for {model_id} / {json_file_path.name} Q{i+1}: {api_error}")

            for j in range(i + 1, len(mcqs)):
                q_text_skipped = mcqs[j].get("question", "SKIPPED_QUESTION")
                q_opts_skipped = format_options(mcqs[j].get("options", {}))
                skipped_prompt_text, _ = get_multimodal_prompt_text(
                    f"* Question: {q_text_skipped}\n* Options:\n{q_opts_skipped}\n\n{answer_format_instruction}",
                    mcqs[j].get("image_details"),
                    mcqs[j].get("table_details"),
                    json_dir,
                    image_root,
                )
                evaluation_results.append(
                    {
                        "model_id": model_id,
                        "input_prompt_to_model": skipped_prompt_text,
                        "model_raw_response": None,
                        "extracted_answer": None,
                        "is_correct": False,
                        "api_error": f"Skipped due to persistent API error on question {i+1}. Last error: {api_error}",
                        "image_paths_sent": [],
                    }
                )
                if progress_bar:
                    progress_bar.update(1)
            break

    return evaluation_results


async def evaluate_single_model_for_file(
    model_id: str,
    data: Dict[str, Any],
    json_path: Path,
    client_set: Dict[str, Any],
    image_root: Optional[Path] = None,
    progress_bar=None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Evaluate one model for one JSON file."""
    try:
        api_client = client_set.get(model_id)
        if not api_client:
            raise ValueError(f"No client found for model {model_id} in the provided client set")

        final_policy = data.get("exam_creation", {}).get("final_policy")
        if not final_policy:
            raise ValueError("'final_policy' not found")

        scenario_text = final_policy.get("scenario")
        mcqs_from_policy = final_policy.get("mcqs")

        if not isinstance(scenario_text, str) or not isinstance(mcqs_from_policy, list) or not mcqs_from_policy:
            raise ValueError("Missing/empty/invalid scenario or MCQs in 'final_policy'")

        model_eval_results = await evaluate_mcq_series_for_model(
            model_id=model_id,
            api_client=api_client,
            scenario_text=scenario_text,
            scenario_images=final_policy.get("scenario_image_details"),
            scenario_tables=final_policy.get("scenario_table_details"),
            mcqs=mcqs_from_policy,
            json_file_path=json_path,
            image_root=image_root,
            progress_bar=progress_bar,
        )
        return model_id, model_eval_results

    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Error evaluating model {model_id} for {json_path.name}: {type(e).__name__} - {e}")

        mcqs_count = len(data.get("exam_creation", {}).get("final_policy", {}).get("mcqs", []))
        if progress_bar:
            progress_bar.update(mcqs_count)

        error_results = [
            {
                "model_id": model_id,
                "input_prompt_to_model": "ERROR_DURING_PROCESSING_PIPELINE",
                "model_raw_response": None,
                "extracted_answer": None,
                "is_correct": False,
                "api_error": f"Processing error: {type(e).__name__} - {str(e)}",
                "image_paths_sent": [],
            }
            for _ in range(mcqs_count)
        ]
        return model_id, error_results


async def process_single_json_for_evaluation(
    json_path: Path,
    pending_models: Optional[List[str]] = None,
    image_root: Optional[Path] = None,
) -> Tuple[str, bool]:
    """Process one JSON file: evaluate pending models and write results back."""
    client_idx: Optional[int] = None

    try:
        if json_path in FILES_TO_TRASH:
            if VERBOSE_LOGGING:
                print(f"Skipping {json_path.name}: Marked for trash due to critical API error")
            return json_path.name, False

        pending_models = pending_models or MODEL_API_IDS.copy()
        if not pending_models or not pending_models[0]:
            if VERBOSE_LOGGING:
                print(f"Skipping {json_path.name}: No model configured")
            return json_path.name, False

        client_idx = await acquire_client_set()
        client_set = CLIENT_POOL[client_idx]
        if VERBOSE_LOGGING:
            print(f"File {json_path.name} acquired client set #{client_idx + 1}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error reading JSON {json_path.name}: {e}")
            return json_path.name, False

        output_data = copy.deepcopy(data)
        final_policy = output_data.get("exam_creation", {}).get("final_policy")
        if not final_policy:
            if VERBOSE_LOGGING:
                print(f"Skipping {json_path.name}: 'final_policy' not found")
            return json_path.name, False

        mcqs = final_policy.get("mcqs", [])
        total_questions = len(mcqs)

        # Remove any previous results for pending models (re-run cleanly)
        for mcq_item in mcqs:
            mcq_item.setdefault("evaluate", [])
            mcq_item["evaluate"] = [
                r for r in mcq_item["evaluate"] if r.get("model_id") not in pending_models
            ]

        total_tasks = len(pending_models) * total_questions
        file_progress = tqdm(
            total=total_tasks,
            desc=f"{json_path.name[:20]:<20}",
            unit="task",
            leave=False,
            disable=not bool(total_tasks),
        )

        semaphore = asyncio.Semaphore(min(MAX_CONCURRENT_MODELS, len(pending_models)))

        async def evaluate_with_semaphore(mid: str):
            async with semaphore:
                return await evaluate_single_model_for_file(mid, data, json_path, client_set, image_root, file_progress)

        try:
            model_tasks = [evaluate_with_semaphore(mid) for mid in pending_models]
            model_results = await asyncio.gather(*model_tasks, return_exceptions=True)

            if json_path in FILES_TO_TRASH:
                file_progress.close()
                return json_path.name, False

            for result in model_results:
                if isinstance(result, Exception):
                    if VERBOSE_LOGGING:
                        print(f"Model evaluation failed: {type(result).__name__} - {result}")
                    continue

                model_id, eval_results = result
                for i, mcq_item in enumerate(final_policy["mcqs"]):
                    if i < len(eval_results):
                        mcq_item["evaluate"].append(eval_results[i])
                    else:
                        mcq_item["evaluate"].append(
                            {
                                "model_id": model_id,
                                "input_prompt_to_model": "ERROR_RESULT_MISMATCH",
                                "model_raw_response": None,
                                "extracted_answer": None,
                                "is_correct": False,
                                "api_error": "Mismatch in returned evaluation results count",
                                "image_paths_sent": [],
                            }
                        )

        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error during evaluation for {json_path.name}: {type(e).__name__} - {e}")
            return json_path.name, False
        finally:
            file_progress.close()

        if json_path in FILES_TO_TRASH:
            return json_path.name, False

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            return json_path.name, True
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"Error writing JSON {json_path.name}: {e}")
            return json_path.name, False

    finally:
        if client_idx is not None:
            await release_client_set(client_idx)
            if VERBOSE_LOGGING:
                print(f"File {json_path.name} released client set #{client_idx + 1}")


async def main_evaluation_async(json_folder_str: str, enable_resume: bool = True, image_root: Optional[Path] = None):
    """Main evaluation entrypoint (supports resume)."""
    global FILES_TO_TRASH
    FILES_TO_TRASH = set()

    input_path = Path(json_folder_str)
    if not input_path.is_dir():
        print(f"Error: '{json_folder_str}' is not a valid directory.")
        return

    json_files = sorted([f for f in input_path.glob("*.json") if f.is_file()])
    if not json_files:
        print(f"No JSON files found in '{json_folder_str}'.")
        return

    print(f"Found {len(json_files)} JSON files")

    # Basic config validation
    if not YOUR_API_KEY:
        print("Error: YOUR_API_KEY is not set. Provide it via --your-api-key or env YOUR_API_KEY.")
        return
    if not API_BASE_URL:
        print("Error: API_BASE_URL is not set.")
        return
    if not MODEL_API_IDS or not MODEL_API_IDS[0]:
        print("Error: MODEL_API_IDS is empty. Provide it via --model-api-id or env MODEL_API_ID.")
        return

    print(f"Models: {', '.join(MODEL_API_IDS)}")
    print(f"Concurrency: {MAX_CONCURRENT_FILES} files, {MAX_CONCURRENT_MODELS} models/file")

    await initialize_client_pool()
    if not CLIENT_POOL:
        print("Error: No client slots were successfully initialized.")
        return

    if enable_resume:
        print("Analyzing completion status (resume mode)...")
        status_summary = analyze_folder_status(input_path)
        print_status_summary(status_summary)

        if status_summary["completed_files"] == status_summary["total_files"]:
            print("All files are already completed.")
            cleanup_client_pool()
            return

        if status_summary["completed_model_tasks"] > 0:
            print(f"Resume mode: skipping {status_summary['completed_model_tasks']} completed tasks")

    file_tasks: List[Tuple[Path, List[str]]] = []
    total_pending_tasks = 0

    for json_file in json_files:
        if enable_resume:
            pending_models, _, errors = get_pending_models_for_file(json_file)
            if errors and VERBOSE_LOGGING:
                print(f"Warning for {json_file.name}: {'; '.join(errors)}")

            if pending_models:
                file_tasks.append((json_file, pending_models))
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    mcq_count = len(data.get("exam_creation", {}).get("final_policy", {}).get("mcqs", []))
                    total_pending_tasks += len(pending_models) * mcq_count
                except Exception:
                    total_pending_tasks += len(pending_models) * 5
        else:
            file_tasks.append((json_file, MODEL_API_IDS))
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                mcq_count = len(data.get("exam_creation", {}).get("final_policy", {}).get("mcqs", []))
                total_pending_tasks += len(MODEL_API_IDS) * mcq_count
            except Exception:
                total_pending_tasks += len(MODEL_API_IDS) * 5

    if not file_tasks:
        print("All tasks are already completed.")
        cleanup_client_pool()
        return

    print(f"Processing {len(file_tasks)} files, ~{total_pending_tasks} total tasks")

    main_progress = tqdm(total=len(file_tasks), desc="Overall Progress", unit="file", position=0)

    async def process_with_progress(file_path: Path, pending: List[str]):
        result = await process_single_json_for_evaluation(file_path, pending, image_root)
        main_progress.update(1)
        return result

    start_time = time.time()
    tasks = [process_with_progress(fp, pm) for fp, pm in file_tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    main_progress.close()
    elapsed = time.time() - start_time

    if FILES_TO_TRASH:
        print(f"Moving {len(FILES_TO_TRASH)} files to test_trash due to critical API errors...")
        for file_path in FILES_TO_TRASH:
            move_file_to_trash(file_path)

    succeeded = sum(1 for r in results if isinstance(r, tuple) and r[1])
    failed = len(results) - succeeded

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Succeeded: {succeeded} files")
    print(f"Failed:    {failed} files")
    if FILES_TO_TRASH:
        print(f"Moved to test_trash: {len(FILES_TO_TRASH)} files")

    if len(file_tasks) > 0:
        print(f"Avg time: {elapsed / len(file_tasks):.1f}s per file")
        print(f"Success rate: {(succeeded / len(file_tasks) * 100):.1f}%")

    if failed > 0 and VERBOSE_LOGGING:
        print("\nFailed files:")
        for i, result in enumerate(results):
            if isinstance(result, Exception) or (isinstance(result, tuple) and not result[1]):
                filename = file_tasks[i][0].name
                print(f"  - {filename}")

    cleanup_client_pool()


# =============================================================================
# CLI entrypoint (non-interactive)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ Evaluation System")

    # CHANGED: accept jsonl-path instead of json-folder-path
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to a JSONL file (one sample per line).")

    parser.add_argument("--model-api-id", type=str, required=True, help="API ID of the model to test.")
    parser.add_argument("--api-base-url", type=str, default=os.getenv("API_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--your-api-key", type=str, default=os.getenv("YOUR_API_KEY", ""))
    parser.add_argument("--image-root-path", type=str, default=None, help="Optional root directory for images.")
    parser.add_argument("--use-url", action="store_true", help="Use image URLs instead of local paths.")
    parser.add_argument("--resume", action="store_true", default=True, help="Enable resume mode.")
    parser.add_argument("--non-interactive", action="store_true", help="Run using only CLI args (no prompts).")

    args = parser.parse_args()

    # Override globals from CLI args (keeps behavior consistent with controller usage)
    MODEL_API_IDS = [args.model_api_id.strip()] if args.model_api_id else [""]
    API_BASE_URL = (args.api_base_url or "").strip()
    YOUR_API_KEY = (args.your_api_key or "").strip()
    IMAGE_ROOT_PATH = Path(args.image_root_path).resolve() if args.image_root_path else None
    USE_URL = bool(args.use_url)
    enable_resume = bool(args.resume)

    jsonl_path = Path(args.jsonl_path).resolve()
    if not jsonl_path.is_file():
        print(f"Error: Invalid JSONL file: '{jsonl_path}'")
        raise SystemExit(1)

    # NEW: expand jsonl into <jsonl_parent>/json/PMCxxx.json
    resolved_path = prepare_json_folder_from_jsonl_to_parent_json_dir(jsonl_path)

    if not resolved_path.is_dir():
        print(f"Error: Invalid directory after jsonl expansion: '{resolved_path}'")
        raise SystemExit(1)

    print("MCQ Evaluation System (Non-Interactive Mode)")
    print("=" * 55)
    print(f"Input JSONL:   {jsonl_path}")
    print(f"Target Folder: {resolved_path}   (generated from JSONL)")
    print(f"Model to Test: {MODEL_API_IDS[0]}")
    print(f"API Base URL:  {API_BASE_URL}")
    print(f"Resume:        {'ON' if enable_resume else 'OFF'}")
    if IMAGE_ROOT_PATH:
        print(f"Image Root:    {IMAGE_ROOT_PATH}")
    print(f"Image Mode:    {'URL' if USE_URL else 'Local Path'}")
    print("Starting evaluation...")

    asyncio.run(main_evaluation_async(str(resolved_path), enable_resume, IMAGE_ROOT_PATH))

