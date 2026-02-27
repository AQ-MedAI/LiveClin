"""Core evaluation engine: async concurrent case evaluation with resume."""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm.asyncio import tqdm

from . import EvalConfig, __version__
from .client import APIClient
from .data import get_case_id
from .utils import (
    build_followup_prompt,
    build_scenario_prompt,
    extract_answer,
)


# ── Single-case evaluation (multi-turn conversation) ────────────────────

async def evaluate_case(
    case: Dict[str, Any],
    config: EvalConfig,
    image_root: Optional[Path],
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate all MCQs for one clinical case via multi-turn conversation.
    Returns a result dict ready for the output JSON.
    """
    policy = case.get("exam_creation", {}).get("final_policy", {})
    scenario = policy.get("scenario", "")
    mcqs = policy.get("mcqs", [])
    scenario_images = policy.get("scenario_image_details")
    scenario_tables = policy.get("scenario_table_details")

    case_result: Dict[str, Any] = {
        "case_id": get_case_id(case),
        "pmc": case.get("pmc"),
        "doi": case.get("doi"),
        "rarity": case.get("Rarity", "unknown"),
        "level1": case.get("Level1", ""),
        "total_mcqs": len(mcqs),
        "correct_mcqs": 0,
        "mcqs": [],
    }

    if not mcqs:
        return case_result

    client = APIClient(config)

    for i, mcq in enumerate(mcqs):
        mcq_result: Dict[str, Any] = {
            "stage": mcq.get("stage", ""),
            "correct_answer": mcq.get("correct_answer", ""),
            "extracted_answer": None,
            "is_correct": False,
            "model_response": None,
            "error": None,
        }

        try:
            if i == 0:
                prompt, images = build_scenario_prompt(
                    scenario, mcq, scenario_images, scenario_tables,
                    use_url=config.use_url, image_root=image_root,
                )
            else:
                prompt, images = build_followup_prompt(
                    mcq, use_url=config.use_url, image_root=image_root,
                )

            resp = await client.send(prompt, images)

            if resp["status"] == "success":
                content = resp["content"]
                mcq_result["model_response"] = content
                mcq_result["extracted_answer"] = extract_answer(content)
                correct = mcq.get("correct_answer", "").upper()
                mcq_result["is_correct"] = (
                    mcq_result["extracted_answer"] is not None
                    and mcq_result["extracted_answer"] == correct
                )
            else:
                mcq_result["error"] = resp.get("message", "Unknown error")
                # On API failure, skip remaining MCQs for this case
                case_result["mcqs"].append(mcq_result)
                if progress:
                    progress.update(1)
                for j in range(i + 1, len(mcqs)):
                    case_result["mcqs"].append({
                        "stage": mcqs[j].get("stage", ""),
                        "correct_answer": mcqs[j].get("correct_answer", ""),
                        "extracted_answer": None,
                        "is_correct": False,
                        "model_response": None,
                        "error": f"Skipped: prior question failed ({mcq_result['error']})",
                    })
                    if progress:
                        progress.update(1)
                break

        except Exception as e:
            mcq_result["error"] = f"{type(e).__name__}: {e}"
            case_result["mcqs"].append(mcq_result)
            if progress:
                progress.update(1)
            for j in range(i + 1, len(mcqs)):
                case_result["mcqs"].append({
                    "stage": mcqs[j].get("stage", ""),
                    "correct_answer": mcqs[j].get("correct_answer", ""),
                    "extracted_answer": None,
                    "is_correct": False,
                    "model_response": None,
                    "error": f"Skipped: prior question raised {type(e).__name__}",
                })
                if progress:
                    progress.update(1)
            break

        case_result["mcqs"].append(mcq_result)
        if progress:
            progress.update(1)

    case_result["correct_mcqs"] = sum(
        1 for m in case_result["mcqs"] if m["is_correct"]
    )
    return case_result


# ── Results file I/O ────────────────────────────────────────────────────

def _init_results(config: EvalConfig) -> Dict[str, Any]:
    return {
        "meta": {
            "model": config.model,
            "dataset": config.dataset,
            "image_mode": config.image_mode,
            "api_base": config.api_base,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "liveclin_version": __version__,
        },
        "summary": {},
        "cases": [],
    }


def _load_existing_results(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_results(results: Dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Main runner ─────────────────────────────────────────────────────────

async def run_evaluation(
    cases: List[Dict[str, Any]],
    config: EvalConfig,
    image_root: Optional[Path],
) -> Dict[str, Any]:
    """
    Run evaluation on all cases with concurrency control and resume support.
    Returns the full results dict (meta + summary + cases).
    """
    results = _init_results(config)
    output_path = config.output_path

    # Resume: load completed case IDs
    completed_ids: Set[str] = set()
    if config.resume:
        existing = _load_existing_results(output_path)
        if existing.get("cases"):
            results["cases"] = existing["cases"]
            completed_ids = {c["case_id"] for c in results["cases"] if c.get("case_id")}
            print(f"Resuming: {len(completed_ids)} cases already completed, skipping.")

    pending = [c for c in cases if get_case_id(c) not in completed_ids]

    if not pending:
        print("All cases already evaluated.")
        results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        return results

    total_mcqs = sum(
        len(c.get("exam_creation", {}).get("final_policy", {}).get("mcqs", []))
        for c in pending
    )

    print(f"Evaluating {len(pending)} cases ({total_mcqs} MCQs), concurrency={config.concurrency}")
    print(f"Model: {config.model}  Image mode: {config.image_mode}")
    print()

    sem = asyncio.Semaphore(config.concurrency)
    progress = tqdm(total=total_mcqs, desc="Evaluating", unit="mcq")
    save_lock = asyncio.Lock()
    start_time = time.time()

    async def eval_with_semaphore(case: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            result = await evaluate_case(case, config, image_root, progress)
        # Incremental save
        async with save_lock:
            results["cases"].append(result)
            _save_results(results, output_path)
        return result

    tasks = [eval_with_semaphore(c) for c in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    progress.close()
    elapsed = time.time() - start_time

    results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    _save_results(results, output_path)

    print()
    print(f"Evaluation completed in {elapsed:.1f}s ({len(pending)} cases)")
    print(f"Results saved to: {output_path}")

    return results
