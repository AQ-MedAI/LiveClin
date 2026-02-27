"""Data loading: HuggingFace download and JSONL parsing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


HF_REPO_ID = "AQ-MedAI/LiveClin"


def ensure_dataset(data_dir: str, dataset: str) -> Path:
    """
    Ensure the dataset JSONL exists locally, downloading from HuggingFace
    if needed. Returns the path to the JSONL file.
    """
    jsonl_path = Path(data_dir) / "data" / dataset / f"{dataset}.jsonl"

    if jsonl_path.is_file():
        return jsonl_path

    print(f"Data not found at {jsonl_path}. Downloading from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=data_dir,
        )
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for auto-download.\n"
            "  pip install huggingface_hub\n"
            "Or clone manually:\n"
            f"  git clone https://huggingface.co/datasets/{HF_REPO_ID} {data_dir}"
        )

    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Download finished but {jsonl_path} not found. "
            f"Check dataset name or data_dir."
        )

    print(f"Download complete: {jsonl_path}")
    return jsonl_path


def get_image_root(data_dir: str, dataset: str) -> Path:
    """Return the image root directory for a dataset."""
    return Path(data_dir) / "data" / dataset / "image"


def load_cases(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load all cases from a JSONL file."""
    cases: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i + 1}: {e}")
    return cases


def get_case_id(case: Dict[str, Any]) -> str:
    """Get a unique identifier for a case (PMC > PMID > DOI > index)."""
    pmc = case.get("pmc")
    if pmc:
        return str(pmc).strip()
    pmid = case.get("pmid") or case.get("PMID")
    if pmid:
        return f"PMID_{pmid}"
    doi = case.get("doi")
    if doi:
        return str(doi).strip()
    return ""
