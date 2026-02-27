# LiveClin

**[ICLR 2026] LiveClin: A Live Clinical Benchmark without Leakage**

<p align="center">
   📃 <a href="https://arxiv.org/abs/2602.16747" target="_blank">Paper</a> &bull;
   🤗 <a href="https://huggingface.co/datasets/AQ-MedAI/LiveClin" target="_blank">Dataset</a> &bull;
   💻 <a href="https://github.com/AQ-MedAI/LiveClin" target="_blank">Code</a>
</p>

## Update

* **[2026.02.27]** Evaluation framework refactored — single CLI entry-point, auto-download from HuggingFace, streamlined results.
* **[2026.02.21]** [Paper](https://arxiv.org/abs/2602.16747) released.
* **[2026.02.15]** LiveClin is published!

## Overview

LiveClin is a contamination-free, continuously updated clinical benchmark for evaluating large language / vision-language models on realistic, multi-stage clinical case reasoning with medical images.

| Statistic               | 2025_H1       |
| ----------------------- | ------------- |
| Clinical cases          | 1,407         |
| Total MCQs              | 6,605         |
| MCQs per case           | 3–6 (avg 4.7) |
| Cases with images       | 1,333 (94.7%) |
| Total images            | 2,970         |
| ICD-10 chapters covered | 16            |
| Rare cases              | 1,181 (84%)   |
| Non-rare cases          | 226 (16%)     |

## Results

![LiveClin_result1](assets/result.png)

## Project Structure

```
LiveClin/                          # This GitHub repo
├── evaluate.py                    # Single CLI entry-point
├── liveclin/                      # Core evaluation package
│   ├── __init__.py                # EvalConfig dataclass
│   ├── client.py                  # Async API client (OpenAI-compatible)
│   ├── runner.py                  # Concurrent evaluation engine
│   ├── analyzer.py                # Results analysis & CLI summary
│   ├── data.py                    # HuggingFace data download & JSONL loading
│   └── utils.py                   # Answer extraction & prompt formatting
├── scripts/
│   └── serve_sglang.py            # SGLang deployment helper
├── requirements.txt
├── assets/
├── LICENSE
└── README.md

AQ-MedAI/LiveClin (HuggingFace)   # Dataset repo (data + images)
└── data/
    ├── demo/                      # 14-case preview subset
    ├── 2025_H1/                   # Full 2025-H1 benchmark
    │   ├── 2025_H1.jsonl
    │   └── image/
    └── 2025_H2/                   # (future)
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/AQ-MedAI/LiveClin.git
cd LiveClin
pip install -r requirements.txt
```

### 2. Evaluate

A single command downloads the data (if needed) and runs the full evaluation:

```bash
# Evaluate via a remote API (images sent as URLs)
python evaluate.py \
    --model gpt-4o \
    --api-base https://api.openai.com/v1 \
    --api-key sk-xxx \
    --image-mode url
```

```bash
# Evaluate a locally-served model (images sent as base64)
python evaluate.py \
    --model Qwen2.5-VL-7B-Instruct \
    --api-base http://localhost:8000/v1 \
    --api-key token \
    --image-mode local
```

The script will:
1. Auto-download the dataset from [HuggingFace](https://huggingface.co/datasets/AQ-MedAI/LiveClin) on first run
2. Run concurrent evaluation across all clinical cases
3. Print a brief summary to the terminal
4. Save detailed results to a JSON file (default: `results/<model>_<dataset>.json`)

### CLI Options

| Flag            | Description                              | Default   |
| --------------- | ---------------------------------------- | --------- |
| `--model`       | Model identifier (required)              | —         |
| `--api-base`    | API base URL (required)                  | —         |
| `--api-key`     | API key (required)                       | —         |
| `--image-mode`  | `url` or `local` (required)              | —         |
| `--dataset`     | Dataset config name                      | `2025_H1` |
| `--concurrency` | Max concurrent evaluations               | `50`      |
| `--output`      | Output JSON path                         | auto      |
| `--resume`      | Resume from existing results             | off       |
| `--temperature` | Sampling temperature                     | `0.0`     |
| `--max-tokens`  | Max tokens per response                  | `16384`   |
| `--data-dir`    | Local data directory                     | `data`    |
| `--jsonl-path`  | Override: direct path to JSONL file      | —         |
| `--image-root`  | Override: direct path to image directory | —         |

#### Data Path Resolution

Data paths are resolved with the following priority (highest first):

1. **`--jsonl-path` / `--image-root`** — Directly specify the JSONL file and image directory. Skips auto-download entirely. Useful when you already have the data locally.
2. **`--data-dir`** — Change the root directory for auto-download (e.g. `--data-dir /mnt/datasets`). The internal directory structure is managed automatically.
3. **Default** — No path arguments needed. The dataset is auto-downloaded from HuggingFace into `data/` on first run.

```bash
# Example: use your own local data
python evaluate.py \
    --model gpt-4o \
    --api-base https://api.openai.com/v1 \
    --api-key sk-xxx \
    --image-mode local \
    --jsonl-path /path/to/your/2025_H1.jsonl \
    --image-root /path/to/your/images/
```

### 3. Deploy Your Own Model (Optional)

If you want to evaluate your own model, deploy it with [SGLang](https://github.com/sgl-project/sglang) to expose an OpenAI-compatible API:

```bash
# Terminal 1: start the model server
python scripts/serve_sglang.py \
    --model-path /path/to/your-model \
    --tp 2 --dp 4 --port 8000

# Terminal 2: run evaluation
python evaluate.py \
    --model your-model-name \
    --api-base http://localhost:8000/v1 \
    --api-key token \
    --image-mode local
```

## Output Format

Results are saved as a single JSON file with the following structure:

```json
{
  "meta": {
    "model": "gpt-4o",
    "dataset": "2025_H1",
    "image_mode": "url",
    "started_at": "...",
    "finished_at": "..."
  },
  "summary": {
    "total_cases": 1407,
    "total_mcqs": 6605,
    "question_accuracy": ...,
    "case_accuracy": ...,
    "by_rarity": { "rare": {...}, "unrare": {...} },
    "by_chapter": { "Chapter 2: Neoplasms": {...}, ... }
  },
  "cases": [...]
}
```

## Load Data with `datasets`

```python
from datasets import load_dataset

ds = load_dataset("AQ-MedAI/LiveClin", "2025_H1", split="test")

case = ds[0]
fp = case["exam_creation"]["final_policy"]
print(fp["scenario"])
for mcq in fp["mcqs"]:
    print(f"[{mcq['stage']}] {mcq['question'][:80]}...")
    print(f"  Answer: {mcq['correct_answer']}")
```

## Citation

Please cite the following if you use LiveClin for training or evaluation:

```bibtex
@misc{wang2026liveclinliveclinicalbenchmark,
      title={LiveClin: A Live Clinical Benchmark without Leakage},
      author={Xidong Wang and Shuqi Guo and Yue Shen and Junying Chen and Jian Wang and Jinjie Gu and Ping Zhang and Lei Liu and Benyou Wang},
      year={2026},
      eprint={2602.16747},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.16747},
}
```
