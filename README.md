# LiveClin
LiveClin: A Live Clinical Benchmark without Leakage

<p align="center">
   📃 <a href="" target="_blank">Paper</a> • 🤗 <a href="" target="_blank">Dataset</a>


## 🌈 Update
* **[2026.02.21]** [Paper]() released.
* **[2026.02.10]** 🎉🎉🎉 LiveClin is published！🎉🎉🎉


## Results

![LiveClin_result1](assets/result.png)



## Project Structure

、、、
LiveClin/
├── assets/                     # (optional) figures, logos, example outputs for documentation
├── data/                       # benchmark datasets (each release/period in its own folder)
│   ├── 2025H1/                 # 2025 first-half dataset
│   │   ├── 2025h1.jsonl        # input JSONL
│   │   └── images/             # images referenced by the dataset JSON/JSONL (after unzip images.zip)
│   └── ...
├── demo/                       # small demo dataset for quick testing
│   ├── demo.jsonl              # demo JSONL input
│   └── images/                 # images referenced by demo samples
├── core.py                     # core evaluation logic (reads JSONL, runs model inference, writes results back)
├── evaluate.py                 # controller script: start SGLang server -> run core.py -> stop server
├── stats_analyzer.py           # analyze evaluated results and generate summary reports
└── README.md                   # documentation
、、、



## Evaluate Pipeline

0. **Prepare Dataset**

   Take 2025H1 as an example
   ```bash
   cd data/2025H1
   unzip images.zip
   ```

1. **Evaluate**

   Please modify JSONL_PATH & IMAGE_ROOT_PATH in evalute.py (Line 15,16) first

   ```bash
   python evalute.py
   ```

2. **Analysis**

   ```bash
   python stats_analyzer.py
   ```
