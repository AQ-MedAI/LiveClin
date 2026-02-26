# LiveClin
LiveClin: A Live Clinical Benchmark without Leakage

<p align="center">
   📃 <a href="https://arxiv.org/abs/2602.16747" target="_blank">Paper</a> • 🤗 <a href="" target="_blank">Dataset</a>


## 🌈 Update
* **[2026.02.21]** [Paper](https://arxiv.org/abs/2602.16747) released.
* **[2026.02.15]** 🎉🎉🎉 LiveClin is published！🎉🎉🎉


## Results

![LiveClin_result1](assets/result.png)



## Project Structure

、、、
LiveClin/
├── assets/
├── data/
│   ├── 2025H1/
│   │   ├── 2025h1.jsonl
│   │   └── images/                # unzip images.zip here
│   └── ...                        # other releases/periods
├── demo/
│   ├── demo.jsonl
│   └── images/
├── api_client.py
├── core.py
├── evaluate.py
├── stats_analyzer.py
└── README.md

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



##  Citation
Please use the following citation if you intend to use our dataset for training or evaluation:

```
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