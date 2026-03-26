# ScratchMath

[![Paper](https://img.shields.io/badge/Paper-AIED%202026-blue)](paper/ScratchMath_AIED2026.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/songdj/ScratchMath)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](LICENSE)

**Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math**

*Dingjie Song, Tianlong Xu, Yi-Fan Zhang, Hang Li, Zhiling Yan, Xing Fan, Haoyang Li, Lichao Sun, Qingsong Wen*

**AIED 2026** (27th International Conference on Artificial Intelligence in Education)

<p align="center">
  <img src="figs/overview.png" width="100%" alt="ScratchMath Overview"/>
</p>

## Overview

**ScratchMath** is a multimodal benchmark for evaluating the ability of Multimodal Large Language Models (MLLMs) to analyze handwritten mathematical scratchwork produced by real students. Unlike existing math benchmarks that focus on problem-solving, ScratchMath targets **error diagnosis** — identifying what type of mistake a student made and explaining why.

The benchmark contains **1,720 samples** of authentic student scratchwork (1,479 primary school + 241 middle school) with expert annotations, supporting two tasks:

- **Error Cause Explanation (ECE):** Generate a free-form explanation of the student's error.
- **Error Cause Classification (ECC):** Classify the error into one of 7 predefined categories.

## Tasks

### Error Cause Explanation (ECE)

Given a math problem, its correct answer, reference solution, the student's incorrect answer, and an image of the student's handwritten scratchwork, the model must explain the specific cause of the student's error.

**Evaluation:** LLM-as-a-Judge (using o3-mini, 88.6% agreement with human judges).

### Error Cause Classification (ECC)

Using the same inputs, the model must classify the error into one of 7 categories:

| # | Category (Chinese) | Category (English) |
|---|---|---|
| 1 | 计算错误 | Calculation Error |
| 2 | 题目理解错误 | Question Comprehension Error |
| 3 | 知识点错误 | Knowledge Gap Error |
| 4 | 答题技巧错误 | Problem-Solving Strategy Error |
| 5 | 手写誊抄错误 | Handwriting Transcription Error |
| 6 | 逻辑推理错误 | Logical Reasoning Error |
| 7 | 注意力与细节错误 | Attention & Detail Error |

**Evaluation:** Weighted-average accuracy.

## Dataset

The dataset is hosted on HuggingFace: [songdj/ScratchMath](https://huggingface.co/datasets/songdj/ScratchMath)

```python
from datasets import load_dataset

ds_primary = load_dataset("songdj/ScratchMath", "primary", split="train")
ds_middle = load_dataset("songdj/ScratchMath", "middle", split="train")
```

| Subset | Samples | Grade Level |
|--------|---------|-------------|
| `primary` | 1,479 | Grades 1-6 |
| `middle` | 241 | Grades 7-9 |

Each sample contains: `question_id`, `question`, `answer`, `solution`, `student_answer`, `student_scratchwork` (image), `error_category`, `error_explanation`.

## Leaderboard

Performance of state-of-the-art MLLMs on ScratchMath (human performance: 83.9% average):

| Model | #Params | ECE Primary | ECE Middle | ECC Primary | ECC Middle | Average |
|-------|---------|-------------|------------|-------------|------------|---------|
| o4-mini* | -- | **71.8** | **69.7** | **40.1** | 47.3 | **57.2** |
| Gemini 2.0 Flash Thinking* | -- | 65.9 | 61.0 | 43.9 | 47.3 | 54.5 |
| Gemini 2.0 Flash | -- | 52.2 | 46.9 | 38.6 | **49.0** | 46.7 |
| QVQ* | 72B | 57.5 | 56.8 | 12.7 | 17.0 | 36.0 |
| Qwen2.5-VL | 72B | 40.0 | 34.0 | 32.5 | 49.4 | 39.0 |
| Gemma-3 | 27B | 38.9 | 26.1 | 32.2 | 46.1 | 35.8 |
| Skywork-R1V* | 38B | 37.5 | 33.6 | 27.7 | 43.2 | 35.5 |
| GPT-4o | -- | 47.7 | 44.8 | 26.1 | 22.0 | 35.2 |
| InternVL2.5 | 78B | 27.1 | 24.5 | 30.7 | 44.8 | 31.8 |

*\* denotes reasoning models. Bold indicates best performance. Full results in the [paper](paper/ScratchMath_AIED2026.pdf).*

## Quick Start

### Installation

```bash
git clone https://github.com/ai-for-edu/ScratchMath.git
cd ScratchMath
pip install -r requirements.txt
```

### Run ECE Evaluation

```bash
export OPENAI_API_KEY="your-key"

# Evaluate with GPT-4o on primary subset
python -m eval.run_ece \
    --model gpt-4o \
    --subset primary \
    --output results/ece_gpt4o_primary.jsonl

# Judge the ECE results
python -m eval.judge_ece \
    --predictions results/ece_gpt4o_primary.jsonl \
    --judge-model o3-mini \
    --output results/ece_gpt4o_primary_judged.jsonl
```

### Run ECC Evaluation

```bash
# Evaluate with GPT-4o on primary subset
python -m eval.run_ecc \
    --model gpt-4o \
    --subset primary \
    --output results/ecc_gpt4o_primary.jsonl
```

### Using Open-Source Models (via vLLM)

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-VL-7B-Instruct

# Run evaluation
python -m eval.run_ece \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --subset primary \
    --output results/ece_qwen25vl_primary.jsonl \
    --api-base http://localhost:8000/v1 \
    --api-key dummy
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{song2026scratchmath,
  title={Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math},
  author={Song, Dingjie and Xu, Tianlong and Zhang, Yi-Fan and Li, Hang and Yan, Zhiling and Fan, Xing and Li, Haoyang and Sun, Lichao and Wen, Qingsong},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence in Education (AIED)},
  year={2026}
}
```

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
