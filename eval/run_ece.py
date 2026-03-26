"""Run Error Cause Explanation (ECE) evaluation on ScratchMath.

Usage:
    python -m eval.run_ece \
        --model gpt-4o \
        --subset primary \
        --output results/ece_gpt4o_primary.jsonl

    # With Chain-of-Thought prompting:
    python -m eval.run_ece \
        --model gpt-4o \
        --subset primary \
        --output results/ece_gpt4o_primary_cot.jsonl \
        --cot

    # Using a custom API base (e.g., vLLM-served open-source model):
    python -m eval.run_ece \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --subset primary \
        --output results/ece_qwen25vl_primary.jsonl \
        --api-base http://localhost:8000/v1
"""

import argparse
import os

from openai import OpenAI
from tqdm import tqdm

from eval.prompts import build_ece_prompt
from eval.utils import build_multimodal_messages, load_dataset_split, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ECE evaluation on ScratchMath"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., gpt-4o, o4-mini)")
    parser.add_argument("--subset", type=str, default="primary",
                        choices=["primary", "middle"],
                        help="Dataset subset to evaluate")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--api-base", type=str, default=None,
                        help="Custom API base URL (for vLLM, etc.)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max output tokens (default: 2048)")
    parser.add_argument("--cot", action="store_true",
                        help="Use Chain-of-Thought prompting")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize client
    client_kwargs = {}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    elif "OPENAI_API_KEY" not in os.environ and not args.api_base:
        raise ValueError(
            "Set OPENAI_API_KEY or provide --api-key / --api-base"
        )
    client = OpenAI(**client_kwargs)

    # Load dataset
    print(f"Loading ScratchMath/{args.subset}...")
    ds = load_dataset_split(args.subset)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Evaluating {len(ds)} samples with model={args.model}")

    results = []
    for i, sample in enumerate(tqdm(ds, desc="ECE Evaluation")):
        prompt = build_ece_prompt(
            question=sample["question"],
            answer=sample["answer"],
            solution=sample["solution"],
            student_answer=sample["student_answer"],
            cot=args.cot,
        )
        messages = build_multimodal_messages(prompt, sample["student_scratchwork"])

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            prediction = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            prediction = f"ERROR: {e}"

        results.append({
            "question_id": sample["question_id"],
            "subset": args.subset,
            "task": "ece",
            "model": args.model,
            "prediction": prediction,
            "ground_truth": sample["error_explanation"],
            "error_category": sample["error_category"],
        })

    save_results(results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
