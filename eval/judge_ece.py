"""LLM-as-a-Judge evaluation for Error Cause Explanation (ECE).

Usage:
    python -m eval.judge_ece \
        --predictions results/ece_gpt4o_primary.jsonl \
        --judge-model o3-mini \
        --output results/ece_gpt4o_primary_judged.jsonl
"""

import argparse
import os

from openai import OpenAI
from tqdm import tqdm

from eval.prompts import build_judge_prompt
from eval.utils import load_results, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge ECE predictions using LLM-as-a-Judge"
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to ECE prediction JSONL file")
    parser.add_argument("--judge-model", type=str, default="o3-mini",
                        help="Judge model name (default: o3-mini)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with judge results")
    parser.add_argument("--api-base", type=str, default=None,
                        help="Custom API base URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples")
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

    # Load predictions
    predictions = load_results(args.predictions)
    if args.limit:
        predictions = predictions[:args.limit]
    print(f"Judging {len(predictions)} predictions with {args.judge_model}")

    correct = 0
    total = 0
    judged_results = []

    for pred in tqdm(predictions, desc="Judging ECE"):
        if pred.get("prediction", "").startswith("ERROR:"):
            judge_result = "INCORRECT"
        else:
            prompt = build_judge_prompt(
                prediction=pred["prediction"],
                ground_truth=pred["ground_truth"],
            )
            try:
                response = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=16,
                )
                judge_output = response.choices[0].message.content.strip().upper()
                judge_result = (
                    "CORRECT" if "CORRECT" in judge_output
                    and "INCORRECT" not in judge_output
                    else "INCORRECT"
                )
            except Exception as e:
                print(f"  Judge error: {e}")
                judge_result = "ERROR"

        is_correct = judge_result == "CORRECT"
        if is_correct:
            correct += 1
        total += 1

        judged_results.append({
            **pred,
            "judge_model": args.judge_model,
            "judge_result": judge_result,
            "is_correct": is_correct,
        })

    save_results(judged_results, args.output)

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"ECE Judge Results")
    print(f"{'='*50}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("Done!")


if __name__ == "__main__":
    main()
