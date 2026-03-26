"""Run Error Cause Classification (ECC) evaluation on ScratchMath.

Usage:
    python -m eval.run_ecc \
        --model gpt-4o \
        --subset primary \
        --output results/ecc_gpt4o_primary.jsonl

    # With Chain-of-Thought prompting:
    python -m eval.run_ecc \
        --model gpt-4o \
        --subset primary \
        --output results/ecc_gpt4o_primary_cot.jsonl \
        --cot
"""

import argparse
import os
import re

from openai import OpenAI
from tqdm import tqdm

from eval.prompts import ERROR_CATEGORIES, build_ecc_prompt
from eval.utils import build_multimodal_messages, load_dataset_split, save_results


def parse_prediction(text: str) -> int:
    """Parse model output to extract predicted category index.

    Args:
        text: Raw model output text.

    Returns:
        Predicted category index (0-6), or -1 if parsing fails.
    """
    # Try to match "Category: N. ..." format
    m = re.search(r"Category:\s*(\d+)", text)
    if m:
        idx = int(m.group(1)) - 1  # 1-indexed to 0-indexed
        if 0 <= idx < len(ERROR_CATEGORIES):
            return idx

    # Try to match any category name directly
    for i, cat in enumerate(ERROR_CATEGORIES):
        if cat in text:
            return i

    return -1


def compute_accuracy(results: list[dict]) -> dict:
    """Compute weighted-average accuracy for ECC.

    Args:
        results: List of result dicts with 'predicted_category' and
                 'ground_truth_category' keys.

    Returns:
        Dict with overall accuracy and per-category accuracy.
    """
    correct = 0
    total = 0
    per_cat = {i: {"correct": 0, "total": 0} for i in range(len(ERROR_CATEGORIES))}

    for r in results:
        gt = r["ground_truth_category"]
        pred = r["predicted_category"]
        per_cat[gt]["total"] += 1
        total += 1
        if pred == gt:
            correct += 1
            per_cat[gt]["correct"] += 1

    overall = correct / total * 100 if total > 0 else 0.0
    per_cat_acc = {}
    for i, cat in enumerate(ERROR_CATEGORIES):
        t = per_cat[i]["total"]
        c = per_cat[i]["correct"]
        per_cat_acc[cat] = c / t * 100 if t > 0 else 0.0

    # Weighted average (weighted by class frequency)
    weighted = sum(
        per_cat_acc[cat] * per_cat[i]["total"]
        for i, cat in enumerate(ERROR_CATEGORIES)
    ) / total if total > 0 else 0.0

    return {
        "overall_accuracy": overall,
        "weighted_accuracy": weighted,
        "per_category": per_cat_acc,
        "total": total,
        "correct": correct,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ECC evaluation on ScratchMath"
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
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max output tokens (default: 512)")
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
    for i, sample in enumerate(tqdm(ds, desc="ECC Evaluation")):
        prompt = build_ecc_prompt(
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
            raw_output = response.choices[0].message.content.strip()
            predicted_category = parse_prediction(raw_output)
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            raw_output = f"ERROR: {e}"
            predicted_category = -1

        results.append({
            "question_id": sample["question_id"],
            "subset": args.subset,
            "task": "ecc",
            "model": args.model,
            "raw_output": raw_output,
            "predicted_category": predicted_category,
            "ground_truth_category": sample["error_category"],
        })

    save_results(results, args.output)

    # Compute and print accuracy
    metrics = compute_accuracy(results)
    print(f"\n{'='*50}")
    print(f"ECC Results ({args.model} on {args.subset})")
    print(f"{'='*50}")
    print(f"Weighted Accuracy: {metrics['weighted_accuracy']:.1f}%")
    print(f"Overall Accuracy:  {metrics['overall_accuracy']:.1f}%")
    print(f"Total: {metrics['total']}, Correct: {metrics['correct']}")
    print(f"\nPer-category accuracy:")
    for cat, acc in metrics["per_category"].items():
        print(f"  {cat}: {acc:.1f}%")
    print("Done!")


if __name__ == "__main__":
    main()
