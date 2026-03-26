"""Utility functions for ScratchMath evaluation."""

import base64
import io
import json
import os
from typing import Optional

from datasets import load_dataset
from PIL import Image


def load_dataset_split(subset: str = "primary", split: str = "train"):
    """Load a ScratchMath dataset subset from HuggingFace.

    Args:
        subset: "primary" or "middle".
        split: Dataset split (default "train").

    Returns:
        A HuggingFace Dataset object.
    """
    return load_dataset("songdj/ScratchMath", subset, split=split)


def encode_image(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string.

    Args:
        image: A PIL Image object.
        format: Image format (default "PNG").

    Returns:
        Base64-encoded string of the image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_results(results: list[dict], output_path: str) -> None:
    """Save evaluation results to a JSONL file.

    Args:
        results: List of result dictionaries.
        output_path: Path to write the JSONL file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {output_path}")


def load_results(path: str) -> list[dict]:
    """Load results from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of result dictionaries.
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def build_multimodal_messages(text_prompt: str, image: Image.Image,
                              detail: str = "auto") -> list[dict]:
    """Build OpenAI-compatible multimodal messages.

    Args:
        text_prompt: The text part of the prompt.
        image: A PIL Image of the student's scratchwork.
        detail: Image detail level ("auto", "low", "high").

    Returns:
        List of message dicts for the OpenAI chat API.
    """
    b64 = encode_image(image)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": detail,
                    },
                },
                {
                    "type": "text",
                    "text": text_prompt,
                },
            ],
        }
    ]
