"""Prompt templates for ScratchMath evaluation."""

# 7 error categories (Chinese labels as used in the dataset)
ERROR_CATEGORIES = [
    "计算错误",          # Calculation Error
    "题目理解错误",      # Question Comprehension Error
    "知识点错误",        # Knowledge Gap Error
    "答题技巧错误",      # Problem-Solving Strategy Error
    "手写誊抄错误",      # Handwriting Transcription Error
    "逻辑推理错误",      # Logical Reasoning Error
    "注意力与细节错误",  # Attention & Detail Error
]

# English labels for display
ERROR_CATEGORIES_EN = [
    "Calculation Error",
    "Question Comprehension Error",
    "Knowledge Gap Error",
    "Problem-Solving Strategy Error",
    "Handwriting Transcription Error",
    "Logical Reasoning Error",
    "Attention & Detail Error",
]

CATEGORY_ZH_TO_EN = dict(zip(ERROR_CATEGORIES, ERROR_CATEGORIES_EN))
CATEGORY_EN_TO_ZH = dict(zip(ERROR_CATEGORIES_EN, ERROR_CATEGORIES))


def build_ece_prompt(question: str, answer: str, solution: str,
                     student_answer: str, cot: bool = False) -> str:
    """Build the Error Cause Explanation prompt.

    Args:
        question: The math problem text.
        answer: The correct answer.
        solution: The step-by-step reference solution.
        student_answer: The student's incorrect answer.
        cot: Whether to use Chain-of-Thought prompting.

    Returns:
        The formatted prompt string (text part; image is sent separately).
    """
    base = (
        "You are an experienced mathematics teacher. A student solved the "
        "following math problem incorrectly. Based on the student's handwritten "
        "scratchwork (provided as an image), the problem statement, the correct "
        "answer, and the student's answer, analyze and explain the specific "
        "reason for the student's error.\n\n"
        f"**Problem:** {question}\n"
        f"**Correct Answer:** {answer}\n"
        f"**Reference Solution:** {solution}\n"
        f"**Student's Answer:** {student_answer}\n\n"
        "The image shows the student's handwritten scratchwork.\n\n"
    )
    if cot:
        base += (
            "Think step by step:\n"
            "1. First, read and understand the student's scratchwork in the image.\n"
            "2. Compare the student's work with the reference solution.\n"
            "3. Identify where the student made errors.\n"
            "4. Explain the specific cause of the error.\n\n"
        )
    base += "Please provide a concise explanation of the student's error cause."
    return base


def build_ecc_prompt(question: str, answer: str, solution: str,
                     student_answer: str, cot: bool = False) -> str:
    """Build the Error Cause Classification prompt.

    Args:
        question: The math problem text.
        answer: The correct answer.
        solution: The step-by-step reference solution.
        student_answer: The student's incorrect answer.
        cot: Whether to use Chain-of-Thought prompting.

    Returns:
        The formatted prompt string (text part; image is sent separately).
    """
    categories_str = "\n".join(
        f"  {i+1}. {zh} ({en})"
        for i, (zh, en) in enumerate(zip(ERROR_CATEGORIES, ERROR_CATEGORIES_EN))
    )

    base = (
        "You are an experienced mathematics teacher. A student solved the "
        "following math problem incorrectly. Based on the student's handwritten "
        "scratchwork (provided as an image), classify the student's error into "
        "exactly one of the following categories.\n\n"
        f"**Problem:** {question}\n"
        f"**Correct Answer:** {answer}\n"
        f"**Reference Solution:** {solution}\n"
        f"**Student's Answer:** {student_answer}\n\n"
        "The image shows the student's handwritten scratchwork.\n\n"
        f"**Error Categories:**\n{categories_str}\n\n"
    )
    if cot:
        base += (
            "Think step by step:\n"
            "1. First, read and understand the student's scratchwork in the image.\n"
            "2. Compare the student's work with the reference solution.\n"
            "3. Identify the type of error.\n"
            "4. Select the most appropriate category.\n\n"
        )
    base += (
        "Respond with ONLY the category number and name in the format: "
        "\"Category: <number>. <Chinese name>\"\n"
        "For example: \"Category: 1. 计算错误\""
    )
    return base


def build_judge_prompt(prediction: str, ground_truth: str) -> str:
    """Build the LLM-as-a-Judge prompt for ECE evaluation.

    Args:
        prediction: The model's generated error explanation.
        ground_truth: The ground truth error explanation.

    Returns:
        The judge prompt string.
    """
    return (
        "You are an expert judge evaluating error cause explanations for "
        "student math mistakes. Compare the predicted explanation with the "
        "ground truth and determine if they describe the same error cause.\n\n"
        "The prediction does NOT need to be word-for-word identical. It should "
        "correctly identify the same fundamental error that the student made. "
        "Minor differences in wording or additional correct details are acceptable.\n\n"
        f"**Ground Truth Explanation:**\n{ground_truth}\n\n"
        f"**Predicted Explanation:**\n{prediction}\n\n"
        "Does the prediction correctly identify the same error cause as the "
        "ground truth? Respond with ONLY \"CORRECT\" or \"INCORRECT\"."
    )
