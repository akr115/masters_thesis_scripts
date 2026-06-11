import re
from typing import Optional
import pandas as pd


def extract_ans_from_answer(answer: str) -> int | str:
    """Extract and clean ground truth from the '#### <value>' suffix.
    Returns int if convertible, otherwise the cleaned string.
    """
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    answer = answer.strip(" $%g")
    answer = re.sub(r",", "", answer)
    try:
        return int(answer)
    except ValueError:
        return answer


def extract_real_answers(response: str) -> Optional[int]:
    """Extract the final numerical answer from a model response.
    Strips <ANS> tags, then looks for a trailing = <number> or bare trailing number.
    Returns int or None if no valid answer found.
    """
    if not response or response == "ANS":
        return None

    cleaned = re.sub(r"(</?ANS>|ANS\s*<|ANS\s*>)", " ", response).strip()

    def parse_number(num_str: str) -> Optional[int]:
        try:
            return int(float(num_str.replace(",", "").replace("%", "")))
        except (ValueError, TypeError):
            return None

    # Final result after equals sign (e.g. "= 42 >>", "= $1,200")
    equals_match = re.search(
        r"=\s*\$?([0-9,]+(?:\.[0-9]+)?%?)\s*(?:>>)?(?:\s*>)?$", cleaned
    )
    if equals_match:
        return parse_number(equals_match.group(1))

    # Edge case: </N</ANS> pattern
    if re.search(r"<\/\d+<\/ANS>", cleaned):
        num_match = re.search(r"<\/(\d+)", cleaned)
        if num_match:
            return parse_number(num_match.group(1))

    # Trailing number (with optional $, commas, %)
    simple_match = re.search(
        r"\$?([0-9,]+(?:\.[0-9]+)?%?)\s*(?:>>)?(?:\s*>)?$", cleaned
    )
    if simple_match:
        return parse_number(simple_match.group(1))

    # Expressions ending in >> or >
    if ">>" in cleaned or cleaned.endswith(">"):
        expr_match = re.search(
            r"(?:\/[0-9]+)?[^=]*?(\d+(?:\.[0-9]+)?%?)\s*(?:>>|>)$", cleaned
        )
        if expr_match:
            return parse_number(expr_match.group(1))

    return None


def evaluate_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate GSM8K responses across a full DataFrame.

    Expected columns:
      - 'output' : model's raw response string
      - 'answer' : ground truth string containing '#### <number>'
    """
    df = df.copy()
    df["expected"] = df["answer"].apply(extract_ans_from_answer)
    df["predicted"] = df["output"].apply(extract_real_answers)
    df["correct"] = df.apply(
        lambda row: (
            isinstance(row["predicted"], int)
            and row["predicted"] == row["expected"]
        ),
        axis=1,
    )

    total = len(df)
    correct = int(df["correct"].sum())
    invalid = int(df["predicted"].isna().sum())
    stats = {
        "dataset": "gsm8k",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid_responses": invalid,
    }
    return df, stats