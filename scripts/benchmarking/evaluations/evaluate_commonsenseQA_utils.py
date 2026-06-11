import re
import pandas as pd


def clean_response(response: str) -> str:
    if not isinstance(response, str):
        return ""
    match = re.search(r"<ANS>(.*?)</ANS>", response, re.IGNORECASE)
    if not match:
        match = re.search(r"ANS\s*>?\s*(\S.*)", response, re.IGNORECASE)

    extracted = match.group(1).strip() if match else response.strip()

    if extracted in {"</ANS>", "<ANS>", "ANS >", ">", ""}:
        extracted = ""

    return extracted


def extract_letter(response: str) -> str | None:
    """Extract a single choice letter (A-E) from a cleaned response.
    Handles formats like 'A', '(A)', 'A.', '(A) some text'.
    """
    if not response:
        return None
    m = re.match(r"^\(?([A-Ea-e])\)?[\s.)]*", response.strip())
    return m.group(1).upper() if m else None


def evaluate_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate CommonsenseQA responses across a full DataFrame.

    Expected columns:
      - 'output'    : model's raw response string
      - 'answerKey' : ground truth letter (A-E)

    Returns (augmented_df, stats) where augmented_df gains 'predicted' and
    'correct' columns and stats contains accuracy summary.
    """
    df = df.copy()
    df["predicted"] = df["output"].apply(lambda r: extract_letter(clean_response(r)))
    df["correct"] = df.apply(
        lambda row: (
            row["predicted"] is not None
            and row["predicted"] == str(row["answerKey"]).strip().upper()
        ),
        axis=1,
    )
    total = len(df)
    correct = int(df["correct"].sum())
    stats = {
        "dataset": "commonsenseqa",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_ans": int(df["predicted"].isna().sum()),
    }
    return df, stats