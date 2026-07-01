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


def _text_to_label(response: str, choices: list) -> str | None:
    """Fallback: exact case-insensitive match of response text against choice texts.
    Strips punctuation first, mirroring NeurIPS evaluate_commonsenseQA.py which does
    re.sub(r"[^\w\s]", "", response) before comparing.
    """
    cleaned = re.sub(r"[^\w\s]", "", response).strip()
    for choice in choices:
        if cleaned.lower() == re.sub(r"[^\w\s]", "", choice["text"]).strip().lower():
            return choice["label"]
    return None


def evaluate_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate CommonsenseQA responses across a full DataFrame.

    Expected columns:
      - 'output'    : model's raw response string
      - 'answerKey' : ground truth letter (A-E)
      - 'question'  : original question dict containing 'choices' list

    Returns (augmented_df, stats) where augmented_df gains 'predicted' and
    'correct' columns and stats contains accuracy summary.
    """
    df = df.copy()

    predicted = []
    for _, row in df.iterrows():
        cleaned = clean_response(str(row["output"]))
        letter = extract_letter(cleaned)
        if letter is None and cleaned:
            # NeurIPS evaluator fallback: match full response text against choice texts
            choices = row.get("question", {}).get("choices", [])
            letter = _text_to_label(cleaned, choices)
        predicted.append(letter)

    df["predicted"] = predicted
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