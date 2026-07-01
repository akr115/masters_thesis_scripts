import re
from typing import Optional
import pandas as pd


def clean_response(response: str) -> Optional[str]:
    """Extract actual response from <ANS> tags, handling all edge-case formats."""
    if not isinstance(response, str):
        return None
    response = response.strip()

    match = re.search(r"<ANS>(.*?)(?:</ANS>)?$", response, re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        # Nested empty tags like <A></A> — return the tag name
        if re.match(r"^<([A-Z0-9]+)>\s*</\1>$", extracted, re.IGNORECASE):
            tag_match = re.match(r"^<([A-Z0-9]+)>", extracted, re.IGNORECASE)
            if tag_match:
                return tag_match.group(1)
        # Parenthesised single char: (A)
        paren_match = re.match(r"^\(([A-Z0-9])\)>?$", extracted, re.IGNORECASE)
        if paren_match:
            return paren_match.group(1)
    else:
        # Fallback: any HTML-like tag content
        content_match = re.search(r"<[^>]+>([^<]+)</[^>]+>", response)
        if content_match:
            return content_match.group(1).strip()
        match = re.search(
            r"ANS\s*>?\s*(<([A-Z0-9/]+)>|\(?([A-Z0-9/]+)\)?)", response, re.IGNORECASE
        )
        extracted = (match.group(2) or match.group(3)) if match else response.strip()

    paren_match = re.match(r"^\(([A-Z0-9])\)$", extracted, re.IGNORECASE)
    if paren_match:
        return paren_match.group(1)

    extracted = re.sub(r"<.*?>", "", extracted).strip()
    extracted = re.sub(r"\s+", " ", extracted).strip()

    if extracted.upper() in {"</ANS>", "<ANS>", "ANS >", ">", "", "ANS"}:
        return None
    return extracted


def _clean_further(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None
    response = re.sub(r"<.*?>", "", response).strip()
    if response in {">", ""}:
        return None
    response = re.sub(r"^\((.*?)\)$", r"\1", response)
    return response if response else None


def _extract_options(text: str) -> list[str]:
    """Parse the Options block from a formatted BBH prompt."""
    start = text.find("Options")
    end = text.find("Print")
    if start == -1 or end == -1 or end <= start:
        return []
    block = text[start + 8: end].replace("\n", "").strip()
    tokens = [t for t in block.split("(") if t]
    return [f"({t}" for t in tokens]


def _find_prompt_type(prompt: str) -> str:
    return "option" if "options" in prompt.lower() else "other"


def _normalize_text(text: str) -> str:
    text = text.lower().replace("(", "").replace(")", "")
    parts = text.split()
    if len(parts) > 1 and parts[0].rstrip(") .").isalpha() and len(parts[0]) <= 2:
        text = " ".join(parts[1:])
    text = "".join(c if c.isalnum() else " " for c in text)
    return " ".join(text.split())


def _match_to_label(
    cleaned: Optional[str],
    answer_key: str,
    choices: list[str],
    prompt: str,
    prompt_type: str,
) -> str:
    """Return '1' (correct), '0' (incorrect), or 'None' (invalid/discarded)."""
    _letters = list("abcdefghijklmnopqrstuvwxyz")

    if cleaned is None:
        return "None"

    cleaned = cleaned.lower()
    answer_key = answer_key.lower()

    if cleaned == answer_key:
        return "1"

    if prompt_type == "option" and "(" in answer_key and ")" in answer_key:
        answer_key = _clean_further(answer_key) or answer_key
        if cleaned == answer_key:
            return "1"
        if cleaned in _letters:
            return "0"
        # Response is a full text — try to match against choices
        norm_resp = _normalize_text(cleaned)
        for i, choice in enumerate(choices):
            if norm_resp in _normalize_text(choice):
                return "1" if _letters[i] == answer_key else "0"
        return "None"  # couldn't map to any choice

    # "other" type — sorting tasks tolerate spacing/comma differences
    if "sort" in prompt.lower():
        c = cleaned.replace(",", "").replace(" ", "")
        a = answer_key.replace(",", "").replace(" ", "")
        if c == a:
            return "1"
    return "0"


def evaluate_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate BBH responses across a full DataFrame.

    Expected columns:
      - 'output'           : model's raw response string
      - 'target'           : ground truth (e.g. '(C)', 'True', 'False')
      - 'formatted_prompt' : the prompt used to query the model
    """
    df = df.copy()

    cleaned_list: list[Optional[str]] = []
    label_list: list[str] = []
    correct_list: list[int] = []
    valid = removed = 0

    for _, row in df.iterrows():
        prompt = str(row["formatted_prompt"])
        prompt_type = _find_prompt_type(prompt)
        choices = _extract_options(prompt) if prompt_type == "option" else []

        cleaned = _clean_further(clean_response(str(row["output"])))
        label = _match_to_label(cleaned, str(row["target"]), choices, prompt, prompt_type)

        cleaned_list.append(cleaned)
        label_list.append(label)
        correct_list.append(1 if label == "1" else 0)

        if cleaned is None or label == "None":
            removed += 1
        else:
            valid += 1

    df["cleaned_response"] = cleaned_list
    df["matched_label"] = label_list
    df["correct"] = correct_list

    total = len(df)
    correct = sum(correct_list)
    stats = {
        "dataset": "bbh",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "valid_responses": valid,
        "removed_responses": removed,
    }
    return df, stats
