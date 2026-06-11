"""
GSM8K scorer: exact numeric match.

Ground truth comes from the '#### <number>' suffix in the answer field.
Prediction is extracted from <ANS>...</ANS> in the model output.
Both are parsed as floats so "375" == "375.0" == "375.00".

Usage:
    python scoring/gsm8k.py outputs/<model>/gsm8k.jsonl
"""

import json
import re
import sys
from pathlib import Path

from utils import extract_ans


def parse_number(text: str | None) -> float | None:
    """Extract the last number from a string (handles commas, $, trailing text)."""
    if text is None:
        return None
    # strip formatting characters
    cleaned = text.replace(",", "").replace("$", "").strip()
    # take the last standalone number in the string
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return float(nums[-1]) if nums else None


def score_row(row: dict) -> bool:
    """Return True if the model's answer matches the ground truth."""
    # ground truth: after '####' in the answer field
    answer_text = row["indata"].get("answer", "")
    gt_match = re.search(r"####\s*(.+)", answer_text)
    if not gt_match:
        return False
    gt = parse_number(gt_match.group(1).strip())

    pred = parse_number(extract_ans(row.get("output")))

    if gt is None or pred is None:
        return False
    return gt == pred


def score_file(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    total = len(rows)
    correct = sum(score_row(r) for r in rows)
    null_outputs = sum(1 for r in rows if r.get("output") is None)
    no_tag = sum(1 for r in rows if extract_ans(r.get("output")) is None and r.get("output") is not None)
    return {
        "dataset": "gsm8k",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "null_outputs": null_outputs,
        "missing_ans_tag": no_tag,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <output.jsonl>", file=sys.stderr)
        sys.exit(1)

    result = score_file(Path(sys.argv[1]))
    print(f"Accuracy : {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
    print(f"Timeouts/nulls : {result['null_outputs']}")
    print(f"Missing <ANS> tag: {result['missing_ans_tag']}")
