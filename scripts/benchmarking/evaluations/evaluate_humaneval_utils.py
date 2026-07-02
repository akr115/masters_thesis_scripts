import os
import re
import pandas as pd
from evaluate import load

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def clean_response(response: str, entry_point: str | None = None) -> str:
    """Strip <ANS> tags and, when entry_point is given, discard everything before
    the function definition so that docstring examples prepended by the chat-mode
    model (e.g. 'foo(1, 2)\n42\ndef foo(...)') don't cause a NameError at eval time.
    """
    cleaned = str(response).replace("<ANS>", "").replace("</ANS>", "")
    if entry_point:
        m = re.search(rf"def\s+{re.escape(entry_point)}\s*\(", cleaned)
        if m:
            cleaned = cleaned[m.start():]
    return cleaned


def evaluate_df(df: pd.DataFrame, k: list[int] = [1]) -> tuple[pd.DataFrame, dict]:
    """Evaluate HumanEval responses using pass@k (default k=1).

    Expected columns:
      - 'output'       : model's raw response string
      - 'test'         : HumanEval test suite string
      - 'entry_point'  : function name (optional; used to trim pre-function preamble)
    """
    code_eval = load("code_eval")

    test_cases = df["test"].tolist()
    candidates = (
        df.apply(
            lambda row: [clean_response(row["output"], row.get("entry_point"))]
            if pd.notna(row["output"]) else [],
            axis=1,
        ).tolist()
    )

    pass_at_k, results = code_eval.compute(
        references=test_cases,
        predictions=candidates,
        k=k,
    )

    passed_by_task = {
        task_id: outcomes[0][1]["passed"] for task_id, outcomes in results.items()
    }
    evaluation_list = [passed_by_task[task_id] for task_id in sorted(passed_by_task)]

    df = df.copy()
    df["correct"] = evaluation_list

    total = len(df)
    correct = sum(evaluation_list)
    stats = {
        "dataset": "humaneval",
        "total": total,
        "correct": correct,
        "accuracy": pass_at_k.get("pass@1", correct / total if total else 0.0),
        **pass_at_k,
    }
    return df, stats