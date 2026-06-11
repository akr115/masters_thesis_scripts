import os
import pandas as pd
from evaluate import load

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def clean_response(response: str) -> str:
    """Strip <ANS> tags from the model response, leaving the raw code."""
    return str(response).replace("<ANS>", "").replace("</ANS>", "")


def evaluate_df(df: pd.DataFrame, k: list[int] = [1]) -> tuple[pd.DataFrame, dict]:
    """Evaluate HumanEval responses using pass@k (default k=1).

    Expected columns:
      - 'output' : model's raw response string
      - 'test'   : HumanEval test suite string
    """
    code_eval = load("code_eval")

    test_cases = df["test"].tolist()
    candidates = (
        df["output"]
        .apply(lambda x: [clean_response(x)] if pd.notna(x) else [])
        .tolist()
    )

    pass_at_k, results = code_eval.compute(
        references=test_cases,
        predictions=candidates,
        k=k,
    )

    evaluation = []
    for result in results.items():
        evaluation.append((result[0], result[1][0][1]["passed"]))

    sorted_evaluation = sorted(evaluation)
    evaluation_list = [x[1] for x in sorted_evaluation]

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