import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from evaluations.evaluate_commonsenseQA_utils import evaluate_df as _eval_commonsenseqa
from evaluations.evaluate_bbh_utils import evaluate_df as _eval_bbh
from evaluations.evaluate_gsm8k_utils import evaluate_df as _eval_gsm8k
from evaluations.evaluate_humaneval_utils import evaluate_df as _eval_humaneval
from evaluations.evaluate_truthfulqa_utils import evaluate_df as _eval_truthfulqa


_EVALUATORS = {
    "commonsenseqa": _eval_commonsenseqa,
    "bbh": _eval_bbh,
    "gsm8k": _eval_gsm8k,
    "humaneval": _eval_humaneval,
    "truthfulqa": _eval_truthfulqa,
}


def evaluate_responses(
    dataset: str,
    rows: List[dict],
    responses: List[str],
    prompts: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a DataFrame from rows + responses and run dataset-specific evaluation.

    Returns (df_with_results, stats_dict).
    """
    evaluator = _EVALUATORS.get(dataset)
    if evaluator is None:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    df = pd.DataFrame(rows)
    df["output"] = responses

    if dataset == "bbh":
        if prompts is None:
            raise ValueError("BBH evaluation requires formatted prompts — pass prompts= to evaluate_responses")
        df["formatted_prompt"] = prompts

    return evaluator(df)


def save_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Append per-row results to the unified rows.jsonl file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("a") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, default=_default) + "\n")