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


def evaluate_responses(
    dataset: str,
    rows: List[dict],
    responses: List[str],
    prompts: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a DataFrame from rows + responses and run dataset-specific evaluation.

    Returns (df_with_results, stats_dict).
    """
    if dataset == "commonsenseqa":
        df = pd.DataFrame(rows)
        df["output"] = responses
        return _eval_commonsenseqa(df)

    elif dataset == "bbh":
        if prompts is None:
            raise ValueError("BBH evaluation requires formatted prompts — pass prompts= to evaluate_responses")
        df = pd.DataFrame(rows)
        df["output"] = responses
        df["formatted_prompt"] = prompts
        return _eval_bbh(df)

    elif dataset == "gsm8k":
        df = pd.DataFrame(rows)
        df["output"] = responses
        return _eval_gsm8k(df)

    elif dataset == "humaneval":
        df = pd.DataFrame(rows)
        df["output"] = responses
        return _eval_humaneval(df)

    elif dataset == "truthfulqa":
        df = pd.DataFrame(rows)
        df["output"] = responses
        return _eval_truthfulqa(df)

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def save_results(df: pd.DataFrame, stats: dict, output_dir: Path, dataset: str, run: int) -> None:
    """Save per-row results as JSONL and append run stats to a summary file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # Raw per-row JSONL — one file per dataset, one line per row, tagged with run index
    rows_path = output_dir / f"{dataset}.jsonl"
    with rows_path.open("a") as f:
        for record in df.assign(run=run).to_dict(orient="records"):
            f.write(json.dumps(record, default=_default) + "\n")

    # Running stats — one line per (dataset, run) pair
    stats_path = output_dir / f"{dataset}_stats.jsonl"
    with stats_path.open("a") as f:
        f.write(json.dumps({"run": run, **stats}, default=_default) + "\n")