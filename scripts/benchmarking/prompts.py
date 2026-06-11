from config import cfg

DATASET_PATHS = {
    "bbh":           cfg.dataset_root / "BIG-Bench-Hard" / "collated_bbh_200_samples.jsonl",
    "commonsenseqa": cfg.dataset_root / "CommonsenseQA"  / "commonsenseqa_200_samples.jsonl",
    "gsm8k":         cfg.dataset_root / "GSM8K"          / "gsm8k_200_samples.jsonl",
    "humaneval":     cfg.dataset_root / "HumanEval"      / "HumanEval.jsonl",
    "truthfulqa":    cfg.dataset_root / "TruthfulQA"     / "truthfulQA_MC_200_samples.jsonl",
}

instruction_general = (
    "\n\nPrint only the answer surrounded by <ANS> and </ANS>. "
    "Never print any extra explanations about how the response was generated."
)

instruction_humaneval = (
    "\n\n# Complete the function implementation based on the provided docstring, "
    "and print only the completed function surrounded by <ANS> and </ANS>. "
    "Never print any extra explanations about how the code was generated."
)


def process_prompt(dataset: str, prompt: dict) -> str:
    """Return the formatted prompt string for one row of a dataset jsonl."""
    if dataset == "bbh":
        text = prompt["input"]
        instruction = instruction_general

    elif dataset == "commonsenseqa":
        q = prompt["question"]
        stem = q["stem"]
        choices = q["choices"]
        options = "\n".join(f"({c['label']}) {c['text']}" for c in choices)
        text = f"Question: {stem}\nOptions:\n{options}"
        instruction = instruction_general

    elif dataset == "gsm8k":
        text = prompt["question"]
        instruction = instruction_general

    elif dataset == "humaneval":
        text = prompt["prompt"]
        instruction = instruction_humaneval

    elif dataset == "truthfulqa":
        question = prompt["question"]
        choices = "\n".join(prompt["mc1_targets"].keys())
        text = f"Question: {question}\nChoices:\n{choices}"
        instruction = instruction_general

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    return text + instruction
