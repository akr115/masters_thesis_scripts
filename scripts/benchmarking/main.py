import argparse
import json
from pathlib import Path

from llama_server import LlamaServer
from prompts import DATASET_PATHS, process_prompt
from evaluation import evaluate_responses, save_results
from perplexity import run_perplexity

DATASETS = ["bbh", "commonsenseqa", "gsm8k", "humaneval", "truthfulqa"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument("--accuracy_evaluation_method", choices=["datasets", "perplexity"])
    parser.add_argument("--iterations", type=int, default=5, help="Number of runs for mean ± std")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Where to save results")
    args = parser.parse_args()

    model_stem = Path(args.model).stem
    output_dir = args.output_dir / model_stem

    print(f"Running benchmarks for model: {model_stem}")

    if args.accuracy_evaluation_method == "datasets":
        with LlamaServer(args.model) as server:
            for run in range(args.iterations):
                for dataset in DATASETS:
                    jsonl_path = DATASET_PATHS[dataset]
                    rows = [
                        json.loads(line)
                        for line in Path(jsonl_path).read_text().splitlines()
                        if line.strip()
                    ]

                    prompts = []
                    responses = []
                    for row in rows:
                        prompt = process_prompt(dataset, row)
                        response = server.complete(prompt)
                        prompts.append(prompt)
                        responses.append(response)

                    df, stats = evaluate_responses(dataset, rows, responses, prompts=prompts)
                    save_results(df, stats, output_dir, dataset, run=run)
                    print(f"Run {run} | {dataset}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    elif args.accuracy_evaluation_method == "perplexity":
        stats = run_perplexity(args.model, output_dir)
        print(f"Perplexity: {stats['ppl']:.4f} ± {stats['ppl_std']:.4f}")