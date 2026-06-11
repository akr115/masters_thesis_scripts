"""
Run instruct task-eval for one model over one or more datasets.

Writes one jsonl file per dataset:
    outputs/<model_stem>/<dataset>.jsonl
Each line: {"indata": <original row>, "formatted_prompt": <str>, "output": <str>}

Usage:
    python run_eval.py --model <path.gguf> --datasets gsm8k
    python run_eval.py --model <path.gguf> --datasets gsm8k,commonsenseqa,bbh,truthfulqa,humaneval
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from prompts import process_prompt, DATASET_PATHS
from llama_server import LlamaServer

OUTPUT_ROOT = Path(__file__).parent / "outputs"


def run_dataset(server: LlamaServer, dataset: str, out_path: Path) -> None:
    jsonl_path = DATASET_PATHS[dataset]
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    total = len(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    with open(out_path, "w") as out_f:
        for i, row in enumerate(rows, 1):
            formatted = process_prompt(dataset, row)
            output = None
            try:
                output = server.complete(formatted)
                ok += 1
            except requests.Timeout:
                print(f"  [{i}/{total}] TIMEOUT", flush=True)
                failed += 1
            except Exception as e:
                print(f"  [{i}/{total}] ERROR: {e}", flush=True)
                failed += 1

            out_f.write(json.dumps({
                "indata": row,
                "formatted_prompt": formatted,
                "output": output,
            }) + "\n")

            if i % 10 == 0 or i == total:
                print(f"  [{i}/{total}] ok={ok} failed={failed}", flush=True)

    print(f"  Done. Written to {out_path}")


def main() -> None:
    all_datasets = list(DATASET_PATHS)

    parser = argparse.ArgumentParser(description="Instruct task-eval over one or more datasets")
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument(
        "--datasets",
        default="gsm8k",
        help=f"Comma-separated list of datasets. Choices: {', '.join(all_datasets)}. Default: gsm8k",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    unknown = [d for d in datasets if d not in DATASET_PATHS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Valid: {all_datasets}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    model_stem = model_path.stem
    out_dir = OUTPUT_ROOT / model_stem

    print(f"Model : {model_path.name}")
    print(f"Output: {out_dir}")
    print(f"Datasets: {datasets}\n")

    with LlamaServer(model_path, port=args.port) as server:
        for dataset in datasets:
            out_path = out_dir / f"{dataset}.jsonl"
            print(f"=== {dataset} ({DATASET_PATHS[dataset].name}) ===")
            t0 = time.monotonic()
            run_dataset(server, dataset, out_path)
            elapsed = time.monotonic() - t0
            print(f"  Elapsed: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()