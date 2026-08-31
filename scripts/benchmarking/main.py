"""
Benchmark one GGUF model: task accuracy, perplexity, and/or runtime performance.

Source your machine-local config.env first (see config.py for the variables):

    source config.env
    python main.py --model model.gguf --accuracy datasets --performance_benchmark

Outputs are appended under <output_dir>/<model_stem>/:
    rows.jsonl        one line per prompt, with per-row scoring and timings
    summary.jsonl     one line per (dataset, run) with accuracy and timing stats
    perplexity.jsonl  one line per perplexity run
    performance.jsonl one line per performance run
"""

import argparse
import json
import signal
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import cfg
from evaluation import evaluate_responses, save_results
from llama_bench import run_llama_bench
from llama_server import LlamaServer, REQUEST_TIMEOUT
from metrics import (
    read_gguf_meta,
    compute_kv_cache_bytes,
    compute_flops_per_decode_token,
    compute_flops_prefill,
    compute_mbu,
)
from perplexity import run_perplexity
from prompts import DATASET_PATHS, process_prompt

DATASETS = ["bbh", "commonsenseqa", "gsm8k", "truthfulqa", "humaneval"]
TIMING_KEYS = ["e2e_latency_s", "ttft_approx_ms", "throughput_tps",
               "prompt_tokens", "generated_tokens", "generation_ms"]

TTFT_PROMPT = "What is the capital of France?"


class RequestTimeout(Exception):
    """Raised by the SIGALRM handler when a request outlives REQUEST_TIMEOUT."""


def _alarm_handler(signum, frame):
    raise RequestTimeout


# A dead or wedged server surfaces as any of these; each is recoverable by
# recording an empty response and restarting the server before the next prompt.
RECOVERABLE_ERRORS = (
    RequestTimeout,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
    requests.exceptions.ConnectionError,
)


def _timing_stats(df: pd.DataFrame) -> dict:
    """Return mean ± std for each timing column as a flat dict."""
    agg = df[TIMING_KEYS].agg(["mean", "std"])
    return {f"{col}_{stat}": agg.loc[stat, col]
            for col in TIMING_KEYS for stat in ("mean", "std")}


def _load_rows(dataset: str) -> list[dict]:
    """Read one dataset's jsonl into a list of row dicts."""
    text = Path(DATASET_PATHS[dataset]).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _collect_responses(server: LlamaServer, dataset: str, rows: list[dict],
                       max_tokens: int) -> tuple[list[str], list[str], list[dict]]:
    """Prompt the server once per row, returning (prompts, responses, timings).

    A prompt that times out or kills the server yields an empty response and
    null timings, and the server is restarted before the next prompt.
    """
    prompts, responses, timings = [], [], []
    for row in tqdm(rows, desc="prompts", leave=False):
        prompt = process_prompt(dataset, row)
        prompts.append(prompt)
        signal.alarm(REQUEST_TIMEOUT)
        try:
            result = server.complete(prompt, max_tokens=max_tokens)
            responses.append(result["content"])
            timings.append({k: result[k] for k in TIMING_KEYS})
        except RECOVERABLE_ERRORS:
            responses.append("")
            timings.append({k: None for k in TIMING_KEYS})
            server.ensure_alive()
        finally:
            signal.alarm(0)
    return prompts, responses, timings


def run_accuracy_benchmark(model_path: str | Path, output_dir: Path,
                           iterations: int) -> None:
    """Evaluate the model on every dataset in DATASETS, `iterations` times each."""
    print(f"Running accuracy evaluation on datasets: {', '.join(DATASETS)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _alarm_handler)

    with LlamaServer(model_path) as server:
        ttlm_s = server.ttlm_s
        print(f"TTLM: {ttlm_s:.2f} s")

        for dataset in tqdm(DATASETS, desc="datasets"):
            rows = _load_rows(dataset)
            # HumanEval functions can require many tokens to implement fully;
            # cap other datasets at llama_n_predict to keep runs bounded.
            max_tokens = -1 if dataset == "humaneval" else cfg.llama_n_predict

            for run in tqdm(range(iterations), desc=dataset, leave=False):
                prompts, responses, timings = _collect_responses(
                    server, dataset, rows, max_tokens)

                df, stats = evaluate_responses(dataset, rows, responses, prompts=prompts)
                df = df.reset_index(drop=True)
                df["dataset"] = dataset
                df["run"] = run
                df["prompt"] = prompts
                df[TIMING_KEYS] = pd.DataFrame(timings)
                save_results(df, output_dir)

                summary = {
                    "dataset": dataset,
                    "run": run,
                    "ttlm_s": ttlm_s,
                    **stats,
                    **_timing_stats(df),
                }
                with (output_dir / "summary.jsonl").open("a") as f:
                    f.write(json.dumps(summary) + "\n")

                print(f"Run {run} | {dataset}: {stats['accuracy']:.1%} "
                      f"({stats['correct']}/{stats['total']})  "
                      f"tps={summary['throughput_tps_mean']:.1f}  "
                      f"ttft≈{summary['ttft_approx_ms_mean']:.0f}ms")


def run_performance_benchmark(model_path: str | Path, output_dir: Path,
                              iterations: int) -> dict:
    """Measure model-size, FLOPs, throughput, TTFT and MBU; append one jsonl record."""
    model_path = Path(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = read_gguf_meta(model_path)
    print(f"[perf] {model_path.name}: arch={meta.arch}, layers={meta.n_layers}, "
          f"heads={meta.n_heads}, kv_heads={meta.n_kv_heads}, head_dim={meta.head_dim}, "
          f"n_params={meta.n_params:,}, size={meta.model_size_bytes / 1e9:.2f} GB")

    kv_bytes      = compute_kv_cache_bytes(meta, ctx_len=cfg.llama_n_ctx, kv_dtype=cfg.llama_kv_cache_type)
    flops_decode  = compute_flops_per_decode_token(meta, seq_len=cfg.llama_n_ctx)
    flops_prefill = compute_flops_prefill(meta, prompt_len=cfg.llama_n_ctx)
    print(f"[perf] KV cache: {kv_bytes / 1e6:.1f} MB  |  "
          f"FLOPs/token: {flops_decode / 1e9:.2f} G  |  "
          f"FLOPs prefill: {flops_prefill / 1e12:.2f} T")

    print("[perf] Running llama-bench …")
    bench = run_llama_bench(model_path)
    tg_ts = bench.get("tg_128", {}).get("avg_ts")
    for key, vals in bench.items():
        phase = "pp" if key.startswith("pp") else "tg"
        print(f"[perf]   {phase} {key}: {vals['avg_ts']:.1f} ± {vals['stddev_ts']:.1f} t/s  "
              f"({vals['latency_ms_per_token']:.2f} ms/tok)")

    print("[perf] Measuring TTFT …")
    with LlamaServer(model_path) as server:
        ttlm_s = server.ttlm_s
        print(f"[perf] TTLM: {ttlm_s:.2f} s")
        ttft_rows = [server.complete_stream(TTFT_PROMPT) for _ in range(iterations)]

    ttft_df = pd.DataFrame(ttft_rows)[["ttft_s", "e2e_latency_s", "throughput_tps", "generated_tokens"]]
    print(ttft_df.to_string(index=False))
    avg_ttft_s   = ttft_df["ttft_s"].mean()
    avg_live_tps = ttft_df["throughput_tps"].mean()

    # Prefer llama-bench's tg throughput; fall back to the live server average.
    mbu_tps = tg_ts or avg_live_tps
    mbu = compute_mbu(meta=meta, kv_cache_bytes=kv_bytes,
                      throughput_tps=mbu_tps, peak_bw_gbps=cfg.peak_memory_bw_gbps) if mbu_tps else None
    print(f"[perf] MBU: {mbu * 100:.1f}%" if mbu else "[perf] MBU: N/A")

    record = {
        "model":                  model_path.stem,
        "kv_cache_bytes":         kv_bytes,
        "kv_cache_ctx_len":       cfg.llama_n_ctx,
        "kv_cache_dtype":         cfg.llama_kv_cache_type,
        "ttlm_s":                 ttlm_s,
        "ttft_avg_s":             avg_ttft_s,
        "ttft_runs":              ttft_df.to_dict(orient="records"),
        "llama_bench":            bench,
        "flops_per_decode_token": flops_decode,
        "flops_prefill":          flops_prefill,
        "mbu":                    mbu,
        "mbu_peak_bw_gbps":       cfg.peak_memory_bw_gbps,
        "model_size_bytes":       meta.model_size_bytes,
        "n_params":               meta.n_params,
        "arch":                   meta.arch,
    }

    out_path = output_dir / "performance.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[perf] Saved to {out_path}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one GGUF model: accuracy, perplexity and/or performance.")
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument("--accuracy", choices=["datasets", "perplexity"],
                        help="Accuracy method: task datasets, or WikiText perplexity")
    parser.add_argument("--performance_benchmark", action="store_true",
                        help="Measure throughput, TTFT, FLOPs, KV cache size and MBU")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Repeats per dataset, and TTFT samples (default: 5)")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not args.accuracy and not args.performance_benchmark:
        parser.error("No benchmark selected. Provide --accuracy and/or --performance_benchmark.")
    return args


def main() -> None:
    args = parse_args()

    model_stem = Path(args.model).stem
    output_dir = args.output_dir / model_stem
    print(f"Model: {model_stem}")

    if args.accuracy == "datasets":
        run_accuracy_benchmark(args.model, output_dir, args.iterations)
    elif args.accuracy == "perplexity":
        stats = run_perplexity(args.model, output_dir)
        print(f"Perplexity: {stats['ppl']:.4f} ± {stats['ppl_std']:.4f}")
    else:
        print("WARNING: --accuracy not set. Skipping accuracy evaluation.")

    if args.performance_benchmark:
        run_performance_benchmark(args.model, output_dir, args.iterations)


if __name__ == "__main__":
    main()
