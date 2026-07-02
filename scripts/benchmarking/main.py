import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import cfg
from llama_server import LlamaServer
from ollama_backend import OllamaBackend
from llama_bench import run_llama_bench
from metrics import (
    read_gguf_meta,
    compute_kv_cache_bytes,
    compute_flops_per_decode_token,
    compute_flops_prefill,
    compute_mbu,
)
from prompts import DATASET_PATHS, process_prompt
from evaluation import evaluate_responses, save_results
from perplexity import run_perplexity

DATASETS = ["bbh", "commonsenseqa", "gsm8k",  "truthfulqa", "humaneval"]
TIMING_KEYS = ["e2e_latency_s", "ttft_approx_ms", "throughput_tps",
               "prompt_tokens", "generated_tokens", "generation_ms"]


def _timing_stats(df: pd.DataFrame) -> dict:
    """Return mean ± std for each timing column as a flat dict."""
    agg = df[TIMING_KEYS].agg(["mean", "std"])
    return {f"{col}_{stat}": agg.loc[stat, col]
            for col in TIMING_KEYS for stat in ("mean", "std")}


def run_performance_benchmark(model_path: str | Path, output_dir: Path) -> dict:
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
        ttft_rows = [server.complete_stream("What is the capital of France?") for _ in range(3)]

    ttft_df = pd.DataFrame(ttft_rows)[["ttft_s", "e2e_latency_s", "throughput_tps", "generated_tokens"]]
    print(ttft_df.to_string(index=False))
    avg_ttft_s   = ttft_df["ttft_s"].mean()
    avg_live_tps = ttft_df["throughput_tps"].mean()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to .gguf model file (required for llama_server backend and performance benchmark)")
    parser.add_argument("--backend", choices=["llama_server", "ollama"], default="llama_server",
                        help="Inference backend for --accuracy datasets (default: llama_server)")
    parser.add_argument("--ollama_model", default=None,
                        help="Ollama model name, e.g. llama3.2:1b-instruct-fp16 (required when --backend ollama)")
    parser.add_argument("--accuracy", choices=["datasets", "perplexity"])
    parser.add_argument("--performance_benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not args.accuracy and not args.performance_benchmark:
        parser.error("No benchmark selected. Provide --accuracy and/or --performance_benchmark.")

    if args.backend == "ollama" and not args.ollama_model:
        parser.error("--ollama_model is required when --backend ollama")

    if (args.performance_benchmark or args.accuracy == "perplexity" or args.backend == "llama_server") and not args.model:
        parser.error("--model is required for llama_server backend, perplexity, and performance benchmark")

    model_stem = args.ollama_model.replace(":", "_") if args.backend == "ollama" else Path(args.model).stem
    output_dir = args.output_dir / model_stem
    print(f"Model: {model_stem}  backend: {args.backend}")

    if not args.accuracy and args.performance_benchmark:
        print("WARNING: --accuracy not set. Skipping accuracy evaluation.")

    if args.accuracy == "datasets":
        print(f"Running accuracy evaluation on datasets: {', '.join(DATASETS)}")
        if args.backend == "ollama":
            backend = OllamaBackend(args.ollama_model)
        else:
            backend = LlamaServer(args.model)
        with backend as server:
            ttlm_s = server.ttlm_s
            if ttlm_s is not None:
                print(f"TTLM: {ttlm_s:.2f} s")
            for dataset in tqdm(DATASETS, desc="datasets"):
                rows = [
                    json.loads(line)
                    for line in Path(DATASET_PATHS[dataset]).read_text().splitlines()
                    if line.strip()
                ]
                # HumanEval functions can require many tokens to implement fully;
                # cap other datasets at llama_n_predict to keep runs bounded.
                max_tok = -1 if dataset == "humaneval" else cfg.llama_n_predict
                for run in tqdm(range(args.iterations), desc=dataset, leave=False):
                    prompts, responses, timings = [], [], []
                    for row in tqdm(rows, desc="prompts", leave=False):
                        prompt = process_prompt(dataset, row)
                        result = server.complete(prompt, max_tokens=max_tok)
                        prompts.append(prompt)
                        responses.append(result["content"])
                        timings.append({k: result[k] for k in TIMING_KEYS})

                    df, stats = evaluate_responses(dataset, rows, responses, prompts=prompts)
                    timing_df = pd.DataFrame(timings)
                    df = df.reset_index(drop=True)
                    df["dataset"] = dataset
                    df["run"]     = run
                    df["prompt"]  = prompts
                    df[TIMING_KEYS] = timing_df

                    save_results(df, output_dir)

                    summary_record = {
                        "dataset": dataset,
                        "run":     run,
                        "ttlm_s":  ttlm_s,
                        **stats,
                        **_timing_stats(df),
                    }
                    with (output_dir / "summary.jsonl").open("a") as f:
                        f.write(json.dumps(summary_record) + "\n")

                    tps = summary_record["throughput_tps_mean"]
                    print(f"Run {run} | {dataset}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})  "
                          f"tps={tps:.1f}  ttft≈{summary_record['ttft_approx_ms_mean']:.0f}ms")

    elif args.accuracy == "perplexity":
        stats = run_perplexity(args.model, output_dir)
        print(f"Perplexity: {stats['ppl']:.4f} ± {stats['ppl_std']:.4f}")

    if args.performance_benchmark:
        run_performance_benchmark(args.model, output_dir)
