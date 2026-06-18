import argparse
import json
from pathlib import Path

from config import cfg
from llama_server import LlamaServer
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

DATASETS = ["bbh", "commonsenseqa", "gsm8k", "humaneval", "truthfulqa"]

# Context length used for KV cache and MBU calculations.
# ELIB (Chen et al. 2025) uses the model's maximum context window so
# comparisons are consistent across runs regardless of actual prompt length.
BENCH_CTX_LEN = 4096


def run_performance_benchmark(model_path: str | Path, output_dir: Path) -> dict:
    """Category E — collect all 6 performance metrics for one model.

    Returns a single dict with all metrics and appends it to performance.jsonl.

    Metric sources:
      1. KV Cache size   — metrics.compute_kv_cache_bytes()         [analytical]
      2. TTFT            — LlamaServer.complete_stream()             [wall-clock]
      3. E2E latency     — llama-bench pp latency_ms_per_token       [synthetic]
      4. Throughput      — llama-bench tg avg_ts                     [synthetic]
      5. FLOPs           — metrics.compute_flops_per_decode_token()  [analytical]
      6. MBU             — metrics.compute_mbu()                     [derived]
    """
    model_path = Path(model_path)
    model_stem = model_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Read GGUF metadata (architecture constants + file size) ───────────────
    print(f"[perf] Reading GGUF metadata: {model_path.name}")
    meta = read_gguf_meta(model_path)
    print(f"[perf]   arch={meta.arch}, layers={meta.n_layers}, "
          f"heads={meta.n_heads}, kv_heads={meta.n_kv_heads}, "
          f"head_dim={meta.head_dim}, n_params={meta.n_params:,}, "
          f"file_size={meta.model_size_bytes / 1e9:.2f} GB")

    # ── Metric 1: KV Cache size ───────────────────────────────────────────────
    kv_bytes = compute_kv_cache_bytes(meta, ctx_len=cfg.llama_n_ctx, kv_dtype=cfg.llama_kv_cache_type)
    print(f"[perf] Metric 1 — KV cache ({cfg.llama_n_ctx} tokens, {cfg.llama_kv_cache_type}): "
          f"{kv_bytes / 1e6:.1f} MB")

    # ── Metric 5: FLOPs (analytical, one-time) ────────────────────────────────
    flops_decode  = compute_flops_per_decode_token(meta, seq_len=BENCH_CTX_LEN)
    flops_prefill = compute_flops_prefill(meta, prompt_len=BENCH_CTX_LEN)
    print(f"[perf] Metric 5 — FLOPs/decode token: {flops_decode / 1e9:.2f} GFLOPs")
    print(f"[perf] Metric 5 — FLOPs prefill ({BENCH_CTX_LEN} tokens): "
          f"{flops_prefill / 1e12:.2f} TFLOPs")

    # ── Metrics 3 & 4: llama-bench (throughput + latency) ────────────────────
    print("[perf] Running llama-bench …")
    bench = run_llama_bench(model_path)
    tg_ts = bench.get("tg_128", {}).get("avg_ts")
    for key, vals in bench.items():
        phase = "Prefill" if key.startswith("pp") else "Decode"
        print(f"[perf]   {phase} {key}: {vals['avg_ts']:.1f} t/s  "
              f"± {vals['stddev_ts']:.1f}  "
              f"({vals['latency_ms_per_token']:.2f} ms/tok)")

    # ── Metric 2: TTFT via streaming ──────────────────────────────────────────
    print("[perf] Starting llama-server for TTFT measurement …")
    ttft_results: list[dict] = []
    with LlamaServer(model_path) as server:
        ttlm_s = server.ttlm_s
        print(f"[perf] TTLM (time to load model): {ttlm_s:.2f} s")

        ttft_prompt = "What is the capital of France?"
        for i in range(3):
            r = server.complete_stream(ttft_prompt)
            ttft_results.append({
                "run":        i,
                "ttft_s":     r["ttft_s"],
                "e2e_s":      r["e2e_latency_s"],
                "throughput": r["throughput_tps"],
                "gen_tokens": r["generated_tokens"],
            })
            print(f"[perf]   TTFT run {i}: {r['ttft_s'] * 1000:.1f} ms  "
                  f"e2e={r['e2e_latency_s']:.2f} s  "
                  f"tps={r['throughput_tps']:.1f}")

    avg_ttft_s = sum(x["ttft_s"] for x in ttft_results) / len(ttft_results)
    live_tps   = [x["throughput"] for x in ttft_results if x["throughput"]]
    avg_live_tps = sum(live_tps) / len(live_tps) if live_tps else None

    # ── Metric 6: MBU ─────────────────────────────────────────────────────────
    # Prefer llama-bench throughput (more stable); fall back to live server.
    mbu_tps = tg_ts or avg_live_tps
    mbu = compute_mbu(
        meta=meta,
        kv_cache_bytes=kv_bytes,
        throughput_tps=mbu_tps,
        peak_bw_gbps=cfg.peak_memory_bw_gbps,
    ) if mbu_tps else None
    print(f"[perf] Metric 6 — MBU: {mbu * 100:.1f}%" if mbu else
          "[perf] Metric 6 — MBU: N/A (no throughput)")

    # ── Assemble and save ─────────────────────────────────────────────────────
    record = {
        "model":                  model_stem,
        # Metric 1
        "kv_cache_bytes":         kv_bytes,
        "kv_cache_ctx_len":       cfg.llama_n_ctx,
        "kv_cache_dtype":         cfg.llama_kv_cache_type,
        # Metric 2
        "ttlm_s":                 ttlm_s,
        "ttft_avg_s":             avg_ttft_s,
        "ttft_runs":              ttft_results,
        # Metrics 3 & 4
        "llama_bench":            bench,
        # Metric 5
        "flops_per_decode_token": flops_decode,
        "flops_prefill":          flops_prefill,
        # Metric 6
        "mbu":                    mbu,
        "mbu_peak_bw_gbps":       cfg.peak_memory_bw_gbps,
        # Metadata for reproducibility
        "model_size_bytes":       meta.model_size_bytes,
        "n_params":               meta.n_params,
        "arch":                   meta.arch,
    }

    out_path = output_dir / "performance.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[perf] Saved to {out_path}")

    return record


def _save_timing(
    timings: list[dict],
    ttlm_s: float,
    output_dir: Path,
    dataset: str,
    run: int,
) -> None:
    """Aggregate per-prompt timings and save to {dataset}_timing.jsonl.

    Follows Husom et al.: collect timing from every real inference call,
    then report mean ± std per run.  None values (fields the server didn't
    return) are excluded from aggregation.
    """
    def _mean_std(key: str) -> tuple[float | None, float | None]:
        vals = [t[key] for t in timings if t.get(key) is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, variance ** 0.5

    e2e_mean,   e2e_std   = _mean_std("e2e_latency_s")
    ttft_mean,  ttft_std  = _mean_std("ttft_approx_ms")
    tps_mean,   tps_std   = _mean_std("throughput_tps")
    ptok_mean,  _         = _mean_std("prompt_tokens")
    gtok_mean,  _         = _mean_std("generated_tokens")

    record = {
        "run":                  run,
        "n_prompts":            len(timings),
        "ttlm_s":               ttlm_s,
        "e2e_latency_s_mean":   e2e_mean,
        "e2e_latency_s_std":    e2e_std,
        "ttft_approx_ms_mean":  ttft_mean,
        "ttft_approx_ms_std":   ttft_std,
        "throughput_tps_mean":  tps_mean,
        "throughput_tps_std":   tps_std,
        "prompt_tokens_mean":   ptok_mean,
        "generated_tokens_mean": gtok_mean,
    }

    out_path = output_dir / f"{dataset}_timing.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Run {run} | {dataset} timing: "
          f"tps={tps_mean:.1f}±{tps_std:.1f}  "
          f"ttft≈{ttft_mean:.0f}ms  "
          f"e2e={e2e_mean:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument(
        "--accuracy",
        choices=["datasets", "perplexity"],
        help="Run accuracy evaluation (datasets or perplexity)",
    )
    parser.add_argument(
        "--performance_benchmark",
        action="store_true",
        help="Run performance benchmark (metrics 1-6)",
    )
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of accuracy runs for mean ± std")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"),
                        help="Where to save results")
    print("Parsing arguments …")
    args = parser.parse_args()

    model_stem = Path(args.model).stem
    output_dir = args.output_dir / model_stem

    print(f"Running benchmarks for model: {model_stem}")

    # ── Accuracy: datasets ────────────────────────────────────────────────────
    if args.accuracy == "datasets":
        with LlamaServer(args.model) as server:
            ttlm_s = server.ttlm_s

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
                    timings: list[dict] = []
                    for row in rows:
                        prompt = process_prompt(dataset, row)
                        result = server.complete(prompt)
                        prompts.append(prompt)
                        responses.append(result["content"])
                        timings.append({
                            "e2e_latency_s":  result["e2e_latency_s"],
                            "ttft_approx_ms": result["ttft_approx_ms"],
                            "throughput_tps": result["throughput_tps"],
                            "prompt_tokens":  result["prompt_tokens"],
                            "generated_tokens": result["generated_tokens"],
                            "generation_ms":  result["generation_ms"],
                        })

                    df, stats = evaluate_responses(dataset, rows, responses, prompts=prompts)
                    save_results(df, stats, output_dir, dataset, run=run)
                    print(f"Run {run} | {dataset}: {stats['accuracy']:.1%} "
                          f"({stats['correct']}/{stats['total']})")

                    # Aggregate timing (Husom et al. style) and save alongside accuracy
                    _save_timing(timings, ttlm_s, output_dir, dataset, run=run)

    # ── Accuracy: perplexity ──────────────────────────────────────────────────
    elif args.accuracy == "perplexity":
        stats = run_perplexity(args.model, output_dir)
        print(f"Perplexity: {stats['ppl']:.4f} ± {stats['ppl_std']:.4f}")

    # ── Performance benchmark (metrics 1-6) ───────────────────────────────────
    if args.performance_benchmark:
        run_performance_benchmark(args.model, output_dir)