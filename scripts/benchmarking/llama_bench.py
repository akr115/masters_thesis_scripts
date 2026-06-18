"""
Category C — llama-bench wrapper.

Runs the llama-bench binary (shipped with llama.cpp) and parses its JSON output
into a structured dict.

Covers:
  Metric 3 — E2E latency (ms/token, from pp and tg phases)
  Metric 4 — Throughput (tokens/s, tg phase)

llama-bench uses synthetic random tokens, not real text, so base and instruct
variants of the same quantization produce statistically identical results.
That is correct for throughput comparison — accuracy differences show up in the
dataset / perplexity tests instead.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from config import cfg

# Default sweep matching ELIB paper (Chen et al. 2025) — can be overridden.
DEFAULT_PP_SIZES = (128, 512)   # prefill (prompt) token counts
DEFAULT_TG_SIZES = (128,)       # generation (decode) token counts
DEFAULT_REPETITIONS = 5         # warmup + averaging handled by llama-bench


def run_llama_bench(
    model_path: str | Path,
    pp_sizes: tuple[int, ...] = DEFAULT_PP_SIZES,
    tg_sizes: tuple[int, ...] = DEFAULT_TG_SIZES,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Run llama-bench and return parsed results keyed by phase and token count.

    Args:
        model_path:  path to the .gguf model file.
        pp_sizes:    prefill (prompt-processing) token counts to benchmark.
        tg_sizes:    token-generation (decode) counts to benchmark.
        repetitions: number of repeated runs per configuration for averaging.

    Returns a flat dict where each key is "{phase}_{tokens}" (e.g. "pp_128",
    "tg_128") and each value is:
        {
            "avg_ts":              float,  # average throughput (tokens/s)
            "stddev_ts":           float,  # stddev of throughput
            "avg_ns":              int,    # average wall-clock time (nanoseconds)
            "latency_ms_per_token": float, # 1000 / avg_ts
        }
    """
    bench_bin = _resolve_bench_bin(model_path)

    cmd = [
        str(bench_bin),
        "-m", str(model_path),
        "-p", ",".join(map(str, pp_sizes)),
        "-n", ",".join(map(str, tg_sizes)),
        "-r", str(repetitions),
        "-o", "json",
    ]
    # llama-bench default is 99 (all layers); -1 is llama-server shorthand for
    # "all" but is not valid for llama-bench's range parser — skip it.
    if cfg.llama_n_gpu_layers >= 0:
        cmd += ["-ngl", str(cfg.llama_n_gpu_layers)]
    if cfg.llama_n_threads > 0:
        cmd += ["-t", str(cfg.llama_n_threads)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    rows: list[dict] = json.loads(result.stdout)

    out: dict[str, Any] = {}
    for row in rows:
        # llama-bench sets n_gen=0 for pure prefill runs and n_prompt=0 for
        # pure decode runs.
        if row.get("n_gen", 0) == 0:
            phase  = "pp"
            tokens = row["n_prompt"]
        else:
            phase  = "tg"
            tokens = row["n_gen"]

        avg_ts = float(row["avg_ts"])
        out[f"{phase}_{tokens}"] = {
            "avg_ts":               avg_ts,
            "stddev_ts":            float(row["stddev_ts"]),
            "avg_ns":               int(row["avg_ns"]),
            "latency_ms_per_token": 1000.0 / avg_ts if avg_ts > 0 else None,
        }

    return out


def _resolve_bench_bin(model_path: str | Path) -> Path:
    """Return the llama-bench binary path.

    Priority:
      1. LLAMA_BENCH in config.env (cfg.llama_bench)
      2. Sibling of llama-server binary (same build directory)
    """
    if cfg.llama_bench is not None:
        if not cfg.llama_bench.is_file():
            raise FileNotFoundError(
                f"llama-bench binary not found at {cfg.llama_bench}\n"
                "  Check LLAMA_BENCH in config.env."
            )
        return cfg.llama_bench

    sibling = cfg.llama_server.parent / "llama-bench"
    if not sibling.is_file():
        raise FileNotFoundError(
            f"llama-bench binary not found at {sibling}\n"
            "  Build llama.cpp or set LLAMA_BENCH in config.env."
        )
    return sibling
