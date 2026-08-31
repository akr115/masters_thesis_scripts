"""
Benchmarking configuration via environment variables.

All settings are read from the process environment — no path guessing.
Source your machine-local config.env before running any benchmark script:

    source config.env && python main.py --model model.gguf --accuracy datasets

Required variables (no defaults — must be set):
    LLAMA_SERVER          Absolute path to the compiled llama-server binary.
    DATASET_ROOT          Absolute path to neurips-edge-llm-challenge-sampled/.

Optional hardware variables (defaults shown):
    LLAMA_N_GPU_LAYERS   99    GPU layers to offload (99=all layers for any realistic model, 0=CPU only, N=exact count).
    LLAMA_N_THREADS       0    Generation threads  (0 = llama.cpp default).
    LLAMA_N_THREADS_BATCH 0    Batch-eval threads  (0 = llama.cpp default).
    LLAMA_BATCH_SIZE    512    Logical batch size.
    LLAMA_N_CTX        4096   Context window size (tokens).
    LLAMA_KV_CACHE_TYPE f16   KV cache element dtype: f16 or q5_0.
                               Must match kv_dtype passed to compute_kv_cache_bytes().

Perplexity evaluation (only needed with --accuracy perplexity):
    WIKITEXT_PATH         Absolute path to wikitext.test.raw (WikiText-2 test split).
                          llama-perplexity binary is derived from the same directory
                          as LLAMA_SERVER — no separate path needed.

Performance benchmarking (only needed with --performance_benchmark):
    PEAK_MEMORY_BW_GBPS   Peak memory bandwidth of the current device in GB/s.
                          M3 Pro ≈ 150, Jetson Orin Nano ≈ 68.
    LLAMA_BENCH           Path to the llama-bench binary.
                          Defaults to sibling of LLAMA_SERVER if unset.

See config.env.example for a ready-to-fill template.
"""
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="config.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Required paths ────────────────────────────────────────────────────────
    llama_server: Path
    dataset_root: Path

    # ── Hardware ──────────────────────────────────────────────────────────────
    llama_n_gpu_layers: int = 99  # 99 = offload all layers (accepted by all llama.cpp binaries)
    llama_n_threads: int = 0        # 0 = llama.cpp chooses based on hardware
    llama_n_threads_batch: int = 0  # 0 = llama.cpp chooses based on hardware
    llama_batch_size: int = 512
    llama_n_ctx: int = 4096         # context window
    llama_kv_cache_type: str = "f16"  # f16 or q5_0; must match kv_dtype in compute_kv_cache_bytes()

    # ── Inference ─────────────────────────────────────────────────────────────
    llama_temperature: float = 0.0   # 0 = deterministic/greedy
    llama_n_predict: int = 512       # max tokens per response

    # ── Perplexity ────────────────────────────────────────────────────────────
    wikitext_path: Optional[Path] = None  # required only for --accuracy perplexity

    # ── Performance benchmarking ──────────────────────────────────────────────
    # Peak memory bandwidth of the current device in GB/s.
    peak_memory_bw_gbps: float = 150.0
    # llama-bench binary path — defaults to sibling of llama-server if unset.
    llama_bench: Optional[Path] = None

    @field_validator("llama_server")
    @classmethod
    def binary_must_exist(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(
                f"llama-server binary not found: {v}\n"
                "  Build llama.cpp, or correct LLAMA_SERVER in config.env."
            )
        return v

    @field_validator("dataset_root")
    @classmethod
    def dataset_dir_must_exist(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(
                f"Dataset root not found: {v}\n"
                "  Set DATASET_ROOT to the neurips-edge-llm-challenge-sampled directory."
            )
        return v


cfg = BenchConfig()