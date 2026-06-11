"""
Benchmarking configuration via environment variables.

All settings are read from the process environment — no path guessing.
Source your machine-local config.env before running any benchmark script:

    source config.env && python run_eval.py --model model.gguf --datasets gsm8k

Required variables (no defaults — must be set):
    LLAMA_SERVER          Absolute path to the compiled llama-server binary.
    DATASET_ROOT          Absolute path to neurips-edge-llm-challenge-sampled/.

Optional hardware variables (defaults shown):
    LLAMA_N_GPU_LAYERS    0    GPU layers to offload (0=CPU only, -1=all).
    LLAMA_N_THREADS       0    Generation threads  (0 = llama.cpp default).
    LLAMA_N_THREADS_BATCH 0    Batch-eval threads  (0 = llama.cpp default).
    LLAMA_BATCH_SIZE    512    Logical batch size.

Perplexity evaluation (only needed with --accuracy_evaluation_method perplexity):
    WIKITEXT_PATH         Absolute path to wikitext.test.raw (WikiText-2 test split).
                          llama-perplexity binary is derived from the same directory
                          as LLAMA_SERVER — no separate path needed.

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
    llama_n_gpu_layers: int = 0
    llama_n_threads: int = 0        # 0 = llama.cpp chooses based on hardware
    llama_n_threads_batch: int = 0  # 0 = llama.cpp chooses based on hardware
    llama_batch_size: int = 512

    # ── Inference ─────────────────────────────────────────────────────────────
    llama_temperature: float = 0.0   # 0 = deterministic/greedy
    llama_n_predict: int = 512       # max tokens per response

    # ── Perplexity ────────────────────────────────────────────────────────────
    wikitext_path: Optional[Path] = None  # required only for --accuracy_evaluation_method perplexity

    @field_validator("llama_server")
    @classmethod
    def binary_must_exist(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(
                f"llama-server binary not found: {v}\n"
                "  Build it with bootstrap.sh, or correct LLAMA_SERVER in config.env."
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