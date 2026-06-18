"""
Category B — Analytical metric helpers.

All compute_* functions are pure / stateless: given architecture constants and
runtime measurements they return a single numeric result, no I/O.

Covers:
  Metric 1 — KV Cache size (bytes)
  Metric 5 — FLOPs per token (and prefill FLOPs)
  Metric 6 — MBU (Model Bandwidth Utilization, 0-1)

Use read_gguf_meta() to pull architecture constants from any GGUF file instead
of hardcoding per-model values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gguf import GGUFReader


# ── GGUF metadata reader ──────────────────────────────────────────────────────

@dataclass
class ModelMeta:
    """Architecture constants extracted from a GGUF file."""
    arch: str           # e.g. "llama"
    n_layers: int       # llama.block_count
    n_heads: int        # llama.attention.head_count
    n_kv_heads: int     # llama.attention.head_count_kv
    head_dim: int       # llama.attention.key_length (or hidden/n_heads)
    hidden_size: int    # llama.embedding_length
    n_params: int       # sum of all tensor element counts
    model_size_bytes: int  # file size on disk (bytes read from memory during inference)


def read_gguf_meta(model_path: str | Path) -> ModelMeta:
    """Read architecture constants and parameter count from a GGUF file.

    All values come from the file itself, so this works for any model family
    that follows the GGUF metadata spec (LLaMA, Mistral, Phi, …).
    """
    path = Path(model_path)
    reader = GGUFReader(str(path))
    fields = reader.fields

    def _int(key: str) -> int:
        field = fields[key]
        return int(field.parts[field.data[0]])

    arch = str(bytes(fields["general.architecture"].parts[-1]), "utf-8")

    n_layers   = _int(f"{arch}.block_count")
    n_heads    = _int(f"{arch}.attention.head_count")
    n_kv_heads = _int(f"{arch}.attention.head_count_kv")
    hidden_size = _int(f"{arch}.embedding_length")

    # head_dim stored directly; fall back to hidden_size / n_heads
    if f"{arch}.attention.key_length" in fields:
        head_dim = _int(f"{arch}.attention.key_length")
    else:
        head_dim = hidden_size // n_heads

    n_params = sum(int(t.n_elements) for t in reader.tensors)
    model_size_bytes = os.path.getsize(path)

    return ModelMeta(
        arch=arch,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        n_params=n_params,
        model_size_bytes=model_size_bytes,
    )


# ── KV cache dtype sizes ──────────────────────────────────────────────────────
# Only the two dtypes used in this thesis are listed.
# f16  → default llama-server KV cache
# q5_0 → quantized weights (KV cache stays f16 unless --cache-type-k/v set)
KV_DTYPE_BYTES: dict[str, float] = {
    "f16":  2.0,
    "q5_0": 0.625,
}


# ── Metric 1: KV Cache size ───────────────────────────────────────────────────

def compute_kv_cache_bytes(
    meta: ModelMeta,
    ctx_len: int,
    kv_dtype: str = "f16",
) -> float:
    """Return KV cache size in bytes for a given context length.

    Formula: 2 × n_layers × n_kv_heads × head_dim × ctx_len × dtype_bytes
    The factor of 2 accounts for one K tensor and one V tensor per layer.

    Args:
        meta:     architecture constants from read_gguf_meta().
        ctx_len:  number of tokens in context (prompt + generated so far).
                  ELIB uses the max context window (4096) for a consistent
                  cross-model comparison; use the actual prompt+generation
                  length for a tighter per-request estimate.
        kv_dtype: "f16" (default) or "q5_0" if --cache-type-k/v flags are set.
    """
    dtype_bytes = KV_DTYPE_BYTES[kv_dtype]
    return 2 * meta.n_layers * meta.n_kv_heads * meta.head_dim * ctx_len * dtype_bytes


# ── Metric 5: FLOPs ──────────────────────────────────────────────────────────

def compute_flops_per_decode_token(meta: ModelMeta, seq_len: int) -> float:
    """FLOPs for one autoregressive decode step (one new token).

    Two components:
      weight_flops    = 2 × n_params
          (one multiply-add per weight across all linear projections)
      attention_flops = 4 × n_layers × n_heads × head_dim × seq_len
          (QK^T dot products + softmax-weighted V aggregation)

    At typical seq_len < 1024 the weight term dominates for 1B models.

    Note: Wanda-pruned models report the same FLOPs because llama.cpp performs
    dense matmul over zero weights — no sparse kernel savings apply.
    """
    weight_flops     = 2 * meta.n_params
    attention_flops  = 4 * meta.n_layers * meta.n_heads * meta.head_dim * seq_len
    return float(weight_flops + attention_flops)


def compute_flops_prefill(meta: ModelMeta, prompt_len: int) -> float:
    """FLOPs for the prefill phase (processing the full prompt in one batch).

    Each token in the prompt passes through all weights, and attention is
    quadratic in prompt length (we follow the standard upper-bound convention
    without the 0.5 causal-masking factor).
    """
    weight_flops     = 2 * meta.n_params * prompt_len
    attention_flops  = 4 * meta.n_layers * meta.n_heads * meta.head_dim * (prompt_len ** 2)
    return float(weight_flops + attention_flops)


# ── Metric 6: MBU (Model Bandwidth Utilization) ───────────────────────────────

def compute_mbu(
    meta: ModelMeta,
    kv_cache_bytes: float,
    throughput_tps: float,
    peak_bw_gbps: float,
) -> float:
    """Return MBU as a fraction (0–1; values above 1 indicate measurement error).

    Formula from ELIB (Chen et al. 2025), Equations 1-3:
        TPOT        = 1 / throughput_tps              (seconds per output token)
        achieved_bw = (model_size_bytes + kv_cache_bytes) / TPOT  (bytes/s)
        MBU         = achieved_bw / peak_bw

    Args:
        meta:            from read_gguf_meta() — provides model_size_bytes.
        kv_cache_bytes:  from compute_kv_cache_bytes() at the benchmark ctx_len.
        throughput_tps:  decode throughput in tokens/s (tg phase from llama-bench
                         or timings["predicted_per_second"] from llama-server).
        peak_bw_gbps:    device peak memory bandwidth in GB/s.
                         Set PEAK_MEMORY_BW_GBPS in config.env per machine:
                         M3 Pro ≈ 150, Jetson Orin Nano ≈ 68.
    """
    tpot        = 1.0 / throughput_tps
    achieved_bw = (meta.model_size_bytes + kv_cache_bytes) / tpot
    peak_bw     = peak_bw_gbps * 1e9
    return achieved_bw / peak_bw
