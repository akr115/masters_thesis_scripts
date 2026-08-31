# masters_thesis_scripts

Scripts and job runners for the thesis pipeline: model conversion, quantization,
pruning, and edge benchmarking of Llama 3.2 variants.

Conversion and quantization use [llama.cpp](https://github.com/ggml-org/llama.cpp);
pruning uses [WANDA](https://github.com/locuslab/wanda).

## Directory layout

- `jobscripts/` — Slurm job scripts for cluster runs (download, quantize, prune).
- `scripts/`
	- `convert_to_gguf.py` — HF safetensors → GGUF (f16).
	- `quantization/` — quantization pipeline helpers.
	- `benchmarking/` — accuracy, perplexity and runtime benchmarking harness.

## Model pipeline

- Converted Llama 3.2 3B and 1B from HF format to GGUF (f16).
- Quantized the 3B GGUF to Q5_0.
- Pruned the 3B model with WANDA (unstructured sparsity) at two ratios:
	- 0.66 — brings the effective parameter count closer to Llama 3.2 1B.
	- 0.20 — low-sparsity reference point, close to lossless.
- Repeated each variant for both the base and Instruct models.

WANDA parameters (see `jobscripts/wanda_prune.sh`):

- `--prune_method wanda`
- `--sparsity_ratio 0.66` — fraction of weights zeroed.
- `--sparsity_type unstructured` — applied to individual weights, not blocks.

Unstructured pruning zeroes weights rather than removing them, so pruned GGUFs
are the same size as the dense model and llama.cpp gains no speedup from the
sparsity. They isolate the *quality* effect of pruning.

We used a lightly modified WANDA workflow to support the newer Llama family with
a larger context window.

## Benchmarking

`scripts/benchmarking/main.py` runs three benchmark families against a GGUF file
via `llama-server` and `llama-bench`:

| Flag | What it measures |
|---|---|
| `--accuracy datasets` | BBH, CommonsenseQA, GSM8K, TruthfulQA, HumanEval |
| `--accuracy perplexity` | WikiText-2 perplexity via `llama-perplexity` |
| `--performance_benchmark` | throughput, TTFT, TTLM, FLOPs, KV cache size, MBU |

Configuration comes from `scripts/benchmarking/config.env`, which `config.py` loads
relative to the working directory — so run benchmarks from that directory. Copy
`config.env.example` to `config.env` and fill in the machine-local paths.

```bash
cd scripts/benchmarking
pip install -r requirements.txt
cp config.env.example config.env   # then edit it
python main.py --model model.gguf --accuracy datasets --performance_benchmark
```

Results are appended under `<output_dir>/<model_stem>/` as `rows.jsonl`,
`summary.jsonl`, `perplexity.jsonl` and `performance.jsonl`. Result directories
are gitignored.

Run the prompt-formatting tests with `pytest` from `scripts/benchmarking/`.

## Typical cluster usage

- Download a model: `jobscripts/download.sh`
- Prune with WANDA: `jobscripts/wanda_prune.sh`
- Convert + quantize to Q5_0: `jobscripts/quantize_q5_0.sh`

## Published models

The GGUF artifacts are published as private repos under the
[`am00r`](https://huggingface.co/am00r) Hugging Face account, grouped into a
collection. Each repo's filename matches its result directory name in the
benchmark outputs.
