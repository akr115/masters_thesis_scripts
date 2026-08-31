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
- `--sparsity_ratio` — fraction of weights zeroed; passed as the first script
  argument (default 0.20).
- `--sparsity_type unstructured` — applied to individual weights, not blocks.

The output directory is tagged with the ratio, so `0.20` and `0.66` produce
`<model>-wanda-sp20` and `<model>-wanda-sp66`.

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

Slurm resolves `#SBATCH --output=` before the job script runs, so the log
directories must exist or the job fails to launch. They are gitignored, so
create them once after cloning:

```bash
mkdir -p results_download results_prune results_quantize results_upload \
         results_1b_conversion results_3b_conversion results_prune_conversion
```

Then, in pipeline order:

| Step | Script |
|---|---|
| Download a model | `jobscripts/download_scripts/download_{1b,3b_base,3b_instruct}.sh` |
| Convert to GGUF f16 | `jobscripts/convert_scripts/*_convert_to_gguf.sh` |
| Prune with WANDA | `jobscripts/wanda_prune.sh [SPARSITY] [MODEL_NAME]` |
| Convert a pruned model | `jobscripts/wanda_pruned_convert_to_gguf.sh` |
| Quantize to Q5_0 | `jobscripts/quantize_q5_0.sh`, `jobscripts/quantize_instruct_q5_0.sh` |
| Upload to the Hub | `jobscripts/upload_scripts/upload_gguf_to_hf.sh [LOCAL_DIR] [REPO_ID]` |

`wanda_prune.sh` takes the sparsity ratio and model directory as arguments, so
one script covers both ratios:

```bash
sbatch jobscripts/wanda_prune.sh 0.20 llama3.2-3b-instruct
sbatch jobscripts/wanda_prune.sh 0.66 llama3.2-3b
```

The upload script authenticates with the token stored by `hf auth login` and
creates repos private by default (`PRIVATE=0` to create them public).
`hf upload-large-folder` is resumable, so a re-submitted job continues where it
stopped.

## Published models

The GGUF artifacts are published as private repos under the
[`am00r`](https://huggingface.co/am00r) Hugging Face account, grouped into a
collection. Each repo's filename matches its result directory name in the
benchmark outputs.
