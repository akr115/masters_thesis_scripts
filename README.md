# masters_thesis_scripts

Scripts and job runners used for the thesis model conversion, quantization, and pruning pipeline.

We used the llama.cpp repo for conversion to GGUF and for quantization:
https://github.com/ggml-org/llama.cpp

## Directory layout

- jobscripts/
	- Slurm job scripts for cluster runs (download, quantize, prune).
- scripts/
	- Local python helpers for model conversion.
	- quantization/
		- Quantization pipeline helpers.

## What has been done so far

- Converted the original Llama 3.2 3B parameter model from HF format to GGUF (f16).
- Quantized the 3B GGUF to Q5_0.
- Pruned the 3B model with WANDA (unstructured sparsity).
	- WANDA chosen for simplicity and as a common baseline in pruning work.
	- Sparsity ratio set to 0.66 to bring parameter count closer to Llama 3.2 1B.
- Converted the Llama 3.2 1B model to GGUF.

## Notes on WANDA parameters

In jobscripts/wanda_prune.sh, the pruning command uses:

- --prune_method wanda: WANDA pruning method.
- --sparsity_ratio 0.66: remove ~66% of weights (targeting a smaller effective model).
- --sparsity_type unstructured: sparsity applied to individual weights, not blocks.

WANDA reference implementation:
https://github.com/locuslab/wanda

We used a lightly modified WANDA workflow to support the newer Llama family with a larger context window.

## Typical usage

- Download a model on the cluster: jobscripts/download.sh
- Prune with WANDA: jobscripts/wanda_prune.sh
- Convert + quantize to Q5_0: jobscripts/quantize_q5_0.sh
