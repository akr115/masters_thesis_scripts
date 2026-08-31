#!/bin/bash
#SBATCH --job-name=hf-download
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=results_download/hf_download_%j.out

module purge
module load Python/3.13.5-GCCcore-14.3.0

BASE="$HOME/masters_thesis"
VENV_DIR="$BASE/venv"

# ── venv ─────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── download ─────────────────────────────────────────
# `original/` holds Meta's consolidated.00.pth (6.4G) — a duplicate of the
# safetensors in a format nothing in this pipeline reads. Excluding it halves
# the download from 12.9G to 6.4G.
hf download meta-llama/Llama-3.2-3B \
  --local-dir $BASE/models/llama3.2-3b \
  --exclude "original/*"

# ── verify ───────────────────────────────────────────
ls -lh $BASE/models/llama3.2-3b
du -sh $BASE/models/llama3.2-3b
