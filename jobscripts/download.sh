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
hf download meta-llama/Llama-3.2-3B \
  --local-dir $BASE/models/llama3.2-3b 

# ── verify ───────────────────────────────────────────
ls -lh $BASE/models/llama3.2-3b