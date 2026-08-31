#!/bin/bash
#SBATCH --job-name=wanda-convert-gguf
#SBATCH --output=results_prune_conversion/wanda_prune_%j.out
#SBATCH --time=02:00:00           # conversion + quantization well within 2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         
#SBATCH --mem=24G                 # f16 GGUF ~6GB; headroom for conversion
#SBATCH --partition=regular       # adjust to your available partition

module purge
module load Python/3.13.5-GCCcore-14.3.0

BASE="$HOME/masters_thesis"


SCRIPTS_DIR="$BASE/masters_thesis_scripts"
VENV_DIR="$BASE/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[JOB] Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[JOB] Installing / updating requirements"
pip install --upgrade pip --quiet
pip install -r "$SCRIPTS_DIR/requirements.txt" --quiet

# Usage: sbatch wanda_pruned_convert_to_gguf.sh [MODEL_NAME]
#   e.g. sbatch wanda_pruned_convert_to_gguf.sh llama3.2-3b-wanda-sp20
export MODEL_NAME="${1:-llama3.2-3b-instruct-wanda-sp20}"

echo "[JOB] Starting conversion pipeline for $MODEL_NAME"

python "$SCRIPTS_DIR/scripts/convert_to_gguf.py"

echo "[JOB] Output files:"
ls -lh "$BASE/${MODEL_NAME}-gguf/"

echo "[JOB] Done."