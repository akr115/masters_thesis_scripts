#!/bin/bash
#SBATCH --job-name=wanda-prune
#SBATCH --output=results_prune/wanda_prune_%j.out
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         
#SBATCH --mem=32G                 
#SBATCH --gpus-per-node=1

module purge
module load Python/3.13.5-GCCcore-14.3.0

# ── 0. Arguments ───────────────────────────────────────────────────────────────
# Usage: sbatch wanda_prune.sh [SPARSITY_RATIO] [MODEL_NAME]
#   e.g. sbatch wanda_prune.sh 0.20 llama3.2-3b-instruct
#        sbatch wanda_prune.sh 0.20 llama3.2-3b
# MODEL_NAME is a directory name under models/.
SPARSITY="${1:-0.20}"
MODEL_NAME="${2:-llama3.2-3b-instruct}"
# Tag used in the output dir name: 0.20 -> 20, 0.66 -> 66
TAG=$(awk -v s="$SPARSITY" 'BEGIN{printf "%02d", s*100}')

# ── 1. Paths ───────────────────────────────────────────────────────────────────
BASE="$HOME/masters_thesis"

SCRIPTS_DIR="$BASE/masters_thesis_scripts"
VENV_DIR="$BASE/venv"

WANDA_DIR="$BASE/wanda"

MODEL_IN="$BASE/models/$MODEL_NAME"
MODEL_OUT="$BASE/models/${MODEL_NAME}-wanda-sp${TAG}"

if [ ! -d "$MODEL_IN" ]; then
    echo "[JOB] ERROR: input model not found: $MODEL_IN"
    echo "[JOB] Available models:"; ls "$BASE/models"
    exit 1
fi

# ── 2. Virtual environment ─────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[JOB] Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"


echo "[JOB] Installing / updating requirements"
pip install --upgrade pip --quiet
pip install -r "$SCRIPTS_DIR/requirements.txt" --quiet


echo "[JOB] Python: $(python --version)"
echo "[JOB] torch: $(python -c 'import torch; print(torch.__version__)')"
echo "[JOB] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[JOB] GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
# ── 3. Run pruning ─────────────────────────────────────────────────────────────
echo "[JOB] Starting WANDA pruning"
echo "[JOB]   sparsity : $SPARSITY (unstructured)"
echo "[JOB]   model in : $MODEL_IN"
echo "[JOB]   model out: $MODEL_OUT"

cd "$WANDA_DIR"

python main.py \
  --model "$MODEL_IN" \
  --prune_method wanda \
  --sparsity_ratio "$SPARSITY" \
  --sparsity_type unstructured \
  --save_model "$MODEL_OUT"

# ── 4. Confirm output ──────────────────────────────────────────────────────────
echo "[JOB] Output model:"
ls -lh "$MODEL_OUT"

echo "[JOB] Done."