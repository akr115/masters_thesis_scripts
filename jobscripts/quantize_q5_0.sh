#!/bin/bash
#SBATCH --job-name=llama_quantize_q5_0
#SBATCH --output=%x_%j.out        # e.g. llama_quantize_q5_0_12345.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00           # conversion + quantization well within 2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         
#SBATCH --mem=24G                 # f16 GGUF ~6GB; headroom for conversion
#SBATCH --partition=regular       # adjust to your available partition

# ── 0. Environment ─────────────────────────────────────────────────────────────
module purge
module load Python/3.13.5-GCCcore-14.3.0

# ── 1. Paths ───────────────────────────────────────────────────────────────────
BASE="$HOME/masters_thesis"
# $HOME resolves to /home2/$USER on Hábrók correctly

SCRIPTS_DIR_QUANT="$BASE/masters_thesis_scripts/scripts/quantization"
SCRIPTS_DIR="$BASE/masters_thesis_scripts"
VENV_DIR="$BASE/venv"

# ── 2. Virtual environment ─────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[JOB] Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[JOB] Installing / updating requirements"
pip install --upgrade pip --quiet
pip install -r "$SCRIPTS_DIR/requirements.txt" --quiet

# ── 3. Run pipeline ────────────────────────────────────────────────────────────
echo "[JOB] Starting quantization pipeline"
python "$SCRIPTS_DIR_QUANT/quantize_q5_0.py"

# ── 4. Confirm output ──────────────────────────────────────────────────────────
echo "[JOB] Output files:"
ls -lh "$BASE/llama3.2-3b-compressed/"

echo "[JOB] Done."