#!/bin/bash
#SBATCH --job-name=llama_instruct_quantize_q5_0
#SBATCH --output=results_quantize/llama_instruct_quantize_q5_0_%j.out
#SBATCH --time=01:00:00           # quantization only (no conversion) — 1h is plenty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G                 # f16 GGUF ~6GB; headroom for quantization
#SBATCH --partition=regular       # adjust to your available partition

# ── environment ─────────────────────────────────────────────────────────────
module purge
module load Python/3.13.5-GCCcore-14.3.0

# ── paths ───────────────────────────────────────────────────────────────────
BASE="$HOME/masters_thesis"
SCRIPTS_DIR_QUANT="$BASE/masters_thesis_scripts/scripts/quantization"
SCRIPTS_DIR="$BASE/masters_thesis_scripts"
VENV_DIR="$BASE/venv"

# ── venv ─────────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[JOB] Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[JOB] Installing / updating requirements"
pip install --upgrade pip --quiet
pip install -r "$SCRIPTS_DIR/requirements.txt" --quiet

# ── pipeline ─────────────────────────────────────────────────────────────────
echo "[JOB] Starting instruct quantization pipeline"
python "$SCRIPTS_DIR_QUANT/quantize_instruct_q5_0.py"

# ── confirm output ───────────────────────────────────────────────────────────
echo "[JOB] Output files:"
ls -lh "$BASE/llama3.2-3b-instruct-gguf/"

echo "[JOB] Done."