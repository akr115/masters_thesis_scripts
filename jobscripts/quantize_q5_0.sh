#!/bin/bash
#SBATCH --job-name=llama_quantize_q5_0
#SBATCH --output=results_quantize/llama_quantize_q5_0_%j.out 
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         
#SBATCH --mem=16G
#SBATCH --partition=regular

# --- environment
module purge
module load Python/3.13.5-GCCcore-14.3.0

# --- paths
BASE="$HOME/masters_thesis"
SCRIPTS_DIR_QUANT="$BASE/masters_thesis_scripts/scripts/quantization"
SCRIPTS_DIR="$BASE/masters_thesis_scripts"
VENV_DIR="$BASE/venv"

# --- venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[JOB] Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[JOB] Installing / updating requirements"
pip install --upgrade pip --quiet
pip install -r "$SCRIPTS_DIR/requirements.txt" --quiet

# --- pipeline
echo "[JOB] Starting quantization pipeline"
python "$SCRIPTS_DIR_QUANT/quantize_q5_0.py"

# --- confirm output
echo "[JOB] Output files:"
ls -lh "$BASE/llama3.2-3b-compressed/"

echo "[JOB] Done."