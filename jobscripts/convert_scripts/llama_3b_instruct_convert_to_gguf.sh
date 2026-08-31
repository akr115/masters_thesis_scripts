#!/bin/bash
#SBATCH --job-name=llama_3b_instruct_convert_to_gguf
#SBATCH --output=results_3b_conversion/llama_3b_instruct_convert_to_gguf_%j.out
#SBATCH --time=01:00:00    
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         
#SBATCH --mem=18G                 
#SBATCH --partition=regular      

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

echo "[JOB] Starting conversion pipeline"

python "$SCRIPTS_DIR/scripts/convert_to_gguf.py"

echo "[JOB] Output files:"
ls -lh "$BASE/llama3.2-3b-instruct-gguf/"

echo "[JOB] Done."