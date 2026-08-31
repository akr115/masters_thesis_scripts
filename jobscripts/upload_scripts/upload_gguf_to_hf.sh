#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --output=results_upload/hf_upload_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load Python/3.13.5-GCCcore-14.3.0

# ── 0. What to upload ──────────────────────────────────────────────────────────
# Usage: sbatch upload_gguf_to_hf.sh [LOCAL_DIR] [REPO_ID]
BASE="$HOME/masters_thesis"
VENV_DIR="$BASE/venv"

LOCAL_DIR="${1:-$BASE/llama3.2-3b-instruct-wanda-sp20-gguf}"
REPO_ID="${2:-am00r/Llama-3.2-3B-Instruct-wanda-sp20-GGUF}"

# Repo is created private on first upload. Set PRIVATE=0 to create it public
# (ignored if the repo already exists — change visibility in the Hub settings).
PRIVATE="${PRIVATE:-1}"
PRIVATE_FLAG=""
[ "$PRIVATE" = "1" ] && PRIVATE_FLAG="--private"

source "$VENV_DIR/bin/activate"

# ── 1. Pre-flight ──────────────────────────────────────────────────────────────
if [ ! -d "$LOCAL_DIR" ]; then
    echo "[JOB] ERROR: local dir not found: $LOCAL_DIR"
    exit 1
fi

echo "[JOB] Uploading to Hugging Face"
echo "[JOB]   local dir : $LOCAL_DIR"
echo "[JOB]   repo id   : $REPO_ID"
echo "[JOB]   private   : $PRIVATE"
echo "[JOB]   size      : $(du -sh "$LOCAL_DIR" | cut -f1)"

# Uses the token stored by `hf auth login` (~/.cache/huggingface/token).
# Override with HF_TOKEN if you need a different one.
echo "[JOB]   account   : $(hf auth whoami)"

# ── 2. Upload ──────────────────────────────────────────────────────────────────
# upload-large-folder is resumable: re-running the job continues where it stopped.
hf upload-large-folder "$REPO_ID" "$LOCAL_DIR" \
    --repo-type model \
    $PRIVATE_FLAG \
    --num-workers 4 \
    --no-bars

echo "[JOB] Done -> https://huggingface.co/$REPO_ID"
