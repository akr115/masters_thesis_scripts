import os
import json
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
 
# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
USER           = os.environ["USER"]
BASE           = Path(f"/home2/{USER}/masters_thesis")
 
HF_MODEL_DIR   = BASE / "models" / "llama3.2-3b"
OUTPUT_DIR     = BASE / "llama3.2-3b-compressed"
GGUF_F16       = OUTPUT_DIR / "llama3.2-3b-f16.gguf"
GGUF_Q5_0      = OUTPUT_DIR / "llama3.2-3b-q5_0.gguf"
METADATA_FILE  = OUTPUT_DIR / "metadata.json"
 
LLAMA_CPP_DIR  = BASE / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
 
 
# ──────────────────────────────────────────────
# Sanity checks before doing any work
# ──────────────────────────────────────────────
def check_paths():
    missing = []
    for p in [HF_MODEL_DIR, CONVERT_SCRIPT, LLAMA_QUANTIZE]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Required paths not found:\n" + "\n".join(f"  {p}" for p in missing)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CHECK] All required paths exist.")
    print(f"  model : {HF_MODEL_DIR}")
    print(f"  output: {OUTPUT_DIR}")
    print(f"  binary: {LLAMA_QUANTIZE}")
 
 
# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def run(cmd: list, desc: str) -> None:
    print(f"\n[STEP] {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True)
 
 
def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)
 
 
# ──────────────────────────────────────────────
# Step 1: Convert HF model -> GGUF f16
# ──────────────────────────────────────────────
def convert_to_gguf():
    run(
        [
            "python3", str(CONVERT_SCRIPT),
            str(HF_MODEL_DIR),
            "--outfile", str(GGUF_F16),
            "--outtype", "f16",
        ],
        desc=f"Convert HF model -> GGUF f16  =>  {GGUF_F16.name}",
    )
 
 
# ──────────────────────────────────────────────
# Step 2: Quantize GGUF f16 -> Q5_0
# ──────────────────────────────────────────────
def quantize_q5_0():
    run(
        [str(LLAMA_QUANTIZE), str(GGUF_F16), str(GGUF_Q5_0), "Q5_0"],
        desc=f"Quantize f16 -> Q5_0  =>  {GGUF_Q5_0.name}",
    )
 
 
# ──────────────────────────────────────────────
# Step 3: Save metadata
# ──────────────────────────────────────────────
def save_metadata(timings: dict):
    metadata = {
        "model_id": "meta-llama/Llama-3.2-3B",
        "quantization": "Q5_0",
        "paths": {
            "hf_model":  str(HF_MODEL_DIR),
            "gguf_f16":  str(GGUF_F16),
            "gguf_q5_0": str(GGUF_Q5_0),
        },
        "sizes_gb": {
            "gguf_f16":  round(file_size_gb(GGUF_F16), 3)  if GGUF_F16.exists()  else None,
            "gguf_q5_0": round(file_size_gb(GGUF_Q5_0), 3) if GGUF_Q5_0.exists() else None,
        },
        "timings_sec": timings,
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[DONE] Metadata saved -> {METADATA_FILE}")
    print(json.dumps(metadata, indent=2))
 
 
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    check_paths()
    timings = {}
 
    with tqdm(total=2, desc="Overall pipeline") as pbar:
        t0 = time.time()
        convert_to_gguf()
        timings["convert_to_gguf"] = round(time.time() - t0, 1)
        pbar.update(1)
 
        t0 = time.time()
        quantize_q5_0()
        timings["quantize_q5_0"] = round(time.time() - t0, 1)
        pbar.update(1)
 
    save_metadata(timings)