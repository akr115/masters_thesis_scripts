import os
import subprocess
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
USER           = os.environ["USER"]
BASE           = Path(f"/home2/{USER}/masters_thesis")

HF_MODEL_DIR   = BASE / "models" / "llama3.2-3b-wanda-pruned"
OUTPUT_DIR     = BASE / "llama3.2-3b-compressed" / "pruned"
GGUF_F16       = OUTPUT_DIR / "llama3.2-3b-wanda-66-f16.gguf"

LLAMA_CPP_DIR  = BASE / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────
def check_paths():
    missing = []
    for p in [HF_MODEL_DIR, CONVERT_SCRIPT]:
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

# ──────────────────────────────────────────────
# Convert
# ──────────────────────────────────────────────
def convert_to_gguf():
    cmd = [
        "python3", str(CONVERT_SCRIPT),
        str(HF_MODEL_DIR),
        "--outfile", str(GGUF_F16),
        "--outtype", "f16",
    ]
    print(f"\n[STEP] Converting pruned model to GGUF f16")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\n[DONE] GGUF saved -> {GGUF_F16}")
    print(f"  Size: {GGUF_F16.stat().st_size / (1024**3):.3f} GB")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    check_paths()
    convert_to_gguf()