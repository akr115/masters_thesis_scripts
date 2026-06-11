import os
import subprocess
from pathlib import Path

HF_MODEL_DIR   = Path("../../llama-3.2-1b-instruct")          # ← adjust this
LLAMA_CPP_DIR  = Path("../../llama.cpp")      # ← adjust this

CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
GGUF_OUT       = HF_MODEL_DIR / (HF_MODEL_DIR.name + "-f16.gguf")

cmd = [
    "python3", str(CONVERT_SCRIPT),
    str(HF_MODEL_DIR),
    "--outfile", str(GGUF_OUT),
    "--outtype", "f16",
]

print(f"Converting {HF_MODEL_DIR.name} → {GGUF_OUT.name}")
subprocess.run(cmd, check=True)
print(f"Done. Size: {GGUF_OUT.stat().st_size / (1024**3):.2f} GB")