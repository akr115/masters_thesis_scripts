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

GGUF_DIR       = BASE / "llama3.2-3b-instruct-gguf"
OUTPUT_DIR     = GGUF_DIR                              # write Q5_0 alongside existing files
METADATA_FILE  = OUTPUT_DIR / "metadata_q5_0.json"

LLAMA_CPP_DIR  = BASE / "llama.cpp"
LLAMA_QUANTIZE = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"


# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────
def find_f16_gguf() -> Path:
    """Return the first *f16*.gguf (or fallback: any .gguf) in GGUF_DIR."""
    candidates = sorted(GGUF_DIR.glob("*f16*.gguf"))
    if not candidates:
        candidates = sorted(GGUF_DIR.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(
            f"No .gguf file found in {GGUF_DIR}. "
            "Place the f16 GGUF there before running this script."
        )
    if len(candidates) > 1:
        print(f"[WARN] Multiple GGUF files found; using: {candidates[0].name}")
    return candidates[0]


def check_paths(gguf_f16: Path) -> None:
    missing = []
    for p in [gguf_f16, LLAMA_QUANTIZE]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Required paths not found:\n" + "\n".join(f"  {p}" for p in missing)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CHECK] All required paths exist.")
    print(f"  input  (f16): {gguf_f16}")
    print(f"  output dir  : {OUTPUT_DIR}")
    print(f"  quantize bin: {LLAMA_QUANTIZE}")


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
# Step 1: Quantize f16 -> Q5_0
# ──────────────────────────────────────────────
def quantize_q5_0(gguf_f16: Path, gguf_q5_0: Path) -> None:
    run(
        [str(LLAMA_QUANTIZE), str(gguf_f16), str(gguf_q5_0), "Q5_0"],
        desc=f"Quantize f16 -> Q5_0  =>  {gguf_q5_0.name}",
    )


# ──────────────────────────────────────────────
# Step 2: Save metadata
# ──────────────────────────────────────────────
def save_metadata(gguf_f16: Path, gguf_q5_0: Path, timings: dict) -> None:
    metadata = {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": "Q5_0",
        "paths": {
            "gguf_f16":  str(gguf_f16),
            "gguf_q5_0": str(gguf_q5_0),
        },
        "sizes_gb": {
            "gguf_f16":  round(file_size_gb(gguf_f16), 3)  if gguf_f16.exists()  else None,
            "gguf_q5_0": round(file_size_gb(gguf_q5_0), 3) if gguf_q5_0.exists() else None,
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
    gguf_f16  = find_f16_gguf()
    # Derive Q5_0 output name from the f16 filename
    gguf_q5_0 = OUTPUT_DIR / gguf_f16.name.replace("f16", "q5_0")
    if gguf_q5_0 == gguf_f16:                          # name had no 'f16' token
        gguf_q5_0 = OUTPUT_DIR / (gguf_f16.stem + "-q5_0.gguf")

    check_paths(gguf_f16)
    timings = {}

    with tqdm(total=1, desc="Overall pipeline") as pbar:
        t0 = time.time()
        quantize_q5_0(gguf_f16, gguf_q5_0)
        timings["quantize_q5_0"] = round(time.time() - t0, 1)
        pbar.update(1)

    save_metadata(gguf_f16, gguf_q5_0, timings)