import json
import re
import subprocess
from pathlib import Path

from config import cfg


def run_perplexity(model_path: str | Path, output_dir: Path) -> dict:
    """Run llama-perplexity on the WikiText test set and save the result.

    The llama-perplexity binary is expected in the same directory as llama-server.
    WIKITEXT_PATH must be set in config.env.

    Returns a stats dict with 'ppl' and 'ppl_std'.
    """
    if cfg.wikitext_path is None:
        raise ValueError(
            "WIKITEXT_PATH is not set in config.env — required for perplexity evaluation."
        )

    perplexity_bin = cfg.llama_server.parent / "llama-perplexity"
    if not perplexity_bin.is_file():
        raise FileNotFoundError(
            f"llama-perplexity binary not found at {perplexity_bin}\n"
            "  Build llama.cpp and ensure llama-perplexity is compiled."
        )

    cmd = [
        str(perplexity_bin),
        "-m", str(model_path),
        "-f", str(cfg.wikitext_path),
        "-ngl", str(cfg.llama_n_gpu_layers),
        "-c", str(cfg.llama_n_ctx), 
        "-b", str(cfg.llama_batch_size),
    ]
    if cfg.llama_n_threads > 0:
        cmd += ["-t", str(cfg.llama_n_threads)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    output = result.stdout + result.stderr

    # Parse "Final estimate: PPL = 5.1234 +/- 0.0567"
    match = re.search(r"PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", output)
    if not match:
        raise RuntimeError(
            f"Could not parse PPL value from llama-perplexity output:\n{output}"
        )

    stats = {
        "model": Path(model_path).stem,
        "ppl": float(match.group(1)),
        "ppl_std": float(match.group(2)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "perplexity.jsonl"
    with stats_path.open("a") as f:
        f.write(json.dumps(stats) + "\n")

    return stats