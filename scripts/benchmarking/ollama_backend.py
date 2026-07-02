import time

import ollama as _ollama

from config import cfg


class OllamaBackend:
    """Accuracy-evaluation backend that uses Ollama instead of llama-server.

    Uses ollama.generate() to match the NeurIPS paper's inference setup exactly.
    The Ollama daemon must be running and the model must already be pulled.

    Interface is a drop-in replacement for LlamaServer: context manager +
    complete() returning the same timing dict.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.ttlm_s = None  # Ollama manages model loading internally

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def complete(self, prompt: str, max_tokens: int | None = None) -> dict:
        options: dict = {"temperature": cfg.llama_temperature}
        if max_tokens is not None and max_tokens > 0:
            options["num_predict"] = max_tokens

        t0 = time.perf_counter()
        resp = _ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
        )
        t_done = time.perf_counter()

        eval_dur_s = (resp.eval_duration or 0) / 1e9
        return {
            "content":          resp.response,
            "e2e_latency_s":    t_done - t0,
            "ttft_approx_ms":   (resp.prompt_eval_duration or 0) / 1e6,
            "throughput_tps":   (resp.eval_count / eval_dur_s) if eval_dur_s > 0 else None,
            "prompt_tokens":    resp.prompt_eval_count,
            "generated_tokens": resp.eval_count,
            "generation_ms":    (resp.eval_duration or 0) / 1e6,
        }
