import json
import subprocess
import requests
import time
from pathlib import Path

from config import cfg

HOST = "127.0.0.1"
PORT = 8080
SERVER_START_TIMEOUT = 120
REQUEST_TIMEOUT = 1200


class LlamaServer:
    def __init__(self, model_path: str | Path, port: int = PORT):
        self.model_path = Path(model_path)
        self.port = port
        self._base = f"http://{HOST}:{port}"
        self._proc = None
        self.ttlm_s: float | None = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *_):
        self._stop()

    def _start(self):
        cmd = [
            str(cfg.llama_server),
            "-m", str(self.model_path),
            "--host", HOST,
            "--port", str(self.port),
            "-c", str(cfg.llama_n_ctx),
            "-b", str(cfg.llama_batch_size),
            "--cache-type-k", cfg.llama_kv_cache_type,
            "--cache-type-v", cfg.llama_kv_cache_type,
            "--no-webui",
            "--jinja",  # use full GGUF Jinja template (adds Cutting Knowledge Date system header)
        ]
        cmd += ["-ngl", str(cfg.llama_n_gpu_layers)]
        if cfg.llama_n_threads > 0:
            cmd += ["-t", str(cfg.llama_n_threads)]
        if cfg.llama_n_threads_batch > 0:
            cmd += ["-tb", str(cfg.llama_n_threads_batch)]

        t0 = time.perf_counter()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._wait_ready()
        self.ttlm_s = time.perf_counter() - t0

    def _wait_ready(self):
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                out = self._proc.stdout.read().decode(errors="ignore")
                raise RuntimeError(f"llama-server exited early (code {self._proc.returncode}):\n{out[-2000:]}")
            try:
                r = requests.get(f"{self._base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self._stop()
        raise RuntimeError(f"llama-server did not become ready within {SERVER_START_TIMEOUT}s")

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def complete(self, prompt: str, max_tokens: int | None = None) -> dict:
        # Use /v1/chat/completions so llama-server applies the model's instruct chat
        # template automatically — matching the inference setup used by the paper
        # (Ollama's generate API also applies the template internally).
        # Sending to /completion instead would bypass the template and degrade
        # instruction-following significantly for instruct models.
        n = max_tokens if max_tokens is not None else cfg.llama_n_predict
        payload: dict = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.llama_temperature,
        }
        if n > 0:
            payload["max_tokens"] = n
        t0 = time.perf_counter()
        r = requests.post(f"{self._base}/v1/chat/completions", json=payload, timeout=REQUEST_TIMEOUT)
        t_done = time.perf_counter()
        r.raise_for_status()
        body = r.json()
        tim = body.get("timings", {})
        usage = body.get("usage", {})
        return {
            "content":          body["choices"][0]["message"]["content"],
            "e2e_latency_s":    t_done - t0,
            "ttft_approx_ms":   tim.get("prompt_ms"),
            "throughput_tps":   tim.get("predicted_per_second"),
            "prompt_tokens":    tim.get("prompt_n") or usage.get("prompt_tokens"),
            "generated_tokens": tim.get("predicted_n") or usage.get("completion_tokens"),
            "generation_ms":    tim.get("predicted_ms"),
        }

    def complete_stream(self, prompt: str) -> dict:
        """Streaming /completion — records wall-clock TTFT from first SSE chunk."""
        payload = {"prompt": prompt, "temperature": cfg.llama_temperature,
                   "n_predict": cfg.llama_n_predict, "stream": True}
        t0 = time.perf_counter()
        ttft_s = None
        content_parts = []
        final_timings = {}

        with requests.post(f"{self._base}/completion", json=payload,
                           stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                chunk = json.loads(raw[6:])
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                content_parts.append(chunk.get("content", ""))
                if chunk.get("stop"):
                    final_timings = chunk.get("timings", {})
                    break

        return {
            "content":          "".join(content_parts),
            "ttft_s":           ttft_s,
            "e2e_latency_s":    time.perf_counter() - t0,
            "throughput_tps":   final_timings.get("predicted_per_second"),
            "prompt_tokens":    final_timings.get("prompt_n"),
            "generated_tokens": final_timings.get("predicted_n"),
            "generation_ms":    final_timings.get("predicted_ms"),
        }
