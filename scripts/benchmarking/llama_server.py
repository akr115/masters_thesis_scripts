from pathlib import Path
from config import cfg
import subprocess
import requests
import time

HOST = "127.0.0.1"
PORT = 8080

SERVER_START_TIMEOUT = 120 # seconds to wait for model to load
REQUEST_TIMEOUT = 20        # seconds per prompt

# This class manages the llama server process and provides a method to send completion requests.
class LlamaServer:
    def __init__(self, model_path: str | Path, port: int = PORT):
        self.model_path = Path(model_path)
        self.port = port
        self._base = f"http://{HOST}:{port}"
        self._proc = None

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
            "-ngl", str(cfg.llama_n_gpu_layers),
            "-b", str(cfg.llama_batch_size),
        ]
        if cfg.llama_n_threads > 0:
            cmd += ["-t", str(cfg.llama_n_threads)]
        if cfg.llama_n_threads_batch > 0:
            cmd += ["-tb", str(cfg.llama_n_threads_batch)]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def _wait_ready(self):
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self._base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self._stop()
        raise RuntimeError(
            f"llama-server did not become ready within {SERVER_START_TIMEOUT}s"
        )

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def complete(self, prompt: str) -> str:
        """POST to /completion and return the generated text string."""
        payload = {
            "prompt": prompt,
            "temperature": cfg.llama_temperature,
            "n_predict": cfg.llama_n_predict,
        }
        r = requests.post(
            f"{self._base}/completion",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["content"]