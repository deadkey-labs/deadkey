"""Zero-VRAM Local CPU Semantic Embedder (`nomic-embed-text` on Ryzen 9 9950X3D).

Runs `llama-server.exe -ngl 0 --embeddings --pooling mean` on isolated port `11458`
using your local `nomic-embed-text` GGUF (`274 MB` RAM, **`0 MiB` RTX 5090 VRAM**!).
Boots in ~1 second, embeds queries in ~6-12 ms, and shuts down cleanly on `Stop Stack`.
"""
from __future__ import annotations

import json
import math
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_NOMIC_GGUF = Path(
    r"C:\Users\user\.ollama\models\blobs\sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
DEFAULT_LLAMA_EXE = Path(r"C:\Users\user\colossus\llama-b12633\llama-server.exe")


def _l2_norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


class LocalEmbedder:
    def __init__(
        self,
        *,
        gguf_path: Path = DEFAULT_NOMIC_GGUF,
        llama_exe: Path = DEFAULT_LLAMA_EXE,
        port: int = 11458,
        log_dir: Optional[Path] = None,
    ):
        self.gguf_path = Path(gguf_path)
        self.llama_exe = Path(llama_exe)
        self.port = port
        self.log_dir = Path(log_dir) if log_dir else (Path(__file__).resolve().parent.parent / "data" / "supervisor_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self.ready = False
        self.dim = 768

    def available(self) -> bool:
        return self.gguf_path.exists() and self.llama_exe.exists()

    def is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.8) as r:
                return json.loads(r.read().decode("utf-8", errors="replace")).get("status") == "ok"
        except Exception:
            return False

    def start(self, timeout_s: float = 15.0) -> bool:
        with self._lock:
            if self.is_healthy():
                self.ready = True
                return True
            if not self.available():
                return False
            self.stop()
            self._log_fh = (self.log_dir / "nomic-embed-cpu.log").open("w", encoding="utf-8", errors="replace")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            cmd = [
                str(self.llama_exe),
                "-m", str(self.gguf_path),
                "-ngl", "0",          # STRICT 0 MiB GPU VRAM — runs purely on Ryzen 9 9950X3D!
                "-c", "2048",
                "-t", "8",
                "--embeddings",
                "--pooling", "mean",
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ]
            self._proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT, creationflags=creationflags)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_healthy():
                self.ready = True
                return True
            time.sleep(0.25)
        self.ready = False
        return False

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=4)
                except Exception:
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=2)
                    except Exception:
                        pass
                self._proc = None
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
                self._log_fh = None
            self.ready = False

    def embed_batch(self, texts: list[str], *, prefix: str = "search_document: ") -> list[list[float]]:
        if not texts or not self.ready:
            return []
        prefixed = [f"{prefix}{(t or '')[:1200]}" for t in texts]
        payload = json.dumps({"model": "nomic-embed-text", "input": prefixed}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        items = sorted(data.get("data") or [], key=lambda x: x.get("index", 0))
        vecs = [_l2_norm(it["embedding"]) for it in items if it.get("embedding")]
        if vecs:
            self.dim = len(vecs[0])
        return vecs

    def embed_query(self, query: str) -> Optional[list[float]]:
        if not query or not self.ready:
            return None
        try:
            vecs = self.embed_batch([query], prefix="search_query: ")
            return vecs[0] if vecs else None
        except Exception:
            return None
