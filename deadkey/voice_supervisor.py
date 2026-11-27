"""Unified Voice Supervisor (9950X3D Whisper STT + Native Streaming Orpheus-Tara TTS).

Manages:
1. Local 9950X3D CPU INT8 `faster-whisper-small` (`0 MiB` GPU VRAM, ~450-550 ms on 2.5s speech)
   for instant browser mic transcription (`/v1/voice/transcribe`).
2. Supervised Native Streaming Orpheus-Tara stack:
   - `llama-server.exe` on port `11456` serving `Orpheus-3b-FT-Q8_0.gguf` (~4.2 GB VRAM)
   - `deadkey/orpheus_worker.py` on port `11457` (CPU SNAC 24kHz deterministic decoder, `0 MiB` VRAM)
   - Clean start/stop lifecycle tied directly to UI Voice Toggle + `Stop Stack` so 100% VRAM releases!
3. Speech-friendly markdown sanitizer so deadkey speaks crisp natural prose (never reads raw markdown/JSON).
"""
from __future__ import annotations

import base64
import http.client
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_ORPHEUS_GGUF = Path(
    r"C:\Users\user\.ollama\models\blobs\sha256-d435342ebf8428b08157445861370f84ffa5f6af8affcb3a9c00f542f39b4980"
)
DEFAULT_LLAMA_EXE = Path(r"C:\Users\user\colossus\llama-b12633\llama-server.exe")
DEFAULT_SNAC_PYTHON = Path(r"C:\Orpheus-FastAPI-LMStudio\venv312\Scripts\python.exe")

_CODEBLOCK_RX = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RX = re.compile(r"`([^`]+)`")
_MD_LINK_RX = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RX = re.compile(r"https?://\S+")
_MD_SYM_RX = re.compile(r"[*_#~>|]+")
_MULTI_SPACE_RX = re.compile(r"\s+")


def sanitize_for_speech(text: str) -> str:
    """Strips markdown syntax and code fences so Orpheus-Tara speaks natural human prose."""
    t = _CODEBLOCK_RX.sub(" I put the code snippet in the chat for you. ", text or "")
    t = _MD_LINK_RX.sub(r"\1", t)
    t = _URL_RX.sub("", t)
    t = _INLINE_CODE_RX.sub(r"\1", t)
    t = _MD_SYM_RX.sub(" ", t)
    t = _MULTI_SPACE_RX.sub(" ", t).strip()
    return t


@dataclass
class VoiceState:
    status: str = "stopped"  # "stopped" | "starting" | "ready" | "error"
    engine: str = "orpheus_native_stream"
    voice: str = "tara"
    llama_port: int = 11456
    snac_port: int = 11457
    stt_ready: bool = False
    stt_model: str = "small (CPU INT8 · 0 MiB VRAM)"
    last_error: str = ""
    load_seconds: Optional[float] = None
    last_ttfa_ms: Optional[float] = None


class VoiceSupervisor:
    def __init__(
        self,
        *,
        orpheus_gguf: Path = DEFAULT_ORPHEUS_GGUF,
        llama_exe: Path = DEFAULT_LLAMA_EXE,
        snac_python: Path = DEFAULT_SNAC_PYTHON,
        llama_port: int = 11456,
        snac_port: int = 11457,
        log_dir: Optional[Path] = None,
    ):
        self.orpheus_gguf = Path(orpheus_gguf)
        self.llama_exe = Path(llama_exe)
        self.snac_python = Path(snac_python)
        self.worker_script = Path(__file__).resolve().parent / "orpheus_worker.py"
        self.llama_port = llama_port
        self.snac_port = snac_port
        self.log_dir = Path(log_dir) if log_dir else (Path(__file__).resolve().parent.parent / "data" / "supervisor_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._llama_proc: Optional[subprocess.Popen] = None
        self._snac_proc: Optional[subprocess.Popen] = None
        self._fhs: list[Any] = []
        self._whisper = None
        self._whisper_lock = threading.Lock()
        self.state = VoiceState(llama_port=llama_port, snac_port=snac_port)

    def available(self) -> dict[str, bool]:
        return {
            "orpheus_gguf": self.orpheus_gguf.exists(),
            "llama_exe": self.llama_exe.exists(),
            "snac_python": self.snac_python.exists(),
            "worker_script": self.worker_script.exists(),
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            if self.state.status == "ready":
                if (self._llama_proc and self._llama_proc.poll() is not None) or (
                    self._snac_proc and self._snac_proc.poll() is not None
                ):
                    self.state.status = "stopped"
            d = asdict(self.state)
            d["available"] = self.available()
            return d

    def start_async(self, voice: str = "tara") -> dict[str, Any]:
        with self._lock:
            self.state.voice = voice or "tara"
            if self.state.status in ("starting", "ready"):
                return self.get_status()
            self.state.status = "starting"
            self.state.last_error = ""
        t = threading.Thread(target=self.start_blocking, args=(voice,), daemon=True)
        t.start()
        return self.get_status()

    def start_blocking(self, voice: str = "tara", timeout_s: float = 60.0) -> dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock:
            self.state.voice = voice or "tara"
            avail = self.available()
            if not all(avail.values()):
                self.state.status = "error"
                self.state.last_error = f"Missing Orpheus component: {avail}"
                return self.get_status()

            self.stop()
            self.state.status = "starting"
            self.state.last_error = ""
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            l_log = (self.log_dir / "orpheus-llama.log").open("w", encoding="utf-8", errors="replace")
            s_log = (self.log_dir / "orpheus-snac.log").open("w", encoding="utf-8", errors="replace")
            self._fhs = [l_log, s_log]

            llama_cmd = [
                str(self.llama_exe),
                "-m", str(self.orpheus_gguf),
                "-ngl", "99",
                "-c", "4096",
                "--host", "127.0.0.1",
                "--port", str(self.llama_port),
            ]
            snac_cmd = [
                str(self.snac_python),
                str(self.worker_script),
                "--port", str(self.snac_port),
                "--llama-port", str(self.llama_port),
            ]
            self._llama_proc = subprocess.Popen(llama_cmd, stdout=l_log, stderr=subprocess.STDOUT, creationflags=creationflags)
            self._snac_proc = subprocess.Popen(snac_cmd, stdout=s_log, stderr=subprocess.STDOUT, creationflags=creationflags)
            my_llama = self._llama_proc
            my_snac = self._snac_proc

        # Warm Whisper on 9950X3D CPU in parallel while Orpheus + SNAC boot!
        threading.Thread(target=self._ensure_whisper, daemon=True).start()

        deadline = time.time() + timeout_s
        llama_ok = False
        snac_ok = False
        aborted = False
        while time.time() < deadline:
            with self._lock:
                if self._llama_proc is not my_llama or self._snac_proc is not my_snac:
                    aborted = True
                    break
                if my_llama.poll() is not None or my_snac.poll() is not None:
                    break
            if not llama_ok:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.llama_port}/health", timeout=1.5) as r:
                        if json.loads(r.read().decode("utf-8", errors="replace")).get("status") == "ok":
                            llama_ok = True
                except Exception:
                    pass
            if not snac_ok:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.snac_port}/health", timeout=1.5) as r:
                        if json.loads(r.read().decode("utf-8", errors="replace")).get("snac_ready"):
                            snac_ok = True
                except Exception:
                    pass
            if llama_ok and snac_ok:
                break
            time.sleep(0.5)

        with self._lock:
            if aborted or self._llama_proc is not my_llama:
                return self.get_status()
            if llama_ok and snac_ok:
                self.state.status = "ready"
                self.state.load_seconds = round(time.perf_counter() - t0, 1)
            else:
                self.stop()
                self.state.status = "error"
                self.state.last_error = f"Voice startup failed (llama_ok={llama_ok}, snac_ok={snac_ok})"
            return self.get_status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            for proc in (self._snac_proc, self._llama_proc):
                if proc is not None:
                    try:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=5)
                        else:
                            proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                            proc.wait(timeout=3)
                        except Exception:
                            pass
            self._snac_proc = None
            self._llama_proc = None
            for fh in self._fhs:
                try:
                    fh.close()
                except Exception:
                    pass
            self._fhs = []
            self.state.status = "stopped"
            return self.get_status()

    # ---- 9950X3D Local Whisper STT (0 MiB GPU VRAM) ----
    def _ensure_whisper(self):
        with self._whisper_lock:
            if self._whisper is not None:
                return self._whisper
            try:
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
                with self._lock:
                    self.state.stt_ready = True
            except Exception as exc:
                with self._lock:
                    self.state.last_error = f"Whisper load failed: {exc}"
                raise
            return self._whisper

    def transcribe_audio(self, audio_b64: str, *, suffix: str = ".wav") -> dict[str, Any]:
        """Transcribes base64 WAV audio locally on CPU INT8 (pure wave+numpy, thread-safe, zero PyAV quirks)."""
        import wave
        import numpy as np

        raw = base64.b64decode(audio_b64.split(",")[-1])
        t0 = time.perf_counter()
        model = self._ensure_whisper()
        with wave.open(io.BytesIO(raw), "rb") as wf:
            n_ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sw != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got {sw * 8}-bit")
        if not frames:
            return {"ok": True, "text": "", "elapsed_ms": 0.0, "duration_s": 0.0, "engine": "faster-whisper-small-cpu-int8"}
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch).mean(axis=1)
        if sr != 16000 and len(pcm) > 1:
            new_len = max(1, int(round(len(pcm) * 16000.0 / sr)))
            xp = np.linspace(0.0, 1.0, len(pcm), dtype=np.float32)
            x = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
            pcm = np.interp(x, xp, pcm).astype(np.float32)
        duration_s = round(len(pcm) / 16000.0, 2)

        with self._whisper_lock:
            segs, _info = model.transcribe(pcm, beam_size=1, vad_filter=False, language="en")
            text = " ".join(s.text.strip() for s in segs).strip()
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "ok": True,
            "text": text,
            "elapsed_ms": elapsed_ms,
            "duration_s": duration_s,
            "engine": "faster-whisper-small-cpu-int8",
        }

    # ---- Native Streaming Orpheus-Tara TTS ----
    def stream_speech_pcm(self, text: str, *, voice: Optional[str] = None) -> Iterator[bytes]:
        """Yields live 24kHz PCM16 audio chunks from Orpheus-Tara worker and records TTFA."""
        clean = sanitize_for_speech(text)
        if not clean:
            return
        v = voice or self.state.voice or "tara"
        payload = json.dumps({"input": clean, "voice": v, "response_format": "pcm"}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", self.snac_port, timeout=120)
        t0 = time.perf_counter()
        first = True
        try:
            conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                err = resp.read(200).decode("utf-8", errors="replace")
                raise RuntimeError(f"Orpheus worker HTTP {resp.status}: {err}")
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                if first:
                    self.state.last_ttfa_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    first = False
                yield chunk
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def synthesize_wav(self, text: str, *, voice: Optional[str] = None) -> tuple[bytes, float]:
        """Synthesizes a clean 24kHz WAV clip via Orpheus-Tara worker (returns bytes, ms)."""
        clean = sanitize_for_speech(text)
        if not clean:
            return b"", 0.0
        v = voice or self.state.voice or "tara"
        payload = json.dumps({"input": clean, "voice": v, "response_format": "wav"}).encode("utf-8")
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.snac_port}/v1/audio/speech",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            wav = r.read()
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.state.last_ttfa_ms = ms
        return wav, ms
