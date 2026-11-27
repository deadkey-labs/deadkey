"""Live Model Supervisor for Deadkey — In-UI Hot Model Switching & Clean Stop-Stack.

Manages the local `llama-server.exe` lifecycle on an isolated port (`11455` by default)
so the user can switch between `gemma4-31b`, `glm47-flash`, `qwen38-35b-apex`,
`inkling2-mini-unbound`, `qwen36-35b-a3b-mtp`, `gemma4-26b-a4b`, and `blackfrost-27b-mtp`
directly from the UI dropdown, and shut down the entire stack cleanly from the browser
window (freeing 100% of RTX 5090 VRAM immediately).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class SupervisorState:
    status: str = "stopped"  # "stopped" | "starting" | "ready" | "error"
    active_model: str = ""
    active_label: str = ""
    protocol: str = "openai_native"
    port: int = 11455
    pid: Optional[int] = None
    vram_used_mib: Optional[int] = None
    vram_total_mib: int = 32607
    vram_free_mib: Optional[int] = None
    last_error: str = ""
    load_seconds: Optional[float] = None
    started_at: Optional[float] = None


def query_vram_mib() -> tuple[Optional[int], int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        parts = [int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(",")]
        return parts[0], parts[1]
    except Exception:
        return None, 32607


class ModelSupervisor:
    def __init__(self, manifest_path: Path, *, port: int = 11455, log_dir: Optional[Path] = None):
        self.manifest_path = Path(manifest_path)
        self.port = port
        self.log_dir = Path(log_dir) if log_dir else (self.manifest_path.parent.parent / "data" / "supervisor_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self.state = SupervisorState(port=port)
        self.models: dict[str, dict[str, Any]] = {}
        self._manifest_mtime: float = 0.0
        self.reload_manifest()

    def reload_manifest(self) -> None:
        try:
            mt = self.manifest_path.stat().st_mtime
            if mt == self._manifest_mtime and self.models:
                return
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.models = {m["name"]: m for m in raw}
            self._manifest_mtime = mt
        except Exception:
            pass

    def list_models(self) -> list[dict[str, Any]]:
        self.reload_manifest()
        out = []
        for name, m in self.models.items():
            out.append({
                "name": name,
                "label": m.get("label") or name,
                "role": m.get("role") or "companion",
                "vram_est_gb": m.get("vram_est_gb"),
                "tok_s_measured": m.get("tok_s_measured"),
                "protocol": m.get("tool_protocol", "openai_native"),
                "exists": Path(m["file"]).exists() and Path(m["exe"]).exists(),
                "active": (name == self.state.active_model and self.state.status in ("starting", "ready")),
            })
        return out

    def get_status(self, refresh_vram: bool = True) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is not None:
                if self.state.status == "ready":
                    self.state.status = "stopped"
                    self.state.pid = None
            if refresh_vram:
                used, total = query_vram_mib()
                self.state.vram_used_mib = used
                self.state.vram_total_mib = total
                self.state.vram_free_mib = (total - used) if used is not None else None
            return asdict(self.state)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            self._proc = None
            had_proc = proc is not None
            if proc is not None:
                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            capture_output=True, timeout=6,
                        )
                    else:
                        proc.terminate()
                    proc.wait(timeout=6)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=4)
                    except Exception:
                        pass
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
                self._log_fh = None
            self.state.status = "stopped"
            self.state.pid = None
            self.state.last_error = ""
            if had_proc:
                time.sleep(0.6)
            return self.get_status(refresh_vram=True)

    def switch_async(self, model_name: str) -> dict[str, Any]:
        """Triggers non-blocking model swap so the UI stays responsive and polls live status."""
        self.reload_manifest()
        if model_name not in self.models:
            raise KeyError(f"Unknown model profile: {model_name}")
        t = threading.Thread(target=self.switch_blocking, args=(model_name,), daemon=True)
        t.start()
        time.sleep(0.15)
        return self.get_status(refresh_vram=False)

    def switch_blocking(self, model_name: str, timeout_s: float = 150.0) -> dict[str, Any]:
        with self._lock:
            if model_name not in self.models:
                self.state.status = "error"
                self.state.last_error = f"Unknown model: {model_name}"
                return self.get_status()
            m = self.models[model_name]
            exe, mfile = m["exe"], m["file"]
            if not Path(exe).exists() or not Path(mfile).exists():
                self.state.status = "error"
                self.state.last_error = f"Missing binary or weights for {model_name}"
                return self.get_status()

            self.stop()
            self.state.status = "starting"
            self.state.active_model = model_name
            self.state.active_label = m.get("label") or model_name
            self.state.protocol = m.get("tool_protocol", "openai_native")
            self.state.last_error = ""
            self.state.load_seconds = None

            engine_type = str(m.get("engine_type") or "llama_server").lower()
            env = os.environ.copy()
            if engine_type == "strata":
                strata_cfg = str(m.get("strata_config") or "")
                strata_term = str(m.get("strata_terminal_dir") or "")
                server_py = Path(strata_term) / "serve" / "server.py"
                if not strata_cfg or not Path(strata_cfg).exists() or not server_py.exists():
                    self.state.status = "error"
                    self.state.last_error = f"Missing Strata server/config for {model_name}"
                    return self.get_status()
                env["PYTHONPATH"] = strata_term + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                env["STRATA_DATA_DIR"] = str(self.log_dir / "strata_data")
                env.setdefault("STRATA_VRAM_FREE", "device")
                cmd = [
                    sys.executable, str(server_py),
                    "--engine", "strata",
                    "--config", strata_cfg,
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--no-api-key",
                ]
            else:
                cmd = [
                    exe, "-m", mfile,
                    "-c", str(m.get("ctx", 32768)),
                    "-ngl", "99",
                    "-np", str(m.get("slots", 2)),
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--jinja",
                    "-ctk", "q8_0", "-ctv", "q8_0",
                ] + list(m.get("extra", []))
                # Optional approved Sleep-Time GGUF LoRA adapter (hot-swappable via POST /lora-adapters!)
                lora_path = m.get("lora_gguf")
                if lora_path and Path(lora_path).exists():
                    scale = float(m.get("lora_scale", 0.0))
                    cmd += ["--lora-scaled", f"{lora_path}:{scale:.2f}"]

            log_p = self.log_dir / f"{engine_type}-{model_name}.log"
            self._log_fh = log_p.open("w", encoding="utf-8", errors="replace")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                cmd, stdout=self._log_fh, stderr=subprocess.STDOUT,
                creationflags=creationflags, env=env,
            )
            my_proc = self._proc
            self.state.pid = my_proc.pid
            t0 = time.perf_counter()

        # Poll /health outside the lock so GET /v1/models/status stays instant
        deadline = time.time() + timeout_s
        healthy = False
        superseded = False
        while time.time() < deadline:
            with self._lock:
                if self._proc is not my_proc:
                    superseded = True
                    break
                if my_proc.poll() is not None:
                    break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        body = json.loads(r.read().decode("utf-8", errors="replace"))
                        if body.get("status") == "ok" or body.get("ok") is True or body.get("ready") is True:
                            healthy = True
                            break
            except Exception:
                pass
            time.sleep(1.0)

        with self._lock:
            if superseded or self._proc is not my_proc:
                return self.get_status(refresh_vram=True)
            if healthy:
                self.state.status = "ready"
                self.state.load_seconds = round(time.perf_counter() - t0, 1)
                self.state.started_at = time.time()
            else:
                reason = (
                    f"{model_name} exited before becoming healthy."
                    if my_proc.poll() is not None
                    else f"{model_name} failed to become healthy within {timeout_s:.0f}s"
                )
                self.stop()
                self.state.status = "error"
                self.state.last_error = reason
            return self.get_status(refresh_vram=True)

    def get_lora_adapters(self) -> list[dict[str, Any]]:
        """Queries live llama-server `/lora-adapters` for runtime A/B Sleep-Time LoRA scaling."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/lora-adapters", timeout=2) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            return []

    def set_lora_scale(self, adapter_id: int, scale: float) -> dict[str, Any]:
        """Hot-adjusts a loaded GGUF LoRA adapter scale (0.0 = frozen base, 1.0 = full adapter) without restarting!"""
        payload = json.dumps([{"id": int(adapter_id), "scale": float(scale)}]).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/lora-adapters",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = r.read().decode("utf-8", errors="replace")
        return {"ok": True, "adapter_id": adapter_id, "scale": scale, "raw": body}
