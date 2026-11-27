"""Deadkey / Deadkey Environment Doctor — 1-command readiness check for any GPU or CPU machine.

Usage:
  python -m deadkey.doctor
  python -m deadkey.doctor --json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _check_gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"ok": False, "detail": "nvidia-smi not found (CPU-only or non-NVIDIA host)"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "ok": True,
            "name": parts[0],
            "vram_total": parts[1] if len(parts) > 1 else "",
            "vram_free": parts[2] if len(parts) > 2 else "",
            "driver": parts[3] if len(parts) > 3 else "",
            "detail": f"{parts[0]} ({parts[1]} total, {parts[2]} free, driver {parts[3]})",
        }
    except Exception as exc:
        return {"ok": False, "detail": f"nvidia-smi error: {exc}"}


def _check_sqlite_fts5() -> dict[str, Any]:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts_test USING fts5(body)")
        conn.close()
        return {"ok": True, "detail": f"SQLite {sqlite3.sqlite_version} + FTS5 enabled (0 MiB GPU VRAM)"}
    except Exception as exc:
        return {"ok": False, "detail": f"SQLite FTS5 unavailable: {exc}"}


def _find_llama_server() -> dict[str, Any]:
    candidates = [
        ROOT / "engines" / "llama-b12780-cu135" / "llama-server.exe",
        ROOT / "engines" / "llama-server.exe",
        ROOT / "engines" / "llama-server",
    ]
    which_hit = shutil.which("llama-server")
    if which_hit:
        candidates.append(Path(which_hit))
    for c in candidates:
        if c.is_file():
            return {"ok": True, "path": str(c), "detail": f"Found {c}"}
    return {"ok": False, "detail": "No local llama-server binary found (install llama.cpp or point DEADKEY_MODELS_PROFILE)"}


def _check_cartridges() -> dict[str, Any]:
    cdir = ROOT / "cartridges"
    found = sorted(p.parent.name for p in cdir.glob("*/manifest.json")) if cdir.exists() else []
    has_blank = "blank-lab" in found
    return {
        "ok": bool(found),
        "cartridges": found,
        "zero_pii_blank_ready": has_blank,
        "detail": f"Cartridges: {', '.join(found) or 'none'} (blank-lab={'ready' if has_blank else 'missing'})",
    }


def _check_profile() -> dict[str, Any]:
    env_p = os.environ.get("DEADKEY_MODELS_PROFILE", "").strip()
    candidates = [
        Path(env_p) if env_p else None,
        ROOT / "profiles" / "models.json",
        ROOT / "profiles" / "laptop_4080_12gb.json",
        ROOT / "profiles" / "laptop_4080_12gb.json",
        ROOT / "profiles" / "models.example.json",
    ]
    for p in candidates:
        if p and p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                on_disk = sum(1 for m in data if Path(str(m.get("file", ""))).is_file())
                return {
                    "ok": True,
                    "profile": str(p),
                    "total": len(data),
                    "on_disk": on_disk,
                    "detail": f"{p.name}: {on_disk}/{len(data)} GGUF models present on disk",
                }
            except Exception as exc:
                return {"ok": False, "detail": f"Invalid JSON in {p}: {exc}"}
    return {"ok": False, "detail": "No model profile JSON found in profiles/"}


def run_doctor() -> dict[str, Any]:
    checks = {
        "python": {
            "ok": sys.version_info >= (3, 10),
            "detail": f"Python {platform.python_version()} ({platform.system()} {platform.machine()})",
        },
        "sqlite_fts5": _check_sqlite_fts5(),
        "gpu": _check_gpu(),
        "llama_server": _find_llama_server(),
        "cartridges": _check_cartridges(),
        "models_profile": _check_profile(),
    }
    core_ok = checks["python"]["ok"] and checks["sqlite_fts5"]["ok"] and checks["cartridges"]["ok"]
    return {"ok": core_ok, "root": str(ROOT), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description="Deadkey / Deadkey Environment Doctor")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    args = ap.parse_args()
    res = run_doctor()
    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1

    print("=" * 68)
    print("  DEADKEY — ENVIRONMENT DOCTOR (0-VRAM Memory + 2-Tier Gate)")
    print("=" * 68)
    for name, info in res["checks"].items():
        tag = "PASS" if info.get("ok") else "WARN"
        print(f"  [{tag}] {name:16s}: {info.get('detail')}")
    print("-" * 68)
    print("  Quickstart:")
    print("    1. Verify kernel : python -m bench.reproduce_all --quick")
    print("    2. Launch UI     : python -m deadkey.server --cartridge cartridges/blank-lab/manifest.json --port 8765")
    print("    3. MCP server    : python -m deadkey.mcp_server --db ./memory.db")
    print("=" * 68)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
