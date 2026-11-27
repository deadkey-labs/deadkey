"""Cartridge-aware configuration with zero hardcoded C:\\legacy-readonly paths.

Every runtime path resolves from AGENT_HOME (or an explicit cartridge directory),
so the exact same engine can run deadkey or the operator's AI Media Studio Director with
complete isolation of memory, identity, voice, and tool packs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CartridgeConfig:
    agent_id: str
    display_name: str
    home_dir: Path
    memory_dir: Path
    db_path: Path
    ledger_path: Path
    audit_dir: Path
    workspace_roots: tuple[Path, ...]
    enabled_packs: tuple[str, ...]
    voice_profile: str = "tara"
    architecture: str = "arch_a"  # "arch_a" (Streaming Orpheus GPU) | "arch_b" (CPU Pocket TTS + Max-Brain GPU)
    provider_profile: str = "qwen36-a3b"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "CartridgeConfig":
        p = Path(manifest_path).resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        home = p.parent
        mem = (home / data.get("memory_dir", "data/memory")).resolve()
        audit = (home / data.get("audit_dir", "data/audit")).resolve()
        roots = tuple((home / r).resolve() if not Path(r).is_absolute() else Path(r)
                      for r in data.get("workspace_roots", ["workspace"]))
        return cls(
            agent_id=data["agent_id"],
            display_name=data["display_name"],
            home_dir=home,
            memory_dir=mem,
            db_path=(home / data.get("db_path", "data/memory.db")).resolve(),
            ledger_path=(mem / data.get("ledger_file", "ledger.jsonl")).resolve(),
            audit_dir=audit,
            workspace_roots=roots,
            enabled_packs=tuple(data.get("enabled_packs", ["core", "memory"])),
            voice_profile=data.get("voice_profile", "tara"),
            architecture=os.environ.get("AGENT_ARCH", data.get("architecture", "arch_a")),
            provider_profile=os.environ.get("AGENT_MODEL_PROFILE", data.get("provider_profile", "qwen36-a3b")),
            extra=data.get("extra", {}),
        )
