"""Deadkey HTTP Server — zero-dependency, cartridge-isolated, Disk-Equals-Truth runtime.

Full Heavy-Engineering Stack:
- ModelSupervisor (`:11455`) — in-UI hot switching across all 7 Colossus models + live VRAM bar.
- VoiceSupervisor (`:11456` Orpheus-3B Q8_0 GGUF + `:11457` CPU SNAC + 9950X3D Whisper INT8 STT).
- ToolRuntime — hardened file/code/python/memory/silicon/screen-vision/web/live-Hermes IPC tools.
- SlotOneCognition (`id_slot=1`) — background reflection with 0% KV eviction against live Slot-0 chat.
- Auto Hill-Climb + Sleep-Time DPO LoRA Trainer Generator (human-approved SHA-pinned RSI).
- Stop Stack + 4-second reload-safe Stop-on-Close releasing 100% of RTX 5090 VRAM (LLM + Orpheus!).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .auto_hillclimb import generate_5090_dpo_trainer, read_hillclimb_ledger, run_live_hillclimb_trial
except ImportError:
    generate_5090_dpo_trainer = run_live_hillclimb_trial = None
    def read_hillclimb_ledger(*_a, **_kw): return []
from .cognition import SlotOneCognition
from .config import CartridgeConfig
from .embedder import LocalEmbedder
from .epistemic_gate import EpistemicCommitGate, EvidenceReceipt, build_file_receipt
from .rd_scout import read_radar, run_rd_scout
from .live_client import LocalLLMClient
from .memory_store import MemoryStore
from .prefetch import SpeculativePrefetcher
from .providers import ParsedModelTurn, build_architecture_profiles
from .sessions import SessionLedger
from .sleep_cycle import run_sleep_consolidation
from .supervisor import ModelSupervisor
from .tool_router import CompiledToolRouter
from .tools_runtime import ToolRuntime
from .turn_machine import TurnMachine
from .vgr_local import LocalVGRRouter
from .voice_supervisor import VoiceSupervisor

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_SAFE_FILE_RX = re.compile(r"[^A-Za-z0-9._-]+")


class CoreRuntime:
    def __init__(
        self,
        *,
        cartridge: CartridgeConfig,
        tools_catalog_path: Path,
        backend_url: str = "http://127.0.0.1:11455",
        tool_protocol: str = "openai_native",
        ui_path: Optional[Path] = None,
        models_manifest_path: Optional[Path] = None,
        cartridges_dir: Optional[Path] = None,
        supervisor_port: int = 11455,
        prefs_path: Optional[Path] = None,
    ):
        self.project_root = Path(__file__).resolve().parent.parent
        self.cartridges_dir = cartridges_dir or (self.project_root / "cartridges")
        self.tools_catalog_path = Path(tools_catalog_path)
        self.ui_path = ui_path or (self.project_root / "ui" / "index.html")
        self.prefs_path = Path(prefs_path) if prefs_path else (self.project_root / "data" / "user_prefs.json")
        self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._shutdown_cb: Optional[Callable[[], None]] = None
        self._shutdown_requested = False
        self._close_timer: Optional[threading.Timer] = None
        self._turns_since_cognition = 0

        self.archs = build_architecture_profiles()
        self.gate = EpistemicCommitGate()
        self.client = LocalLLMClient(base_url=backend_url, tool_protocol=tool_protocol)

        env_profile = os.environ.get("DEADKEY_MODELS_PROFILE", "").strip()
        m_manifest = (
            models_manifest_path
            or (Path(env_profile) if env_profile else (self.project_root / "profiles" / "colossus_models.json"))
        )
        self.supervisor: Optional[ModelSupervisor] = (
            ModelSupervisor(m_manifest, port=supervisor_port) if m_manifest.exists() else None
        )
        self.voice = VoiceSupervisor(log_dir=self.project_root / "data" / "supervisor_logs")
        self.embedder = LocalEmbedder(log_dir=self.project_root / "data" / "supervisor_logs")
        self.cognition = SlotOneCognition(self)

        self.memory: Optional[MemoryStore] = None
        self.sessions: Optional[SessionLedger] = None
        self.tools_runtime: Optional[ToolRuntime] = None
        self._uploads_cache: dict[str, dict[str, Any]] = {}
        self.load_cartridge(cartridge)

    def cancel_pending_close_shutdown(self) -> None:
        with self._lock:
            if self._close_timer is not None:
                self._close_timer.cancel()
                self._close_timer = None
                self._shutdown_requested = False

    def schedule_close_shutdown(self, delay_s: float = 4.0) -> None:
        """Grace-period shutdown for browser `pagehide` beacon so F5/Reload NEVER kills the stack."""
        def _fire():
            with self._lock:
                self._close_timer = None
                if not self._shutdown_requested:
                    return
            self._stop_gpu_processes()
            if self._shutdown_cb is not None:
                self._shutdown_cb()

        with self._lock:
            if self._close_timer is not None:
                self._close_timer.cancel()
            self._shutdown_requested = True
            self._close_timer = threading.Timer(delay_s, _fire)
            self._close_timer.daemon = True
            self._close_timer.start()

    def _stop_gpu_processes(self) -> None:
        try:
            self.voice.stop()
        except Exception:
            pass
        try:
            self.embedder.stop()
        except Exception:
            pass
        if self.supervisor is not None:
            try:
                self.supervisor.stop()
            except Exception:
                pass

    def load_cartridge(self, cartridge: CartridgeConfig) -> None:
        with self._lock:
            if self.memory is not None:
                self.memory.close()
            if self.sessions is not None:
                self.sessions.close_db()

            self.cartridge = cartridge
            self.active_arch = "arch_a_orpheus_vllm" if cartridge.architecture == "arch_a" else "arch_b_pocket_cpu_maxbrain"
            self.memory = MemoryStore(cartridge.db_path)
            self.sessions = SessionLedger(cartridge.home_dir / "data" / "sessions.db")
            self.active_session_id = self.sessions.ensure_active(f"{cartridge.display_name} — Live Session")

            tools_p = cartridge.home_dir / "tools_catalog.json"
            if not tools_p.exists():
                tools_p = self.tools_catalog_path
            self.router = CompiledToolRouter.from_json(tools_p, max_tools=10)
            is_private = bool(cartridge.extra.get("cloud_privacy_wall", True))
            self.vgr = LocalVGRRouter(
                cloud_privacy_wall=is_private,
                allow_cloud_escalation=(not is_private),
            )
            roots = {f"root_{i}": r for i, r in enumerate(cartridge.workspace_roots)}
            roots["home"] = cartridge.home_dir
            self.uploads_dir = cartridge.home_dir / "data" / "uploads"
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            roots["uploads"] = self.uploads_dir
            self.roots = roots
            self.prefetcher = SpeculativePrefetcher(roots=roots, memory=self.memory, embedder=self.embedder)
            self.approvals_path = cartridge.home_dir / "data" / "approvals.jsonl"
            self.pins_path = cartridge.home_dir / "data" / "pins.json"
            self.partner_path = cartridge.home_dir / "data" / "partner_queue.json"
            self.tools_runtime = ToolRuntime(self)

    def close(self) -> None:
        with self._lock:
            self._stop_gpu_processes()
            if self.memory is not None:
                try:
                    self.memory.close()
                except Exception:
                    pass
                self.memory = None
            if self.sessions is not None:
                try:
                    self.sessions.close_db()
                except Exception:
                    pass
                self.sessions = None

    # ---- Preferences ----
    def read_prefs(self) -> dict[str, Any]:
        defaults = {
            "default_model": "gemma4-31b",
            "stop_on_close": False,
            "theme": "deadkey",
            "auto_load_on_launch": True,
            "voice_auto_speak": False,
            "voice_name": "tara",
            "hands_free_voice": False,
            "hands_free_pause_ms": 1500,
        }
        if not self.prefs_path.exists():
            return defaults
        try:
            saved = json.loads(self.prefs_path.read_text(encoding="utf-8"))
            defaults.update(saved)
        except Exception:
            pass
        return defaults

    def update_prefs(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            cur = self.read_prefs()
            for k in (
                "default_model", "stop_on_close", "theme", "auto_load_on_launch",
                "voice_auto_speak", "voice_name", "hands_free_voice", "hands_free_pause_ms",
            ):
                if k in patch:
                    cur[k] = patch[k]
            self.prefs_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
            return cur

    # ---- Cartridges ----
    def list_cartridges(self) -> list[dict[str, Any]]:
        out = []
        if self.cartridges_dir.exists():
            for mf in sorted(self.cartridges_dir.glob("*/manifest.json")):
                try:
                    c = CartridgeConfig.from_manifest(mf)
                    out.append({
                        "agent_id": c.agent_id,
                        "display_name": c.display_name,
                        "manifest_path": str(mf),
                        "privacy_wall": bool(c.extra.get("cloud_privacy_wall", True)),
                        "active": (c.agent_id == self.cartridge.agent_id),
                    })
                except Exception:
                    pass
        if not any(x["agent_id"] == self.cartridge.agent_id for x in out):
            out.insert(0, {
                "agent_id": self.cartridge.agent_id,
                "display_name": self.cartridge.display_name,
                "manifest_path": str(self.cartridge.home_dir / "manifest.json"),
                "privacy_wall": bool(self.cartridge.extra.get("cloud_privacy_wall", True)),
                "active": True,
            })
        return out

    def switch_cartridge(self, agent_id: str) -> dict[str, Any]:
        for c in self.list_cartridges():
            if c["agent_id"] == agent_id:
                cfg = CartridgeConfig.from_manifest(c["manifest_path"])
                self.load_cartridge(cfg)
                return {"ok": True, "agent_id": cfg.agent_id, "display_name": cfg.display_name}
        raise KeyError(f"Unknown cartridge: {agent_id}")

    # ---- Model Supervisor ----
    def switch_model(self, model_name: str, *, blocking: bool = False) -> dict[str, Any]:
        if self.supervisor is None:
            raise RuntimeError("ModelSupervisor not configured")
        if model_name not in self.supervisor.models:
            raise KeyError(f"Unknown model: {model_name}")
        m = self.supervisor.models[model_name]
        proto = m.get("tool_protocol", "openai_native")
        dt = bool(m.get("disable_thinking", True))
        self.client.set_endpoint(
            f"http://127.0.0.1:{self.supervisor.port}",
            tool_protocol=proto,
            disable_thinking=dt,
        )
        prev_model = self.supervisor.state.active_model
        self.update_prefs({"default_model": model_name})
        status = self.supervisor.switch_blocking(model_name) if blocking else self.supervisor.switch_async(model_name)
        if self.sessions is not None and prev_model and prev_model != model_name:
            self.sessions.append_message(
                session_id=self.active_session_id,
                role="system",
                speaker="system",
                content=f"[model switch] Switched live engine to {m.get('label') or model_name} (protocol={proto}).",
            )
        return status

    # ---- Pinned Workbench Notes ----
    def read_pins(self) -> list[dict[str, Any]]:
        if not self.pins_path.exists():
            return []
        try:
            return json.loads(self.pins_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def add_pin(self, title: str, text: str, *, source: str = "workbench") -> dict[str, Any]:
        with self._lock:
            pins = self.read_pins()
            pid = f"PIN-{int(time.time() * 1000) % 1000000:06d}"
            item = {
                "id": pid,
                "title": (title or "Pinned Note").strip()[:80],
                "text": text.strip()[:2000],
                "source": source,
                "ts": time.time(),
            }
            pins.insert(0, item)
            self.pins_path.write_text(json.dumps(pins[:40], indent=2, ensure_ascii=False), encoding="utf-8")
            return item

    def delete_pin(self, pin_id: str) -> bool:
        with self._lock:
            pins = self.read_pins()
            new_pins = [p for p in pins if p.get("id") != pin_id]
            if len(new_pins) == len(pins):
                return False
            self.pins_path.write_text(json.dumps(new_pins, indent=2, ensure_ascii=False), encoding="utf-8")
            return True

    def _pins_context_block(self) -> str:
        pins = self.read_pins()[:6]
        if not pins:
            return ""
        return "\n".join(f"- [{p['title']}]: {p['text'][:220]}" for p in pins)

    # ---- Drag-and-Drop / Clipboard Uploads ----
    def save_upload(self, filename: str, content_b64: str, mime: str = "") -> dict[str, Any]:
        raw = base64.b64decode(content_b64.split(",")[-1])
        safe = _SAFE_FILE_RX.sub("_", Path(filename or "pasted.txt").name) or "pasted.txt"
        ts = int(time.time() * 1000)
        dest = self.uploads_dir / f"{ts}_{safe}"
        dest.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()[:12]
        ext = dest.suffix.lower()
        is_image = ext in _IMAGE_EXTS or (mime or "").startswith("image/")
        line_count: Optional[int] = None
        preview = ""
        if not is_image:
            text = raw.decode("utf-8", errors="replace")
            line_count = len(text.splitlines())
            preview = text[:1600]
        rec = {
            "id": f"UP-{sha}",
            "filename": safe,
            "stored_name": dest.name,
            "path": str(dest),
            "bytes": len(raw),
            "lines": line_count,
            "sha": sha,
            "mime": mime or ("image/png" if is_image else "text/plain"),
            "is_image": is_image,
            "preview": preview,
            "url": f"/v1/uploads/{urllib.parse.quote(dest.name)}",
        }
        self._uploads_cache[rec["id"]] = rec
        self.prefetcher.refresh_file_index()
        return rec

    def _resolve_attachments(self, attachment_ids: list[str]) -> tuple[list[EvidenceReceipt], list[dict[str, Any]], str]:
        receipts: list[EvidenceReceipt] = []
        vision_parts: list[dict[str, Any]] = []
        text_notes: list[str] = []
        for aid in (attachment_ids or []):
            rec = self._uploads_cache.get(aid)
            if not rec:
                sha_part = aid.replace("UP-", "")
                for f in self.uploads_dir.iterdir():
                    if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()[:12] == sha_part:
                        ext = f.suffix.lower()
                        is_img = ext in _IMAGE_EXTS
                        raw = f.read_bytes()
                        rec = {
                            "id": aid, "filename": f.name, "path": str(f), "bytes": len(raw),
                            "lines": None if is_img else len(raw.decode("utf-8", errors="replace").splitlines()),
                            "sha": sha_part, "mime": "image/png" if is_img else "text/plain",
                            "is_image": is_img, "preview": "" if is_img else raw.decode("utf-8", errors="replace")[:1600],
                        }
                        break
            if not rec:
                continue
            p = Path(rec["path"])
            if rec["is_image"] and p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                vision_parts.append({"type": "image_url", "image_url": {"url": f"data:{rec['mime']};base64,{b64}"}})
                receipts.append(EvidenceReceipt(
                    tool="uploaded_image", args={"filename": rec["filename"]}, ok=True,
                    result_text=f"Attached image {rec['filename']} ({rec['bytes']} bytes, sha {rec['sha']}).",
                    duration_ms=1, resolved_path=str(p), root_label="uploads",
                    byte_count=rec["bytes"], sha256_short=rec["sha"],
                ))
                text_notes.append(f"[Attached image: {rec['filename']} | {rec['bytes']} bytes | sha {rec['sha']}]")
            elif p.exists():
                r = build_file_receipt("uploaded_file", {"path": str(p)}, p, self.roots)
                receipts.append(r)
                text_notes.append(
                    f"[Attached file: {rec['filename']} | {r.byte_count} bytes | {r.line_count} lines | sha {r.sha256_short}]\n"
                    f"```\n{rec['preview']}\n```"
                )
        return receipts, vision_parts, "\n\n".join(text_notes)

    # ---- Approvals ----
    def read_approvals(self) -> list[dict[str, Any]]:
        if not self.approvals_path.exists():
            return []
        rows = []
        for ln in self.approvals_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
        return rows

    def update_approval(self, prop_id: str, new_status: str) -> dict[str, Any] | None:
        rows = self.read_approvals()
        updated = None
        for r in rows:
            if r.get("id") == prop_id:
                r["status"] = new_status
                r["decided_ts"] = time.time()
                updated = r
        if updated is not None:
            self.approvals_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
            )
        return updated

    # ---- Partner / Hermes / Neon Collaboration Bridge ----
    def read_partner_notes(self) -> list[dict[str, Any]]:
        if not self.partner_path.exists():
            return []
        try:
            return json.loads(self.partner_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def add_partner_note(self, partner: str, title: str, body: str, *, priority: str = "normal") -> dict[str, Any]:
        with self._lock:
            items = self.read_partner_notes()
            nid = f"COLLAB-{int(time.time() * 1000) % 1000000:06d}"
            item = {
                "id": nid,
                "partner": partner,
                "title": (title or "Handoff Note").strip()[:100],
                "body": body.strip()[:3000],
                "priority": priority,
                "status": "queued",
                "ts": time.time(),
            }
            items.insert(0, item)
            self.partner_path.write_text(json.dumps(items[:50], indent=2, ensure_ascii=False), encoding="utf-8")
            return item

    def export_redacted_brief(self, partner: str = "neon") -> dict[str, Any]:
        sup = self.supervisor.get_status(refresh_vram=True) if self.supervisor else {}
        v_st = self.voice.get_status()
        stats = self.memory.stats() if self.memory else {}
        recent_msgs = self.sessions.get_messages(self.active_session_id)[-6:] if self.sessions else []
        receipt_lines = []
        for m in recent_msgs:
            for r in (m.get("receipts") or []):
                receipt_lines.append(
                    f"- {r.get('tool')} | path={Path(str(r.get('path') or '')).name} | "
                    f"bytes={r.get('bytes')} | lines={r.get('lines')} | sha={r.get('sha')}"
                )
        queued = [x for x in self.read_partner_notes() if x.get("partner") == partner][:5]
        q_lines = [f"- [{x['id']}] {x['title']}: {x['body'][:180]}" for x in queued]

        brief = (
            f"# Redacted Technical Handoff Brief -> {partner.upper()}\n"
            f"- Cartridge: `{self.cartridge.agent_id}` (Privacy Wall: ACTIVE — personal journal/memories excluded)\n"
            f"- Architecture: `{self.active_arch}`\n"
            f"- Active Model: `{sup.get('active_model') or 'offline'}` "
            f"(VRAM: {sup.get('vram_used_mib')} / {sup.get('vram_total_mib')} MiB)\n"
            f"- Voice Stack: `{v_st.get('status')}` ({v_st.get('engine')} / {v_st.get('voice')})\n"
            f"- Memory DB Stats (counts only): ledger={stats.get('ledger_total', 0)}, "
            f"semantic={stats.get('semantic_active', 0)}, outcomes={stats.get('outcomes', 0)}\n\n"
            f"## Recent Verified Disk Receipts\n"
            + ("\n".join(receipt_lines) if receipt_lines else "- (none in current session)")
            + "\n\n## Queued Partner Tasks\n"
            + ("\n".join(q_lines) if q_lines else "- (none queued)")
        )
        return {"partner": partner, "privacy_wall": True, "brief_markdown": brief}

    # ---- Unified Chat Execution (Sync + Real SSE Streaming + Outcome Logging + Slot-1 Cognition) ----
    def _execute_chat(
        self,
        user_text: str,
        *,
        session_id: Optional[str] = None,
        speaker: str = "user",
        door: str = "text",
        attachments: Optional[list[str]] = None,
        on_sse: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        sid = session_id or self.active_session_id
        att_receipts, vision_parts, att_text = self._resolve_attachments(attachments or [])
        prior_msgs = self.sessions.get_messages(sid) if self.sessions else []
        history = [{"role": m["role"], "content": m["content"]} for m in prior_msgs if m.get("role") in ("user", "assistant")]

        stored_user_text = user_text if not att_receipts else f"{user_text}\n[attached: {', '.join(Path(r.resolved_path or '').name for r in att_receipts)}]"
        if self.sessions is not None:
            self.sessions.append_message(session_id=sid, role="user", speaker=speaker, content=stored_user_text)

        if on_sse:
            on_sse({"event": "step", "phase": "prefetch", "detail": "Checking disk & valid-time memory..."})

        pf = self.prefetcher.prefetch(user_text, memory_k=3)
        if on_sse:
            on_sse({
                "event": "prefetch",
                "elapsed_ms": pf.elapsed_ms,
                "files": [Path(r.resolved_path or "").name for r in pf.receipts],
                "memory_hits": len(pf.memory_hits),
            })

        backend_info = self.client.discover(timeout=0.5)
        vgr_info: dict[str, Any] = {"action": "accept_local", "privacy_fence_enforced": True}
        telemetry_dict: dict[str, Any] = {}
        thinking_accum: list[str] = []

        def tool_fn(name: str, args: dict[str, Any]) -> EvidenceReceipt:
            assert self.tools_runtime is not None
            return self.tools_runtime.execute(name, args, user_text=user_text)

        ctx_parts: list[str] = []
        if att_text:
            ctx_parts.append(att_text)
        extra_receipts = pf.receipts + att_receipts
        pf_block = pf.format_context_block()
        if pf_block:
            ctx_parts.append(pf_block)
        if door == "voice":
            ctx_parts.append("[voice mode: reply in warm, natural spoken prose (2-4 concise sentences, no markdown tables or code fences unless asked).]")
        context_suffix = "\n\n".join(ctx_parts)

        # Determine lane on RAW user_text BEFORE streaming so EVIDENCE_GATED turns buffer cleanly
        # until Disk-Equals-Truth gate approves (100% text == voice sync; never speak unverified draft!).
        from .tool_router import TurnLane
        pre_lane = self.router.classify_lane(user_text)
        allow_live_delta_stream = (pre_lane == TurnLane.FAST_CONVERSATIONAL)
        stream_attempt = {"n": 0}

        if backend_info.reachable:
            def model_fn(sys_prompt, msgs, tools, _proto):
                stream_attempt["n"] += 1
                live_stream = (on_sse is not None and allow_live_delta_stream and stream_attempt["n"] == 1)
                if live_stream:
                    final_turn: Optional[ParsedModelTurn] = None
                    final_tel = None
                    for chunk in self.client.chat_stream(
                        system_prompt=sys_prompt, messages=msgs, tools=tools, slot_id=0,
                        max_tokens=900, temperature=0.55, request_logprobs=True,
                    ):
                        ctype = chunk.get("type")
                        if ctype == "thinking":
                            thinking_accum.append(chunk["content"])
                            on_sse({"event": "thinking", "content": chunk["content"]})
                        elif ctype == "delta":
                            on_sse({"event": "delta", "content": chunk["content"]})
                        elif ctype == "final":
                            final_turn = chunk["turn"]
                            final_tel = chunk["telemetry"]
                    assert final_turn is not None and final_tel is not None
                    telemetry_dict.update(final_tel.__dict__)
                    vgr_dec = self.vgr.decide(
                        first_token_logprob=final_tel.first_token_logprob,
                        top_logprob_margin=final_tel.top_logprob_margin,
                        has_verified_receipts=bool(extra_receipts),
                        contains_private_memory=bool(self.cartridge.extra.get("cloud_privacy_wall", True)),
                    )
                    vgr_info.update(vgr_dec.__dict__)
                    return final_turn
                else:
                    turn, tel = self.client.chat_once(
                        system_prompt=sys_prompt, messages=msgs, tools=tools, slot_id=0,
                        max_tokens=700, temperature=0.55, request_logprobs=True,
                    )
                    telemetry_dict.update(tel.__dict__)
                    vgr_dec = self.vgr.decide(
                        first_token_logprob=tel.first_token_logprob,
                        top_logprob_margin=tel.top_logprob_margin,
                        has_verified_receipts=bool(extra_receipts),
                        contains_private_memory=bool(self.cartridge.extra.get("cloud_privacy_wall", True)),
                    )
                    vgr_info.update(vgr_dec.__dict__)
                    return turn
        else:
            def model_fn(_sys, _msgs, _tools, _proto):
                if att_receipts:
                    r0 = att_receipts[0]
                    txt = (
                        f"[Local Verifier — no GPU model loaded] Verified attached file "
                        f"{Path(r0.resolved_path or '').name}: {r0.byte_count} bytes"
                        + (f", {r0.line_count} lines" if r0.line_count is not None else "")
                        + f" (sha {r0.sha256_short}). Pick a model from the top bar to chat live."
                    )
                elif pf.receipts:
                    r0 = pf.receipts[0]
                    txt = (
                        f"[Local Verifier — no GPU model loaded] Prefetched {Path(r0.resolved_path or '').name}: "
                        f"{r0.byte_count} bytes, {r0.line_count} lines (sha {r0.sha256_short})."
                    )
                elif pf.memory_hits:
                    top = pf.memory_hits[0]
                    txt = (
                        f"[Local Verifier — no GPU model loaded] Top valid-time recall ({top.id}, {top.status}): "
                        f"{top.text[:220]}"
                    )
                else:
                    txt = (
                        "[Local Verifier — no GPU model loaded] Deadkey kernel, pinned tool router, "
                        "and Disk-Equals-Truth gate are live. Choose any model in the top bar to load onto your RTX 5090."
                    )
                if on_sse is not None and allow_live_delta_stream:
                    on_sse({"event": "delta", "content": txt})
                return ParsedModelTurn(text=txt)

        tm = TurnMachine(
            router=self.router, memory=self.memory, gate=self.gate,
            embedder=self.embedder, tool_protocol=self.client.tool_protocol,
        )
        pinned = self.router.pinned_session_tools()
        res = tm.run_turn(
            user_text=user_text,
            door=door,
            model_fn=model_fn,
            tool_fn=tool_fn,
            speaker=speaker,
            pinned_tools=pinned,
            history=history,
            extra_receipts=extra_receipts,
            extra_user_content=vision_parts if (vision_parts and backend_info.reachable) else None,
            context_suffix=context_suffix,
            pinned_notes_text=self._pins_context_block(),
            on_event=on_sse,
        )
        # If turn was buffered (EVIDENCE_GATED or reprompted), emit final verified text cleanly as ONE delta!
        if on_sse is not None and (not allow_live_delta_stream or stream_attempt["n"] > 1):
            on_sse({"event": "replace_turn", "content": res.final_text})

        all_receipts = res.receipts
        rcpt_dicts = [
            {
                "tool": r.tool, "ok": r.ok, "path": r.resolved_path, "root": r.root_label,
                "bytes": r.byte_count, "lines": r.line_count, "sha": r.sha256_short,
            }
            for r in all_receipts
        ]

        final_text = res.final_text
        verified = res.verdict.passed
        if self.sessions is not None:
            self.sessions.append_message(
                session_id=sid, role="assistant", speaker=self.cartridge.agent_id,
                content=final_text, lane=res.lane.value, ttfa_ms=res.wall_ms,
                receipts=rcpt_dicts, verified=verified,
            )

        # Record live outcome into SQLite for continuous RSI & Sleep-Time DPO distillation!
        if self.memory is not None:
            self.memory.record_outcome(
                turn_id=f"T-{int(time.time() * 1000)}",
                asked=user_text,
                attempted=final_text,
                tools=[{"name": r.tool, "ok": r.ok} for r in all_receipts],
                worked=verified,
                guard_fired=(not verified or res.attempts > 1),
                guard_check=",".join(res.verdict.violations[:3]),
            )

        # Auto-trigger non-blocking Slot-1 Background Cognition every 3 live turns (0% Slot-0 KV eviction!)
        if backend_info.reachable:
            self._turns_since_cognition += 1
            if self._turns_since_cognition >= 3:
                self._turns_since_cognition = 0
                self.cognition.trigger_async(session_id=sid)

        payload = {
            "session_id": sid,
            "reply": final_text,
            "thinking": "".join(thinking_accum).strip(),
            "lane": res.lane.value,
            "verified": verified,
            "wall_ms": res.wall_ms,
            "prefetch_ms": pf.elapsed_ms,
            "routed_tools": [t.name for t in pinned],
            "schema_savings_pct": res.route.savings_pct,
            "receipts": rcpt_dicts,
            "salience_hits": [{"id": h.id, "status": h.status, "text": h.text[:140]} for h in pf.memory_hits],
            "vgr": vgr_info,
            "telemetry": telemetry_dict,
            "backend_reachable": backend_info.reachable,
        }
        if on_sse is not None:
            on_sse({"event": "final_meta", **payload})
        return payload

    def handle_chat(
        self,
        user_text: str,
        *,
        session_id: Optional[str] = None,
        speaker: str = "user",
        door: str = "text",
        attachments: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        return self._execute_chat(
            user_text, session_id=session_id, speaker=speaker, door=door, attachments=attachments, on_sse=None,
        )


def make_handler(rt: CoreRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj: Any, status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _check_origin_allowed(self) -> bool:
            origin = (self.headers.get("Origin") or "").strip()
            if not origin:
                return True
            parsed = urllib.parse.urlparse(origin)
            host = (parsed.hostname or "").lower()
            return host in ("127.0.0.1", "localhost", "::1") or host.endswith(".ts.net")

        def _read_json(self) -> dict[str, Any]:
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                return {}
            if n <= 0 or n > 25 * 1024 * 1024:
                return {}
            raw = self.rfile.read(n).decode("utf-8", errors="replace").strip()
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                rt.cancel_pending_close_shutdown()
                if rt.ui_path.exists():
                    body = rt.ui_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._send_json({"error": "UI not found"}, 404)
                return

            if path.startswith("/v1/uploads/"):
                fname = urllib.parse.unquote(path[len("/v1/uploads/"):])
                target = (rt.uploads_dir / Path(fname).name).resolve()
                if target.parent == rt.uploads_dir.resolve() and target.exists():
                    ext = target.suffix.lower()
                    ctype = {
                        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp", ".gif": "image/gif",
                    }.get(ext, "text/plain; charset=utf-8")
                    data = target.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send_json({"error": "Upload not found"}, 404)
                return

            if path == "/manifest.webmanifest":
                manifest = {
                    "name": "Deadkey — Deadkey",
                    "short_name": "deadkey",
                    "start_url": "/",
                    "display": "standalone",
                    "background_color": "#121412",
                    "theme_color": "#d9863b",
                    "description": "Local-first voice, vision & Disk-Equals-Truth companion on Colossus",
                }
                raw = json.dumps(manifest).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/manifest+json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/v1/health":
                rt.cancel_pending_close_shutdown()
                b = rt.client.discover(timeout=0.35)
                sup = rt.supervisor.get_status(refresh_vram=True) if rt.supervisor else None
                self._send_json({
                    "ok": True,
                    "agent_id": rt.cartridge.agent_id,
                    "display_name": rt.cartridge.display_name,
                    "active_arch": rt.active_arch,
                    "active_session_id": rt.active_session_id,
                    "backend": b.__dict__,
                    "supervisor": sup,
                    "voice": rt.voice.get_status(),
                    "cognition": {"running": rt.cognition.is_running(), "recent": rt.cognition.history[:3]},
                    "prefs": rt.read_prefs(),
                    "memory_stats": rt.memory.stats() if rt.memory else {},
                    "pinned_core_tokens_est": (len(rt.memory.pinned_core_prompt()) // 4) if rt.memory else 0,
                    "tool_catalog_size": len(rt.router.tools),
                    "prefetched_files_indexed": len(rt.prefetcher._index),
                })
                return

            if path == "/v1/models":
                if rt.supervisor is None:
                    self._send_json({"models": [], "status": None})
                    return
                self._send_json({
                    "models": rt.supervisor.list_models(),
                    "status": rt.supervisor.get_status(refresh_vram=True),
                    "prefs": rt.read_prefs(),
                })
                return

            if path == "/v1/models/status":
                st = rt.supervisor.get_status(refresh_vram=True) if rt.supervisor else {"status": "unconfigured"}
                self._send_json(st)
                return

            if path == "/v1/lora":
                adapters = rt.supervisor.get_lora_adapters() if rt.supervisor else []
                self._send_json({"adapters": adapters})
                return

            if path == "/v1/voice/status":
                self._send_json(rt.voice.get_status())
                return

            if path == "/v1/cognition/status":
                self._send_json({"running": rt.cognition.is_running(), "history": rt.cognition.history})
                return

            if path == "/v1/hillclimb":
                ledger = read_hillclimb_ledger(rt.cartridge.home_dir / "data" / "hill_climb_ledger.jsonl")
                self._send_json({"trials": ledger[-15:]})
                return

            if path == "/v1/rd/radar":
                items = read_radar(rt.cartridge.home_dir / "data" / "rd_radar.jsonl", limit=20)
                self._send_json({"items": items})
                return

            if path == "/v1/cartridges":
                self._send_json({"active": rt.cartridge.agent_id, "cartridges": rt.list_cartridges()})
                return

            if path == "/v1/prefs":
                self._send_json(rt.read_prefs())
                return

            if path == "/v1/pins":
                self._send_json({"pins": rt.read_pins()})
                return

            if path == "/v1/partner":
                self._send_json({"notes": rt.read_partner_notes()})
                return

            if path == "/v1/sessions":
                self._send_json({
                    "active": rt.active_session_id,
                    "sessions": rt.sessions.list_sessions() if rt.sessions else [],
                })
                return

            if path.startswith("/v1/sessions/") and path.endswith("/messages"):
                sid = path.split("/")[3]
                self._send_json({
                    "session_id": sid,
                    "messages": rt.sessions.get_messages(sid) if rt.sessions else [],
                })
                return

            if path == "/v1/memory/recall":
                q = (qs.get("q") or [""])[0]
                k = int((qs.get("k") or ["6"])[0])
                qvec = rt.embedder.embed_query(q) if (rt.embedder and rt.embedder.ready) else None
                hits = rt.memory.recall(q, k=k, query_vec=qvec) if rt.memory else []
                self._send_json({"query": q, "hybrid_vectors": bool(qvec), "hits": [h.__dict__ for h in hits]})
                return

            if path == "/v1/arch":
                self._send_json({"active": rt.active_arch, "profiles": rt.archs})
                return

            if path == "/.well-known/agent.json":
                from .a2a import build_agent_card
                host_hdr = self.headers.get("Host") or "127.0.0.1:8765"
                self._send_json(build_agent_card(rt, base_url=f"http://{host_hdr}"))
                return

            if path == "/v1/approvals":
                self._send_json({"approvals": rt.read_approvals()})
                return

            self._send_json({"error": f"Not found: {path}"}, 404)

        def do_DELETE(self):
            if not self._check_origin_allowed():
                self._send_json({"error": "Forbidden cross-origin request"}, 403)
                return
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path.startswith("/v1/pins/"):
                pid = path.split("/")[3]
                ok = rt.delete_pin(pid)
                self._send_json({"ok": ok, "id": pid}, 200 if ok else 404)
                return
            self._send_json({"error": f"Not found: {path}"}, 404)

        def do_POST(self):
            if not self._check_origin_allowed():
                self._send_json({"error": "Forbidden cross-origin request"}, 403)
                return
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            body = self._read_json()

            if path == "/v1/chat":
                txt = (body.get("text") or "").strip()
                if not txt:
                    self._send_json({"error": "text is required"}, 400)
                    return
                try:
                    res = rt.handle_chat(
                        txt,
                        session_id=body.get("session_id"),
                        speaker=body.get("speaker") or "user",
                        door=body.get("door") or "text",
                        attachments=body.get("attachments") or [],
                    )
                    self._send_json(res)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 502)
                return

            if path == "/v1/chat/stream":
                txt = (body.get("text") or "").strip()
                if not txt:
                    self._send_json({"error": "text is required"}, 400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def send_sse(ev: dict[str, Any]) -> None:
                    ev_name = ev.get("event") or "message"
                    line = f"event: {ev_name}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    try:
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    except OSError:
                        pass

                try:
                    rt._execute_chat(
                        txt,
                        session_id=body.get("session_id"),
                        speaker=body.get("speaker") or "user",
                        door=body.get("door") or "text",
                        attachments=body.get("attachments") or [],
                        on_sse=send_sse,
                    )
                except Exception as exc:
                    send_sse({"event": "error", "message": str(exc)})
                return

            # ---- Live Voice Mode Endpoints ----
            if path == "/v1/voice/start":
                v = str(body.get("voice") or rt.read_prefs().get("voice_name") or "tara")
                blocking = bool(body.get("blocking", False))
                st = rt.voice.start_blocking(voice=v) if blocking else rt.voice.start_async(voice=v)
                rt.update_prefs({"voice_name": v, "voice_auto_speak": True})
                self._send_json({"ok": True, "voice": st})
                return

            if path == "/v1/voice/stop":
                st = rt.voice.stop()
                rt.update_prefs({"voice_auto_speak": False, "hands_free_voice": False})
                self._send_json({"ok": True, "voice": st})
                return

            if path == "/v1/voice/transcribe":
                b64 = str(body.get("audio_base64") or "")
                if not b64:
                    self._send_json({"error": "audio_base64 is required"}, 400)
                    return
                try:
                    res = rt.voice.transcribe_audio(b64, suffix=str(body.get("suffix") or ".wav"))
                    self._send_json(res)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 500)
                return

            if path == "/v1/voice/speak":
                txt = str(body.get("text") or "").strip()
                if not txt:
                    self._send_json({"error": "text is required"}, 400)
                    return
                voice = str(body.get("voice") or rt.read_prefs().get("voice_name") or "tara")
                fmt = str(body.get("format") or "pcm").lower()
                if rt.voice.get_status().get("status") != "ready":
                    self._send_json({"error": "Voice engine is not running — enable Voice Mode first."}, 503)
                    return
                headers_sent = False
                try:
                    if fmt == "wav":
                        wav, ms = rt.voice.synthesize_wav(txt, voice=voice)
                        self.send_response(200)
                        self.send_header("Content-Type", "audio/wav")
                        self.send_header("Content-Length", str(len(wav)))
                        self.send_header("X-Orpheus-Ms", str(ms))
                        self.end_headers()
                        headers_sent = True
                        self.wfile.write(wav)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("X-Sample-Rate", "24000")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        headers_sent = True
                        for chunk in rt.voice.stream_speech_pcm(txt, voice=voice):
                            self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass  # Normal on browser barge-in!
                except Exception as exc:
                    if not headers_sent:
                        try:
                            self._send_json({"error": str(exc)}, 500)
                        except OSError:
                            pass
                return

            # ---- Slot-1 Background Cognition & Live Hill-Climb ----
            if path == "/v1/cognition/run":
                rep = rt.cognition.run_once(session_id=body.get("session_id"))
                self._send_json({"ok": rep.ok, "report": rep.__dict__})
                return

            if path == "/v1/hillclimb/run":
                axis = str(body.get("axis") or "flags")
                label = str(body.get("label") or ("prefetch_ab" if axis == "memory" else "temp_0.35_vs_0.55"))
                out = run_live_hillclimb_trial(rt, axis=axis, candidate_label=label)
                self._send_json({"ok": True, "outcome": out})
                return

            if path == "/v1/rd/scout":
                rep = run_rd_scout(rt.cartridge.home_dir / "data" / "rd_radar.jsonl")
                self._send_json(rep)
                return

            if path == "/v1/screen/frame":
                b64 = str(body.get("image_base64") or "")
                if not b64:
                    self._send_json({"error": "image_base64 is required"}, 400)
                    return
                try:
                    raw = base64.b64decode(b64.split(",")[-1])
                    dest = rt.uploads_dir / "latest_screen.png"
                    dest.write_bytes(raw)
                    sha = hashlib.sha256(raw).hexdigest()[:12]
                    self._send_json({"ok": True, "bytes": len(raw), "sha": sha})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/v1/upload":
                fname = str(body.get("filename") or "pasted.txt")
                b64 = str(body.get("content_base64") or "")
                mime = str(body.get("mime") or "")
                if not b64:
                    self._send_json({"error": "content_base64 is required"}, 400)
                    return
                try:
                    rec = rt.save_upload(fname, b64, mime=mime)
                    self._send_json({"ok": True, "attachment": rec})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/v1/models/switch":
                mname = str(body.get("model") or "").strip()
                if not mname:
                    self._send_json({"error": "model is required"}, 400)
                    return
                try:
                    st = rt.switch_model(mname, blocking=bool(body.get("blocking", False)))
                    self._send_json({"ok": True, "status": st})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/v1/models/stop":
                if rt.supervisor is None:
                    self._send_json({"ok": True, "status": {"status": "stopped"}})
                    return
                st = rt.supervisor.stop()
                self._send_json({"ok": True, "status": st})
                return

            if path == "/v1/lora/scale":
                if rt.supervisor is None:
                    self._send_json({"error": "Supervisor not configured"}, 400)
                    return
                try:
                    res = rt.supervisor.set_lora_scale(int(body.get("id", 0)), float(body.get("scale", 0.0)))
                    self._send_json(res)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/v1/cartridges/switch":
                aid = str(body.get("agent_id") or "").strip()
                try:
                    res = rt.switch_cartridge(aid)
                    self._send_json(res)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                return

            if path == "/v1/prefs":
                self._send_json({"ok": True, "prefs": rt.update_prefs(body)})
                return

            if path == "/v1/pins":
                txt = str(body.get("text") or "").strip()
                if not txt:
                    self._send_json({"error": "text is required"}, 400)
                    return
                item = rt.add_pin(str(body.get("title") or "Pinned Note"), txt, source=str(body.get("source") or "workbench"))
                self._send_json({"ok": True, "pin": item})
                return

            if path == "/v1/partner/note":
                item = rt.add_partner_note(
                    partner=str(body.get("partner") or "hermes"),
                    title=str(body.get("title") or "Collaboration Note"),
                    body=str(body.get("body") or ""),
                    priority=str(body.get("priority") or "normal"),
                )
                self._send_json({"ok": True, "note": item})
                return

            if path == "/v1/partner/ask-hermes":
                prompt = str(body.get("prompt") or "").strip()
                if not prompt:
                    self._send_json({"error": "prompt is required"}, 400)
                    return
                assert rt.tools_runtime is not None
                rcpt = rt.tools_runtime.execute("ask_hermes", {"prompt": prompt})
                self._send_json({"ok": rcpt.ok, "reply": rcpt.result_text, "duration_ms": rcpt.duration_ms})
                return

            if path == "/v1/partner/export-brief":
                partner = str(body.get("partner") or "neon")
                self._send_json({"ok": True, **rt.export_redacted_brief(partner)})
                return

            if path == "/a2a":
                from .a2a import handle_a2a_rpc
                host_hdr = self.headers.get("Host") or "127.0.0.1:8765"
                res = handle_a2a_rpc(rt, body, base_url=f"http://{host_hdr}")
                self._send_json(res, 200)
                return

            if path == "/v1/proof/compare":
                prompt = str(body.get("prompt") or "").strip()
                if not prompt:
                    self._send_json({"error": "prompt is required"}, 400)
                    return
                from .semantic_critic import SemanticEpistemicCritic, evaluate_hybrid
                from .tool_router import EVIDENCE_GATE_RX, TurnLane
                import urllib.request as _ureq

                llm_base = rt.client.base_url.rstrip("/")
                lane = TurnLane.EVIDENCE_GATED if EVIDENCE_GATE_RX.search(prompt) else TurnLane.FAST_CONVERSATIONAL
                active_mems: list[str] = []
                if bool(body.get("use_memory", True)) and rt.memory is not None:
                    qvec = rt.embedder.embed_query(prompt) if (rt.embedder and rt.embedder.ready) else None
                    hits = rt.memory.recall(prompt, k=3, query_vec=qvec)
                    active_mems = [h.text for h in hits]

                sys_blank = (
                    "You are a concise, accurate engineering assistant. "
                    "Answer directly using only verified facts and tool receipts provided in context."
                )
                if active_mems:
                    sys_mem = sys_blank + "\n\n[ACTIVE BI-TEMPORAL MEMORY]\n" + "\n".join(f"- {m}" for m in active_mems)
                else:
                    sys_mem = sys_blank

                def _call(msgs: list[dict[str, str]]) -> tuple[str, float]:
                    t0 = time.perf_counter()
                    req = _ureq.Request(
                        f"{llm_base}/v1/chat/completions",
                        data=json.dumps({
                            "model": "local-model",
                            "messages": msgs,
                            "temperature": 0.0,
                            "seed": 42,
                            "max_tokens": 512,
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with _ureq.urlopen(req, timeout=90) as r:
                        d = json.loads(r.read().decode("utf-8", errors="replace"))
                    txt = (d["choices"][0]["message"].get("content") or "").strip()
                    txt = re.sub(r"(?is)<think>.*?</think>", "", txt).strip()
                    return txt, round((time.perf_counter() - t0) * 1000.0, 1)

                try:
                    raw_reply, raw_ms = _call([{"role": "system", "content": sys_blank}, {"role": "user", "content": prompt}])
                    # Tier 1 Frozen deterministic check (0.01ms)
                    t1_t0 = time.perf_counter()
                    v_t1 = rt.gate.evaluate(
                        user_text=prompt, draft_reply=raw_reply, lane=lane, receipts=[], attempt=2, max_attempts=2
                    )
                    t1_ms = round((time.perf_counter() - t1_t0) * 1000.0, 3)

                    # Tier 1 + Tier 2 Hybrid Semantic Critic
                    critic = SemanticEpistemicCritic(base_url=llm_base, max_tokens=512)
                    v_t2, cv_t2 = evaluate_hybrid(
                        rt.gate, critic,
                        user_text=prompt, draft_reply=raw_reply, lane=lane,
                        receipts=[], active_memories=active_mems, attempt=2, max_attempts=2,
                    )
                    self._send_json({
                        "ok": True,
                        "prompt": prompt,
                        "lane": lane.value,
                        "active_memories": active_mems,
                        "raw": {"reply": raw_reply, "ms": raw_ms},
                        "tier1": {
                            "reply": v_t1.cleaned_text,
                            "passed": v_t1.passed,
                            "action": v_t1.action,
                            "violations": v_t1.violations,
                            "ms": t1_ms,
                        },
                        "tier1_plus_tier2": {
                            "reply": v_t2.cleaned_text,
                            "passed": v_t2.passed,
                            "action": v_t2.action,
                            "violations": v_t2.violations,
                            "critic_invoked": cv_t2.invoked,
                            "critic_type": cv_t2.violation_type,
                            "critic_reason": cv_t2.reason,
                            "critic_ms": cv_t2.elapsed_ms,
                        },
                    })
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 502)
                return

            if path == "/v1/shutdown":
                reason = str(body.get("reason") or "ui_stop_stack")
                if reason == "window_closed" and not bool(body.get("dry_run", False)):
                    rt.schedule_close_shutdown(delay_s=4.0)
                    self._send_json({"ok": True, "stopping": True, "grace_seconds": 4.0, "reason": reason})
                    return
                rt._shutdown_requested = True
                rt._stop_gpu_processes()
                self._send_json({"ok": True, "stopping": True, "reason": reason, "vram_released": True})
                if rt._shutdown_cb is not None and not bool(body.get("dry_run", False)):
                    threading.Timer(0.25, rt._shutdown_cb).start()
                return

            if path == "/v1/sessions":
                action = (body.get("action") or "open").lower()
                if action == "open" and rt.sessions is not None:
                    sid = rt.sessions.open_session(body.get("title") or "New Session")
                    rt.active_session_id = sid
                    self._send_json({"ok": True, "session_id": sid})
                elif action == "resume":
                    rt.active_session_id = str(body.get("session_id") or rt.active_session_id)
                    self._send_json({"ok": True, "active": rt.active_session_id})
                elif action == "close" and rt.sessions is not None:
                    sid = str(body.get("session_id") or rt.active_session_id)
                    ok = rt.sessions.close_session(sid, str(body.get("summary") or ""))
                    self._send_json({"ok": ok, "session_id": sid})
                else:
                    self._send_json({"error": f"Unknown action: {action}"}, 400)
                return

            if path == "/v1/arch":
                key = str(body.get("arch") or "")
                if key not in rt.archs:
                    self._send_json({"error": f"Unknown arch: {key}"}, 400)
                    return
                rt.active_arch = key
                self._send_json({"ok": True, "active": rt.active_arch, "profile": rt.archs[key]})
                return

            if path == "/v1/sleep/run" and rt.memory is not None:
                sleep_dir = rt.cartridge.home_dir / "data" / "sleep"
                rep = run_sleep_consolidation(
                    rt.memory,
                    sleep_dir,
                    approvals_path=rt.approvals_path,
                )
                trainer_p = generate_5090_dpo_trainer(sleep_dir, Path(rep.dpo_output_path), rep.dpo_sha256)
                out = rep.__dict__.copy()
                out["trainer_script"] = str(trainer_p)
                self._send_json({"ok": True, "report": out})
                return

            if path.startswith("/v1/approvals/"):
                pid = path.split("/")[3]
                status = str(body.get("status") or "approved")
                upd = rt.update_approval(pid, status)
                if not upd:
                    self._send_json({"error": f"Approval {pid} not found"}, 404)
                else:
                    self._send_json({"ok": True, "approval": upd})
                return

            self._send_json({"error": f"Not found: {path}"}, 404)

    return Handler


def create_server(rt: CoreRuntime, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(rt))
    srv.daemon_threads = True
    rt._shutdown_cb = srv.shutdown
    return srv


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Deadkey portable server")
    ap.add_argument("--cartridge", default=str(root / "cartridges/blank-lab/manifest.json"))
    ap.add_argument("--tools", default=str(root / "cartridges/blank-lab/tools_catalog.json"))
    ap.add_argument("--backend", default="http://127.0.0.1:11455")
    ap.add_argument("--protocol", choices=("openai_native", "xml_json_bypass"), default="openai_native")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--auto-model", default="", help="Optional model profile to load immediately on startup")
    ap.add_argument("--auto-voice", action="store_true", help="Also boot native Orpheus-Tara voice engine on startup")
    args = ap.parse_args(argv)

    cfg = CartridgeConfig.from_manifest(args.cartridge)
    rt = CoreRuntime(
        cartridge=cfg,
        tools_catalog_path=Path(args.tools),
        backend_url=args.backend,
        tool_protocol=args.protocol,
    )
    if args.auto_model and rt.supervisor is not None:
        m = args.auto_model
        if m == "saved":
            m = rt.read_prefs().get("default_model") or "gemma4-31b"
        if m in rt.supervisor.models:
            print(f"[deadkey] Auto-loading {m} on RTX 5090 (port {rt.supervisor.port})...")
            rt.switch_model(m, blocking=False)

    if args.auto_voice or rt.read_prefs().get("voice_auto_speak"):
        print("[deadkey] Auto-starting Native Streaming Orpheus-Tara Voice Engine...")
        rt.voice.start_async(voice=rt.read_prefs().get("voice_name") or "tara")

    # Boot 0-VRAM local CPU semantic embedder in background + backfill any new memory vectors!
    def _boot_embedder():
        if rt.embedder.start(timeout_s=15.0) and rt.memory is not None:
            n = rt.memory.index_missing_vectors(rt.embedder)
            print(f"[deadkey] Hybrid 768-d CPU Semantic Memory ready (0 MiB GPU VRAM, backfilled {n} vectors).")
    threading.Thread(target=_boot_embedder, daemon=True).start()

    srv = create_server(rt, host=args.host, port=args.port)
    print(f"[deadkey] Serving {cfg.display_name} ({cfg.agent_id}) at http://{args.host}:{args.port} (backend={args.backend})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        rt.close()
        print("[deadkey] Stack stopped cleanly. GPU VRAM released (LLM + Voice).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
