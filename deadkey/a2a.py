"""Zero-dependency A2A (Agent-to-Agent Protocol) Peer Adapter for Deadkey / Deadkey.

Exposes:
  - GET  /.well-known/agent.json  — Standard A2A Agent Card for peer discovery (Hermes, OpenClaw, LangGraph, Tailscale peers)
  - POST /a2a                     — JSON-RPC 2.0 task execution (`agent/capabilities`, `tasks/send`)

Privacy Wall Guarantee:
  - `epistemic_audit` (auditing caller-provided drafts + receipts via Tier 1 & optional Tier 2 Semantic Critic)
    is stateless and safe on ANY cartridge.
  - `memory_recall` / `memory_record` over A2A are STRICTLY BLOCKED whenever the active cartridge
    enforces `cloud_privacy_wall=True` (private personal memories NEVER cross peer boundaries!).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .epistemic_gate import EvidenceReceipt
from .semantic_critic import SemanticEpistemicCritic, evaluate_hybrid
from .tool_router import TurnLane


def build_agent_card(rt: Any, base_url: str = "http://127.0.0.1:8765") -> dict[str, Any]:
    is_private = bool(rt.cartridge.extra.get("cloud_privacy_wall", True))
    sup = rt.supervisor.get_status(refresh_vram=False) if getattr(rt, "supervisor", None) else {}
    skills = [
        {
            "id": "epistemic_audit",
            "name": "Two-Tier Epistemic Audit",
            "description": (
                "Audits a candidate model draft against verified EvidenceReceipts and optional active "
                "memory policies using the 0.01 ms deterministic Tier 1 Disk-Equals-Truth gate and "
                "optional local Tier 2 Semantic Epistemic Critic."
            ),
            "input_schema": {
                "user_text": "string",
                "draft_reply": "string",
                "receipts": "list[dict] (optional)",
                "active_memories": "list[str] (optional)",
                "semantic": "bool (default false)",
            },
        },
        {
            "id": "health_telemetry",
            "name": "Local Hardware & Runtime Telemetry",
            "description": "Reports active model, engine, VRAM availability, and zero-VRAM memory DB stats.",
        },
    ]
    if not is_private:
        skills.append({
            "id": "memory_recall",
            "name": "Zero-VRAM Bi-Temporal Memory Recall",
            "description": "Valid-time SQLite FTS5 + hybrid CPU vector recall (zero stale superseded leaks).",
            "input_schema": {"query": "string", "k": "int (default 5)"},
        })
    return {
        "protocol": "a2a/1.0",
        "name": f"Deadkey Peer ({rt.cartridge.display_name})",
        "description": "Local-first sovereign AI runtime with 0-VRAM bi-temporal memory and 2-tier epistemic verification.",
        "url": base_url.rstrip("/") + "/a2a",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "privacy": {
            "cartridge_id": rt.cartridge.agent_id,
            "privacy_wall": is_private,
            "memory_exposed_over_a2a": (not is_private),
        },
        "runtime": {
            "active_model": sup.get("active_model") or "offline",
            "backend_url": rt.client.base_url,
        },
        "skills": skills,
    }


def _rpc_ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_err(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def handle_a2a_rpc(rt: Any, payload: dict[str, Any], base_url: str = "http://127.0.0.1:8765") -> dict[str, Any]:
    req_id = payload.get("id")
    method = str(payload.get("method") or "").strip()
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return _rpc_err(req_id, -32602, "params must be a JSON object")

    if method == "agent/capabilities":
        return _rpc_ok(req_id, build_agent_card(rt, base_url=base_url))

    if method != "tasks/send":
        return _rpc_err(req_id, -32601, f"Unsupported A2A method: {method}")

    task_id = str(params.get("id") or f"a2a-{uuid.uuid4().hex[:10]}")
    skill = str(params.get("skill") or "epistemic_audit").strip()
    inp = params.get("input") or {}
    if not isinstance(inp, dict):
        return _rpc_err(req_id, -32602, "params.input must be a JSON object")

    t0 = time.perf_counter()
    is_private = bool(rt.cartridge.extra.get("cloud_privacy_wall", True))

    if skill == "health_telemetry":
        sup = rt.supervisor.get_status(refresh_vram=False) if getattr(rt, "supervisor", None) else {}
        stats = rt.memory.stats() if getattr(rt, "memory", None) else {}
        return _rpc_ok(req_id, {
            "id": task_id,
            "status": {"state": "completed"},
            "skill": skill,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "output": {
                "cartridge_id": rt.cartridge.agent_id,
                "privacy_wall": is_private,
                "active_model": sup.get("active_model") or "offline",
                "memory_stats": stats,
            },
        })

    if skill == "memory_recall":
        if is_private:
            return _rpc_err(
                req_id,
                -32003,
                "Privacy Wall active: private cartridge memory cannot be queried over A2A.",
                {"cartridge_id": rt.cartridge.agent_id, "privacy_wall": True},
            )
        q = str(inp.get("query") or "").strip()
        k = max(1, min(20, int(inp.get("k") or 5)))
        qvec = rt.embedder.embed_query(q) if (getattr(rt, "embedder", None) and rt.embedder.ready) else None
        hits = rt.memory.recall(q, k=k, query_vec=qvec) if getattr(rt, "memory", None) else []
        return _rpc_ok(req_id, {
            "id": task_id,
            "status": {"state": "completed"},
            "skill": skill,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "output": {"query": q, "hits": [h.__dict__ for h in hits]},
        })

    if skill == "epistemic_audit":
        user_text = str(inp.get("user_text") or "")
        draft_reply = str(inp.get("draft_reply") or "")
        if not draft_reply:
            return _rpc_err(req_id, -32602, "input.draft_reply is required for epistemic_audit")
        raw_rcpts = inp.get("receipts") or []
        receipts: list[EvidenceReceipt] = []
        for r in raw_rcpts:
            if isinstance(r, dict):
                bc = r.get("byte_count") if r.get("byte_count") is not None else r.get("byte_size")
                lc = r.get("line_count")
                sha = r.get("sha256_short") or r.get("sha256_prefix")
                vnums = {str(x) for x in (bc, lc) if x is not None}
                receipts.append(EvidenceReceipt(
                    tool=str(r.get("tool") or r.get("tool_name") or "peer_tool"),
                    args=dict(r.get("args") or {}),
                    ok=bool(r.get("ok", True)),
                    result_text=str(r.get("result_text") or ""),
                    duration_ms=int(r.get("duration_ms") or 1),
                    resolved_path=r.get("resolved_path"),
                    byte_count=bc,
                    line_count=lc,
                    sha256_short=sha,
                    verified_numbers=vnums,
                ))
        active_mems = [str(x) for x in (inp.get("active_memories") or []) if str(x).strip()]
        use_semantic = bool(inp.get("semantic", False))
        critic = SemanticEpistemicCritic(base_url=rt.client.base_url) if use_semantic else None
        lane = TurnLane.EVIDENCE_GATED if str(inp.get("lane") or "evidence_gated") != "fast" else TurnLane.FAST_CONVERSATIONAL
        verdict, c_verdict = evaluate_hybrid(
            rt.gate,
            critic,
            user_text=user_text,
            draft_reply=draft_reply,
            lane=lane,
            receipts=receipts,
            active_memories=active_mems,
            attempt=int(inp.get("attempt") or 1),
            max_attempts=int(inp.get("max_attempts") or 2),
        )
        return _rpc_ok(req_id, {
            "id": task_id,
            "status": {"state": "completed"},
            "skill": skill,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "output": {
                "action": verdict.action,
                "violations": verdict.violations,
                "cleaned_text": verdict.cleaned_text,
                "reprompt_instruction": verdict.reprompt_instruction,
                "tier2_invoked": c_verdict.invoked,
                "tier2_elapsed_ms": c_verdict.elapsed_ms,
                "tier2_type": c_verdict.violation_type,
                "tier2_reason": c_verdict.reason,
            },
        })

    return _rpc_err(req_id, -32602, f"Unknown A2A skill: {skill}")
