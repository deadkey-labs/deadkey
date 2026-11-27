"""Standalone, Zero-Dependency MCP Server for Deadkey (Bi-Temporal Memory + Disk-Equals-Truth Gate).

Speaks standard Model Context Protocol (JSON-RPC 2.0 over stdio) so ANY agent harness
— Hermes Agent, Google Antigravity, Claude Code, OpenClaw, or custom local/frontier agents —
can plug directly into:
  1. `memory_recall`       — Sub-millisecond bi-temporal valid-time FTS5/RRF recall (0 MiB VRAM)
  2. `memory_record`       — Append verified observations or semantic facts with valid-time stamps
  3. `memory_supersede`    — Retire/disprove stale memories (SUPERSEDENCE / DISPROVAL) so old lies never return
  4. `verify_file_receipt` — Deterministic Disk-Equals-Truth file receipt (exact bytes, lines, SHA-256)
  5. `epistemic_check`     — 0.01 ms Disk-Equals-Truth commit gate auditing any draft reply against receipts

Usage:
  python -m deadkey.mcp_server --db ./cartridges/default/data/memory.db --root .
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .epistemic_gate import EpistemicCommitGate, EvidenceReceipt, build_file_receipt
from .memory_store import MemoryStore
from .semantic_critic import SemanticEpistemicCritic, evaluate_hybrid
from .tool_router import TurnLane

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "deadkey-epistemic-memory", "version": "0.1.0"}

TOOL_SPECS = [
    {
        "name": "memory_recall",
        "description": "Search bi-temporal valid-time memory (FTS5 + RRF, 0 MiB GPU VRAM). Superseded/disproved records are excluded by default.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "k": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                "include_inactive": {"type": "boolean", "description": "Include retired/superseded records", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_record",
        "description": "Record a verified observation or semantic fact into bi-temporal SQLite memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Fact or observation text"},
                "kind": {"type": "string", "enum": ["observation", "semantic"], "default": "observation"},
                "category": {"type": "string", "default": "general"},
                "source": {"type": "string", "default": "mcp_client"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "memory_supersede",
        "description": "Mark an existing observation (J-*) or semantic fact (SF-*) ID as SUPERSEDED or DISPROVED and record the corrected active truth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "description": "Existing observation (J-*) or semantic fact (SF-*) ID to supersede/disprove"},
                "transition": {"type": "string", "enum": ["SUPERSEDENCE", "DISPROVAL"], "default": "SUPERSEDENCE"},
                "correction": {"type": "string", "description": "Verified replacement truth"},
                "reason": {"type": "string", "description": "Optional reason why the old fact was retired", "default": ""},
                "source": {"type": "string", "default": "mcp_client"},
            },
            "required": ["target_id", "correction"],
        },
    },
    {
        "name": "verify_file_receipt",
        "description": "Read a file inside allowed workspace roots and return a deterministic Disk-Equals-Truth receipt (exact bytes, lines, SHA-256).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path inside allowed roots"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "epistemic_check",
        "description": "Audit a draft assistant reply against Disk-Equals-Truth rules and verified tool receipts (catches unread numbers, fake write claims, and promise-only tails).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_text": {"type": "string", "description": "Original user prompt"},
                "draft_reply": {"type": "string", "description": "Candidate assistant reply to audit"},
                "evidence_gated": {"type": "boolean", "description": "True if the turn asked to check/read/verify disk state", "default": False},
                "receipts": {
                    "type": "array",
                    "description": "Optional list of verified receipts ({tool, ok, byte_count, line_count, sha256_short, result_text})",
                    "items": {"type": "object"},
                },
                "active_memories": {
                    "type": "array",
                    "description": "Optional list of verified active memory strings for Tier 2 semantic policy checking",
                    "items": {"type": "string"},
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Enable Tier 2 Semantic Epistemic Critic via local OpenAI endpoint (default false)",
                    "default": False,
                },
                "llm_url": {
                    "type": "string",
                    "description": "Optional local OpenAI-compatible base URL for Tier 2 critic (default http://127.0.0.1:11455)",
                    "default": "http://127.0.0.1:11455",
                },
            },
            "required": ["user_text", "draft_reply"],
        },
    },
]


class EpistemicMemoryMCPServer:
    def __init__(self, db_path: Path, roots: dict[str, Path]):
        self.store = MemoryStore(db_path)
        self.roots = {k: v.resolve() for k, v in roots.items()}
        self.gate = EpistemicCommitGate()

    def close(self) -> None:
        self.store.close()

    def _resolve_in_roots(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if not p.is_absolute():
            first_root = next(iter(self.roots.values()))
            p = (first_root / p).resolve()
        else:
            p = p.resolve()
        for root in self.roots.values():
            try:
                p.relative_to(root)
                return p
            except ValueError:
                continue
        raise PermissionError(f"Path outside allowed workspace roots: {raw_path}")

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "memory_recall":
            hits = self.store.recall(
                str(args.get("query") or ""),
                k=int(args.get("k") or 5),
                include_inactive=bool(args.get("include_inactive", False)),
            )
            return {
                "ok": True,
                "count": len(hits),
                "hits": [
                    {
                        "id": h.id,
                        "status": h.status,
                        "valid_from": h.valid_from,
                        "score": round(h.score, 4),
                        "text": h.text,
                    }
                    for h in hits
                ],
            }

        if name == "memory_record":
            text = str(args.get("text") or "").strip()
            if not text:
                raise ValueError("text is required")
            kind = str(args.get("kind") or "observation").lower()
            source = str(args.get("source") or "mcp_client")
            if kind == "semantic":
                fid = self.store.add_semantic_fact(text, category=str(args.get("category") or "general"), source=source)
                return {"ok": True, "id": fid, "kind": "semantic"}
            eid = f"J-{uuid.uuid4().hex[:8]}"
            self.store.add_event(event_id=eid, kind="observation", text=text, source=source)
            return {"ok": True, "id": eid, "kind": "observation"}

        if name == "memory_supersede":
            return self.store.supersede(
                target_id=str(args.get("target_id") or "").strip(),
                correction=str(args.get("correction") or "").strip(),
                transition=str(args.get("transition") or "SUPERSEDENCE").upper(),
                source=str(args.get("source") or "mcp_client"),
                reason=str(args.get("reason") or ""),
            )

        if name == "verify_file_receipt":
            import hashlib
            target = self._resolve_in_roots(str(args.get("path") or ""))
            rcpt = build_file_receipt("verify_file_receipt", {"path": str(target)}, target, self.roots)
            full_sha = hashlib.sha256(target.read_bytes()).hexdigest() if (rcpt.ok and target.is_file()) else ""
            return {
                "ok": rcpt.ok,
                "resolved_path": rcpt.resolved_path,
                "root": rcpt.root_label,
                "byte_count": rcpt.byte_count,
                "line_count": rcpt.line_count,
                "sha256": full_sha,
                "sha256_short": rcpt.sha256_short,
                "preview": rcpt.result_text[:1500],
            }

        if name == "epistemic_check":
            raw_rcpts = args.get("receipts") or []
            rcpts = [
                EvidenceReceipt(
                    tool=str(r.get("tool") or "tool"),
                    args=dict(r.get("args") or {}),
                    ok=bool(r.get("ok", True)),
                    result_text=str(r.get("result_text") or ""),
                    duration_ms=float(r.get("duration_ms") or 0.0),
                    byte_count=r.get("byte_count"),
                    line_count=r.get("line_count"),
                    sha256_short=r.get("sha256_short"),
                )
                for r in raw_rcpts
                if isinstance(r, dict)
            ]
            lane = TurnLane.EVIDENCE_GATED if bool(args.get("evidence_gated", False)) else TurnLane.FAST_CONVERSATIONAL
            active_mems = [str(x) for x in (args.get("active_memories") or []) if str(x).strip()]
            use_sem = bool(args.get("semantic", False))
            critic = SemanticEpistemicCritic(base_url=str(args.get("llm_url") or "http://127.0.0.1:11455")) if use_sem else None
            verdict, cv = evaluate_hybrid(
                self.gate,
                critic,
                user_text=str(args.get("user_text") or ""),
                draft_reply=str(args.get("draft_reply") or ""),
                lane=lane,
                receipts=rcpts,
                active_memories=active_mems,
            )
            return {
                "passed": verdict.passed,
                "action": verdict.action,
                "violations": verdict.violations,
                "cleaned_text": verdict.cleaned_text,
                "reprompt_instruction": verdict.reprompt_instruction,
                "semantic_critic": {
                    "invoked": cv.invoked,
                    "grounded": cv.grounded,
                    "violation_type": cv.violation_type,
                    "reason": cv.reason,
                    "elapsed_ms": cv.elapsed_ms,
                },
            }

        raise KeyError(f"Unknown MCP tool: {name}")

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}

        # Notifications have no id
        if method == "notifications/initialized":
            return None

        try:
            if method == "initialize":
                res = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                }
            elif method == "ping":
                res = {}
            elif method == "tools/list":
                res = {"tools": TOOL_SPECS}
            elif method == "tools/call":
                tname = str(params.get("name") or "")
                targs = params.get("arguments") or {}
                out = self.call_tool(tname, targs)
                res = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": False}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            if req_id is None:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": res}
        except Exception as exc:
            if req_id is None:
                return None
            if method == "tools/call":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"ok": False, "error": str(exc)})}],
                        "isError": True,
                    },
                }
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(exc)}}

    def serve_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="Deadkey Standalone MCP Server (Bi-Temporal Memory + Epistemic Gate)")
    ap.add_argument("--db", default="./cartridges/default/data/memory.db", help="Path to SQLite bi-temporal memory DB")
    ap.add_argument("--root", action="append", default=[], help="Allowed workspace root for verify_file_receipt (repeatable)")
    args = ap.parse_args()

    roots_list = [Path(r) for r in args.root] if args.root else [Path.cwd()]
    roots = {f"root_{i}": p for i, p in enumerate(roots_list)}
    srv = EpistemicMemoryMCPServer(Path(args.db), roots)
    try:
        srv.serve_stdio()
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
