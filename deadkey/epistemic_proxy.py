"""Model-Agnostic OpenAI-Compatible Epistemic & Bi-Temporal Memory Middleware Proxy.

Point ANY OpenAI-compatible client (Hermes Agent, OpenWebUI, Continue, Cursor, custom scripts)
at `http://127.0.0.1:8788/v1` while forwarding inference to ANY upstream backend
(Strata 125B, llama.cpp, vLLM, Ollama, or frontier cloud APIs):

  Client -> Epistemic Proxy (:8788) -> Upstream OpenAI Backend (e.g. Strata :11455)
              ├── Injects top-K bi-temporal valid-time memories (0 MiB VRAM)
              ├── Audits non-streamed responses via Disk-Equals-Truth EpistemicCommitGate (0.01 ms)
              └── Appends `x_epistemic_audit` metadata & blocks fabricated numbers/fake writes!

Usage:
  python -m deadkey.epistemic_proxy --port 8788 --upstream http://127.0.0.1:11455 --db ./data/memory.db
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .epistemic_gate import EpistemicCommitGate
from .memory_store import MemoryStore
from .semantic_critic import SemanticEpistemicCritic, evaluate_hybrid
from .tool_router import EVIDENCE_GATE_RX, TurnLane


class EpistemicProxyRuntime:
    def __init__(
        self,
        *,
        upstream_url: str,
        db_path: Path,
        recall_k: int = 3,
        semantic_critic: bool = False,
        critic_url: str | None = None,
    ):
        self.upstream_url = upstream_url.rstrip("/")
        self.store = MemoryStore(db_path)
        self.gate = EpistemicCommitGate()
        self.recall_k = recall_k
        self.critic = (
            SemanticEpistemicCritic(base_url=(critic_url or self.upstream_url))
            if semantic_critic
            else None
        )

    def close(self) -> None:
        self.store.close()

    def proxy_chat_completions(self, payload: dict[str, Any], auth_header: str = "") -> tuple[int, dict[str, Any]]:
        messages = list(payload.get("messages") or [])
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    last_user = c.strip()
                    break

        # 1. Bi-temporal valid-time memory injection (0 MiB GPU VRAM)
        hits = self.store.recall(last_user, k=self.recall_k) if last_user else []
        if hits:
            mem_block = "\n".join(f"- [{h.id} · {h.status} · {h.valid_from}] {h.text}" for h in hits)
            injected = {
                "role": "system",
                "content": (
                    "[Verified Bi-Temporal Memory (active valid-time records only)]\n"
                    f"{mem_block}\n"
                    "Rule: Follow Disk-Equals-Truth strictly. Never invent file sizes, line counts, or tool actions."
                ),
            }
            messages = [injected] + messages

        fwd = dict(payload)
        fwd["messages"] = messages
        fwd["stream"] = False  # Buffered audit ensures verified-before-commit parity

        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        req = urllib.request.Request(
            f"{self.upstream_url}/v1/chat/completions",
            data=json.dumps(fwd).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        # 2. Disk-Equals-Truth Epistemic Commit Gate audit
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            draft = msg.get("content") or ""
            has_tool_calls = bool(msg.get("tool_calls"))
            if draft and not has_tool_calls:
                lane = TurnLane.EVIDENCE_GATED if EVIDENCE_GATE_RX.search(last_user) else TurnLane.FAST_CONVERSATIONAL
                active_mems = [h.text for h in hits]
                verdict, cv = evaluate_hybrid(
                    self.gate,
                    self.critic,
                    user_text=last_user,
                    draft_reply=draft,
                    lane=lane,
                    receipts=[],
                    active_memories=active_mems,
                    attempt=2,
                    max_attempts=2,
                )
                msg["content"] = verdict.cleaned_text
                data["x_epistemic_audit"] = {
                    "passed": verdict.passed,
                    "action": verdict.action,
                    "lane": lane.value,
                    "violations": verdict.violations,
                    "recalled_ids": [h.id for h in hits],
                    "semantic_critic": {
                        "enabled": self.critic is not None,
                        "invoked": cv.invoked,
                        "grounded": cv.grounded,
                        "violation_type": cv.violation_type,
                        "elapsed_ms": cv.elapsed_ms,
                    },
                }
        return status, data


def create_proxy_server(rt: EpistemicProxyRuntime, host: str = "127.0.0.1", port: int = 8788) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"ok": True, "proxy": "deadkey-epistemic-proxy", "upstream": rt.upstream_url})
                return
            if self.path == "/v1/models":
                try:
                    with urllib.request.urlopen(f"{rt.upstream_url}/v1/models", timeout=5) as r:
                        self._send_json(json.loads(r.read().decode("utf-8", errors="replace")), r.status)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 502)
                return
            self._send_json({"error": f"Not found: {self.path}"}, 404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json({"error": f"Unsupported endpoint: {self.path}"}, 404)
                return
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n <= 0 or n > 25 * 1024 * 1024:
                self._send_json({"error": "Invalid Content-Length"}, 400)
                return
            payload = json.loads(self.rfile.read(n).decode("utf-8", errors="replace"))
            try:
                status, data = rt.proxy_chat_completions(payload, auth_header=self.headers.get("Authorization") or "")
                self._send_json(data, status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 502)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser(description="Deadkey OpenAI Epistemic & Memory Middleware Proxy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--upstream", default="http://127.0.0.1:11455", help="Upstream OpenAI-compatible base URL")
    ap.add_argument("--db", default="./cartridges/default/data/memory.db")
    ap.add_argument("--semantic-critic", action="store_true", help="Enable Tier 2 local Semantic Epistemic Critic")
    ap.add_argument("--critic-url", default=None, help="Optional separate OpenAI-compatible URL for Tier 2 critic")
    args = ap.parse_args()

    rt = EpistemicProxyRuntime(
        upstream_url=args.upstream,
        db_path=Path(args.db),
        semantic_critic=bool(args.semantic_critic),
        critic_url=args.critic_url,
    )
    srv = create_proxy_server(rt, host=args.host, port=args.port)
    print(f"[epistemic-proxy] Listening on http://{args.host}:{args.port}/v1 -> {args.upstream}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
