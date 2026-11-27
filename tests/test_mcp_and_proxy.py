"""End-to-End Tests for Deadkey's Standalone MCP Server & OpenAI Epistemic Proxy."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deadkey.epistemic_proxy import EpistemicProxyRuntime, create_proxy_server
from deadkey.mcp_server import EpistemicMemoryMCPServer


class TestMCPAndEpistemicProxy(unittest.TestCase):
    def test_01_mcp_server_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            demo = root / "notes.md"
            demo.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

            mcp = EpistemicMemoryMCPServer(root / "mem.db", {"ws": root})
            try:
                # 1. initialize + tools/list
                r_init = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
                self.assertEqual(r_init["result"]["serverInfo"]["name"], "deadkey-epistemic-memory")
                r_list = mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
                names = {t["name"] for t in r_list["result"]["tools"]}
                self.assertEqual(names, {"memory_recall", "memory_record", "memory_supersede", "verify_file_receipt", "epistemic_check"})

                # 2. Record -> Supersede -> Recall (retired fact MUST disappear, correction MUST surface!)
                r_rec = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "memory_record", "arguments": {"text": "Server runs on port 9999."}},
                })
                old_id = json.loads(r_rec["result"]["content"][0]["text"])["id"]

                r_sup = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "memory_supersede", "arguments": {
                        "target_id": old_id, "transition": "DISPROVAL", "correction": "Server actually runs on port 8765."
                    }},
                })
                sup_out = json.loads(r_sup["result"]["content"][0]["text"])
                self.assertTrue(sup_out["ok"])

                r_q = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "memory_recall", "arguments": {"query": "Server port", "k": 5}},
                })
                hits = json.loads(r_q["result"]["content"][0]["text"])["hits"]
                self.assertTrue(any("8765" in h["text"] for h in hits))
                self.assertFalse(any(h["id"] == old_id for h in hits))

                # 2b. Semantic fact (SF-*) Record -> Supersede -> Recall (regression caught by Hermes Agent!)
                r_sf = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 51, "method": "tools/call",
                    "params": {"name": "memory_record", "arguments": {
                        "text": "Demo GPU target is RTX 4090 24GB", "kind": "semantic", "category": "hermes_demo"
                    }},
                })
                sf_old_id = json.loads(r_sf["result"]["content"][0]["text"])["id"]
                self.assertTrue(sf_old_id.startswith("SF-"))

                r_sf_sup = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 52, "method": "tools/call",
                    "params": {"name": "memory_supersede", "arguments": {
                        "target_id": sf_old_id,
                        "transition": "SUPERSEDENCE",
                        "correction": "Demo GPU target is RTX 5090 32GB GDDR7",
                        "reason": "Upgraded hardware",
                    }},
                })
                sf_sup_out = json.loads(r_sf_sup["result"]["content"][0]["text"])
                self.assertTrue(sf_sup_out["ok"])
                self.assertEqual(sf_sup_out["retired_table"], "semantic_facts")

                r_sf_q = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 53, "method": "tools/call",
                    "params": {"name": "memory_recall", "arguments": {"query": "Demo GPU target", "k": 5}},
                })
                sf_hits = json.loads(r_sf_q["result"]["content"][0]["text"])["hits"]
                self.assertEqual(len(sf_hits), 1)
                self.assertIn("RTX 5090", sf_hits[0]["text"])
                self.assertFalse(any(h["id"] == sf_old_id for h in sf_hits))

                # 3. Deterministic file receipt (exact bytes + full 64-char sha256) + Epistemic Gate audit
                r_file = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {"name": "verify_file_receipt", "arguments": {"path": "notes.md"}},
                })
                f_out = json.loads(r_file["result"]["content"][0]["text"])
                self.assertTrue(f_out["ok"])
                self.assertEqual(f_out["line_count"], 3)
                self.assertEqual(len(f_out["sha256"]), 64)

                r_bad = mcp.handle_request({
                    "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "epistemic_check", "arguments": {
                        "user_text": "How many bytes is notes.md?",
                        "draft_reply": "notes.md is 99,999 bytes.",
                        "evidence_gated": True,
                        "receipts": [],
                    }},
                })
                bad_out = json.loads(r_bad["result"]["content"][0]["text"])
                self.assertFalse(bad_out["passed"])
                self.assertIn("check_without_tool", bad_out["violations"])
            finally:
                mcp.close()

    def test_02_openai_epistemic_proxy_blocks_hallucination(self):
        captured_reqs: list[dict] = []

        class MockUpstream(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                captured_reqs.append(body)
                last_u = body["messages"][-1]["content"]
                reply = "app.py is 42,000 bytes." if "app.py" in last_u else "Doing great!"
                raw = json.dumps({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            up_srv = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
            up_port = up_srv.server_address[1]
            threading.Thread(target=up_srv.serve_forever, daemon=True).start()

            rt = EpistemicProxyRuntime(upstream_url=f"http://127.0.0.1:{up_port}", db_path=root / "proxy.db")
            rt.store.add_event(event_id="J-000001", kind="observation", text="Primary engine is Strata CUDA 13.")
            px_srv = create_proxy_server(rt, host="127.0.0.1", port=0)
            px_port = px_srv.server_address[1]
            threading.Thread(target=px_srv.serve_forever, daemon=True).start()
            try:
                # 1. Hallucinated byte count MUST be caught and blocked by the proxy!
                req1 = urllib.request.Request(
                    f"http://127.0.0.1:{px_port}/v1/chat/completions",
                    data=json.dumps({"model": "any-frontier-model", "messages": [{"role": "user", "content": "Read app.py and tell me its exact size."}]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req1, timeout=5) as r1:
                    d1 = json.loads(r1.read().decode("utf-8"))
                self.assertFalse(d1["x_epistemic_audit"]["passed"])
                self.assertNotIn("42,000", d1["choices"][0]["message"]["content"])

                # 2. Clean casual turn passes AND injects valid-time memory into upstream!
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:{px_port}/v1/chat/completions",
                    data=json.dumps({"model": "any-frontier-model", "messages": [{"role": "user", "content": "Which Strata engine are we using?"}]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=5) as r2:
                    d2 = json.loads(r2.read().decode("utf-8"))
                self.assertTrue(d2["x_epistemic_audit"]["passed"])
                self.assertIn("J-000001", d2["x_epistemic_audit"]["recalled_ids"])
                self.assertIn("Strata CUDA 13", captured_reqs[-1]["messages"][0]["content"])
            finally:
                px_srv.shutdown()
                px_srv.server_close()
                rt.close()
                up_srv.shutdown()
                up_srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
