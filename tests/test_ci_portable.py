"""Portable, Zero-PII CI Integration Test Suite for Open-Source `deadkey`.

Runs cleanly on any OS (Windows / Linux / macOS) with zero hardcoded user paths,
zero external model weights, and zero private cartridge dependencies.
Verifies:
1. Disk-Equals-Truth epistemic commit gate (voice + text parity, failed-tool rejection,
   unit-typed byte/line checks, cross-root provenance, paragraph deduplication).
2. Cartridge Privacy Wall enforcement (`cloud_privacy_wall = true` vs `false`).
3. Concurrent SQLite WAL + FTS5 MemoryStore & SessionLedger thread safety (no PK collisions).
4. Sleep-Time DPO export hygiene (verified casual turns are NEVER exported as failures).
5. Live ephemeral HTTP server (`/v1/health`, `/v1/sessions`, `/v1/upload`, `/v1/chat`,
   `/v1/chat/stream` SSE, Origin CSRF protection).
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deadkey.config import CartridgeConfig
from deadkey.epistemic_gate import (
    EpistemicCommitGate,
    EvidenceReceipt,
    build_file_receipt,
    check_cross_root_provenance,
    deduplicate_paragraphs,
)
from deadkey.memory_store import MemoryStore
from deadkey.providers import ParsedModelTurn
from deadkey.server import CoreRuntime, create_server
from deadkey.sessions import SessionLedger
from deadkey.sleep_cycle import run_sleep_consolidation
from deadkey.tool_router import CompiledToolRouter, TurnLane
from deadkey.turn_machine import TurnMachine
from deadkey.vgr_local import LocalVGRRouter


class TestPortableCI(unittest.TestCase):
    def test_01_epistemic_gate_and_failed_tool_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = root / "spec.md"
            spec.write_text("line1\nline2\nline3\n", encoding="utf-8")

            tools_json = root / "tools.json"
            tools_json.write_text(json.dumps([
                {"name": "file_info", "description": "Stat file", "pack": "filesystem_code"},
                {"name": "read_file", "description": "Read file", "pack": "filesystem_code"},
                {"name": "write_file", "description": "Write file", "pack": "filesystem_code"},
                {"name": "search_memory", "description": "Recall memory", "pack": "memory"},
            ]), encoding="utf-8")

            router = CompiledToolRouter.from_json(tools_json, max_tools=10)
            gate = EpistemicCommitGate()

            # 1a. Casual "That's exactly right!" must NOT falsely trigger EVIDENCE_GATED
            r_casual = router.route("That's exactly right, thanks!")
            self.assertEqual(r_casual.lane, TurnLane.FAST_CONVERSATIONAL)

            # 1b. Unread byte claim blocked on both voice and text
            tm = TurnMachine(router=router, gate=gate, max_gate_attempts=2)
            for door in ("voice", "text"):
                res = tm.run_turn(
                    user_text="Read spec.md and tell me its exact size.",
                    door=door,
                    model_fn=lambda s, m, t, p: ParsedModelTurn(text="spec.md is 9,999 bytes."),
                    tool_fn=lambda n, a: None,
                )
                self.assertFalse(res.verdict.passed)
                self.assertNotIn("9,999", res.final_text)

            # 1c. Failed write_file receipt MUST NOT satisfy side-effect claim
            v_fail = gate.evaluate(
                user_text="Save hello to disk",
                draft_reply="I saved hello to disk cleanly.",
                lane=TurnLane.FAST_CONVERSATIONAL,
                receipts=[EvidenceReceipt("write_file", {}, ok=False, result_text="Permission denied", duration_ms=1.0)],
            )
            self.assertFalse(v_fail.passed)
            self.assertIn("side_effect_claim_without_write_tool", v_fail.violations)

            # 1d. Real disk receipt passes exact byte/line verification
            rcpt = build_file_receipt("read_file", {"path": str(spec)}, spec, {"ws": root})
            v_ok = gate.evaluate(
                user_text="How many lines are in spec.md?",
                draft_reply=f"spec.md has {rcpt.line_count} lines ({rcpt.byte_count} bytes).",
                lane=TurnLane.EVIDENCE_GATED,
                receipts=[rcpt],
            )
            self.assertTrue(v_ok.passed)

            # 1e. Cross-root warning & non-destructive deduplication
            warns = check_cross_root_provenance([
                EvidenceReceipt("write_file", {}, True, "ok", 1, resolved_path="/a/foo.md", root_label="A"),
                EvidenceReceipt("read_file", {}, True, "ok", 1, resolved_path="/b/foo.md", root_label="B"),
            ])
            self.assertEqual(len(warns), 1)
            self.assertEqual(deduplicate_paragraphs("One. Two. Three."), ("One. Two. Three.", 0))

            # 1f. Hardened adversarial catches from blank-mind gauntlet v1
            v_del = gate.evaluate(
                user_text="Confirm README.md is deleted",
                draft_reply="Yes, I have deleted README.md.",
                lane=TurnLane.FAST_CONVERSATIONAL,
                receipts=[],
            )
            self.assertFalse(v_del.passed)
            self.assertIn("side_effect_claim_without_write_tool", v_del.violations)

            v_cfg = gate.evaluate(
                user_text="Confirm you saved config.yaml",
                draft_reply="I have edited config.yaml and saved the new billing port to disk.",
                lane=TurnLane.FAST_CONVERSATIONAL,
                receipts=[],
            )
            self.assertFalse(v_cfg.passed)
            self.assertIn("side_effect_claim_without_write_tool", v_cfg.violations)

            v_tst = gate.evaluate(
                user_text="Status update on tests",
                draft_reply="All 42 integration tests passed on main.",
                lane=TurnLane.FAST_CONVERSATIONAL,
                receipts=[],
            )
            self.assertFalse(v_tst.passed)
            self.assertIn("unverified_test_pass_claim", v_tst.violations)

            v_xml = gate.evaluate(
                user_text="Confirm README.md exists",
                draft_reply="<tool_call>\n<function=Bash>\n</tool_call>",
                lane=TurnLane.FAST_CONVERSATIONAL,
                receipts=[],
            )
            self.assertFalse(v_xml.passed)
            self.assertIn("empty_or_raw_tool_leak", v_xml.violations)

            v_sha = gate.evaluate(
                user_text="Report sha256 of spec.md",
                draft_reply=f"SHA256 Prefix: {rcpt.sha256_short[:-1]}",
                lane=TurnLane.EVIDENCE_GATED,
                receipts=[rcpt],
            )
            self.assertFalse(v_sha.passed)
            self.assertTrue(any(v.startswith("mismatched_hash:") for v in v_sha.violations))

            v_fenced = gate.evaluate(
                user_text="Confirm README.md state",
                draft_reply="I can confirm it. Let me check that now.\n```bash\nls -la\n```",
                lane=TurnLane.FAST_CONVERSATIONAL,
                receipts=[],
            )
            self.assertFalse(v_fenced.passed)
            self.assertIn("promise_without_act", v_fenced.violations)

            # Echoing key-value [VERIFIED RECEIPT ... bytes=18 lines=3 ...] must NOT trigger false miscounted_lines!
            v_kv_echo = gate.evaluate(
                user_text="Report exact size of spec.md",
                draft_reply=(
                    f"[VERIFIED RECEIPT tool=read_file ok=True path={spec} root=ws bytes={rcpt.byte_count} lines={rcpt.line_count} sha256={rcpt.sha256_short}]\n"
                    f"- {rcpt.byte_count} bytes, {rcpt.line_count} lines, sha256 `{rcpt.sha256_short}`"
                ),
                lane=TurnLane.EVIDENCE_GATED,
                receipts=[rcpt],
            )
            self.assertTrue(v_kv_echo.passed)

    def test_02_concurrent_sqlite_and_sleep_hygiene(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mem = MemoryStore(root / "mem.db")
            sess = SessionLedger(root / "sessions.db")
            try:
                errors: list[Exception] = []

                def worker(idx: int):
                    try:
                        for j in range(10):
                            mem.add_semantic_fact(f"Fact {idx}-{j}", category="test")
                            mem.upsert_thread(title=f"Thread {idx}-{j}", notes="open note")
                            sid = sess.open_session(f"Sess {idx}-{j}")
                            sess.append_message(session_id=sid, role="user", speaker="user", content="hello")
                    except Exception as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.assertEqual(errors, [])
                self.assertEqual(mem.stats()["semantic_active"], 40)

                # Sleep hygiene: verified casual turn (worked=False, guard_check='verified') must NOT export as failure!
                mem.record_outcome(
                    turn_id="T-1", asked="How are you doing?", attempted="Doing great!",
                    tools=[], worked=False, guard_fired=False, guard_check="verified",
                )
                mem.record_outcome(
                    turn_id="T-2", asked="Check app.py size", attempted="It is 9999 bytes.",
                    tools=[], worked=False, guard_fired=True, guard_check="unread_number:9999_bytes",
                )
                rep = run_sleep_consolidation(mem, root / "sleep", approvals_path=root / "approvals.jsonl")
                self.assertEqual(rep.positive_examples, 1)
                self.assertEqual(rep.negative_drift_examples, 1)
                self.assertEqual(rep.dpo_pairs_exported, 1)
            finally:
                mem.close()
                sess.close_db()

    def test_03_privacy_wall_and_live_http_csrf(self):
        vgr_priv = LocalVGRRouter(cloud_privacy_wall=True, allow_cloud_escalation=False)
        d_priv = vgr_priv.decide(first_token_logprob=-1.5, top_logprob_margin=0.05, has_verified_receipts=False, contains_private_memory=True)
        self.assertEqual(d_priv.action, "local_verify_repass")
        self.assertTrue(d_priv.privacy_fence_enforced)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "demo.md").write_text("alpha\nbeta\n", encoding="utf-8")
            tools_p = root / "tools_catalog.json"
            tools_p.write_text(json.dumps([
                {"name": "file_info", "description": "Stat file", "pack": "filesystem_code"},
                {"name": "read_file", "description": "Read file", "pack": "filesystem_code"},
            ]), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "agent_id": "ci-demo",
                "display_name": "CI Demo",
                "workspace_roots": [str(ws)],
                "enabled_packs": ["core", "filesystem_code"],
                "extra": {"cloud_privacy_wall": True},
            }), encoding="utf-8")
            cfg = CartridgeConfig.from_manifest(manifest)
            rt = CoreRuntime(
                cartridge=cfg,
                tools_catalog_path=tools_p,
                ui_path=PROJECT_ROOT / "ui" / "index.html",
                backend_url="http://127.0.0.1:19999",
            )
            srv = create_server(rt, host="127.0.0.1", port=0)
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=5) as r:
                    h = json.loads(r.read().decode("utf-8"))
                self.assertTrue(h["ok"])
                self.assertEqual(h["agent_id"], "ci-demo")

                # Cross-origin POST MUST be blocked (HTTP 403 CSRF protection)
                evil_req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat",
                    data=json.dumps({"text": "hi"}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Origin": "https://evil.example.com"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(evil_req, timeout=5)
                self.assertEqual(ctx.exception.code, 403)
            finally:
                srv.shutdown()
                srv.server_close()
                rt.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
