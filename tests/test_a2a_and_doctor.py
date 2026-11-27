"""Unit & integration tests for A2A Peer Protocol, Privacy Wall enforcement, and Environment Doctor."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deadkey.a2a import build_agent_card, handle_a2a_rpc
from deadkey.config import CartridgeConfig
from deadkey.doctor import run_doctor
from deadkey.server import CoreRuntime


class TestA2AAndDoctor(unittest.TestCase):
    def test_01_doctor_reports_core_ok(self):
        res = run_doctor()
        self.assertTrue(res["ok"], f"Doctor failed: {res}")
        self.assertTrue(res["checks"]["sqlite_fts5"]["ok"])
        self.assertTrue(res["checks"]["python"]["ok"])

    def test_02_a2a_privacy_wall_and_epistemic_audit(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "manifest.json").write_text(json.dumps({
                "agent_id": "private_vault",
                "display_name": "Private Vault",
                "db_path": "data/mem.db",
                "extra": {"cloud_privacy_wall": True},
            }), encoding="utf-8")
            (p / "tools_catalog.json").write_text("[]", encoding="utf-8")
            rt = CoreRuntime(
                cartridge=CartridgeConfig.from_manifest(p / "manifest.json"),
                tools_catalog_path=p / "tools_catalog.json",
            )
            try:
                card = build_agent_card(rt)
                self.assertEqual(card["protocol"], "a2a/1.0")
                self.assertTrue(card["privacy"]["privacy_wall"])
                self.assertFalse(card["privacy"]["memory_exposed_over_a2a"])

                blocked = handle_a2a_rpc(rt, {
                    "id": "t1", "method": "tasks/send",
                    "params": {"skill": "memory_recall", "input": {"query": "anything"}},
                })
                self.assertIn("error", blocked)
                self.assertEqual(blocked["error"]["code"], -32003)

                audited = handle_a2a_rpc(rt, {
                    "id": "t2", "method": "tasks/send",
                    "params": {
                        "skill": "epistemic_audit",
                        "input": {
                            "user_text": "What is the exact byte size of config.yaml?",
                            "draft_reply": "config.yaml is 4,096 bytes and 112 lines.",
                            "semantic": False,
                        },
                    },
                })
                out = audited["result"]["output"]
                self.assertNotEqual(out["action"], "commit")
                self.assertTrue(out["violations"])
            finally:
                rt.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
