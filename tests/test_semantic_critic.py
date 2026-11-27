"""Deterministic Unit Tests for Tier 2 Semantic Epistemic Critic & Hybrid Gate."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deadkey.epistemic_gate import EpistemicCommitGate, build_file_receipt
from deadkey.semantic_critic import SemanticEpistemicCritic, evaluate_hybrid, should_invoke_critic
from deadkey.tool_router import TurnLane


class TestSemanticEpistemicCritic(unittest.TestCase):
    def test_01_selective_trigger_skips_benign_general_chat(self):
        self.assertFalse(should_invoke_critic(
            "Explain Python decorators in two sentences.",
            "A decorator wraps a function to extend its behavior without modifying it.",
        ))
        self.assertFalse(should_invoke_critic(
            "What is the time complexity of binary search on a sorted array?",
            "Binary search runs in O(log n) time.",
        ))
        self.assertTrue(should_invoke_critic(
            "Roughly how many kilobytes is server.py?",
            "It is about fifteen kilobytes.",
        ))
        self.assertTrue(should_invoke_critic(
            "Tell me if that's allowed.",
            "Sure!",
            active_memories=["Customer exports must stay in eu-central-1."],
        ))

    def test_02_hybrid_catches_paraphrased_fabrication_and_skips_tier1_blocks(self):
        gate = EpistemicCommitGate()
        calls = {"n": 0}

        def mock_judge(msgs):
            calls["n"] += 1
            body = msgs[-1]["content"]
            if "about fifteen kilobytes" in body:
                return json.dumps({
                    "grounded": False,
                    "violation_type": "paraphrased_fabrication",
                    "reason": "Estimated unread file size in words without a receipt.",
                })
            return json.dumps({"grounded": True, "violation_type": "clean", "reason": "Grounded or honest refusal."})

        critic = SemanticEpistemicCritic(transport_fn=mock_judge)

        # 1. Paraphrased spelled-out fabrication slips past Tier 1, gets caught by Tier 2!
        v1, cv1 = evaluate_hybrid(
            gate, critic,
            user_text="Roughly how many kilobytes is server.py? Spell it out in words.",
            draft_reply="server.py is about fifteen kilobytes.",
            lane=TurnLane.FAST_CONVERSATIONAL,
            receipts=[],
            attempt=1,
            max_attempts=2,
        )
        self.assertEqual(calls["n"], 1)
        self.assertFalse(v1.passed)
        self.assertEqual(v1.action, "reprompt")
        self.assertFalse(cv1.grounded)
        self.assertIn("semantic_critic:paraphrased_fabrication", v1.violations)

        # 2. Attempt 2 still fabricated -> hard fail-closed replace!
        v2, _ = evaluate_hybrid(
            gate, critic,
            user_text="Roughly how many kilobytes is server.py? Spell it out in words.",
            draft_reply="server.py is about fifteen kilobytes.",
            lane=TurnLane.FAST_CONVERSATIONAL,
            receipts=[],
            attempt=2,
            max_attempts=2,
        )
        self.assertFalse(v2.passed)
        self.assertEqual(v2.action, "replace")
        self.assertNotIn("fifteen", v2.cleaned_text.lower())

        # 3. Tier 1 obvious violation stops at Tier 1 WITHOUT invoking Tier 2!
        calls_before = calls["n"]
        v_t1, cv_t1 = evaluate_hybrid(
            gate, critic,
            user_text="Tell me the exact byte size of server.py.",
            draft_reply="server.py is 99,999 bytes.",
            lane=TurnLane.EVIDENCE_GATED,
            receipts=[],
            attempt=1,
            max_attempts=2,
        )
        self.assertFalse(v_t1.passed)
        self.assertEqual(calls["n"], calls_before)
        self.assertFalse(cv_t1.invoked)

        # 4. Benign grounded file receipt passes cleanly (FPR = 0%)!
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "demo.md"
            f.write_text("alpha\nbeta\n", encoding="utf-8")
            rcpt = build_file_receipt("read_file", {"path": str(f)}, f, {"ws": Path(td)})
            v_ok, cv_ok = evaluate_hybrid(
                gate, critic,
                user_text="Report exact bytes and lines of demo.md.",
                draft_reply=f"demo.md has {rcpt.byte_count} bytes and {rcpt.line_count} lines.",
                lane=TurnLane.EVIDENCE_GATED,
                receipts=[rcpt],
                attempt=1,
                max_attempts=2,
            )
            self.assertTrue(v_ok.passed)
            self.assertEqual(v_ok.action, "commit")
            self.assertTrue(cv_ok.grounded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
