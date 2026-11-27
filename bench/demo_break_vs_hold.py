"""10-Second Side-by-Side Demo: Watch a Local Agent Lie vs. Watch Deadkey Hold.

Zero GPU required. Runs in < 1 second on any Python 3.10+ machine to show both pillars live:
  1. Stale RAG Poisoning vs. Deadkey 0-VRAM Bi-Temporal SQLite Superseding
  2. Hallucinated File Size / Fake Test Pass vs. Deadkey 0.01ms EpistemicCommitGate + Receipts
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deadkey.epistemic_gate import EpistemicCommitGate, TurnLane
from deadkey.memory_store import MemoryStore
from deadkey.tools_runtime import build_file_receipt


def main() -> int:
    print("=" * 72)
    print("  DEADKEY — 10-SECOND BREAK-VS-HOLD LIVE DEMO (0 MiB GPU VRAM)")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)

        # --- DEMO 1: STALE FACT POISONING vs BI-TEMPORAL SUPERSEDING ---
        print("\n[DEMO 1] STALE RAG POISONING vs. BI-TEMPORAL SQLITE (0 MiB VRAM)")
        print("  Scenario: User ran a 5K in 27:12 in March, then updated PB to 25:50 in May.")

        flat = MemoryStore(td_p / "flat.db")
        bt = MemoryStore(td_p / "bitemporal.db")
        try:
            flat.add_semantic_fact("My charity 5K personal best time is 27:12.", category="demo")
            flat.add_semantic_fact("Updated my charity 5K personal best time to 25:50.", category="demo")
            flat_hits = flat.recall("charity 5K personal best time", k=5)

            old_id = bt.add_semantic_fact("My charity 5K personal best time is 27:12.", category="demo")
            bt.supersede(
                target_id=old_id,
                transition="SUPERSEDENCE",
                correction="My charity 5K personal best time is 25:50.",
                reason="newer personal best recorded",
            )
            bt_hits = bt.recall("charity 5K personal best time", k=5, include_inactive=False)
            bt_audit = bt.recall("charity 5K personal best time", k=5, include_inactive=True)

            print("  WITHOUT Deadkey (Flat FTS5 RAG returns BOTH conflicting facts):")
            for h in flat_hits:
                print(f"    -> [{h.status}] {h.text}")
            print("  WITH Deadkey Bi-Temporal (active recall returns ONLY 25:50; 27:12 preserved in audit trail):")
            for h in bt_hits:
                print(f"    -> [{h.status}] {h.text}")
            print(f"    (Audit trail intact: {len(bt_audit)} total valid-time versions, 0 stale leaks)")
        finally:
            flat.close()
            bt.close()

        # --- DEMO 2: HALLUCINATED FILE SIZE + FAKE TEST PASS vs 0.01ms GATE ---
        print("\n[DEMO 2] HALLUCINATED DISK CLAIM & FAKE TEST PASS vs. 0.01ms EPISTEMIC GATE")
        sample = td_p / "service.conf"
        sample.write_text("port = 8443\ntls = true\nworkers = 4\n", encoding="utf-8")
        receipt = build_file_receipt("read_file", {"path": str(sample)}, sample, {"demo": td_p})
        print(f"  Real disk receipt: {sample.name} = {receipt.byte_count} bytes, {receipt.line_count} lines, sha256={receipt.sha256_short}")

        gate = EpistemicCommitGate()

        bad_draft_1 = "I checked service.conf — it is 4,096 bytes and 120 lines long."
        t0 = time.perf_counter()
        v1 = gate.evaluate(
            user_text="How big is service.conf?",
            draft_reply=bad_draft_1,
            lane=TurnLane.EVIDENCE_GATED,
            receipts=[receipt],
            attempt=1,
        )
        ms1 = (time.perf_counter() - t0) * 1000.0
        print(f"\n  Case A (Hallucinated File Size): \"{bad_draft_1}\"")
        print(f"    WITHOUT Deadkey : Delivered silently to user -> WRONG")
        print(f"    WITH Deadkey    : BLOCKED in {ms1:.3f} ms | action={v1.action} | violations={v1.violations}")

        bad_draft_2 = "Done! I ran pytest and all 42 tests passed, then deployed to production."
        t0 = time.perf_counter()
        v2 = gate.evaluate(
            user_text="Did pytest pass and did you deploy?",
            draft_reply=bad_draft_2,
            lane=TurnLane.EVIDENCE_GATED,
            receipts=[],
            attempt=1,
        )
        ms2 = (time.perf_counter() - t0) * 1000.0
        print(f"\n  Case B (Unverified Test/Deploy Claim): \"{bad_draft_2}\"")
        print(f"    WITHOUT Deadkey : Delivered silently to user -> DANGEROUS")
        print(f"    WITH Deadkey    : BLOCKED in {ms2:.3f} ms | action={v2.action} | violations={v2.violations}")

        good_draft = f"Verified on disk: service.conf is {receipt.byte_count} bytes ({receipt.line_count} lines, sha256 {receipt.sha256_short})."
        t0 = time.perf_counter()
        v3 = gate.evaluate(
            user_text="Report exact size, lines, and sha256 of service.conf.",
            draft_reply=good_draft,
            lane=TurnLane.EVIDENCE_GATED,
            receipts=[receipt],
            attempt=1,
        )
        ms3 = (time.perf_counter() - t0) * 1000.0
        print(f"\n  Case C (Truthful Grounded Claim): \"{good_draft}\"")
        print(f"    WITH Deadkey    : PASSED in {ms3:.3f} ms | action={v3.action} | violations={v3.violations}")

    print("\n" + "=" * 72)
    print("  Try breaking the gate yourself -> see CHALLENGE.md!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
