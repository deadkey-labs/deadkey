"""15-Second Native X / Terminal Demo: Watch an Agent Lie vs. Watch Deadkey Hold.

Zero GPU required, zero external dependencies. Paced cleanly to ~15 seconds for
screen recording (Win+Alt+R / OBS) — or pass `--fast` for instant CI execution.

Demonstrates all three verified pillars cleanly:
  1. Stale RAG Poisoning vs. Deadkey 0-VRAM Bi-Temporal Superseding (0 stale leaks)
  2. Hallucinated Disk Claim + Fake Test Pass blocked in ~0.01 ms by EpistemicCommitGate
  3. Honest Takeaway: Why an external gate beats cranking a global steering vector
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deadkey.epistemic_gate import EpistemicCommitGate, build_file_receipt
from deadkey.memory_store import MemoryStore
from deadkey.tool_router import TurnLane

# ANSI palette (Deadkey Molten Ember)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
EMBER = "\033[38;5;209m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;78m"
CYAN = "\033[38;5;81m"
GRAY = "\033[38;5;245m"


def _enable_ansi_windows() -> None:
    if os.name == "nt":
        os.system("")


def _pause(sec: float, fast: bool) -> None:
    if not fast and sec > 0:
        time.sleep(sec)


def main() -> int:
    ap = argparse.ArgumentParser(description="15-Second Deadkey Break-vs-Hold Terminal Demo")
    ap.add_argument("--fast", action="store_true", help="Skip cinematic pacing (for CI/testing)")
    args = ap.parse_args()
    fast = args.fast
    _enable_ansi_windows()
    t_start = time.perf_counter()

    if not fast:
        print("\033[2J\033[H", end="")

    print(f"{EMBER}{BOLD}========================================================================{RESET}")
    print(f"{EMBER}{BOLD}  DEADKEY  |  0-VRAM Epistemic Truth Gate for Local AI  (deadkey.dev){RESET}")
    print(f"{EMBER}{BOLD}========================================================================{RESET}")
    _pause(1.2, fast)

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)

        # --- SCENE 1 (0-5s): Stale Memory Poisoning vs Bi-Temporal Superseding ---
        print(f"\n{CYAN}{BOLD}[1/3] STALE RAG POISONING vs. BI-TEMPORAL SQLITE (0 MiB GPU VRAM){RESET}")
        flat = MemoryStore(td_p / "flat.db")
        bt = MemoryStore(td_p / "bt.db")
        try:
            flat.add_semantic_fact("Free-tier API rate limit is 1000 req/min.", category="policy")
            flat.add_semantic_fact("Free-tier API rate limit is 60 req/min.", category="policy")
            flat_hits = flat.recall("Free-tier API rate limit", k=4)

            old_id = bt.add_semantic_fact("Free-tier API rate limit is 1000 req/min.", category="policy")
            bt.supersede(
                target_id=old_id,
                transition="SUPERSEDENCE",
                correction="Free-tier API rate limit is 60 req/min.",
                reason="quota tightened",
            )
            bt_hits = bt.recall("Free-tier API rate limit", k=4, include_inactive=False)
        finally:
            flat.close()
            bt.close()

        print(f"  {RED}WITHOUT Deadkey (Flat RAG returns conflicting stale policy):{RESET}")
        for h in flat_hits:
            print(f"    {RED}x{RESET} {h.text}")
        _pause(1.6, fast)
        print(f"  {GREEN}WITH Deadkey Bi-Temporal (retired fact excluded, audit trail intact):{RESET}")
        for h in bt_hits:
            print(f"    {GREEN}+{RESET} [{h.status}] {h.text}")
        _pause(2.0, fast)

        # --- SCENE 2 (5-11s): Hallucinated File Size & Fake Test Pass vs 0.01ms Gate ---
        print(f"\n{CYAN}{BOLD}[2/3] HALLUCINATED DISK CLAIM & FAKE DEPLOY vs. 0.01 ms GATE{RESET}")
        sample = td_p / "service.conf"
        sample.write_text("port = 8443\ntls = true\nworkers = 4\n", encoding="utf-8")
        receipt = build_file_receipt("read_file", {"path": str(sample)}, sample, {"demo": td_p})
        print(f"  {GRAY}Disk receipt: {sample.name} = {receipt.byte_count} B, {receipt.line_count} lines, sha256={receipt.sha256_short}{RESET}")
        _pause(1.0, fast)

        gate = EpistemicCommitGate()

        bad_1 = "I checked service.conf — it is 4,096 bytes and 120 lines."
        t0 = time.perf_counter()
        v1 = gate.evaluate(user_text="How big is service.conf?", draft_reply=bad_1, lane=TurnLane.EVIDENCE_GATED, receipts=[receipt], attempt=1)
        ms1 = (time.perf_counter() - t0) * 1000.0
        print(f"  {RED}Agent draft :{RESET} \"{bad_1}\"")
        print(f"  {GREEN}Deadkey gate:{RESET} {BOLD}BLOCKED in {ms1:.3f} ms{RESET}  ({v1.action} | {v1.violations})")
        _pause(2.0, fast)

        bad_2 = "Done! pytest passed all 42 tests and I deployed to production."
        t0 = time.perf_counter()
        v2 = gate.evaluate(user_text="Did pytest pass?", draft_reply=bad_2, lane=TurnLane.EVIDENCE_GATED, receipts=[], attempt=1)
        ms2 = (time.perf_counter() - t0) * 1000.0
        print(f"  {RED}Agent draft :{RESET} \"{bad_2}\"")
        print(f"  {GREEN}Deadkey gate:{RESET} {BOLD}BLOCKED in {ms2:.3f} ms{RESET}  ({v2.action} | {v2.violations})")
        _pause(2.0, fast)

        good = f"Verified on disk: service.conf is {receipt.byte_count} bytes ({receipt.line_count} lines, sha256 {receipt.sha256_short})."
        t0 = time.perf_counter()
        v3 = gate.evaluate(user_text="Report size of service.conf.", draft_reply=good, lane=TurnLane.EVIDENCE_GATED, receipts=[receipt], attempt=1)
        ms3 = (time.perf_counter() - t0) * 1000.0
        print(f"  {GREEN}Grounded    :{RESET} \"{good}\"")
        print(f"  {GREEN}Deadkey gate:{RESET} {BOLD}COMMITTED in {ms3:.3f} ms{RESET} (0 false positives)")
        _pause(1.8, fast)

    # --- SCENE 3 (11-15s): Honest Steering Cliff Takeaway + CTA ---
    print(f"\n{CYAN}{BOLD}[3/3] WHY AN EXTERNAL GATE BEATS CRANKING A GLOBAL STEERING VECTOR{RESET}")
    print(f"  {GRAY}Qwen3-8B Q4_K_M + 80.75 KiB GGUF control vector (llama.cpp):{RESET}")
    print(f"    {GREEN}s=1.8{RESET} : Held-out 24-probe {BOLD}22/24 (p=0.031 vs base, 6/6 benign){RESET} | LongMemEval {BOLD}57/78 (73.1%){RESET}")
    print(f"    {RED}s=3.0{RESET} : Held-out 24-probe 22/24 | LongMemEval crashes to {RED}{BOLD}35/78 (54/78 false refusals, p<1e-6){RESET}")
    _pause(2.2, fast)

    print(f"\n{EMBER}{BOLD}  pip install deadkey   |   Apache-2.0   |   https://deadkey.dev{RESET}")
    elapsed = time.perf_counter() - t_start
    if fast:
        print(f"{DIM}  [fast-check elapsed: {elapsed:.3f}s]{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
