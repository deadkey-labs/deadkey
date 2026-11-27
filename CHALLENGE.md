# Break Deadkey — Open Adversarial Probe Challenge

**We do not claim 100% universal perfection.** Our 24-probe blind gauntlet demonstrates *runtime invariance* across 10+ local models (`8B–31B`), not that the gate is unbreakable.

If you can write an adversarial prompt + model output that sneaks an **unverified disk claim, fake tool/action execution, or stale superseded fact** past Deadkey's 2-Tier Epistemic Gate — **or** triggers a false positive on an ordinary benign prompt — **open a PR or Issue and help harden the kernel.**

## How to Submit a Probe That Breaks Deadkey

1. Run the existing verification suite locally (`< 2 seconds`, no GPU needed):
   ```bash
   python -m bench.demo_break_vs_hold
   python -m bench.reproduce_all --quick
   ```
2. Inspect the current 24-probe blind suite in `bench/blind_abstract_gauntlet.py`.
3. Add a minimal reproducible test case (JSON or Python test) specifying:
   - **Category:** `paraphrased_fabrication`, `stale_memory_leak`, `implicit_action`, `cot_rationalization`, or `benign_false_positive`
   - **Input Context:** active bi-temporal memories + `EvidenceReceipt`s (if any)
   - **Model Draft:** exact text that slipped past Tier 1 (`EpistemicCommitGate`) or Tier 1+2 (`SemanticEpistemicCritic`)
   - **Expected Verdict:** `BLOCK` (if fabricated/stale) or `PASS` (if benign)
4. Open a PR adding your probe to `bench/community_probes.json` (remember DCO sign-off: `git commit -s` — see [CONTRIBUTING.md](CONTRIBUTING.md)).

## What Happens When You Break It
- **Permanent Attribution:** Every verified bypass or false-positive PR is merged into the public held-out evaluation suite and credited in the **Deadkey Epistemic Leaderboard & Hall of Breakers**.
- **No Moving the Goalposts:** We publish the before/after scores on community-submitted probes openly alongside our own receipts.

*Weights rationalize. Deadkey doesn't — especially when smart skeptics try to break the latch.*
