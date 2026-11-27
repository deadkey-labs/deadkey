"""Sleep-Time Compute & Sandboxed Nightly LoRA/DPO Consolidation Engine.

Implements safe, compute-frugal overnight growth:
1. Distills verified outcomes, tendencies, and supersession transitions from `MemoryStore`.
2. Builds a local-only DPO preference dataset (`sleep_dpo_pairs.jsonl`) for overnight
   RTX 5090 LoRA consolidation (rank 8/16, ~3-8 min), turning Disk-Equals-Truth habits
   from nagging prompt rules into clean weight reflexes — while base weights remain frozen.
3. Enforces a SHA-gated human approval queue (`approvals.jsonl`): NO LoRA adapter or
   self-authored tool ever activates without passing `hill_climb.py` + the operator's approval.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .memory_store import MemoryStore


@dataclass
class SleepReport:
    ts: float
    outcomes_analyzed: int
    positive_examples: int
    negative_drift_examples: int
    dpo_pairs_exported: int
    dpo_output_path: str
    dpo_sha256: str
    self_model_evidence_lines: list[str]
    approval_proposal_id: str | None


def run_sleep_consolidation(
    store: MemoryStore,
    output_dir: str | Path,
    *,
    approvals_path: str | Path | None = None,
) -> SleepReport:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = store.conn

    # 1. Harvest verified positive vs drift/guard-fired turns from outcomes
    rows = cur.execute(
        "SELECT turn_id, ts, asked, attempted, tools_json, worked, reaction, guard_fired, guard_check "
        "FROM outcomes ORDER BY ts DESC LIMIT 500"
    ).fetchall()

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for r in rows:
        asked = (r["asked"] or "").strip()
        attempted = (r["attempted"] or "").strip()
        if not asked or not attempted:
            continue
        tools = json.loads(r["tools_json"] or "[]")
        all_ok = all((t.get("ok") if isinstance(t, dict) else True) for t in tools) if tools else False
        gcheck = (r["guard_check"] or "").strip().lower()
        is_real_guard_fail = bool(r["guard_fired"]) or (
            bool(gcheck) and gcheck not in ("verified", "ok", "pass", "neutral", "none")
        )
        tool_failed = bool(tools) and (r["worked"] == 0 or not all_ok)
        if not is_real_guard_fail and not tool_failed and r["reaction"] != "neg" and (
            r["worked"] == 1 or gcheck == "verified" or r["reaction"] == "pos"
        ):
            positives.append({"asked": asked, "response": attempted, "tools": tools, "turn_id": r["turn_id"]})
        elif is_real_guard_fail or tool_failed:
            negatives.append({
                "asked": asked, "response": attempted,
                "reason": r["guard_check"] or ("failed_tool" if tool_failed else "guard_fired"),
                "turn_id": r["turn_id"],
            })

    # 2. Also harvest explicit SUPERSEDENCE / DISPROVAL lessons from the ledger
    corrections = cur.execute(
        "SELECT id, timestamp, event_type, target_id, entry FROM ledger_events "
        "WHERE event_type IN ('SUPERSEDENCE', 'DISPROVAL') ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()

    # 3. Synthesize clean DPO pairs (chosen = truth-verified habit, rejected = drift/unverified habit)
    pairs: list[dict[str, Any]] = []
    for neg in negatives[:80]:
        pairs.append({
            "kind": "epistemic_correction",
            "prompt": neg["asked"],
            "rejected": neg["response"],
            "chosen": (
                "I don't have a verified tool receipt for that yet — let me run the exact read/probe "
                "and report only what disk returns verbatim."
            ),
            "meta": {"source_turn": neg["turn_id"], "failure_class": neg["reason"]},
        })
    for c in corrections[:40]:
        pairs.append({
            "kind": "belief_revision",
            "prompt": f"Recall the corrected state regarding {c['target_id']}.",
            "chosen": f"Note the {c['event_type'].lower()} ({c['id']}): {c['entry']}",
            "rejected": f"Repeating the superseded record {c['target_id']} without naming the correction.",
            "meta": {"ledger_id": c["id"], "target_id": c["target_id"]},
        })

    dpo_path = out_dir / "sleep_dpo_pairs.jsonl"
    raw_dpo = "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + ("\n" if pairs else "")
    dpo_path.write_text(raw_dpo, encoding="utf-8")
    dpo_sha = hashlib.sha256(raw_dpo.encode("utf-8")).hexdigest()[:12]

    # 4. Build concise behavioral self-model evidence lines (bounded, evidence-backed)
    tendencies = cur.execute("SELECT text, evidence FROM tendencies ORDER BY id DESC LIMIT 5").fetchall()
    ev_lines = [
        f"outcomes_analyzed={len(rows)} verified_positive={len(positives)} drift_or_neg={len(negatives)}",
        f"ledger_corrections_indexed={len(corrections)} dpo_pairs_ready={len(pairs)}",
    ]
    for t in tendencies[:3]:
        ev_lines.append(f"tendency [{t['evidence'] or 'record'}]: {t['text'][:140]}")

    # 5. Queue a SHA-gated approval entry (never auto-activates!)
    prop_id = None
    if approvals_path is not None and pairs:
        ap = Path(approvals_path)
        ap.parent.mkdir(parents=True, exist_ok=True)
        existing = [ln for ln in ap.read_text(encoding="utf-8").splitlines() if ln.strip()] if ap.exists() else []
        prop_id = f"SLEEP-{len(existing) + 1:04d}"
        rec = {
            "id": prop_id,
            "ts": time.time(),
            "kind": "sleep_lora_dataset",
            "status": "pending_approval",
            "artifact": str(dpo_path),
            "sha256": dpo_sha,
            "pair_count": len(pairs),
            "note": "Ready for 5-min RTX 5090 QLoRA/DPO consolidation + sandbox hill_climb.py verification.",
        }
        with ap.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return SleepReport(
        ts=time.time(),
        outcomes_analyzed=len(rows),
        positive_examples=len(positives),
        negative_drift_examples=len(negatives),
        dpo_pairs_exported=len(pairs),
        dpo_output_path=str(dpo_path),
        dpo_sha256=dpo_sha,
        self_model_evidence_lines=ev_lines,
        approval_proposal_id=prop_id,
    )
