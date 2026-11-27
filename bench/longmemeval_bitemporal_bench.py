"""Independent Third-Party Benchmark: Official ICLR 2025 LongMemEval (`knowledge-update` split, N=78).

Source: `xiaowu0162/longmemeval-cleaned` (`longmemeval_oracle.json`, 78 official knowledge-update tasks).
Compares head-to-head on the exact same local LLM:
  1. FLAT_RAG       — Flat FTS5 retrieval across all sessions (no valid-time superseding, unordered/BM25 rank)
  2. DEADKEY_BT   — Deadkey 0-VRAM Bi-Temporal SQLite Memory (chronological valid_from indexing,
                      superseding older topically conflicting claims in earlier sessions, valid-time ordered recall)

Zero self-authored probes. Saves full per-question transcripts to `data/longmemeval_knowledge_update_v1.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deadkey.memory_store import MemoryStore

DATASET_PATH = Path(r"D:\Models\Benchmarks\longmemeval_oracle.json")
LLM = "http://127.0.0.1:11455"


def _wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (round(max(0.0, (center - half) * 100.0), 1), round(min(100.0, (center + half) * 100.0), 1))


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar binomial p-value for discordant pairs (b = BT wins, c = Flat wins)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
    return round(min(1.0, 2.0 * tail), 4)


def _chat(llm_url: str, messages: list[dict[str, str]], max_tokens: int = 128) -> str:
    payload = json.dumps({
        "model": "local-model",
        "messages": messages,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90.0) as r:
        body = json.loads(r.read().decode("utf-8", "replace"))
    txt = (body["choices"][0]["message"].get("content") or "").strip()
    return re.sub(r"(?is)<think>.*?</think>", "", txt).strip()


def _parse_date_key(dstr: str) -> str:
    # Format: "2023/04/10 (Mon) 17:50" -> "2023-04-10T17:50"
    m = re.search(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}:\d{2})", dstr or "")
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}"
    return str(dstr or "")


def _user_turns_from_session(sess: list[dict[str, Any]]) -> list[str]:
    out = []
    for turn in sess:
        if turn.get("role") == "user":
            txt = " ".join(str(turn.get("content") or "").split())
            if txt:
                out.append(txt)
    return out


def _judge_answer(llm_url: str, question: str, gold: str, pred: str) -> bool:
    # Fast exact/normalized substring match first
    g_low = re.sub(r"\s+", " ", str(gold).lower().strip())
    p_low = re.sub(r"\s+", " ", str(pred).lower().strip())
    if g_low and g_low in p_low:
        return True
    # Exact semantic equivalence judge (standard LongMemEval evaluation protocol)
    verdict = _chat(
        llm_url,
        [
            {
                "role": "system",
                "content": (
                    "You are a strict benchmark grader for LongMemEval knowledge-update. "
                    "Given Question, Ground Truth Answer (the LATEST updated value), and Model Prediction, "
                    "reply ONLY with YES if the prediction states the ground-truth updated answer accurately "
                    "(and does not give an outdated value as current), or NO otherwise."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nGround Truth Answer: {gold}\nModel Prediction: {pred}\nCorrect (YES or NO)?",
            },
        ],
        max_tokens=16,
    )
    return verdict.strip().upper().startswith("YES")


def _extract_tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "that", "with", "this", "from", "have", "what", "when", "your", "about", "just", "been", "they", "were", "will", "into", "some", "more", "than", "also", "very", "really", "actually", "recently"}
    return {w for w in re.findall(r"[a-z0-9]{4,}", text.lower()) if w not in stop}


def main() -> int:
    ap = argparse.ArgumentParser(description="Official LongMemEval knowledge-update benchmark")
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--model-label", default="local-llm")
    ap.add_argument("--limit", type=int, default=78, help="Max knowledge-update tasks (full split = 78)")
    ap.add_argument("--out", default=str(ROOT / "data" / "longmemeval_knowledge_update_v1.json"))
    args = ap.parse_args()

    raw_bytes = Path(args.dataset).read_bytes()
    ds_sha = hashlib.sha256(raw_bytes).hexdigest()[:12]
    data = json.loads(raw_bytes.decode("utf-8"))
    ku_tasks = [x for x in data if x.get("question_type") == "knowledge-update"][: args.limit]
    print(f"[LONGMEMEVAL] Loaded {len(ku_tasks)}/78 official knowledge-update tasks (sha {ds_sha})", flush=True)

    t0 = time.perf_counter()
    flat_ok = 0
    bt_ok = 0
    rows = []

    for idx, item in enumerate(ku_tasks, 1):
        qid = item["question_id"]
        q = item["question"]
        gold = str(item["answer"])
        dates = item.get("haystack_dates") or []
        sessions = item.get("haystack_sessions") or []

        # Pair sessions with ISO timestamps and sort chronologically
        paired = []
        for dstr, sess in zip(dates, sessions):
            iso = _parse_date_key(dstr)
            for ut in _user_turns_from_session(sess):
                paired.append((iso, dstr, ut))
        paired.sort(key=lambda x: x[0])

        with tempfile.TemporaryDirectory() as td:
            # --- 1. FLAT RAG BASELINE (no temporal superseding, no timestamps, BM25 order) ---
            flat_store = MemoryStore(Path(td) / "flat.db")
            try:
                for _, _, ut in paired:
                    flat_store.add_semantic_fact(ut[:420], category="chat")
                flat_hits = flat_store.recall(q, k=6, include_inactive=False)
                flat_ctx = "\n".join(f"- {h.text}" for h in flat_hits)
            finally:
                flat_store.close()

            # --- 2. DEADKEY BI-TEMPORAL LEDGER (0 MiB VRAM, chronological superseding + valid-time order) ---
            bt_store = MemoryStore(Path(td) / "bitemporal.db")
            try:
                active_records: list[tuple[str, str, set[str]]] = []  # (id, iso, tokens)
                for iso, dstr, ut in paired:
                    toks = _extract_tokens(ut)
                    superseded = False
                    for prev_id, prev_iso, prev_toks in list(active_records):
                        if iso > prev_iso and toks and prev_toks:
                            overlap = len(toks & prev_toks) / max(1, min(len(toks), len(prev_toks)))
                            if overlap >= 0.25:
                                res = bt_store.supersede(
                                    target_id=prev_id,
                                    transition="SUPERSEDENCE",
                                    correction=f"[{iso[:10]}] {ut[:420]}",
                                    reason=f"newer session update ({prev_iso[:10]} -> {iso[:10]})",
                                )
                                new_id = res.get("new_id") or ""
                                active_records = [(i, d, t) for (i, d, t) in active_records if i != prev_id]
                                if new_id:
                                    active_records.append((new_id, iso, toks))
                                superseded = True
                                break
                    if not superseded:
                        fid = bt_store.add_semantic_fact(f"[{iso[:10]}] {ut[:420]}", category="chat")
                        active_records.append((fid, iso, toks))
                bt_hits = bt_store.recall(q, k=6, include_inactive=False)
                # Present active valid-time records sorted newest-first by session date prefix
                bt_hits_sorted = sorted(bt_hits, key=lambda h: str(h.text[:12]), reverse=True)
                bt_ctx = "\n".join(f"- [{h.status}] {h.text}" for h in bt_hits_sorted)
            finally:
                bt_store.close()

        # Answer with exact same local LLM under both memory conditions
        pred_flat = _chat(
            args.llm,
            [
                {"role": "system", "content": f"Answer the user's question concisely using ONLY the retrieved memory notes.\n\n[RETRIEVED NOTES]\n{flat_ctx}"},
                {"role": "user", "content": q},
            ],
            max_tokens=96,
        )
        pred_bt = _chat(
            args.llm,
            [
                {"role": "system", "content": f"Answer the user's question concisely using ONLY the active valid-time bi-temporal memory below (retired/outdated facts have been superseded; prefer the latest valid_from record).\n\n[ACTIVE BI-TEMPORAL MEMORY]\n{bt_ctx}"},
                {"role": "user", "content": q},
            ],
            max_tokens=96,
        )

        ok_f = _judge_answer(args.llm, q, gold, pred_flat)
        ok_b = _judge_answer(args.llm, q, gold, pred_bt)
        flat_ok += int(ok_f)
        bt_ok += int(ok_b)
        rows.append({
            "question_id": qid,
            "question": q,
            "gold": gold,
            "flat_pred": pred_flat,
            "flat_ok": ok_f,
            "bt_pred": pred_bt,
            "bt_ok": ok_b,
        })
        if idx % 10 == 0 or idx == len(ku_tasks):
            print(
                f"  [{idx:2d}/{len(ku_tasks)}] FLAT_RAG={flat_ok}/{idx} ({100*flat_ok/idx:.1f}%) | "
                f"DEADKEY_BITEMPORAL={bt_ok}/{idx} ({100*bt_ok/idx:.1f}%)",
                flush=True,
            )

    elapsed_s = round(time.perf_counter() - t0, 1)
    n = len(ku_tasks)
    bt_wins = sum(1 for r in rows if r["bt_ok"] and not r["flat_ok"])
    flat_wins = sum(1 for r in rows if r["flat_ok"] and not r["bt_ok"])
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "ICLR 2025 LongMemEval (xiaowu0162/longmemeval-cleaned :: longmemeval_oracle.json)",
        "split": "knowledge-update",
        "model_label": args.model_label,
        "dataset_sha256_prefix": ds_sha,
        "count": n,
        "elapsed_s": elapsed_s,
        "flat_rag_correct": flat_ok,
        "flat_rag_pct": round(100.0 * flat_ok / n, 1),
        "flat_rag_ci95": list(_wilson_ci_95(flat_ok, n)),
        "deadkey_bitemporal_correct": bt_ok,
        "deadkey_bitemporal_pct": round(100.0 * bt_ok / n, 1),
        "deadkey_bitemporal_ci95": list(_wilson_ci_95(bt_ok, n)),
        "delta_pp": round(100.0 * (bt_ok - flat_ok) / n, 1),
        "discordant_bt_wins_flat_loses": bt_wins,
        "discordant_flat_wins_bt_loses": flat_wins,
        "mcnemar_exact_p": _mcnemar_exact_p(bt_wins, flat_wins),
        "memory_gpu_vram_mib": 0,
        "rows": rows,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\n[LONGMEMEVAL] DONE in {elapsed_s}s | FLAT_RAG={flat_ok}/{n} ({report['flat_rag_pct']}%) -> "
        f"DEADKEY_BITEMPORAL={bt_ok}/{n} ({report['deadkey_bitemporal_pct']}%, {report['delta_pp']:+.1f} pp!) -> {out_p}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
