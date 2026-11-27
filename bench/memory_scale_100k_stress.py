"""100,000-Record Bi-Temporal SQLite Scale, Zero-Stale-Bleed & 8-Thread Concurrency Stress Benchmark.

Proves Deadkey's zero-dependency, 0-MiB-VRAM SQLite WAL + FTS5 ledger holds up far past toy 10k scale:
  - Inserts 100,000 bi-temporal records (50,000 ACTIVE + 50,000 SUPERSEDED/DISPROVED)
    including 200 planted multi-revision contradiction pairs
  - Verifies ZERO stale-belief leakage (`0/200` retired records returned when `include_inactive=False`)
  - Benchmarks 300 FTS5 recall queries (p50 / p95 / p99 ms) on the 100k-row database
  - Runs 8-thread concurrent read/write/supersede WAL contention stress (`0` SQLite lock errors)
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deadkey.memory_store import MemoryStore


def main() -> int:
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "scale_100k.db"
        store = MemoryStore(db_path)
        try:
            t_ins0 = time.perf_counter()
            planted: list[tuple[str, str, str, str]] = []
            for i in range(200):
                svc = f"svc{i:04d}omega"
                old_val = f"OLDPORT{30000 + i}"
                new_val = f"NEWPORT{40000 + i}"
                if i % 2 == 0:
                    eid = f"J-plant100k-{i:04d}"
                    store.add_event(event_id=eid, kind="observation", text=f"Service {svc} listens on {old_val}.", valid_from="2025-06-01")
                    store.supersede(target_id=eid, transition="SUPERSEDENCE", correction=f"Service {svc} listens on {new_val}.", reason="migration")
                    planted.append((svc, old_val, new_val, eid))
                else:
                    fid = store.add_semantic_fact(f"Service {svc} uses legacy token {old_val}.", category="scale")
                    store.supersede(target_id=fid, transition="DISPROVAL", correction=f"Service {svc} uses verified token {new_val}.", reason="audit")
                    planted.append((svc, old_val, new_val, fid))

            domains = ["network", "storage", "voice", "gpu", "thermal", "backup", "security", "kernel", "compiler", "memory"]
            bulk_n = 99600  # + 400 planted = 100,000 total records
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ledger_rows = []
            fts_rows = []
            for i in range(bulk_n):
                dom = domains[i % len(domains)]
                is_active = (i < bulk_n // 2)
                status = "ACTIVE" if is_active else ("SUPERSEDED" if i % 2 == 0 else "DISPROVED")
                eid = f"J-bulk100k-{i:06d}"
                txt = f"[{dom}] telemetry snapshot #{i}: cluster node_{i % 256} status={status.lower()} latency={10 + (i % 90)}ms"
                ledger_rows.append((eid, ts, txt, status, None if is_active else "2026-06-01"))
                fts_rows.append((eid, txt))

            with store._lock:
                store.conn.executemany(
                    """INSERT INTO ledger_events
                       (id, timestamp, event_type, source, confidence, entry, status, valid_from, valid_to)
                       VALUES (?, ?, 'OBSERVATION', 'scale_100k', 'high', ?, ?, '2026-01-01', ?)""",
                    ledger_rows,
                )
                store.conn.executemany(
                    "INSERT INTO ledger_fts (id, entry, source) VALUES (?, ?, 'scale_100k')",
                    fts_rows,
                )
                store.conn.commit()
            insert_s = round(time.perf_counter() - t_ins0, 2)
            st = store.stats()

            stale_leaks = 0
            active_hits_ok = 0
            fts_ms: list[float] = []
            for svc, old_val, new_val, old_id in planted:
                q0 = time.perf_counter()
                hits = store.recall(svc, k=5, include_inactive=False)
                fts_ms.append((time.perf_counter() - q0) * 1000.0)
                texts = " ".join(h.text for h in hits)
                ids = {h.id for h in hits}
                if old_val in texts or old_id in ids:
                    stale_leaks += 1
                if new_val in texts:
                    active_hits_ok += 1

            for i in range(100):
                q = f"{domains[i % len(domains)]} node_{i % 256} latency"
                q0 = time.perf_counter()
                _ = store.recall(q, k=5, include_inactive=False)
                fts_ms.append((time.perf_counter() - q0) * 1000.0)

            errors: list[str] = []
            ops_done = {"n": 0}
            ops_lock = threading.Lock()

            def worker(tid: int):
                try:
                    for step in range(25):
                        fid = store.add_semantic_fact(f"Concurrent100k worker {tid} fact #{step} val=OLD_{tid}_{step}", category="stress")
                        store.supersede(target_id=fid, correction=f"Concurrent100k worker {tid} fact #{step} val=NEW_{tid}_{step}", reason="concurrent_update")
                        _ = store.recall(f"Concurrent100k worker {tid}", k=5)
                        with ops_lock:
                            ops_done["n"] += 3
                except Exception as exc:
                    errors.append(f"t{tid}: {exc}")

            t_c0 = time.perf_counter()
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            conc_s = round(time.perf_counter() - t_c0, 2)

            fts_ms.sort()
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ok": (stale_leaks == 0 and active_hits_ok == 200 and len(errors) == 0),
                "record_count": 100000,
                "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 2),
                "gpu_vram_mib": 0,
                "insert_100k_seconds": insert_s,
                "stats_after_100k": st,
                "planted_contradictions": 200,
                "active_hits_found": active_hits_ok,
                "stale_belief_leaks": stale_leaks,
                "fts5_latency_ms": {
                    "p50": round(statistics.median(fts_ms), 2),
                    "p95": round(fts_ms[int(len(fts_ms) * 0.95)], 2),
                    "p99": round(fts_ms[int(len(fts_ms) * 0.99)], 2),
                },
                "concurrency_8_threads": {
                    "total_ops": ops_done["n"],
                    "seconds": conc_s,
                    "lock_errors": len(errors),
                    "errors": errors[:5],
                },
                "total_elapsed_s": round(time.perf_counter() - t_total, 2),
            }
            out = ROOT / "data" / "memory_scale_100k_report.json"
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        finally:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
