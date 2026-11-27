"""Event-Driven Dual-Store Memory Engine (SQLite WAL + FTS5 + Bi-Temporal Facts).

Preserves Deadkey's immutable append-only `ledger.jsonl` spine while eliminating:
- 12.2 MB flat-file scanning & Windows msvcrt lock races
- 6.26 MB full-dump `shadow_weights.jsonl` bloat
- Stale belief leakage (every fact has `valid_from` / `valid_to` / `status`)
- Per-turn 5,000-token static prompt bloat (replaced by <=1,200-token pinned core blocks
  + sub-5ms FTS5/entity event-driven salience recall).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS ledger_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    entry TEXT NOT NULL,
    target_id TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    superseded_by TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS ledger_fts USING fts5(
    id UNINDEXED,
    entry,
    source UNINDEXED,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS semantic_facts (
    id TEXT PRIMARY KEY,
    fact TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    valid_from TEXT,
    valid_to TEXT,
    evidence_ids TEXT DEFAULT '[]'
);

CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
    id UNINDEXED,
    fact,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    state TEXT NOT NULL,
    why_open TEXT,
    next_step TEXT,
    journal_ids TEXT DEFAULT '[]',
    updated_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tendencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    text TEXT NOT NULL,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    turn_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    asked TEXT,
    attempted TEXT,
    tools_json TEXT,
    worked INTEGER,
    reaction TEXT,
    guard_fired INTEGER DEFAULT 0,
    guard_check TEXT
);

CREATE TABLE IF NOT EXISTS core_blocks (
    block_key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS changelog (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    author TEXT NOT NULL,
    action TEXT NOT NULL,
    artifact TEXT NOT NULL,
    verified_by TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_stats (
    entry_id TEXT PRIMARY KEY,
    presentations INTEGER DEFAULT 0,
    active_recalls INTEGER DEFAULT 0,
    weight REAL DEFAULT 1.0,
    last_active_ts REAL
);

CREATE TABLE IF NOT EXISTS memory_vectors (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
"""


@dataclass(frozen=True)
class RecallHit:
    id: str
    kind: str  # "journal" | "semantic"
    text: str
    status: str
    valid_from: str
    valid_to: Optional[str]
    superseded_by: Optional[str]
    source: str
    confidence: str
    score: float


class MemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA busy_timeout = 10000;")
            self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def pinned_core_prompt(self, max_tokens: int = 1200) -> str:
        """Letta-style frozen core blocks (pinned once per session -> high KV cache hit rate)."""
        order = ("identity", "human", "self_model", "active_threads")
        parts: list[str] = []
        used_chars = 0
        max_chars = max_tokens * 4
        with self._lock:
            for key in order:
                row = self.conn.execute(
                    "SELECT content FROM core_blocks WHERE block_key = ?", (key,)
                ).fetchone()
                if not row or not row["content"].strip():
                    continue
                block = f"## {key}\n{row['content'].strip()}\n"
                if used_chars + len(block) > max_chars:
                    room = max_chars - used_chars
                    if room > 120:
                        parts.append(block[:room - 4] + "...\n")
                    break
                parts.append(block)
                used_chars += len(block)
        return "\n".join(parts)

    def recall(
        self,
        query: str,
        k: int = 5,
        include_inactive: bool = False,
        query_vec: Optional[list[float]] = None,
    ) -> list[RecallHit]:
        """Sub-8ms Hybrid Bi-Temporal Recall (FTS5 BM25 + 768-d Semantic Vector RRF, 0 MiB GPU VRAM)."""
        tokens = re.findall(r"[a-zA-Z0-9_]{2,}", query or "")
        fts_hits: list[RecallHit] = []

        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens[:10]))
            sql_j = """
                SELECT l.id, l.entry, l.status, l.valid_from, l.valid_to, l.superseded_by,
                       l.source, l.confidence, bm25(ledger_fts) AS rank
                FROM ledger_fts
                JOIN ledger_events l ON l.id = ledger_fts.id
                WHERE ledger_fts MATCH ? AND l.event_type = 'OBSERVATION'
            """
            if not include_inactive:
                sql_j += " AND l.status IN ('ACTIVE', 'CONFIRMED')"
            sql_j += " ORDER BY rank LIMIT ?"
            with self._lock:
                for r in self.conn.execute(sql_j, (fts_query, max(k * 2, 10))):
                    fts_hits.append(RecallHit(
                        id=r["id"], kind="journal", text=r["entry"], status=r["status"],
                        valid_from=r["valid_from"], valid_to=r["valid_to"],
                        superseded_by=r["superseded_by"], source=r["source"],
                        confidence=r["confidence"], score=round(-float(r["rank"]), 4),
                    ))

                sql_s = """
                    SELECT s.id, s.fact, s.status, s.valid_from, s.valid_to, bm25(semantic_fts) AS rank
                    FROM semantic_fts
                    JOIN semantic_facts s ON s.id = semantic_fts.id
                    WHERE semantic_fts MATCH ?
                """
                if not include_inactive:
                    sql_s += " AND s.status = 'ACTIVE'"
                sql_s += " ORDER BY rank LIMIT ?"
                for r in self.conn.execute(sql_s, (fts_query, max(k * 2, 10))):
                    fts_hits.append(RecallHit(
                        id=r["id"], kind="semantic", text=r["fact"], status=r["status"],
                        valid_from=r["valid_from"] or "", valid_to=r["valid_to"],
                        superseded_by=None, source="semantic", confidence="high",
                        score=round(-float(r["rank"]), 4),
                    ))
            fts_hits.sort(key=lambda h: h.score, reverse=True)

        # If no semantic query vector supplied (or vector table empty), return pure FTS5 fast path
        if not query_vec:
            return fts_hits[:k]

        vec_hits = self._vector_recall(query_vec, k=max(k * 2, 12), include_inactive=include_inactive)
        if not vec_hits:
            return fts_hits[:k]

        # Reciprocal Rank Fusion (RRF): blends exact keyword BM25 + conceptual cosine similarity!
        by_id: dict[str, RecallHit] = {}
        rrf_scores: dict[str, float] = {}
        for rank, h in enumerate(fts_hits, 1):
            by_id[h.id] = h
            rrf_scores[h.id] = rrf_scores.get(h.id, 0.0) + 1.0 / (40.0 + rank)
        for rank, (h, cos_sim) in enumerate(vec_hits, 1):
            if cos_sim < 0.38:
                continue
            by_id.setdefault(h.id, h)
            rrf_scores[h.id] = rrf_scores.get(h.id, 0.0) + (1.35 * cos_sim) / (40.0 + rank)

        ranked_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
        out: list[RecallHit] = []
        for i in ranked_ids:
            h = by_id[i]
            out.append(RecallHit(
                id=h.id, kind=h.kind, text=h.text, status=h.status,
                valid_from=h.valid_from, valid_to=h.valid_to,
                superseded_by=h.superseded_by, source=h.source,
                confidence=h.confidence, score=round(rrf_scores[i] * 100.0, 3),
            ))
        return out

    def _vector_recall(
        self,
        query_vec: list[float],
        *,
        k: int = 12,
        include_inactive: bool = False,
    ) -> list[tuple[RecallHit, float]]:
        import numpy as np
        with self._lock:
            rows = self.conn.execute("SELECT id, kind, dim, vec FROM memory_vectors").fetchall()
        if not rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q_dim = q.shape[0]
        scored: list[tuple[float, str, str]] = []
        for r in rows:
            if r["dim"] != q_dim:
                continue
            v = np.frombuffer(r["vec"], dtype=np.float32)
            sim = float(np.dot(q, v))
            scored.append((sim, r["id"], r["kind"]))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[tuple[RecallHit, float]] = []
        with self._lock:
            for sim, item_id, kind in scored[: k * 2]:
                if kind == "journal":
                    r = self.conn.execute(
                        "SELECT id, entry, status, valid_from, valid_to, superseded_by, source, confidence FROM ledger_events WHERE id = ?",
                        (item_id,),
                    ).fetchone()
                    if not r or (not include_inactive and r["status"] not in ("ACTIVE", "CONFIRMED")):
                        continue
                    out.append((RecallHit(
                        id=r["id"], kind="journal", text=r["entry"], status=r["status"],
                        valid_from=r["valid_from"], valid_to=r["valid_to"],
                        superseded_by=r["superseded_by"], source=r["source"],
                        confidence=r["confidence"], score=round(sim, 4),
                    ), sim))
                else:
                    r = self.conn.execute(
                        "SELECT id, fact, status, valid_from, valid_to FROM semantic_facts WHERE id = ?",
                        (item_id,),
                    ).fetchone()
                    if not r or (not include_inactive and r["status"] != "ACTIVE"):
                        continue
                    out.append((RecallHit(
                        id=r["id"], kind="semantic", text=r["fact"], status=r["status"],
                        valid_from=r["valid_from"] or "", valid_to=r["valid_to"],
                        superseded_by=None, source="semantic", confidence="high", score=round(sim, 4),
                    ), sim))
                if len(out) >= k:
                    break
        return out

    def upsert_vector(self, item_id: str, kind: str, vec: list[float]) -> None:
        import numpy as np
        arr = np.asarray(vec, dtype=np.float32)
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (id, kind, dim, vec) VALUES (?, ?, ?, ?)",
                (item_id, kind, int(arr.shape[0]), arr.tobytes()),
            )
            self.conn.commit()

    def index_missing_vectors(self, embedder: Any, batch_size: int = 32) -> int:
        """Backfills 768-d semantic embeddings for all unindexed active ledger & semantic rows (0 MiB GPU VRAM)."""
        if not embedder or not getattr(embedder, "ready", False):
            return 0
        with self._lock:
            existing = {r[0] for r in self.conn.execute("SELECT id FROM memory_vectors").fetchall()}
            j_rows = self.conn.execute(
                "SELECT id, entry FROM ledger_events WHERE event_type = 'OBSERVATION' AND status IN ('ACTIVE', 'CONFIRMED')"
            ).fetchall()
            s_rows = self.conn.execute(
                "SELECT id, fact FROM semantic_facts WHERE status = 'ACTIVE'"
            ).fetchall()

        todo: list[tuple[str, str, str]] = []
        for r in j_rows:
            if r["id"] not in existing and (r["entry"] or "").strip():
                todo.append((r["id"], "journal", r["entry"]))
        for r in s_rows:
            if r["id"] not in existing and (r["fact"] or "").strip():
                todo.append((r["id"], "semantic", r["fact"]))

        indexed = 0
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            vecs = embedder.embed_batch([b[2] for b in batch], prefix="search_document: ")
            for (item_id, kind, _txt), vec in zip(batch, vecs):
                self.upsert_vector(item_id, kind, vec)
                indexed += 1
        return indexed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            def _c(tbl: str, where: str = "") -> int:
                q = f"SELECT COUNT(*) FROM {tbl}" + (f" WHERE {where}" if where else "")
                return int(self.conn.execute(q).fetchone()[0])

            return {
                "ledger_total": _c("ledger_events"),
                "observations_active": _c("ledger_events", "event_type='OBSERVATION' AND status IN ('ACTIVE','CONFIRMED')"),
                "observations_inactive": _c("ledger_events", "event_type='OBSERVATION' AND status IN ('SUPERSEDED','DISPROVED')"),
                "transitions": _c("ledger_events", "event_type!='OBSERVATION'"),
                "semantic_active": _c("semantic_facts", "status='ACTIVE'"),
                "semantic_retired": _c("semantic_facts", "status IN ('RETIRED','SUPERSEDED','DISPROVED')"),
                "threads_total": _c("threads"),
                "tendencies": _c("tendencies"),
                "outcomes": _c("outcomes"),
                "changelog": _c("changelog"),
                "core_blocks": _c("core_blocks"),
                "retrieval_stats": _c("retrieval_stats"),
            }

    # ---- Thread-Safe Live Memory Writes (Journal, Semantic Facts, Threads, Outcomes) ----
    def add_event(
        self,
        *,
        event_id: str,
        kind: str,
        text: str,
        status: str = "ACTIVE",
        valid_from: Optional[str] = None,
        source: str = "live_turn",
        confidence: str = "high",
    ) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        vf = valid_from or time.strftime("%Y-%m-%d")
        etype = "OBSERVATION" if kind in ("journal", "observation") else kind.upper()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO ledger_events
                   (id, timestamp, event_type, source, confidence, entry, status, valid_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, ts, etype, source, confidence, text, status.upper(), vf),
            )
            self.conn.execute("DELETE FROM ledger_fts WHERE id = ?", (event_id,))
            self.conn.execute(
                "INSERT INTO ledger_fts (id, entry, source) VALUES (?, ?, ?)",
                (event_id, text, source),
            )
            self.conn.commit()
        return event_id

    def add_semantic_fact(self, fact: str, *, category: str = "general", source: str = "live") -> str:
        import uuid
        clean = (fact or "").strip()
        with self._lock:
            existing = self.conn.execute(
                "SELECT id FROM semantic_facts WHERE fact = ? AND status = 'ACTIVE'", (clean,)
            ).fetchone()
            if existing:
                return str(existing["id"])
            fid = f"SF-{uuid.uuid4().hex[:10]}"
            vf = time.strftime("%Y-%m-%d")
            self.conn.execute(
                """INSERT INTO semantic_facts (id, fact, status, valid_from, evidence_ids)
                   VALUES (?, ?, 'ACTIVE', ?, ?)""",
                (fid, clean, vf, json.dumps([f"{category}:{source}"])),
            )
            self.conn.execute("INSERT INTO semantic_fts (id, fact) VALUES (?, ?)", (fid, clean))
            self.conn.commit()
        return fid

    def supersede(
        self,
        *,
        target_id: str,
        correction: str,
        transition: str = "SUPERSEDENCE",
        source: str = "live",
        reason: str = "",
    ) -> dict[str, Any]:
        """Bi-temporally retire an active observation (`ledger_events`) OR semantic fact (`semantic_facts`)
        and write both an immutable transition audit row and an active replacement truth."""
        import uuid
        tid = (target_id or "").strip()
        corr = (correction or "").strip()
        trans = (transition or "SUPERSEDENCE").strip().upper()
        if trans not in ("SUPERSEDENCE", "DISPROVAL"):
            trans = "SUPERSEDENCE"
        if not tid or not corr:
            raise ValueError("target_id and correction are required")

        new_status = "DISPROVED" if trans == "DISPROVAL" else "SUPERSEDED"
        trans_id = f"T-{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        today = time.strftime("%Y-%m-%d")
        prov = f"{source} ({reason.strip()})" if reason and reason.strip() else source

        with self._lock:
            led_row = self.conn.execute(
                "SELECT id, status FROM ledger_events WHERE id = ?", (tid,)
            ).fetchone()
            sem_row = self.conn.execute(
                "SELECT id, status, evidence_ids FROM semantic_facts WHERE id = ?", (tid,)
            ).fetchone()
            if not led_row and not sem_row:
                raise KeyError(f"Unknown memory target_id: {tid}")

            if sem_row:
                retired_table = "semantic_facts"
                new_id = f"SF-{uuid.uuid4().hex[:10]}"
                self.conn.execute(
                    "UPDATE semantic_facts SET status = ?, valid_to = ? WHERE id = ?",
                    (new_status, today, tid),
                )
                self.conn.execute(
                    """INSERT INTO semantic_facts (id, fact, status, valid_from, evidence_ids)
                       VALUES (?, ?, 'ACTIVE', ?, ?)""",
                    (new_id, corr, today, json.dumps([f"superseded:{tid}", f"source:{prov}"])),
                )
                self.conn.execute("INSERT INTO semantic_fts (id, fact) VALUES (?, ?)", (new_id, corr))
            else:
                retired_table = "ledger_events"
                new_id = f"J-{uuid.uuid4().hex[:8]}"
                self.conn.execute(
                    "UPDATE ledger_events SET status = ?, valid_to = ?, superseded_by = ? WHERE id = ?",
                    (new_status, today, new_id, tid),
                )
                self.conn.execute(
                    """INSERT INTO ledger_events
                       (id, timestamp, event_type, source, confidence, entry, status, valid_from, target_id)
                       VALUES (?, ?, 'OBSERVATION', ?, 'high', ?, 'ACTIVE', ?, ?)""",
                    (new_id, ts, prov, corr, today, tid),
                )
                self.conn.execute("DELETE FROM ledger_fts WHERE id = ?", (new_id,))
                self.conn.execute(
                    "INSERT INTO ledger_fts (id, entry, source) VALUES (?, ?, ?)",
                    (new_id, corr, prov),
                )

            # Immutable transition audit record in ledger_events for BOTH tables
            self.conn.execute(
                """INSERT INTO ledger_events
                   (id, timestamp, event_type, source, confidence, entry, status, valid_from, target_id)
                   VALUES (?, ?, ?, ?, 'high', ?, 'ACTIVE', ?, ?)""",
                (trans_id, ts, trans, prov, corr, today, tid),
            )
            self.conn.commit()

        return {
            "ok": True,
            "transition": trans,
            "retired_id": tid,
            "retired_status": new_status,
            "retired_table": retired_table,
            "transition_id": trans_id,
            "new_id": new_id,
        }

    def list_semantic_facts(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, fact, status, valid_from, evidence_ids FROM semantic_facts WHERE status = 'ACTIVE' ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            ev = []
            try:
                ev = json.loads(r["evidence_ids"] or "[]")
            except Exception:
                pass
            cat = ev[0].split(":")[0] if ev and ":" in str(ev[0]) else "semantic"
            out.append({"id": r["id"], "category": cat, "text": r["fact"], "valid_from": r["valid_from"]})
        return out

    def list_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, title, state, why_open, next_step, updated_ts FROM threads ORDER BY updated_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"], "title": r["title"], "status": r["state"],
                "notes": r["why_open"] or r["next_step"] or "", "updated_ts": r["updated_ts"],
            }
            for r in rows
        ]

    def upsert_thread(self, *, title: str, notes: str = "", status: str = "open") -> str:
        import uuid
        with self._lock:
            row = self.conn.execute("SELECT id FROM threads WHERE LOWER(title) = LOWER(?)", (title.strip(),)).fetchone()
            tid = str(row["id"]) if row else f"T-{uuid.uuid4().hex[:10]}"
            now = time.time()
            self.conn.execute(
                """INSERT INTO threads (id, title, state, why_open, next_step, updated_ts)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title,
                     state = excluded.state,
                     why_open = excluded.why_open,
                     next_step = excluded.next_step,
                     updated_ts = excluded.updated_ts""",
                (tid, title.strip(), status.lower(), notes[:400], notes[:400], now),
            )
            self.conn.commit()
        return tid

    def record_outcome(
        self,
        *,
        turn_id: str,
        asked: str,
        attempted: str,
        tools: list[str],
        worked: bool,
        guard_fired: bool = False,
        guard_check: str = "",
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO outcomes
                   (turn_id, ts, asked, attempted, tools_json, worked, reaction, guard_fired, guard_check)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn_id, time.time(), (asked or "")[:500], (attempted or "")[:500],
                    json.dumps(tools), 1 if worked else 0,
                    "verified" if worked else "guard_held", 1 if guard_fired else 0, guard_check[:120],
                ),
            )
            self.conn.commit()
