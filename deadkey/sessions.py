"""Clean SQLite Sessions Ledger — zero timestamp collapse, zero dropped speaker fields.

Fixes all three serving-layer defects from CHANGELOG.jsonl C-0014/C-0016/C-0031:
- Preserves exact per-message `ts` (never overwritten by archive stamp).
- Preserves `speaker` (`user` | `hermes` | `agent`) end-to-end.
- Attaches immutable Verification Receipts directly to each turn row.
- `close()` requires a concise summary in the agent's own words.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

SESSIONS_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'live',
    summary TEXT DEFAULT '',
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    closed_ts REAL
);

CREATE TABLE IF NOT EXISTS session_messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    speaker TEXT NOT NULL,
    lane TEXT DEFAULT '',
    content TEXT NOT NULL,
    ttfa_ms REAL,
    receipts_json TEXT DEFAULT '[]',
    verified INTEGER DEFAULT 1,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""


@dataclass
class SessionInfo:
    id: str
    title: str
    state: str
    summary: str
    created_ts: float
    updated_ts: float
    turns: int
    receipts: int


class SessionLedger:
    def __init__(self, db_path: str | Path):
        import threading
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA busy_timeout = 10000;")
            self.conn.executescript(SESSIONS_SQL)

    def close_db(self) -> None:
        with self._lock:
            self.conn.close()

    def _next_id(self) -> str:
        with self._lock:
            rows = self.conn.execute("SELECT id FROM sessions").fetchall()
            max_n = 0
            for r in rows:
                s = str(r["id"] or "")
                if s.startswith("S-") and s[2:].isdigit():
                    max_n = max(max_n, int(s[2:]))
            return f"S-{max_n + 1:04d}"

    def ensure_active(self, title: str = "Active Session") -> str:
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM sessions WHERE state = 'live' ORDER BY updated_ts DESC LIMIT 1"
            ).fetchone()
            if row:
                return str(row["id"])
            return self.open_session(title=title)

    def open_session(self, title: str = "New Session") -> str:
        with self._lock:
            sid = self._next_id()
            now = time.time()
            self.conn.execute(
                "INSERT INTO sessions (id, title, state, summary, created_ts, updated_ts) VALUES (?, ?, 'live', '', ?, ?)",
                (sid, title, now, now),
            )
            self.conn.commit()
            return sid

    def close_session(self, sid: str, summary: str) -> bool:
        summary = (summary or "").strip()
        if not summary:
            raise ValueError("Closing a session requires a non-empty summary in the agent's own words.")
        with self._lock:
            now = time.time()
            cur = self.conn.execute(
                "UPDATE sessions SET state = 'closed', summary = ?, updated_ts = ?, closed_ts = ? WHERE id = ?",
                (summary, now, now, sid),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        speaker: str,
        content: str,
        lane: str = "",
        ttfa_ms: Optional[float] = None,
        receipts: Optional[list[dict[str, Any]]] = None,
        verified: bool = True,
    ) -> int:
        with self._lock:
            now = time.time()
            cur = self.conn.execute(
                "INSERT INTO session_messages (session_id, ts, role, speaker, lane, content, ttfa_ms, receipts_json, verified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, now, role, speaker, lane, content, ttfa_ms,
                 json.dumps(receipts or [], ensure_ascii=False), 1 if verified else 0),
            )
            self.conn.execute("UPDATE sessions SET updated_ts = ? WHERE id = ?", (now, session_id))
            self.conn.commit()
            return int(cur.lastrowid)

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, title, state, summary, created_ts, updated_ts FROM sessions ORDER BY updated_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                sid = r["id"]
                msgs = self.conn.execute(
                    "SELECT role, receipts_json FROM session_messages WHERE session_id = ?", (sid,)
                ).fetchall()
                turns = sum(1 for m in msgs if m["role"] == "user")
                rcpts = sum(len(json.loads(m["receipts_json"] or "[]")) for m in msgs)
                out.append(asdict(SessionInfo(
                    id=sid, title=r["title"], state=r["state"], summary=r["summary"] or "",
                    created_ts=r["created_ts"], updated_ts=r["updated_ts"], turns=turns, receipts=rcpts,
                )))
            return out

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT seq, ts, role, speaker, lane, content, ttfa_ms, receipts_json, verified "
                "FROM session_messages WHERE session_id = ? ORDER BY seq ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "ts": r["ts"],
                "role": r["role"],
                "speaker": r["speaker"],
                "lane": r["lane"],
                "content": r["content"],
                "ttfa_ms": r["ttfa_ms"],
                "receipts": json.loads(r["receipts_json"] or "[]"),
                "verified": bool(r["verified"]),
            }
            for r in rows
        ]
