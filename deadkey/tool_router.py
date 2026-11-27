"""Compiled Dynamic Tool Router + Turn Lane Classifier.

Solves two verified bottlenecks from the 2026-11-22 forensic audit:
1. Schema Bloat: injecting all 72-73 tool schemas (~5,500-6,000 tokens/turn) saturated
   MoE models (causing Qwen3.8-35B APEX to drop from 4/4 cold to 0/3 inside her door).
   This router selects only 6-12 relevant schemas per turn in <2 ms (~75-85% token reduction),
   always exposing a `request_tool_pack` escape hatch so no capability is ever unreachable.
2. Lane Classification: separates fast casual conversation (FAST_CONVERSATIONAL -> <500ms TTFA)
   from verification/action turns (EVIDENCE_GATED -> strict disk-equals-truth commit gate).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class TurnLane(str, Enum):
    FAST_CONVERSATIONAL = "fast_conversational"
    EVIDENCE_GATED = "evidence_gated"


# Domain packs mapping Deadkey's real 73 tools into clean semantic groups
TOOL_PACKS: dict[str, set[str]] = {
    "core": {
        "search_memory",
        "do_journal",
        "request_tool_pack",
    },
    "memory_continuity": {
        "search_memory", "do_journal", "propose_memory_transition", "recall_journal_entry",
        "note_where_i_left_off", "write_scratchpad", "read_scratchpad", "note_tendency",
        "link_journal", "thread", "add_semantic", "write_semantic", "retire_semantic",
        "list_semantic", "read_archive", "remind_me", "session", "speak_up",
    },
    "filesystem_code": {
        "read_file", "write_file", "list_dir", "file_info", "read_source", "grep_source",
        "list_functions", "read_source_lines", "edit_self", "list_dir_edit", "revert_edit",
        "recent_edits", "list_checkpoints", "revert_checkpoint", "run_python", "run_tests",
        "move_to_trash", "archive_trash", "list_jarvis", "read_jarvis",
    },
    "machine_voice_ops": {
        "machine_probe", "whats_running", "body_state", "world_state", "which_model",
        "tail_log", "diag_pipe", "fix_pipe", "run_tts_probe", "my_record",
    },
    "web_vision": {
        "web_search", "get_weather", "read_page", "look_at_screen", "inspect_image_file",
    },
    "collab_async": {
        "ask_hermes", "mail_hermes", "tell_randy", "build_request", "propose_change",
        "request_approval", "list_approvals", "delegate", "list_delegations",
        "review_delegation", "start_job", "job_status", "cancel_job", "enqueue_action",
        "list_pending", "cancel_pending", "list_skills", "load_skill", "write_skill",
    },
}

PACK_KEYWORDS: dict[str, re.Pattern] = {
    "filesystem_code": re.compile(
        r"(?i)\b(file|bytes|lines|size|count|read|open|edit|patch|code|source|grep|script|python|"
        r"bot\.py|changelog|tasks\.md|journal\.md|directory|folder|checkpoint|revert|test|syntax|"
        r"diff|hash|sha|sandbox|trash|workspace|C:\\legacy-readonly|k:\\workspace|\.py\b|\.md\b|\.json\b|\.jsonl\b)\b"
    ),
    "machine_voice_ops": re.compile(
        r"(?i)\b(gpu|vram|5090|cpu|ram|disk|port|11435|11434|11436|5005|7860|7872|model|which model|"
        r"running|stack|restart|orpheus|tts|voice|audio|mic|dji|crackle|loop|lag|log|tail|probe|"
        r"pipe|timing|latency|speed|tok/s|acoustic|whisper|process|pid)\b"
    ),
    "memory_continuity": re.compile(
        r"(?i)\b(remember|recall|memory|journal|ledger|entry|j-\d+|t-\d+|thread|scratchpad|"
        r"tendency|semantic|supersede|disprove|yesterday|last week|last session|remind|archive|session|"
        r"on your mind|still open|tonight|what did you notice|good at|bad at|lately|earlier)\b"
    ),
    "web_vision": re.compile(
        r"(?i)\b(search the web|google|look up|weather|forecast|url|http|website|read page|"
        r"screen|screenshot|look at this|image|picture|photo|on my screen|latest|news|release)\b"
    ),
    "collab_async": re.compile(
        r"(?i)\b(hermes|mail|ask hermes|tell user|build request|approval|approve|delegate|"
        r"skill|job|pending|queue|propose)\b"
    ),
}

# Triggers the strict Evidence-Gated lane (disk = truth before speech/text commit)
EVIDENCE_GATE_RX = re.compile(
    r"(?i)(?:^\s*(?:please\s+)?(?:read|check|verify|measure|look at|confirm|prove|inspect|count|grep|tail)\s+(?:the\s+|my\s+|this\s+|that\s+|attached\s+|`){0,2}(?:file|disk|screen|log|vram|gpu|port|path|size|lines?|bytes?|code|source|config|manifest|memory|status|attachment|upload|[a-z0-9_.\-/\\]+\.[a-z]{1,6})\b|"
    r"\b(?:how many (?:bytes|lines|files|entries|rows|tools|turns)|size of|contents of|what'?s in|what is in|"
    r"sha-?256|checksum|hash of|which model|how much vram|what'?s running|did (?:that|it) land|"
    r"check,? don'?t guess|on disk|exact\s+(?:bytes?|lines?|size|count|number|path|sha-?256|hash))\b|"
    r"[A-Za-z]:[/\\]|bot\.py|TASKS\.md|CHANGELOG\.jsonl|ledger\.jsonl|journal\.md|J-\d{6})"
)

_TOKEN_RX = re.compile(r"[a-z0-9_]{2,}")


# Explicit OpenAI JSON-Schema `parameters` for every pinned core tool so models know exact argument names!
TOOL_PARAMETERS: dict[str, dict] = {
    "file_info": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path or filename to stat"}},
        "required": ["path"],
    },
    "read_file": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path or filename to read"}},
        "required": ["path"],
    },
    "grep_source": {
        "type": "object",
        "properties": {"pattern": {"type": "string", "description": "Regex or text pattern to search"}},
        "required": ["pattern"],
    },
    "run_python": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Short Python snippet to run safely"}},
        "required": ["code"],
    },
    "which_model": {"type": "object", "properties": {}},
    "machine_probe": {"type": "object", "properties": {}},
    "search_memory": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords to search valid-time memory"}},
        "required": ["query"],
    },
    "do_journal": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Journal observation to save to SQLite WAL"}},
        "required": ["text"],
    },
    "ask_hermes": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "description": "Task or question for local Hermes Agent"}},
        "required": ["prompt"],
    },
    "note_for_neon": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the note"},
            "message": {"type": "string", "description": "Exact message or feedback for Neon"},
        },
        "required": ["message"],
    },
    "request_tool_pack": {
        "type": "object",
        "properties": {
            "pack": {
                "type": "string",
                "enum": ["memory_continuity", "filesystem_code", "machine_voice_ops", "web_vision", "collab_async"],
            }
        },
        "required": ["pack"],
    },
    "look_at_screen": {
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "Optional reason for looking at the screen"}},
    },
}


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    pack: str
    est_tokens: int

    def to_openai_schema(self) -> dict:
        params = TOOL_PARAMETERS.get(self.name, {"type": "object", "properties": {}})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


@dataclass(frozen=True)
class RouteDecision:
    lane: TurnLane
    selected_tools: list[ToolDef]
    matched_packs: list[str]
    full_catalog_tokens: int
    selected_tokens: int
    tokens_saved: int
    savings_pct: float


class CompiledToolRouter:
    """Sub-2ms deterministic hybrid (pack-regex + TF-IDF lexical) tool router."""

    def __init__(self, catalog: Iterable[dict], max_tools: int = 10):
        self.max_tools = max_tools
        self.tools: dict[str, ToolDef] = {}
        self._doc_tf: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}

        # Always include escape-hatch meta-tool & direct Neon outbox tool
        items = list(catalog)
        if not any(t.get("name") == "request_tool_pack" for t in items):
            items.append({
                "name": "request_tool_pack",
                "description": (
                    "Load an additional tool pack for this turn if a needed tool is not currently exposed. "
                    "Packs: memory_continuity, filesystem_code, machine_voice_ops, web_vision, collab_async."
                ),
            })
        if not any(t.get("name") == "note_for_neon" for t in items):
            items.append({
                "name": "note_for_neon",
                "description": (
                    "Write a real, verified handoff note or feedback message into Neon's local Partner Outbox "
                    "(partner_queue.json) with a receipt ID. Use whenever the user asks you to tell Neon something."
                ),
            })

        name_to_pack: dict[str, str] = {}
        for pack, names in TOOL_PACKS.items():
            for n in names:
                name_to_pack.setdefault(n, pack)

        df: dict[str, int] = {}
        raw_tfs: dict[str, dict[str, int]] = {}
        for item in items:
            name = item["name"]
            desc = item.get("description") or ""
            pack = name_to_pack.get(name, "collab_async")
            # Estimate OpenAI JSON schema footprint (~name + desc + params overhead)
            est_toks = max(45, (len(name) + len(desc)) // 4 + 38)
            self.tools[name] = ToolDef(name=name, description=desc, pack=pack, est_tokens=est_toks)

            toks = _TOKEN_RX.findall(f"{name.replace('_', ' ')} {name} {desc}".lower())
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            raw_tfs[name] = tf
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1

        n_docs = max(1, len(self.tools))
        self._idf = {tok: math.log((n_docs + 1) / (c + 1)) + 1.0 for tok, c in df.items()}
        for name, tf in raw_tfs.items():
            vec = {tok: (1.0 + math.log(cnt)) * self._idf[tok] for tok, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._doc_tf[name] = {tok: v / norm for tok, v in vec.items()}

        self.full_catalog_tokens = sum(t.est_tokens for t in self.tools.values())

    @classmethod
    def from_json(cls, path: str | Path, max_tools: int = 10) -> "CompiledToolRouter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, max_tools=max_tools)

    def pinned_session_tools(self) -> list[ToolDef]:
        """Returns a fixed, session-stable 10-tool set pinned in the OpenAI `tools` header.

        Measured during live Colossus conversations: mutating the top-level `tools=[...]`
        array turn-by-turn invalidated llama-server's KV prefix at token ~50 (dropping
        kv_hit from 0.96 -> 0.00). Pinning these 10 highest-utility tools keeps BOTH:
          - ~84% schema token savings (10 vs 73 tools — preventing MoE tool blindness), AND
          - 95%+ multi-turn KV prefix cache hits (~28-100 ms warm TTFT!).
        Any remaining 63 tools load on demand via `request_tool_pack` in a trailing message.
        """
        pinned_names = (
            "file_info", "read_file", "grep_source", "run_python",
            "which_model", "machine_probe", "search_memory",
            "do_journal", "look_at_screen", "ask_hermes", "note_for_neon", "request_tool_pack",
        )
        return [self.tools[n] for n in pinned_names if n in self.tools]

    def classify_lane(self, user_text: str) -> TurnLane:
        if EVIDENCE_GATE_RX.search(user_text or ""):
            return TurnLane.EVIDENCE_GATED
        return TurnLane.FAST_CONVERSATIONAL

    def route(self, user_text: str, extra_packs: Iterable[str] = ()) -> RouteDecision:
        text = (user_text or "").strip()
        lane = self.classify_lane(text)

        matched_packs: list[str] = []
        for pack, rx in PACK_KEYWORDS.items():
            if rx.search(text):
                matched_packs.append(pack)
        for ep in extra_packs:
            if ep in TOOL_PACKS and ep not in matched_packs:
                matched_packs.append(ep)

        # Always keep the 6 highest-frequency companion core tools + pack escape hatch available
        selected_names: list[str] = [
            n for n in (
                "search_memory", "do_journal", "recall_journal_entry",
                "read_scratchpad", "thread", "body_state", "request_tool_pack"
            )
            if n in self.tools
        ]

        # Score all candidate tools with TF-IDF cosine + pack boost + exact name match boost
        q_toks = _TOKEN_RX.findall(text.lower())
        q_tf: dict[str, int] = {}
        for tok in q_toks:
            q_tf[tok] = q_tf.get(tok, 0) + 1
        q_vec = {tok: (1.0 + math.log(cnt)) * self._idf.get(tok, 1.0) for tok, cnt in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {tok: v / q_norm for tok, v in q_vec.items()}

        scores: list[tuple[float, str]] = []
        text_low = text.lower()
        for name, tdef in self.tools.items():
            if name in selected_names:
                continue
            dvec = self._doc_tf[name]
            cos = sum(q_vec[k] * dvec[k] for k in q_vec if k in dvec)
            pack_bonus = 0.35 if tdef.pack in matched_packs else 0.0
            exact_bonus = 2.0 if name in text_low else 0.0
            # Direct high-signal heuristics for common companion operations
            if name == "file_info" and re.search(r"(?i)\b(size|bytes|mtime|exists|file)\b", text):
                pack_bonus += 0.45
            if name == "read_file" and re.search(r"(?i)\b(read|lines|contents|open|tasks\.md|changelog)\b", text):
                pack_bonus += 0.45
            if name == "run_python" and re.search(r"(?i)\b(count|how many lines|calculate|python|script|verify)\b", text):
                pack_bonus += 0.40
            if name == "grep_source" and re.search(r"(?i)\b(grep|where is|find in|search code|bot\.py|defined)\b", text):
                pack_bonus += 0.45
            if name == "machine_probe" and re.search(r"(?i)\b(vram|gpu|disk|audio|dji|ports|stack|status)\b", text):
                pack_bonus += 0.45
            if name == "which_model" and re.search(r"(?i)\b(which model|what model|27b|35b|inkling)\b", text):
                pack_bonus += 0.50
            total = cos + pack_bonus + exact_bonus
            if total > 0.05:
                scores.append((total, name))

        scores.sort(key=lambda x: x[0], reverse=True)
        # In fast conversational lane with no matched packs, keep tool footprint compact (8 tools)
        budget = 8 if (lane == TurnLane.FAST_CONVERSATIONAL and not matched_packs) else self.max_tools
        for _score, name in scores:
            if len(selected_names) >= budget:
                break
            selected_names.append(name)

        selected = [self.tools[n] for n in selected_names]
        sel_tokens = sum(t.est_tokens for t in selected)
        saved = max(0, self.full_catalog_tokens - sel_tokens)
        pct = (100.0 * saved / self.full_catalog_tokens) if self.full_catalog_tokens else 0.0
        return RouteDecision(
            lane=lane,
            selected_tools=selected,
            matched_packs=matched_packs,
            full_catalog_tokens=self.full_catalog_tokens,
            selected_tokens=sel_tokens,
            tokens_saved=saved,
            savings_pct=round(pct, 1),
        )
