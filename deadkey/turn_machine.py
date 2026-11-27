"""Unified Turn State Machine (Voice + Text Door).

Eliminates the Voice-vs-Text guard schism and the 3.1-second dead sleep found
in bot.py:8196-8613. Both doors execute through the exact same state machine:
  1. Route turn (CompiledToolRouter -> lane + session-pinned 10 tools for 95%+ KV hits!)
  2. Event-driven salience recall (MemoryStore -> <=250 tok fresh valid-time cards)
  3. Provider call (native OpenAI tools OR InklingBypassAdapter for llama.cpp inkling2_loop)
  4. Deterministic tool execution with immutable EvidenceReceipts (disk = truth!)
  5. EpistemicCommitGate verification BEFORE final commit (with SSE `replace_turn` flag)
"""
from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .epistemic_gate import EpistemicCommitGate, EvidenceReceipt, GateVerdict
from .memory_store import MemoryStore, RecallHit
from .providers import InklingBypassAdapter, ParsedModelTurn
from .tool_router import CompiledToolRouter, RouteDecision, ToolDef, TurnLane


@dataclass
class TurnResult:
    door: str  # "voice" | "text"
    lane: TurnLane
    final_text: str
    receipts: list[EvidenceReceipt]
    verdict: GateVerdict
    route: RouteDecision
    salience_hits: list[RecallHit]
    attempts: int
    wall_ms: float
    sse_events: list[dict[str, Any]] = field(default_factory=list)


ModelCallable = Callable[[str, list[dict[str, Any]], list[dict[str, Any]], str], ParsedModelTurn]
ToolCallable = Callable[[str, dict[str, Any]], EvidenceReceipt]
EventCallable = Callable[[dict[str, Any]], None]


class TurnMachine:
    def __init__(
        self,
        *,
        router: CompiledToolRouter,
        memory: Optional[MemoryStore] = None,
        gate: Optional[EpistemicCommitGate] = None,
        embedder: Optional[Any] = None,
        tool_protocol: str = "openai_native",  # or "xml_json_bypass" for Inkling 2 Mini on llama.cpp
        max_tool_rounds: int = 6,
        max_gate_attempts: int = 2,
    ):
        self.router = router
        self.memory = memory
        self.gate = gate or EpistemicCommitGate()
        self.embedder = embedder
        self.tool_protocol = tool_protocol
        self.max_tool_rounds = max_tool_rounds
        self.max_gate_attempts = max_gate_attempts

    def run_turn(
        self,
        *,
        user_text: str,
        door: str,
        model_fn: ModelCallable,
        tool_fn: ToolCallable,
        speaker: str = "user",
        pinned_tools: Optional[list[ToolDef]] = None,
        history: Optional[list[dict[str, Any]]] = None,
        extra_receipts: Optional[list[EvidenceReceipt]] = None,
        extra_user_content: Optional[list[dict[str, Any]]] = None,
        context_suffix: str = "",
        pinned_notes_text: str = "",
        on_event: Optional[EventCallable] = None,
    ) -> TurnResult:
        t0 = time.perf_counter()
        # CRITICAL: Classify lane & route ONLY on raw `user_text` (NEVER on memory recall IDs like J-20261122 or pinned notes!)
        route = self.router.route(user_text)
        active_tools = pinned_tools if pinned_tools is not None else route.selected_tools
        qvec = self.embedder.embed_query(user_text) if (self.embedder and getattr(self.embedder, "ready", False)) else None
        salience_hits = self.memory.recall(user_text, k=3, query_vec=qvec) if self.memory else []

        def _emit(ev: dict[str, Any]) -> None:
            sse_events.append(ev)
            if on_event is not None:
                on_event(ev)

        # Build compact, prefix-stable system prompt (never mutate mid-session -> 95%+ KV hit!)
        sys_parts = []
        if self.memory:
            sys_parts.append(self.memory.pinned_core_prompt(max_tokens=1200))
        if self.tool_protocol == "xml_json_bypass":
            sys_parts.append(InklingBypassAdapter.render_tool_block(active_tools))
        system_prompt = "\n\n".join(p for p in sys_parts if p)

        messages: list[dict[str, Any]] = []
        if history:
            for hm in history[-12:]:
                role = hm.get("role")
                content = hm.get("content") or ""
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        tail_lines = [f"- [{h.id} | {h.status} | {h.valid_from}] {h.text[:180]}" for h in salience_hits]
        user_payload = user_text
        if speaker.lower() != "user":
            user_payload = f"[from: {speaker} - NOT the user] {user_payload}"
        if context_suffix:
            user_payload += "\n\n" + context_suffix
        if pinned_notes_text:
            user_payload += "\n\n[pinned workbench notes]\n" + pinned_notes_text
        if tail_lines:
            user_payload += "\n\n[salience recall — valid-time evidence]\n" + "\n".join(tail_lines)

        if extra_user_content:
            # Multimodal user turn (e.g. pasted screenshot for Gemma 4 native vision)
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_payload}] + list(extra_user_content),
            })
        else:
            messages.append({"role": "user", "content": user_payload})

        tool_schemas = [
            t.to_openai_schema()
            for t in active_tools
        ] if self.tool_protocol == "openai_native" else []

        receipts: list[EvidenceReceipt] = list(extra_receipts or [])
        sse_events: list[dict[str, Any]] = []
        attempts = 0
        final_verdict: Optional[GateVerdict] = None

        _emit({"event": "step", "phase": "routing", "lane": route.lane.value, "tools": len(active_tools)})

        for attempt in range(1, self.max_gate_attempts + 1):
            attempts = attempt
            draft_text = ""

            for _round in range(self.max_tool_rounds + 1):
                turn_out = model_fn(system_prompt, messages, tool_schemas, self.tool_protocol)
                if self.tool_protocol == "xml_json_bypass" and not turn_out.tool_calls and "<tool_call>" in turn_out.text:
                    turn_out = InklingBypassAdapter.parse_response(turn_out.text)

                if turn_out.tool_calls:
                    oai_tcs = []
                    for idx_tc, tc in enumerate(turn_out.tool_calls):
                        cid = tc.get("id") or f"call_{attempt}_{_round}_{idx_tc}"
                        tc["_cid"] = cid
                        oai_tcs.append({
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": _json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                            },
                        })
                    if self.tool_protocol == "openai_native":
                        messages.append({"role": "assistant", "content": turn_out.text or "", "tool_calls": oai_tcs})
                    else:
                        xml_calls = "\n".join(
                            f'<tool_call>{_json.dumps({"name": tc["name"], "arguments": tc.get("arguments") or {}}, ensure_ascii=False)}</tool_call>'
                            for tc in turn_out.tool_calls
                        )
                        messages.append({"role": "assistant", "content": xml_calls})
                    deferred_vision_msgs: list[dict] = []
                    for tc in turn_out.tool_calls:
                        _emit({"event": "step", "phase": "verifying", "tool": tc["name"]})
                        rcpt = tool_fn(tc["name"], tc.get("arguments") or {})
                        receipts.append(rcpt)
                        _emit({
                            "event": "tool_receipt",
                            "tool": rcpt.tool,
                            "ok": rcpt.ok,
                            "path": rcpt.resolved_path,
                            "bytes": rcpt.byte_count,
                            "lines": rcpt.line_count,
                            "sha": rcpt.sha256_short,
                            "root": rcpt.root_label,
                        })
                        if self.tool_protocol == "openai_native":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["_cid"],
                                "name": tc["name"],
                                "content": rcpt.format_for_llm(),
                            })
                            # Defer multimodal image attachment until AFTER all contiguous role="tool" messages!
                            if tc["name"] == "look_at_screen" and rcpt.ok and rcpt.resolved_path:
                                import base64 as _b64
                                from pathlib import Path as _P
                                img_p = _P(rcpt.resolved_path)
                                if img_p.exists():
                                    b64_img = _b64.b64encode(img_p.read_bytes()).decode("ascii")
                                    deferred_vision_msgs.append({
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "[Live screen capture attached from look_at_screen — describe what you actually see.]"},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                                        ],
                                    })
                        else:
                            messages.append({"role": "user", "content": rcpt.format_for_llm()})
                    if deferred_vision_msgs:
                        messages.extend(deferred_vision_msgs)
                    continue

                draft_text = turn_out.text
                break

            streamed = (route.lane == TurnLane.FAST_CONVERSATIONAL and attempt == 1 and bool(draft_text))
            if streamed and on_event is None:
                sse_events.append({"event": "delta", "content": draft_text})

            verdict = self.gate.evaluate(
                user_text=user_text,
                draft_reply=draft_text,
                lane=route.lane,
                receipts=receipts,
                attempt=attempt,
                max_attempts=self.max_gate_attempts,
                streamed_already=streamed,
            )
            final_verdict = verdict

            if verdict.action == "commit":
                if not streamed and on_event is None:
                    sse_events.append({"event": "delta", "content": verdict.cleaned_text})
                elif verdict.cleaned_text != draft_text:
                    _emit({"event": "replace_turn", "content": verdict.cleaned_text, "reason": verdict.violations})
                _emit({"event": "done", "door": door, "lane": route.lane.value, "verified": True})
                break

            if verdict.action == "reprompt":
                _emit({"event": "step", "phase": "reverify", "reason": verdict.violations})
                messages.append({"role": "assistant", "content": draft_text})
                messages.append({"role": "user", "content": f"[verification gate] {verdict.reprompt_instruction}"})
                continue

            _emit({
                "event": "replace_turn" if verdict.sse_replace_required else "delta",
                "content": verdict.cleaned_text,
                "reason": verdict.violations,
            })
            _emit({"event": "done", "door": door, "lane": route.lane.value, "verified": False})
            break

        assert final_verdict is not None
        return TurnResult(
            door=door,
            lane=route.lane,
            final_text=final_verdict.cleaned_text,
            receipts=receipts,
            verdict=final_verdict,
            route=route,
            salience_hits=salience_hits,
            attempts=attempts,
            wall_ms=round((time.perf_counter() - t0) * 1000, 2),
            sse_events=sse_events,
        )
