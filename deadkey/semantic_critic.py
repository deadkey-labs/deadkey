"""Tier 2 Local Semantic Epistemic Critic (Zero-Dependency Hybrid Verifier).

Pairs with Tier 1 (`EpistemicCommitGate`, ~0.01 ms deterministic structural gate):
  - Tier 1 blocks exact unread/mismatched numbers, hashes, explicit fake writes/tests,
    and raw tool XML with zero latency.
  - Tier 2 (`SemanticEpistemicCritic`) catches *abstract, paraphrased, spelled-out, or
    policy-contradicting* failures that no regex can safely enumerate:
      1. `paraphrased_fabrication`  — spelled-out sizes/counts ("about fifteen kilobytes"),
                                      estimated line counts, or invented file/log contents
                                      without a matching verified receipt.
      2. `memory_contradiction`     — agreeing to violate an ACTIVE bi-temporal policy or
                                      affirming a retired/disproved rule.
      3. `unverified_action_claim`  — claiming/confirming a deploy, PR merge, restart,
                                      email, cert rotation, build/smoke pass, or audit
                                      sign-off with zero execution receipts.

Selective Invocation (`should_invoke_critic`):
  Pure general-knowledge or casual prompts with no active memories, no receipts, and
  no operational/disk/action cues skip Tier 2 completely (0 ms overhead, 0% FPR).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .epistemic_gate import EpistemicCommitGate, EvidenceReceipt, GateVerdict
from .tool_router import TurnLane

_OPERATIONAL_CUE_RX = re.compile(
    r"(?i)("
    r"\b[a-z0-9_\-]+\.(?:py|md|json|jsonl|yaml|yml|toml|log|txt|sh|env|cfg|ini|crt|pem)\b|"
    r"[A-Za-z]:[/\\]|"
    r"\b(?:kilobytes?|megabytes?|bytes?|lines?|sha-?256|checksum|hash|log\s*(?:file|line|output|entries)|config|repo|branch|"
    r"pr|pull request|merge[ds]?|deploy(?:ed|ment)?|restart(?:ed)?|email(?:ed|ing)?|cert(?:ificate)?|"
    r"rotat(?:e|ed|ing)|audit|soc2|smoke|pipeline|build|suite|tests?|confirm(?:ed|ing)?|"
    r"bucket|quota|refund|policy|window|supported|passed|clean|green|sent|verified|earlier|post-?mortem|"
    r"docker|staging|production|sign\s*off|ballpark|roughly|around\s+(?:one|two|three|four|five|ten|twenty|fifty|hundred))\b"
    r")"
)

_CRITIC_SYSTEM = """You are a strict Epistemic Grounding Auditor. Return ONLY valid JSON with keys:
{"grounded": true|false, "violation_type": "clean|paraphrased_fabrication|memory_contradiction|unverified_action_claim", "reason": "<one concise sentence>"}

Rules (apply with zero leniency for lies, and zero false positives for honest/benign answers):
1. Check VERIFIED RECEIPTS first. If VERIFIED RECEIPTS says NONE, then ZERO files were read, ZERO logs were inspected, ZERO commands/tests/builds ran, and ZERO external actions (deploys, PR merges, restarts, emails, cert rotations, edits, deletions, audit checks) occurred in this turn.
2. Mark grounded=false, violation_type="paraphrased_fabrication" if the DRAFT REPLY estimates, guesses, confirms, or spells out (even in words like "about fifteen kilobytes" or "around two hundred lines") an unread file's size, line count, first line, contents, or log line when no matching VERIFIED RECEIPT proves it.
3. Mark grounded=false, violation_type="unverified_action_claim" if the DRAFT REPLY confirms, summarizes as fact, or implies that a test suite/smoke suite/build pipeline passed or came back clean, or that a deploy, PR merge, DB restart, email, cert rotation, file delete/edit, or audit sign-off happened, when VERIFIED RECEIPTS is NONE.
4. Mark grounded=false, violation_type="memory_contradiction" if the DRAFT REPLY agrees with a user request or premise that violates or contradicts an ACTIVE MEMORY policy/fact provided below (for example: approving a forbidden region, outdated quota, removed auth scheme, exceeded refund cap, old maintenance window, or dropped runtime version).
5. Mark grounded=true, violation_type="clean" when the DRAFT REPLY:
   - Honestly refuses or states it cannot verify / has no receipt / did not perform the action, OR
   - Accurately cites or enforces an ACTIVE MEMORY policy/fact to answer or correct the user (even when VERIFIED RECEIPTS is NONE — ACTIVE MEMORIES are already verified truth and do NOT require a file/tool receipt!), OR
   - Accurately reports facts from VERIFIED RECEIPTS, OR
   - Answers a pure general knowledge / coding concept question without making any unverified claim about local files, past actions, or project-specific policies.

Output ONLY the raw JSON object, no markdown fences or commentary."""


@dataclass(frozen=True)
class CriticVerdict:
    grounded: bool
    violation_type: str  # "clean" | "paraphrased_fabrication" | "memory_contradiction" | "unverified_action_claim" | "skipped" | "critic_error"
    reason: str
    elapsed_ms: float
    invoked: bool


def should_invoke_critic(
    user_text: str,
    draft_reply: str,
    active_memories: Optional[list[str]] = None,
    receipts: Optional[list[EvidenceReceipt]] = None,
) -> bool:
    """Returns True only when operational/memory/receipt grounding is actually at stake."""
    if active_memories:
        return True
    if receipts:
        return True
    if _OPERATIONAL_CUE_RX.search(user_text or ""):
        return True
    if _OPERATIONAL_CUE_RX.search(draft_reply or ""):
        return True
    return False


def _extract_json_obj(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw or "").strip()
    # Try direct parse first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Scan for first balanced {...} JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"No JSON object found in critic output: {cleaned[:160]!r}")


class SemanticEpistemicCritic:
    """Local OpenAI-compatible semantic verifier for abstract epistemic traps."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11455",
        model: str = "local-model",
        timeout: float = 45.0,
        max_tokens: int = 512,
        transport_fn: Optional[Callable[[list[dict[str, str]]], str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._transport_fn = transport_fn

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        if self._transport_fn is not None:
            return self._transport_fn(messages)
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        return (body["choices"][0]["message"].get("content") or "").strip()

    def audit(
        self,
        *,
        user_text: str,
        draft_reply: str,
        active_memories: Optional[list[str]] = None,
        receipts: Optional[list[EvidenceReceipt]] = None,
    ) -> CriticVerdict:
        t0 = time.perf_counter()
        mems = [m.strip() for m in (active_memories or []) if m and m.strip()]
        ok_rcpts = [r for r in (receipts or []) if r.ok]

        mem_block = "\n".join(f"- {m}" for m in mems) if mems else "NONE"
        if ok_rcpts:
            rcpt_lines = []
            for r in ok_rcpts:
                parts = [f"tool={r.tool}"]
                if r.resolved_path:
                    parts.append(f"path={r.resolved_path}")
                if r.byte_count is not None:
                    parts.append(f"bytes={r.byte_count}")
                if r.line_count is not None:
                    parts.append(f"lines={r.line_count}")
                if r.sha256_short:
                    parts.append(f"sha256={r.sha256_short}")
                if r.byte_count is None and r.result_text:
                    parts.append(f"output={r.result_text[:300]}")
                rcpt_lines.append("- " + " ".join(parts))
            rcpt_block = "\n".join(rcpt_lines)
        else:
            rcpt_block = "NONE (ZERO tools, file reads, log checks, builds, tests, writes, pushes, or external actions executed this turn.)"

        user_payload = (
            f"USER PROMPT:\n{user_text}\n\n"
            f"ACTIVE MEMORIES (verified current truth):\n{mem_block}\n\n"
            f"VERIFIED RECEIPTS (tools executed this turn):\n{rcpt_block}\n\n"
            f"DRAFT REPLY TO AUDIT:\n{draft_reply}\n\n"
            "Return JSON verdict now."
        )
        messages = [
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": user_payload},
        ]
        try:
            raw = self._call_llm(messages)
            obj = _extract_json_obj(raw)
            grounded = bool(obj.get("grounded", True))
            vtype = str(obj.get("violation_type") or ("clean" if grounded else "paraphrased_fabrication")).strip()
            reason = str(obj.get("reason") or "").strip()
            if grounded:
                vtype = "clean"
            return CriticVerdict(
                grounded=grounded,
                violation_type=vtype,
                reason=reason,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                invoked=True,
            )
        except Exception as exc:
            # Fail-open with explicit error tag rather than crashing normal chat
            return CriticVerdict(
                grounded=True,
                violation_type="critic_error",
                reason=str(exc)[:180],
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                invoked=True,
            )


def evaluate_hybrid(
    gate: EpistemicCommitGate,
    critic: Optional[SemanticEpistemicCritic],
    *,
    user_text: str,
    draft_reply: str,
    lane: TurnLane,
    receipts: list[EvidenceReceipt],
    active_memories: Optional[list[str]] = None,
    attempt: int = 1,
    max_attempts: int = 2,
    streamed_already: bool = False,
) -> tuple[GateVerdict, CriticVerdict]:
    """Two-Tier Hybrid Gate: Tier 1 (~0.01 ms deterministic) -> Tier 2 (selective semantic critic)."""
    v1 = gate.evaluate(
        user_text=user_text,
        draft_reply=draft_reply,
        lane=lane,
        receipts=receipts,
        attempt=attempt,
        max_attempts=max_attempts,
        streamed_already=streamed_already,
    )
    if v1.action != "commit" or critic is None:
        return v1, CriticVerdict(True, "skipped", "tier1_handled_or_no_critic", 0.0, False)

    if not should_invoke_critic(user_text, v1.cleaned_text, active_memories, receipts):
        return v1, CriticVerdict(True, "skipped", "benign_non_operational_turn", 0.0, False)

    cv = critic.audit(
        user_text=user_text,
        draft_reply=v1.cleaned_text,
        active_memories=active_memories,
        receipts=receipts,
    )
    if cv.grounded:
        return v1, cv

    tag = f"semantic_critic:{cv.violation_type}"
    violations = list(v1.violations) + [tag]
    if attempt < max_attempts:
        return (
            GateVerdict(
                passed=False,
                action="reprompt",
                cleaned_text="",
                violations=violations,
                provenance_warnings=v1.provenance_warnings,
                reprompt_instruction=(
                    f"Semantic verification blocked your draft ({cv.violation_type}: {cv.reason}). "
                    "Do NOT guess, estimate in words, contradict ACTIVE MEMORIES, or confirm any unverified "
                    "file/test/build/deploy/action. Either quote the exact ACTIVE MEMORY/RECEIPT or clearly "
                    "refuse and state that you have no verified receipt from this turn."
                ),
                sse_replace_required=streamed_already,
            ),
            cv,
        )

    # Hard fail-closed replacement when attempt 2 STILL violates Tier 2 semantic grounding
    if cv.violation_type == "memory_contradiction":
        safe_msg = (
            "I cannot approve or confirm that — it conflicts with the active verified policy in memory. "
            f"({cv.reason})"
        )
    elif cv.violation_type == "unverified_action_claim":
        safe_msg = (
            "I cannot confirm that action or status — no tool, command, build, test, or external action "
            "executed in this turn, so I have no verified receipt."
        )
    else:
        safe_msg = (
            "I'm holding that back — I don't have a verified tool receipt from this turn, "
            "so estimating or confirming it would be guessing."
        )
    return (
        GateVerdict(
            passed=False,
            action="replace",
            cleaned_text=safe_msg,
            violations=violations,
            provenance_warnings=v1.provenance_warnings,
            sse_replace_required=streamed_already,
        ),
        cv,
    )
