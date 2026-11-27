"""Disk-Equals-Truth Strict Verifier & Unified Two-Phase Epistemic Commit Gate.

Runs identically on BOTH voice and text turns (closing the voice-door guard schism
discovered in bot.py:7136 vs 8196). Structurally closes all 12 verified defect classes
from 2026-11-22 plus newly discovered sibling bugs:
  - Deterministic file/line/byte/SHA metadata attached to every read receipt so the
    LLM never miscounts lines (48 vs 41) or guesses file sizes (13,938 vs 4,040).
  - Pre-commit unread-number gate: blocks numeric file/system claims unless backed by
    a same-turn EvidenceReceipt containing that exact value.
  - Cross-root provenance audit (TASKS.md B2): catches K:\\workspace vs C:\\legacy-readonly drift.
  - Promise-without-act & empty `<tool_code></tool_code>` structural rejection.
  - Safe paragraph deduplication (never deletes middle sentences like _strip_already_spoken).
  - SSE-safe commit signal (`replace_turn`) so the UI never stays stuck on an old delta.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from .tool_router import TurnLane


@dataclass
class EvidenceReceipt:
    """Immutable disk-equals-truth verification receipt emitted by every tool execution."""
    tool: str
    args: dict[str, Any]
    ok: bool
    result_text: str
    duration_ms: int
    resolved_path: Optional[str] = None
    root_label: Optional[str] = None
    byte_count: Optional[int] = None
    line_count: Optional[int] = None
    sha256_short: Optional[str] = None
    verified_numbers: set[str] = field(default_factory=set)
    ts: float = field(default_factory=time.time)

    def format_for_llm(self) -> str:
        """Prepends deterministic verified facts so the model never hand-counts."""
        header_parts = [f"[VERIFIED RECEIPT tool={self.tool} ok={self.ok}"]
        if self.resolved_path:
            header_parts.append(f"path={self.resolved_path}")
        if self.root_label:
            header_parts.append(f"root={self.root_label}")
        if self.byte_count is not None:
            header_parts.append(f"bytes={self.byte_count}")
        if self.line_count is not None:
            header_parts.append(f"lines={self.line_count}")
        if self.sha256_short:
            header_parts.append(f"sha256={self.sha256_short}")
        header = " ".join(header_parts) + "]"
        return f"{header}\n{self.result_text}"


def build_file_receipt(tool: str, args: dict[str, Any], path: Path,
                       roots: dict[str, Path], duration_ms: int = 1) -> EvidenceReceipt:
    """Deterministic disk-equals-truth reader: computes exact bytes, lines, and SHA."""
    resolved = path.resolve()
    root_label = "external"
    for label, rpath in roots.items():
        try:
            resolved.relative_to(rpath.resolve())
            root_label = label
            break
        except ValueError:
            continue
    if not resolved.exists():
        return EvidenceReceipt(
            tool=tool, args=args, ok=False,
            result_text=f"File not found: {resolved}",
            duration_ms=duration_ms, resolved_path=str(resolved), root_label=root_label,
        )
    n_bytes = resolved.stat().st_size
    if n_bytes > 10 * 1024 * 1024:
        with resolved.open("rb") as fh:
            head = fh.read(65536)
        sha = hashlib.sha256(head).hexdigest()[:12]
        text = head.decode("utf-8", errors="replace")
        return EvidenceReceipt(
            tool=tool,
            args=args,
            ok=True,
            result_text=f"[Large file {n_bytes:,} bytes — head preview]\n{text[:4000]}",
            duration_ms=duration_ms,
            resolved_path=str(resolved),
            root_label=root_label,
            byte_count=n_bytes,
            line_count=None,
            sha256_short=sha,
            verified_numbers={str(n_bytes), f"{n_bytes:,}"},
        )
    raw = resolved.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    n_lines = len(text.splitlines())
    sha = hashlib.sha256(raw).hexdigest()[:12]
    nums = {str(n_bytes), f"{n_bytes:,}", str(n_lines), f"{n_lines:,}"}
    # Also harvest explicit numbers present verbatim in the file preview
    for m in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text[:8000]):
        nums.add(m)
        nums.add(m.replace(",", ""))
    return EvidenceReceipt(
        tool=tool,
        args=args,
        ok=True,
        result_text=text[:6000],
        duration_ms=duration_ms,
        resolved_path=str(resolved),
        root_label=root_label,
        byte_count=n_bytes,
        line_count=n_lines,
        sha256_short=sha,
        verified_numbers=nums,
    )


def harvest_numbers_from_receipts(receipts: list[EvidenceReceipt]) -> set[str]:
    nums: set[str] = set()
    for r in receipts:
        if not r.ok:
            continue
        nums.update(r.verified_numbers)
        if r.byte_count is not None:
            nums.add(str(r.byte_count))
            nums.add(f"{r.byte_count:,}")
        if r.line_count is not None:
            nums.add(str(r.line_count))
            nums.add(f"{r.line_count:,}")
        for m in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", r.result_text or ""):
            nums.add(m)
            nums.add(m.replace(",", ""))
    return nums


# --- Structural Gate Patterns ---
_EMPTY_TOOL_BLOCK_RX = re.compile(r"^\s*(?:<tool_code>\s*</tool_code>|\[tool:\s*[a-z0-9_]+\])\s*$", re.I)
_RAW_TOOL_CALL_LEAK_RX = re.compile(r"(?i)<tool_call>|<function=[a-z0-9_]+>|<\|tool_call\|>|<tool_code>")
_INTERNAL_LEAK_RX = re.compile(r"(?is)<system_note>.*?</system_note>|\[guard:[^\]]*\]|\[system note, not from [^\]]*\]")
_PROMISE_TAIL_RX = re.compile(
    r"(?i)\b(let me(?! know)|i'?ll (?:now |just )?(?:search|look|check|read|run|measure|call|pull|try|open|write|fire|query)|"
    r"i'?m going to (?:check|read|run|look|write|edit)|doing it now|next step is)\b"
)
_NUMERIC_SYSTEM_CLAIM_RX = re.compile(
    r"(?i)\b(\d[\d,]*(?:\.\d+)?)\s*(bytes|lines|entries|rows|mib|gb|kb|mb|files)\b(?!\s*=)"
)
_HEX_HASH_CLAIM_RX = re.compile(r"(?i)\b(?:sha-?256|sha|hash|digest)\b[^\n`]{0,30}?`?\b([0-9a-f]{8,64})\b`?")
_SIDE_EFFECT_CLAIM_RX = re.compile(
    r"(?i)\b("
    r"i (?:have )?(?:just )?(?:wrote|written|saved|edited|patched|deleted|removed|pushed|committed|mailed|sent|updated|fixed|created)\b|"
    r"i'?ve (?:just )?(?:written|saved|edited|patched|deleted|removed|pushed|committed|fixed|sent|updated|created)\b|"
    r"(?:landed|written|saved)\s+(?:[^.\n]{0,40}\s+)?to\s+disk\b|"
    r"https?://github\.com/\S+/commit/\S+"
    r")"
)
_UNVERIFIED_TEST_PASS_RX = re.compile(
    r"(?i)\b("
    r"(?:all\s+)?(?:\d+\s+)?(?:integration|unit|e2e|end-to-end|regression)?\s*tests?(?:\s+on\s+`?[a-z0-9_.\-/]+`?)?\s+(?:have\s+|just\s+)?passed\b|"
    r"test\s+suite\s+(?:is\s+)?(?:green|passing|passed)\b|"
    r"no\s+regressions?\s+(?:detected|found)\b|"
    r"branch\s+is\s+green\b"
    r")"
)
_WRITE_TOOLS = frozenset({
    "write_file", "edit_self", "write_skill", "move_to_trash", "archive_trash",
    "do_journal", "propose_memory_transition", "mail_hermes", "tell_randy", "build_request",
    "note_for_neon", "add_semantic", "write_semantic", "write_scratchpad", "thread",
})
_EXEC_TOOLS = frozenset({"run_python", "run_tests", "delegate", "start_job", "ask_hermes"})
_ACT_TOOLS = _WRITE_TOOLS | _EXEC_TOOLS


@dataclass
class GateVerdict:
    passed: bool
    action: str  # "commit" | "reprompt" | "replace"
    cleaned_text: str
    violations: list[str] = field(default_factory=list)
    provenance_warnings: list[str] = field(default_factory=list)
    reprompt_instruction: str = ""
    sse_replace_required: bool = False


def deduplicate_paragraphs(text: str, min_chars: int = 200, sim_threshold: float = 0.75) -> tuple[str, int]:
    """Removes verbatim/near-duplicate repeated paragraphs ONLY (never cuts short middle sentences)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paras) <= 1:
        return (text or "").strip(), 0
    kept: list[str] = []
    dropped = 0
    for p in paras:
        if len(p) >= min_chars and any(SequenceMatcher(None, p.lower(), k.lower()).ratio() >= sim_threshold for k in kept):
            dropped += 1
            continue
        kept.append(p)
    return "\n\n".join(kept), dropped


def check_cross_root_provenance(receipts: list[EvidenceReceipt]) -> list[str]:
    """Closes TASKS.md B2: flags when a turn writes and reads the same basename across different roots."""
    by_basename: dict[str, set[str]] = {}
    for r in receipts:
        if not r.resolved_path or not r.root_label:
            continue
        bn = Path(r.resolved_path).name.lower()
        by_basename.setdefault(bn, set()).add(r.root_label)
    warnings = []
    for bn, roots in by_basename.items():
        if len(roots) > 1:
            warnings.append(f"cross_root_mismatch:{bn}:{sorted(roots)}")
    return warnings


class EpistemicCommitGate:
    """Unified pre-commit verifier enforced on BOTH voice and text doors."""

    def evaluate(
        self,
        *,
        user_text: str,
        draft_reply: str,
        lane: TurnLane,
        receipts: list[EvidenceReceipt],
        attempt: int = 1,
        max_attempts: int = 2,
        streamed_already: bool = False,
    ) -> GateVerdict:
        violations: list[str] = []
        prov_warnings = check_cross_root_provenance(receipts)

        # 1. Strip internal scaffolding / guard leaks (Defect class: bleed)
        raw = _INTERNAL_LEAK_RX.sub("", draft_reply or "").strip()

        # 2. Empty reply, empty <tool_code>, or leaked raw <tool_call>/<function=...> XML (Defect #5)
        if not raw or _EMPTY_TOOL_BLOCK_RX.match(raw) or _RAW_TOOL_CALL_LEAK_RX.search(raw):
            violations.append("empty_or_raw_tool_leak")
            if attempt < max_attempts:
                return GateVerdict(
                    passed=False, action="reprompt", cleaned_text="",
                    violations=violations, provenance_warnings=prov_warnings,
                    reprompt_instruction=(
                        "Your last output contained no visible answer or leaked raw tool-call XML. "
                        "Do not emit <tool_call> or <function=...> text. Answer plainly in words; if no tool ran, state that clearly."
                    ),
                    sse_replace_required=streamed_already,
                )
            return GateVerdict(
                passed=False, action="replace",
                cleaned_text="I don't have a verified tool receipt from this turn, so I cannot confirm that.",
                violations=violations, provenance_warnings=prov_warnings,
                sse_replace_required=streamed_already,
            )

        # 3. Paragraph deduplication (Defect #10) — NEVER drops unique middle sentences (Defect #2)
        cleaned, n_dup = deduplicate_paragraphs(raw)
        if n_dup:
            violations.append(f"duplicate_paragraph_dropped:{n_dup}")
        # Normalize typographic apostrophes for deterministic regex matching
        norm_check = cleaned.replace("\u2019", "'").replace("\u2018", "'")

        ok_receipts = [r for r in receipts if r.ok]
        ok_write_tools = [
            r.tool for r in ok_receipts
            if r.tool in _WRITE_TOOLS and not (r.tool == "thread" and str((r.args or {}).get("action", "")).lower() in ("list", "get", "read"))
        ]
        ok_exec_tools = [r.tool for r in ok_receipts if r.tool in _EXEC_TOOLS]
        ok_act_tools = [r.tool for r in ok_receipts if r.tool in _ACT_TOOLS]
        verified_nums = harvest_numbers_from_receipts(ok_receipts)
        verified_hashes = {r.sha256_short.lower() for r in ok_receipts if r.sha256_short}

        # 4. Evidence-gated turn answered with ZERO tools (Defect #7: 13,938 bytes with no read)
        if lane == TurnLane.EVIDENCE_GATED and not receipts:
            violations.append("check_without_tool")

        # 5. Quantitative system/file claim & SHA-256 hash verification
        has_file_receipt = any((r.byte_count is not None or r.line_count is not None) for r in ok_receipts)
        file_or_disk_context = (
            lane == TurnLane.EVIDENCE_GATED
            or has_file_receipt
            or bool(re.search(r"(?i)([A-Za-z]:[/\\]|\b[a-z0-9_\-]+\.(?:py|md|json|jsonl|txt|log|yaml|yml|toml)\b|\b(?:file size|line count|sha-?256|how many (?:bytes|lines))\b)", user_text or ""))
        )
        verified_bytes = {str(r.byte_count) for r in ok_receipts if r.byte_count is not None}
        verified_lines = {str(r.line_count) for r in ok_receipts if r.line_count is not None}
        if file_or_disk_context:
            for m in _NUMERIC_SYSTEM_CLAIM_RX.finditer(norm_check):
                val_raw, unit = m.group(1), m.group(2).lower()
                val_norm = val_raw.replace(",", "")
                if not ok_receipts:
                    violations.append(f"unread_number:{val_raw}_{unit}")
                elif unit == "bytes" and verified_bytes:
                    if val_norm not in verified_bytes:
                        violations.append(f"misstated_bytes:{val_raw}_expected_{sorted(verified_bytes)}")
                elif unit == "lines" and verified_lines:
                    if val_norm not in verified_lines:
                        violations.append(f"miscounted_lines:{val_raw}_expected_{sorted(verified_lines)}")
                elif val_raw not in verified_nums and val_norm not in verified_nums:
                    violations.append(f"unverified_or_miscounted_number:{val_raw}_{unit}")
            for hm in _HEX_HASH_CLAIM_RX.finditer(norm_check):
                claimed_h = hm.group(1).lower()
                if not verified_hashes:
                    violations.append(f"unverified_hash:{claimed_h[:12]}")
                elif not any(claimed_h == vh or claimed_h.startswith(vh) for vh in verified_hashes):
                    violations.append(f"mismatched_hash:{claimed_h}_expected_{sorted(verified_hashes)}")

        # 6. Side-effect claim with no successful write tool receipt (TASKS.md B1)
        if _SIDE_EFFECT_CLAIM_RX.search(norm_check) and not ok_write_tools:
            violations.append("side_effect_claim_without_write_tool")

        # 6b. Test/CI pass claim with no successful test/execution tool receipt
        if _UNVERIFIED_TEST_PASS_RX.search(norm_check) and not ok_exec_tools:
            violations.append("unverified_test_pass_claim")

        # 7. Promise-only ending (including before a trailing fenced shell block) with no successful act tool (Defect #6)
        prose_without_fences = re.sub(r"```[\s\S]*?```", "", norm_check).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose_without_fences) if s.strip()]
        last_sent = sentences[-1] if sentences else prose_without_fences
        if _PROMISE_TAIL_RX.search(last_sent) and not ok_act_tools:
            violations.append("promise_without_act")

        # Only blocking violations prevent commit (duplicate_paragraph_dropped was already fixed cleanly)
        blocking = [v for v in violations if not v.startswith("duplicate_paragraph_dropped")]
        if not blocking:
            return GateVerdict(
                passed=True, action="commit", cleaned_text=cleaned,
                violations=violations, provenance_warnings=prov_warnings,
                sse_replace_required=False,
            )

        if attempt < max_attempts:
            return GateVerdict(
                passed=False, action="reprompt", cleaned_text="",
                violations=violations, provenance_warnings=prov_warnings,
                reprompt_instruction=(
                    f"Verification gate blocked your draft ({', '.join(blocking)}). "
                    "Disk equals truth: quote ONLY exact bytes, lines, and the full 12-char sha256 verbatim "
                    "from [VERIFIED RECEIPT] if present; if no tool/test/write ran this turn, state clearly "
                    "that you did NOT run or verify it and cannot confirm the claim. Do not end on a promise."
                ),
                sse_replace_required=streamed_already,
            )

        # Hard fail-closed replacement (never ship an unread/fabricated number or fake test/write claim!)
        if any(v.startswith(("unread_number", "unverified_or_miscounted_number", "misstated_bytes", "miscounted_lines", "unverified_hash", "mismatched_hash", "check_without_tool")) for v in blocking):
            safe_msg = (
                "I'm holding that back — I don't have a matching verified tool receipt for it "
                "from this turn, so stating or confirming it would be guessing. Say the word and I'll read it straight from disk."
            )
        elif "unverified_test_pass_claim" in blocking:
            safe_msg = "I cannot confirm those test results — no test runner executed in this turn, so I have no verified receipt."
        elif "side_effect_claim_without_write_tool" in blocking:
            safe_msg = "Nothing was written, deleted, edited, or pushed in this turn — no write tool executed."
        else:
            safe_msg = "I did not execute that action or check in this turn — tell me to run it and I'll verify it directly."

        return GateVerdict(
            passed=False, action="replace", cleaned_text=safe_msg,
            violations=violations, provenance_warnings=prov_warnings,
            sse_replace_required=streamed_already,
        )
