"""Model-Agnostic Inference & Voice Adapters (Supports Arch A & Arch B + Inkling 2 Mini).

Key capabilities:
1. Native OpenAI tool-calling (vLLM `--tool-call-parser inkling`, SGLang, llama-server).
2. Grammar-Bypass Structured Tool Adapter (`InklingBypassAdapter`) specifically unlocking
   `Inkling 2 Mini Unbound-v2` on llama.cpp builds where `--jinja` throws HTTP 500
   `Unexpected empty grammar stack` (CHANGELOG.jsonl C-0032).
3. Multi-slot prefix isolation (`slot_id=0` pinned live chat, `slot_id=1` async cognition)
   so quiet reflections NEVER evict Deadkey's live KV prefix cache.
4. Pluggable Voice Adapters comparing:
   - Arch A: Streaming Orpheus-Tara (Direct Token -> CUDA SNAC streaming, ~180 ms TTFB)
   - Arch B: Kyutai Pocket TTS (100% CPU on 9950X3D, zero-shot Tara clone, ~100 ms TTFB, 0 MiB VRAM)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .tool_router import ToolDef


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str  # "qwen_moe" | "qwen_dense" | "inkling2_loop" | "gemma4" | "glm4"
    engine: str  # "vllm_wsl2" | "sglang_wsl2" | "llama_server_native"
    base_url: str
    model_id: str
    ctx: int
    tool_protocol: str  # "openai_native" | "xml_json_bypass"
    est_vram_mib: int
    est_tok_s: int
    extra_server_flags: tuple[str, ...] = ()
    supports_multi_slot: bool = True


@dataclass
class ParsedModelTurn:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit_ratio: Optional[float] = None
    protocol_used: str = "openai_native"


_BYPASS_CALL_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)


class InklingBypassAdapter:
    """Bypasses llama.cpp's broken Jinja tool-grammar stack for `inkling2_loop` GGUFs.

    Instead of passing `tools=[...]` (which triggers HTTP 500 `Unexpected empty grammar stack`
    on llama.cpp b12633), renders only the routed 6-10 compact tool specs into a structured
    protocol block and parses `<tool_call>{...}</tool_call>` emissions strictly with JSON validation.
    """

    @staticmethod
    def render_tool_block(tools: list[ToolDef]) -> str:
        lines = [
            "[TOOL PROTOCOL — STRICT]",
            "When you need to check disk, memory, or machine state, emit ONE OR MORE blocks:",
            '<tool_call>{"name": "<tool_name>", "arguments": { ... }}</tool_call>',
            "Never invent a number or file result. Emit the call, stop, and wait for the [VERIFIED RECEIPT].",
            "Available tools this turn:",
        ]
        for t in tools:
            lines.append(f"- {t.name}: {t.description[:140]}")
        return "\n".join(lines)

    @staticmethod
    def parse_response(raw_content: str) -> ParsedModelTurn:
        calls: list[dict[str, Any]] = []
        valid_spans: list[tuple[int, int]] = []
        for m in _BYPASS_CALL_RX.finditer(raw_content or ""):
            try:
                obj = json.loads(m.group(1))
                name = (obj.get("name") or "").strip()
                args = obj.get("arguments") or {}
                if name and isinstance(args, dict):
                    calls.append({"name": name, "arguments": args})
                    valid_spans.append(m.span())
            except json.JSONDecodeError:
                continue
        # Remove parsed tool_call blocks from visible prose
        visible = raw_content or ""
        for start, end in reversed(valid_spans):
            visible = visible[:start] + visible[end:]
        return ParsedModelTurn(
            text=visible.strip(),
            tool_calls=calls,
            protocol_used="xml_json_bypass",
        )


@dataclass
class VoiceLatencyReport:
    architecture: str
    tts_engine: str
    stt_engine: str
    vram_voice_mib: int
    endpoint_ms: float
    stt_ms: float
    llm_ttft_ms: float
    tts_first_frame_ms: float
    transport_jbuf_ms: float

    @property
    def total_ttfa_ms(self) -> float:
        return round(
            self.endpoint_ms + self.stt_ms + self.llm_ttft_ms
            + self.tts_first_frame_ms + self.transport_jbuf_ms, 1
        )


def build_architecture_profiles() -> dict[str, dict[str, Any]]:
    """Canonical Architecture A & Architecture B A/B specifications for Colossus."""
    return {
        "arch_a_orpheus_vllm": {
            "label": "Architecture A: Split-Brain Streaming Kernel (Orpheus-Tara GPU SNAC + vLLM/SGLang WSL2)",
            "host_topology": "WSL2/Docker (vLLM/SGLang + Streaming Orpheus SNAC) + Native Windows Audio Gateway",
            "gpu_total_mib": 32607,
            "voice": {
                "stt": "Moonshine-Medium / faster-whisper-turbo streaming (9950X3D CPU)",
                "tts": "Orpheus-3B Tara FP8 + Direct CUDA SNAC Token-Streamer (ctx=2048 f16 KV)",
                "tts_vram_mib": 2800,
                "target_ttfa_ms": 460,
                "breakdown_ms": {"endpoint": 150, "stt": 40, "llm_ttft": 90, "tts_first_pcm": 140, "transport": 40},
            },
            "candidate_models": [
                "inkling2-mini-vllm",
                "qwen38-35b-apex-nvfp4",
                "qwen36-35b-a3b-q4-mtp",
                "gemma4-26b-a4b-fp8",
            ],
            "cognition_isolation": "RadixAttention / llama-server -np 2 (Slot 1 isolated, zero chat KV eviction)",
        },
        "arch_b_pocket_cpu_maxbrain": {
            "label": "Architecture B: Sovereign CPU-Voice (9950X3D) + Full 32 GB RTX 5090 Brain",
            "host_topology": "9950X3D CPU (Kyutai Pocket TTS Tara-Clone + Streaming STT) + Full 32 GB GPU Brain & Co-Res Cognition",
            "gpu_total_mib": 32607,
            "voice": {
                "stt": "faster-whisper-large-v3-turbo INT8 streaming (9950X3D CPU)",
                "tts": "Kyutai Pocket TTS (~100M Mimi codec, 5s Tara voice clone, 100% 9950X3D CPU)",
                "tts_vram_mib": 0,
                "target_ttfa_ms": 370,
                "breakdown_ms": {"endpoint": 140, "stt": 40, "llm_ttft": 60, "tts_first_pcm": 90, "transport": 40},
            },
            "candidate_models": [
                "inkling2-mini-q6-65k",
                "qwen38-35b-apex-fp8-65k",
                "qwen38-27b-blackfrost-q6-65k",
                "gemma4-31b-fp8-65k",
            ],
            "cognition_isolation": "Permanently co-resident 4B-7B GPU cognition/fact-extractor (~3.8 GB VRAM) beside 25 GB brain",
        },
    }
