"""Local Verification-Gated Routing (VGR-Local) + Hard Cloud Privacy Fence.

Inspired by NVIDIA NeMo Switchyard's first-token logprob routing, adapted for
Colossus's Hard Constraint #2 (Local-First / Privacy Wall):

- Evaluates first-token logprob probability & top-1/top-2 logprob margin from
  the local inference engine (`logprobs=True`, thinking off).
- If confidence >= threshold: commit immediately at minimum compute & latency.
- If confidence < threshold on a **private** cartridge (e.g., `deadkey`):
  NEVER escalates to cloud! Instead triggers a **100% local verification pass**
  (disk receipt check / deeper memory recall / optional local heavy gear).
- If confidence < threshold on an explicitly **non-private opt-in** cartridge
  (e.g., `studio-director` for ComfyUI/FFmpeg coding): permits optional cloud
  escalation ONLY after verifying zero private memory tokens are present.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VGRDecision:
    action: str  # "accept_local" | "local_verify_repass" | "opt_in_cloud_escalate"
    token_prob: Optional[float]
    margin: Optional[float]
    reason: str
    privacy_fence_enforced: bool


class LocalVGRRouter:
    def __init__(
        self,
        *,
        min_prob: float = 0.65,
        min_margin: float = 0.40,
        cloud_privacy_wall: bool = True,
        allow_cloud_escalation: bool = False,
    ):
        self.min_prob = min_prob
        self.min_margin = min_margin
        self.cloud_privacy_wall = cloud_privacy_wall
        self.allow_cloud_escalation = allow_cloud_escalation

    def decide(
        self,
        *,
        first_token_logprob: Optional[float],
        top_logprob_margin: Optional[float],
        has_verified_receipts: bool = False,
        contains_private_memory: bool = True,
    ) -> VGRDecision:
        if first_token_logprob is None:
            return VGRDecision(
                action="accept_local",
                token_prob=None,
                margin=top_logprob_margin,
                reason="Backend did not emit logprobs; defaulting to local epistemic gate.",
                privacy_fence_enforced=True,
            )

        prob = round(math.exp(first_token_logprob), 4)
        low_prob = prob < self.min_prob
        low_margin = (top_logprob_margin is not None) and (top_logprob_margin < self.min_margin)

        if not low_prob and not low_margin:
            return VGRDecision(
                action="accept_local",
                token_prob=prob,
                margin=top_logprob_margin,
                reason=f"High local confidence (p={prob}, margin={top_logprob_margin}).",
                privacy_fence_enforced=True,
            )

        if has_verified_receipts:
            return VGRDecision(
                action="accept_local",
                token_prob=prob,
                margin=top_logprob_margin,
                reason=f"Low lexical certainty (p={prob}) offset by deterministic disk receipts.",
                privacy_fence_enforced=True,
            )

        # Hard cryptographic-policy privacy wall: personal memory NEVER leaves the box!
        if self.cloud_privacy_wall or contains_private_memory or not self.allow_cloud_escalation:
            return VGRDecision(
                action="local_verify_repass",
                token_prob=prob,
                margin=top_logprob_margin,
                reason=f"Low local confidence (p={prob}, margin={top_logprob_margin}); triggering local verification pass (privacy wall held).",
                privacy_fence_enforced=True,
            )

        return VGRDecision(
            action="opt_in_cloud_escalate",
            token_prob=prob,
            margin=top_logprob_margin,
            reason=f"Non-private opt-in cartridge low confidence (p={prob}); eligible for Switchyard cloud escalation.",
            privacy_fence_enforced=False,
        )
