"""Automated Local R&D Radar Scout for Colossus (Zero-Dependency Stdlib).

Keeps deadkey, the user, Hermes, and Neon abreast of bleeding-edge local AI, Blackwell `sm_120`,
inference engine, voice, and open-weight GGUF breakthroughs without manual X/GitHub scrolling.

Checks public GitHub releases & HuggingFace model feeds, deduplicates against
`cartridges/blank-lab/data/rd_radar.jsonl`, and writes a clean, Disk-Equals-Truth radar brief
that can be triggered manually via `/v1/rd/scout` or on a recurring schedule!
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GITHUB_REPOS = (
    ("ggml-org/llama.cpp", "inference_engine"),
    ("Niko1221/Strata", "moe_tier_engine"),
    ("vllm-project/vllm", "blackwell_engine"),
    ("huggingface/trl", "dpo_lora_training"),
    ("huggingface/peft", "lora_adapters"),
    ("unslothai/unsloth", "fast_lora_5090"),
)

HF_QUERIES = (
    ("gemma-4 gguf", "model_gemma4"),
    ("qwen3.8 gguf", "model_qwen38"),
    ("inkling-2 gguf", "model_inkling2"),
    ("glm-4.7 gguf", "model_glm47"),
    ("orpheus tts", "voice_tts"),
)


def _fetch_json(url: str, timeout: float = 8.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "deadkeyCore-RDScout/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def run_rd_scout(radar_path: Path) -> dict[str, Any]:
    """Scans tracked engines & model families and appends unseen updates to `rd_radar.jsonl`."""
    radar_path.parent.mkdir(parents=True, exist_ok=True)
    seen_keys: set[str] = set()
    existing: list[dict[str, Any]] = []
    if radar_path.exists():
        for ln in radar_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.strip():
                try:
                    row = json.loads(ln)
                    existing.append(row)
                    if row.get("key"):
                        seen_keys.add(str(row["key"]))
                except Exception:
                    pass

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_items: list[dict[str, Any]] = []
    errors: list[str] = []

    # 1. GitHub latest releases for core inference & training engines
    for repo, tag in GITHUB_REPOS:
        try:
            data = _fetch_json(f"https://api.github.com/repos/{repo}/releases/latest", timeout=6.0)
            rel_tag = str(data.get("tag_name") or data.get("name") or "")
            key = f"gh:{repo}:{rel_tag}"
            if rel_tag and key not in seen_keys:
                item = {
                    "ts": now_iso,
                    "key": key,
                    "category": tag,
                    "source": f"github:{repo}",
                    "title": f"{repo} released {rel_tag}",
                    "published": str(data.get("published_at") or ""),
                    "url": str(data.get("html_url") or ""),
                    "summary": str(data.get("body") or "")[:240].replace("\r", " ").replace("\n", " "),
                }
                new_items.append(item)
                seen_keys.add(key)
        except Exception as exc:
            errors.append(f"{repo}: {exc}")

    # 2. HuggingFace recently updated GGUF / Voice repos
    for query, tag in HF_QUERIES:
        try:
            url = f"https://huggingface.co/api/models?search={urllib.parse.quote(query)}&sort=downloads&direction=-1&limit=3"
            models = _fetch_json(url, timeout=6.0)
            if isinstance(models, list):
                for m in models[:2]:
                    mid = str(m.get("id") or "")
                    last_mod = str(m.get("lastModified") or "")[:10]
                    key = f"hf:{mid}:{last_mod}"
                    if mid and key not in seen_keys:
                        item = {
                            "ts": now_iso,
                            "key": key,
                            "category": tag,
                            "source": "huggingface",
                            "title": f"HF: {mid}",
                            "published": last_mod,
                            "url": f"https://huggingface.co/{mid}",
                            "summary": f"downloads={m.get('downloads', 0)}, likes={m.get('likes', 0)}",
                        }
                        new_items.append(item)
                        seen_keys.add(key)
        except Exception as exc:
            errors.append(f"hf({query}): {exc}")

    if new_items:
        with radar_path.open("a", encoding="utf-8") as fh:
            for it in new_items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "ts": now_iso,
        "new_count": len(new_items),
        "total_tracked": len(existing) + len(new_items),
        "new_items": new_items,
        "errors": errors,
        "radar_path": str(radar_path),
    }


def read_radar(radar_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not radar_path.exists():
        return []
    rows = []
    for ln in radar_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows[-limit:][::-1]


if __name__ == "__main__":
    p = Path(__file__).resolve().parent.parent / "cartridges" / "blank-lab" / "data" / "rd_radar.jsonl"
    print(json.dumps(run_rd_scout(p), indent=2))
