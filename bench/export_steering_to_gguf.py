"""Export a Latent Deadkey PyTorch steering vector (.pt) to a native llama.cpp GGUF Control Vector.

Produces a sparse, ultra-compact GGUF v3 control vector file compatible with:
  llama-server / llama-cli --control-vector <file.gguf>
                           --control-vector-scaled <file.gguf>:<scale>

Layer indexing contract in llama.cpp (`common/common.cpp` + `src/llama-adapter.cpp`):
  - Tensor names are `direction.<layer_idx>` (`layer_idx >= 1`, F32, shape `[n_embd]`).
  - `common_control_vector_load_one` places `direction.k` at offset `(k - 1) * n_embd`,
    zero-filling any omitted earlier layers automatically.
  - `llama_adapter_cvec::apply` copies `data[(il - 1) * n_embd]` into `tensors[il]`
    for `il = 1 .. n_layer - 1` (`tensors[0]` is null).
  - During the forward pass, `apply_to(ctx, cur, il)` adds `tensors[il]` to the output
    hidden state of decoder block `il`, matching PyTorch `model.model.layers[il]` 1:1.

Therefore PyTorch 0-based layer `l` (`l >= 1`) maps directly to `direction.{l}`!
Optionally supports `--layer-offset` (default `0`) for empirical A/B layer-alignment checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import gguf


def export_pt_to_gguf(
    pt_path: Path,
    out_path: Path,
    *,
    model_hint: str = "qwen3",
    scale: float = 1.0,
    layer_offset: int = 0,
    dense_zeros: bool = False,
    total_layers: int | None = None,
) -> dict[str, Any]:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    vectors: dict[int, torch.Tensor] = payload["vectors_fp16"]
    selected = sorted(int(k) for k in vectors.keys())
    if not selected:
        raise ValueError(f"No layer vectors found in {pt_path}")

    first_vec = next(iter(vectors.values())).detach().cpu().float().numpy().reshape(-1)
    n_embd = int(first_vec.shape[0])

    mapped: dict[int, np.ndarray] = {}
    for l in selected:
        idx = int(l) + int(layer_offset)
        if idx < 1:
            raise ValueError(f"llama.cpp control vector tensor index must be >= 1, got {idx} (layer {l} + offset {layer_offset})")
        vec = vectors[l].detach().cpu().float().numpy().reshape(-1)
        if vec.shape[0] != n_embd:
            raise ValueError(f"Layer {l} dim mismatch: {vec.shape[0]} != {n_embd}")
        mapped[idx] = (vec * float(scale)).astype(np.float32, copy=False)

    max_layer = max(mapped.keys())
    layer_count = int(total_layers) if total_layers is not None else max_layer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="controlvector")
    writer.add_string("controlvector.model_hint", model_hint)
    writer.add_uint32("controlvector.layer_count", layer_count)
    writer.add_string("general.name", f"Deadkey Latent Epistemic Control Vector ({model_hint})")
    writer.add_string("general.author", "Deadkey Labs")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string(
        "general.description",
        f"Sparse activation-space epistemic steering vector across layers {sorted(mapped.keys())} "
        f"(extracted from {payload.get('n_pairs', 'unknown')} zero-PII contrastive pairs, scale={scale}).",
    )

    written_tensors: list[dict[str, Any]] = []
    if dense_zeros:
        zero_vec = np.zeros((n_embd,), dtype=np.float32)
        for idx in range(1, layer_count + 1):
            arr = mapped.get(idx, zero_vec)
            tname = f"direction.{idx}"
            writer.add_tensor(tname, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            written_tensors.append({"name": tname, "norm": round(float(np.linalg.norm(arr)), 4)})
    else:
        for idx in sorted(mapped.keys()):
            arr = mapped[idx]
            tname = f"direction.{idx}"
            writer.add_tensor(tname, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            written_tensors.append({"name": tname, "norm": round(float(np.linalg.norm(arr)), 4)})

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    raw_bytes = out_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return {
        "source_pt": str(pt_path),
        "out_gguf": str(out_path),
        "bytes": len(raw_bytes),
        "kib": round(len(raw_bytes) / 1024.0, 2),
        "sha256": sha256,
        "model_hint": model_hint,
        "n_embd": n_embd,
        "scale_baked": float(scale),
        "layer_offset": int(layer_offset),
        "layer_count": layer_count,
        "tensor_count": len(written_tensors),
        "tensors": written_tensors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Deadkey .pt steering vector to llama.cpp GGUF control vector")
    ap.add_argument("--in", dest="in_path", default="data/deadkey_qwen3_8b_steering_v1.pt")
    ap.add_argument("--out", dest="out_path", default="data/deadkey_qwen3_8b_cvec_v1.gguf")
    ap.add_argument("--model-hint", default="qwen3")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale baked into exported F32 tensors (default 1.0)")
    ap.add_argument("--layer-offset", type=int, default=0, help="Optional integer offset applied to layer indices")
    ap.add_argument("--dense-zeros", action="store_true", help="Write explicit zero tensors for non-selected layers")
    ap.add_argument("--total-layers", type=int, default=None, help="Override controlvector.layer_count")
    args = ap.parse_args()

    receipt = export_pt_to_gguf(
        Path(args.in_path),
        Path(args.out_path),
        model_hint=args.model_hint,
        scale=args.scale,
        layer_offset=args.layer_offset,
        dense_zeros=args.dense_zeros,
        total_layers=args.total_layers,
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
