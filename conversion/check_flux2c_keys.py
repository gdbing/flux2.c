#!/usr/bin/env python3
"""
Check whether a transformer safetensors file has the key schema required by flux2.c.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


REQUIRED_KEYS = [
    "x_embedder.weight",
    "context_embedder.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.linear.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
    "transformer_blocks.0.attn.norm_q.weight",
    "transformer_blocks.0.attn.norm_k.weight",
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.attn.to_k.weight",
    "transformer_blocks.0.attn.to_v.weight",
    "transformer_blocks.0.attn.to_out.0.weight",
    "transformer_blocks.0.attn.norm_added_q.weight",
    "transformer_blocks.0.attn.norm_added_k.weight",
    "transformer_blocks.0.attn.add_q_proj.weight",
    "transformer_blocks.0.attn.add_k_proj.weight",
    "transformer_blocks.0.attn.add_v_proj.weight",
    "transformer_blocks.0.attn.to_add_out.weight",
    "transformer_blocks.0.ff.linear_in.weight",
    "transformer_blocks.0.ff.linear_out.weight",
    "transformer_blocks.0.ff_context.linear_in.weight",
    "transformer_blocks.0.ff_context.linear_out.weight",
    "single_transformer_blocks.0.attn.norm_q.weight",
    "single_transformer_blocks.0.attn.norm_k.weight",
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    "single_transformer_blocks.0.attn.to_out.weight",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        help=(
            "Path to a transformer safetensors file, a sharded safetensors "
            "index JSON, or a transformer directory containing either."
        ),
    )
    p.add_argument(
        "--show-sample",
        type=int,
        default=20,
        help="Print the first N keys for quick inspection (default: %(default)s)",
    )
    return p.parse_args()


def load_keys_from_safetensors(path: Path) -> list[str]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return sorted(k for k in header.keys() if k != "__metadata__")


def load_keys_from_index(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid shard index file (missing dict weight_map): {path}")
    return sorted(weight_map.keys())


def resolve_input_path(path: Path) -> tuple[Path, str]:
    if path.is_dir():
        single_file = path / "diffusion_pytorch_model.safetensors"
        index_file = path / "diffusion_pytorch_model.safetensors.index.json"
        if single_file.exists():
            return single_file, "safetensors"
        if index_file.exists():
            return index_file, "index"
        raise FileNotFoundError(
            "Directory does not contain diffusion_pytorch_model.safetensors "
            "or diffusion_pytorch_model.safetensors.index.json"
        )

    if path.name.endswith(".safetensors.index.json"):
        return path, "index"
    if path.suffix == ".safetensors":
        return path, "safetensors"

    raise ValueError(
        "Unsupported input path. Use a .safetensors file, a "
        ".safetensors.index.json file, or a transformer directory."
    )


def main() -> int:
    args = parse_args()
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"Error: file not found: {path}")
        return 1

    try:
        resolved_path, input_kind = resolve_input_path(path)
        if input_kind == "index":
            keys = load_keys_from_index(resolved_path)
        else:
            keys = load_keys_from_safetensors(resolved_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1

    key_set = set(keys)

    print(f"Input: {resolved_path}")
    print(f"Input type: {input_kind}")
    print(f"Tensor count: {len(keys)}")
    print("")

    missing = [k for k in REQUIRED_KEYS if k not in key_set]
    if missing:
        print("Missing required flux2.c keys:")
        for k in missing:
            print(f"  - {k}")
        print("")
        print("Result: NOT flux2.c-compatible yet")
    else:
        print("All required flux2.c keys are present.")
        print("Result: Looks flux2.c-compatible")

    if args.show_sample > 0:
        n = min(args.show_sample, len(keys))
        print("")
        print(f"Sample keys (first {n}):")
        for k in keys[:n]:
            print(f"  - {k}")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
