#!/usr/bin/env python3
"""
Convert a FLUX-style single-file checkpoint into a Diffusers transformer folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - provided by conversion env
    safe_open = None

try:
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint
except ImportError as exc:
    print(
        "Error: Flux2Transformer2DModel is not available in this diffusers build.\n"
        "Use this project environment with `uv run ...` so the pinned dependency is used.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

DEFAULT_CONFIG_4B = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_CONFIG_9B = "black-forest-labs/FLUX.2-klein-9B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a single-file FLUX checkpoint (for example Civitai) into "
            "a Diffusers transformer directory."
        )
    )
    p.add_argument("--src", required=True, help="Input .safetensors file")
    p.add_argument("--out", required=True, help="Output transformer directory")
    p.add_argument(
        "--config",
        default="auto",
        help=(
            "Model config repo/path, or `auto` to detect 4B vs 9B from checkpoint "
            "tensor shapes (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--subfolder",
        default="transformer",
        help="Subfolder within --config containing transformer config (default: %(default)s)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="Dtype used for loading/saving (default: %(default)s)",
    )
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def _strip_prefix_if_present(checkpoint: dict[str, object], prefix: str) -> dict[str, object]:
    if not any(k.startswith(prefix) for k in checkpoint.keys()):
        return checkpoint
    remapped: dict[str, object] = {}
    for key, value in checkpoint.items():
        if key.startswith(prefix):
            remapped[key[len(prefix) :]] = value
        else:
            remapped[key] = value
    return remapped


def _float8_dtypes() -> set[torch.dtype]:
    out: set[torch.dtype] = set()
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        if hasattr(torch, name):
            out.add(getattr(torch, name))
    return out


def _normalize_checkpoint_for_diffusers(
    checkpoint: dict[str, object], *, target_dtype: torch.dtype
) -> dict[str, object]:
    checkpoint = _strip_prefix_if_present(checkpoint, "model.diffusion_model.")
    fp8_types = _float8_dtypes()
    scale_suffix = "_scale"
    scale2_suffix = "_scale_2"
    removed_comfy = 0
    removed_input_scale = 0
    removed_weight_scale = 0
    removed_weight_scale_2 = 0
    dequantized_weights = 0
    dequantized_uint8_weights = 0
    cast_fp8_without_scale = 0

    # Remove Comfy-specific metadata keys that Diffusers does not understand.
    for key in list(checkpoint.keys()):
        if key.endswith(".comfy_quant"):
            checkpoint.pop(key)
            removed_comfy += 1
        elif key.endswith(".input_scale"):
            checkpoint.pop(key)
            removed_input_scale += 1

    # Fold quantization scales into `*.weight` tensors for Comfy checkpoints.
    for key in [k for k in checkpoint.keys() if k.endswith(".weight")]:
        weight = checkpoint.get(key)
        if not isinstance(weight, torch.Tensor):
            continue

        scale_key = f"{key}{scale_suffix}"
        scale2_key = f"{key}{scale2_suffix}"
        scale = checkpoint.get(scale_key)
        scale2 = checkpoint.get(scale2_key)

        # Variant 1: FP8 tensor weights with scalar/tensor weight scale.
        if weight.dtype in fp8_types and isinstance(scale, torch.Tensor):
            scale_tensor = scale.to(dtype=torch.float32, device=weight.device)
            if scale_tensor.numel() == 1:
                dequantized = weight.to(torch.float32) * float(scale_tensor.item())
            else:
                dequantized = weight.to(torch.float32) * scale_tensor

            if isinstance(scale2, torch.Tensor):
                scale2_tensor = scale2.to(dtype=torch.float32, device=weight.device)
                factor = float(scale2_tensor.item()) if scale2_tensor.numel() == 1 else scale2_tensor
                dequantized = dequantized * factor

            checkpoint[key] = dequantized.to(target_dtype)
            checkpoint.pop(scale_key, None)
            removed_weight_scale += 1
            if scale2_key in checkpoint:
                checkpoint.pop(scale2_key, None)
                removed_weight_scale_2 += 1
            dequantized_weights += 1
            continue

        # Variant 2: UINT8 packed weights with group scales and optional extra scalar scale.
        if weight.dtype == torch.uint8 and isinstance(scale, torch.Tensor):
            w = weight.to(torch.float32)
            s = scale.to(dtype=torch.float32, device=weight.device)
            dequantized: torch.Tensor | None = None

            # Comfy packed-4bit path: uint8 stores two 4-bit values per byte.
            if (
                isinstance(scale2, torch.Tensor)
                and w.ndim == 2
                and s.ndim == 2
                and s.shape[0] == w.shape[0]
                and s.shape[1] > 0
                and (w.shape[1] * 2) % s.shape[1] == 0
            ):
                low = (weight & 0x0F).to(torch.float32)
                high = (weight >> 4).to(torch.float32)
                unpacked = torch.empty(
                    (w.shape[0], w.shape[1] * 2), dtype=torch.float32, device=weight.device
                )
                unpacked[:, 0::2] = low
                unpacked[:, 1::2] = high
                group_size = unpacked.shape[1] // s.shape[1]
                expanded_scale = s.repeat_interleave(group_size, dim=1)
                dequantized = (unpacked - 8.0) * expanded_scale
            elif s.shape == w.shape:
                dequantized = ((w - 128.0) / 127.0) * s
            elif w.ndim == 2 and s.ndim == 2 and s.shape[0] == w.shape[0] and s.shape[1] > 0:
                if w.shape[1] % s.shape[1] == 0:
                    group_size = w.shape[1] // s.shape[1]
                    expanded_scale = s.repeat_interleave(group_size, dim=1)
                    dequantized = ((w - 128.0) / 127.0) * expanded_scale

            if dequantized is not None:
                if isinstance(scale2, torch.Tensor):
                    scale2_tensor = scale2.to(dtype=torch.float32, device=weight.device)
                    factor = float(scale2_tensor.item()) if scale2_tensor.numel() == 1 else scale2_tensor
                    dequantized = dequantized * factor
                checkpoint[key] = dequantized.to(target_dtype)
                checkpoint.pop(scale_key, None)
                removed_weight_scale += 1
                if scale2_key in checkpoint:
                    checkpoint.pop(scale2_key, None)
                    removed_weight_scale_2 += 1
                dequantized_uint8_weights += 1

    # Drop any remaining weight scale auxiliaries that were not consumed.
    for key in list(checkpoint.keys()):
        if key.endswith(".weight_scale"):
            checkpoint.pop(key)
            removed_weight_scale += 1
        elif key.endswith(".weight_scale_2"):
            checkpoint.pop(key)
            removed_weight_scale_2 += 1

    # Safety net for any remaining fp8 tensors without explicit scales.
    for key, value in list(checkpoint.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype not in fp8_types:
            continue
        checkpoint[key] = value.to(target_dtype)
        cast_fp8_without_scale += 1

    if any(
        (
            removed_comfy,
            removed_input_scale,
            removed_weight_scale,
            removed_weight_scale_2,
            dequantized_weights,
            dequantized_uint8_weights,
            cast_fp8_without_scale,
        )
    ):
        print(
            "Applied checkpoint normalization: "
            f"removed {removed_comfy} `.comfy_quant`, "
            f"{removed_input_scale} `.input_scale`, "
            f"{removed_weight_scale} `.weight_scale`, "
            f"{removed_weight_scale_2} `.weight_scale_2`; "
            f"dequantized {dequantized_weights} fp8 weights, "
            f"{dequantized_uint8_weights} uint8-grouped weights"
            + (
                f", cast {cast_fp8_without_scale} unscaled fp8 tensors"
                if cast_fp8_without_scale
                else ""
            )
            + "."
        )

    return checkpoint


def _resolve_local_config(model_size: str) -> str | None:
    repo_root = Path(__file__).resolve().parent.parent
    local_map = {
        "4B": repo_root / "flux-klein-model",
        "9B": repo_root / "flux-klein-9b",
    }
    local = local_map.get(model_size)
    if local is None:
        return None
    if (local / "transformer" / "config.json").exists():
        return str(local)
    return None


def _detect_model_size_from_checkpoint(src: Path) -> str | None:
    if safe_open is None:
        return None

    key_candidates = (
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_img.linear.weight",
        "img_in.weight",
        "txt_in.weight",
    )
    prefixes = ("", "model.", "model.diffusion_model.")

    try:
        with safe_open(str(src), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for suffix in key_candidates:
                for prefix in prefixes:
                    key = f"{prefix}{suffix}"
                    if key not in keys:
                        continue
                    shape = tuple(f.get_slice(key).get_shape())
                    if len(shape) < 2:
                        continue
                    hidden_size = int(shape[1])
                    if hidden_size == 3072:
                        return "4B"
                    if hidden_size == 4096:
                        return "9B"
    except Exception as exc:
        print(
            f"Warning: could not inspect checkpoint for auto config detection ({exc}); "
            "falling back to 4B config."
        )
    return None


def resolve_config(config_arg: str, src: Path) -> str:
    if config_arg != "auto":
        return config_arg

    model_size = _detect_model_size_from_checkpoint(src)
    if model_size == "9B":
        config = _resolve_local_config("9B") or DEFAULT_CONFIG_9B
        print(f"Auto-detected model size: 9B (hidden size 4096). Using config: {config}")
        return config
    if model_size == "4B":
        config = _resolve_local_config("4B") or DEFAULT_CONFIG_4B
        print(f"Auto-detected model size: 4B (hidden size 3072). Using config: {config}")
        return config

    fallback = _resolve_local_config("4B") or DEFAULT_CONFIG_4B
    print(f"Auto-detection failed; falling back to config: {fallback}")
    return fallback


def _count_shards(index_file: Path) -> int | None:
    try:
        data = json.loads(index_file.read_text())
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return None
        return len(set(weight_map.values()))
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not src.exists():
        print(f"Error: source checkpoint not found: {src}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_name(args.dtype)
    config = resolve_config(args.config, src)

    print(f"Loading single-file checkpoint: {src}")
    print(f"Using config: {config} (subfolder={args.subfolder})")
    try:
        model = Flux2Transformer2DModel.from_single_file(
            str(src),
            config=config,
            subfolder=args.subfolder,
            torch_dtype=dtype,
        )
    except Exception as exc:
        print(
            "Direct conversion failed "
            f"({exc.__class__.__name__}: {exc}). "
            "Retrying with checkpoint normalization..."
        )
        checkpoint = load_single_file_checkpoint(str(src))
        checkpoint = _normalize_checkpoint_for_diffusers(checkpoint, target_dtype=dtype)
        model = Flux2Transformer2DModel.from_single_file(
            checkpoint,
            config=config,
            subfolder=args.subfolder,
            torch_dtype=dtype,
        )

    print(f"Saving converted transformer to: {out}")
    model.save_pretrained(str(out), safe_serialization=True)

    out_file = out / "diffusion_pytorch_model.safetensors"
    index_file = out / "diffusion_pytorch_model.safetensors.index.json"
    if out_file.exists():
        print(f"Done: {out_file}")
    elif index_file.exists():
        shard_count = _count_shards(index_file)
        if shard_count is None:
            print(f"Done: {index_file} (sharded output)")
        else:
            print(f"Done: {index_file} (sharded output, {shard_count} files)")
    else:
        print(
            "Warning: conversion finished but neither a single safetensors file nor shard index "
            f"was found in {out}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
