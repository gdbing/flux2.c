#!/usr/bin/env python3
"""
Convert a FLUX-style single-file checkpoint into a Diffusers transformer folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Model config repo or path (default: %(default)s)",
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


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not src.exists():
        print(f"Error: source checkpoint not found: {src}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_name(args.dtype)

    print(f"Loading single-file checkpoint: {src}")
    print(f"Using config: {args.config} (subfolder={args.subfolder})")
    try:
        model = Flux2Transformer2DModel.from_single_file(
            str(src),
            config=args.config,
            subfolder=args.subfolder,
            torch_dtype=dtype,
        )
    except KeyError as exc:
        # Some Civitai FLUX exports keep all keys under `model.diffusion_model.*`.
        # Diffusers' Flux2 special-key remap currently expects this prefix removed.
        if "double_blocks" not in str(exc) and "single_blocks" not in str(exc):
            raise
        print(
            "Hit Diffusers key-mapping mismatch; retrying with prefix normalization "
            "(stripping `model.diffusion_model.`)..."
        )
        checkpoint = load_single_file_checkpoint(str(src))
        checkpoint = _strip_prefix_if_present(checkpoint, "model.diffusion_model.")
        model = Flux2Transformer2DModel.from_single_file(
            checkpoint,
            config=args.config,
            subfolder=args.subfolder,
            torch_dtype=dtype,
        )

    print(f"Saving converted transformer to: {out}")
    model.save_pretrained(str(out), safe_serialization=True)

    out_file = out / "diffusion_pytorch_model.safetensors"
    if out_file.exists():
        print(f"Done: {out_file}")
    else:
        print(
            "Warning: conversion finished but output safetensors file was not found "
            f"at {out_file}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
