#!/usr/bin/env python3
"""
Convert a FLUX-style single-file checkpoint into a Diffusers transformer folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from statistics import median
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


@dataclass
class NormalizationReport:
    removed_comfy: int = 0
    removed_input_scale: int = 0
    removed_weight_scale: int = 0
    removed_weight_scale_2: int = 0
    dequantized_fp8_weights: int = 0
    dequantized_nvfp4_weights: int = 0
    dequantized_uint8_weights: int = 0
    cast_unscaled_fp8_tensors: int = 0
    nvfp4_weight_keys: list[str] = field(default_factory=list)


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
    p.add_argument(
        "--verify-nvfp4",
        action="store_true",
        help=(
            "Run strict sanity checks for decoded NVFP4 weights and abort if "
            "statistics look suspicious."
        ),
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


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """
    Reverse cuBLAS 32_4_4 swizzled layout used for NVFP4 block scales.
    Adapted from Comfy Kitchen's reference implementation.
    """
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def _dequantize_nvfp4_weight(
    packed_weight: torch.Tensor,
    block_scales: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Decode Comfy NVFP4 weights (E2M1 FP4 packed x2) into dense float tensors.
    """
    # FP4 E2M1 lookup table used by Comfy/PyTorch AO.
    # Indices 0-7 are positive values, 8-15 are negative counterparts.
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=output_dtype,
        device=packed_weight.device,
    ).unsqueeze(1)

    lo = packed_weight & 0x0F
    hi = packed_weight >> 4
    unpacked_nibbles = torch.stack([hi, lo], dim=-1).view(*packed_weight.shape[:-1], -1)
    fp4_values = torch.nn.functional.embedding(unpacked_nibbles.int(), lut).squeeze(-1)

    rows, cols = fp4_values.shape
    block_size = 16
    fp4_values_blocked = fp4_values.reshape(rows, -1, block_size)
    num_blocks_per_row = cols // block_size

    unswizzled_scales = _from_blocked(block_scales, num_rows=rows, num_cols=num_blocks_per_row)
    total_scale = per_tensor_scale.to(output_dtype) * unswizzled_scales.to(output_dtype)
    dequantized = fp4_values_blocked * total_scale.unsqueeze(-1)
    return dequantized.reshape(rows, cols).to(output_dtype)


def _normalize_checkpoint_for_diffusers(
    checkpoint: dict[str, object], *, target_dtype: torch.dtype
) -> tuple[dict[str, object], NormalizationReport]:
    checkpoint = _strip_prefix_if_present(checkpoint, "model.diffusion_model.")
    fp8_types = _float8_dtypes()
    scale_suffix = "_scale"
    scale2_suffix = "_scale_2"
    report = NormalizationReport()

    # Remove Comfy-specific metadata keys that Diffusers does not understand.
    for key in list(checkpoint.keys()):
        if key.endswith(".comfy_quant"):
            checkpoint.pop(key)
            report.removed_comfy += 1
        elif key.endswith(".input_scale"):
            checkpoint.pop(key)
            report.removed_input_scale += 1

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
            report.removed_weight_scale += 1
            if scale2_key in checkpoint:
                checkpoint.pop(scale2_key, None)
                report.removed_weight_scale_2 += 1
            report.dequantized_fp8_weights += 1
            continue

        # Variant 2: UINT8 packed weights with group scales and optional extra scalar scale.
        if weight.dtype == torch.uint8 and isinstance(scale, torch.Tensor):
            is_nvfp4 = False
            s = scale.to(dtype=torch.float32, device=weight.device)
            dequantized: torch.Tensor | None = None

            # Comfy NVFP4 packed path: uint8 stores two FP4 (E2M1) values per byte,
            # with per-tensor scale (`weight_scale_2`) and swizzled block scales.
            if (
                isinstance(scale2, torch.Tensor)
                and weight.ndim == 2
                and s.ndim == 2
                and s.shape[0] == weight.shape[0]
                and s.shape[1] > 0
                and scale.dtype in fp8_types
            ):
                # Heuristic sanity check for NVFP4 block geometry:
                # unpacked columns must be divisible by 16 and match 1x16 block scales.
                unpacked_cols = weight.shape[1] * 2
                if unpacked_cols % 16 == 0:
                    dequantized = _dequantize_nvfp4_weight(
                        packed_weight=weight,
                        block_scales=scale,
                        per_tensor_scale=scale2,
                        output_dtype=target_dtype,
                    )
                    is_nvfp4 = True
                    report.dequantized_nvfp4_weights += 1
                    report.nvfp4_weight_keys.append(key)

            # Fallback path for non-NVFP4 uint8 schemes (best-effort).
            if dequantized is None:
                w = weight.to(torch.float32)
                if s.shape == w.shape:
                    dequantized = ((w - 128.0) / 127.0) * s
                elif w.ndim == 2 and s.ndim == 2 and s.shape[0] == w.shape[0] and s.shape[1] > 0:
                    if w.shape[1] % s.shape[1] == 0:
                        group_size = w.shape[1] // s.shape[1]
                        expanded_scale = s.repeat_interleave(group_size, dim=1)
                        dequantized = ((w - 128.0) / 127.0) * expanded_scale

                if dequantized is not None and isinstance(scale2, torch.Tensor):
                    scale2_tensor = scale2.to(dtype=torch.float32, device=weight.device)
                    factor = float(scale2_tensor.item()) if scale2_tensor.numel() == 1 else scale2_tensor
                    dequantized = dequantized * factor

            if dequantized is not None:
                checkpoint[key] = dequantized.to(target_dtype)
                checkpoint.pop(scale_key, None)
                report.removed_weight_scale += 1
                if scale2_key in checkpoint:
                    checkpoint.pop(scale2_key, None)
                    report.removed_weight_scale_2 += 1
                if not is_nvfp4:
                    report.dequantized_uint8_weights += 1

    # Drop any remaining weight scale auxiliaries that were not consumed.
    for key in list(checkpoint.keys()):
        if key.endswith(".weight_scale"):
            checkpoint.pop(key)
            report.removed_weight_scale += 1
        elif key.endswith(".weight_scale_2"):
            checkpoint.pop(key)
            report.removed_weight_scale_2 += 1

    # Safety net for any remaining fp8 tensors without explicit scales.
    for key, value in list(checkpoint.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype not in fp8_types:
            continue
        checkpoint[key] = value.to(target_dtype)
        report.cast_unscaled_fp8_tensors += 1

    if any(
        (
            report.removed_comfy,
            report.removed_input_scale,
            report.removed_weight_scale,
            report.removed_weight_scale_2,
            report.dequantized_fp8_weights,
            report.dequantized_nvfp4_weights,
            report.dequantized_uint8_weights,
            report.cast_unscaled_fp8_tensors,
        )
    ):
        print(
            "Applied checkpoint normalization: "
            f"removed {report.removed_comfy} `.comfy_quant`, "
            f"{report.removed_input_scale} `.input_scale`, "
            f"{report.removed_weight_scale} `.weight_scale`, "
            f"{report.removed_weight_scale_2} `.weight_scale_2`; "
            f"dequantized {report.dequantized_fp8_weights} fp8 weights, "
            f"{report.dequantized_nvfp4_weights} nvfp4 weights, "
            f"{report.dequantized_uint8_weights} uint8-grouped weights"
            + (
                f", cast {report.cast_unscaled_fp8_tensors} unscaled fp8 tensors"
                if report.cast_unscaled_fp8_tensors
                else ""
            )
            + "."
        )

    return checkpoint, report


def _tensor_stats(tensor: torch.Tensor) -> tuple[float, float, float, float]:
    t = tensor.to(torch.float32)
    return float(t.min()), float(t.max()), float(t.mean()), float(t.std())


def _verify_nvfp4_decode(checkpoint: dict[str, object], report: NormalizationReport) -> None:
    if report.dequantized_nvfp4_weights == 0:
        print("NVFP4 verification: no NVFP4 weights decoded in this checkpoint.")
        return

    # Use non-NVFP4 dense weights as baseline for expected distribution scale.
    baseline_stds: list[float] = []
    for key, value in checkpoint.items():
        if key in report.nvfp4_weight_keys:
            continue
        if not key.endswith(".weight"):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim < 2:
            continue
        if not value.dtype.is_floating_point:
            continue
        _, _, _, std = _tensor_stats(value)
        if std > 0:
            baseline_stds.append(std)

    baseline_std = median(baseline_stds) if baseline_stds else 0.02
    print(
        "NVFP4 verification baseline: "
        f"median non-NVFP4 weight std={baseline_std:.6f} "
        f"(sampled {len(baseline_stds)} layers)."
    )

    suspicious: list[str] = []
    # Keep output concise but representative.
    sample_keys = report.nvfp4_weight_keys[:12]
    for key in sample_keys:
        value = checkpoint.get(key)
        if not isinstance(value, torch.Tensor):
            suspicious.append(f"{key}: missing decoded tensor")
            continue
        t_min, t_max, mean, std = _tensor_stats(value)
        mean_ratio = abs(mean) / (std + 1e-12)
        std_ratio = std / (baseline_std + 1e-12)
        print(
            "NVFP4 verify "
            f"{key}: min={t_min:.5f} max={t_max:.5f} mean={mean:.6f} std={std:.6f} "
            f"(mean/std={mean_ratio:.4f}, std/baseline={std_ratio:.3f})"
        )

        if std <= 1e-6:
            suspicious.append(f"{key}: near-zero std ({std:.6g})")
        if mean_ratio > 0.08:
            suspicious.append(f"{key}: strong mean bias ({mean_ratio:.4f})")
        if std_ratio < 0.08 or std_ratio > 12.0:
            suspicious.append(f"{key}: std ratio out of range ({std_ratio:.3f})")

    if suspicious:
        preview = "; ".join(suspicious[:6])
        raise ValueError(
            "NVFP4 verification failed; decoded weights look suspicious. "
            f"Examples: {preview}"
        )

    print(
        "NVFP4 verification passed: "
        f"checked {len(sample_keys)} decoded layer(s), no suspicious statistics."
    )


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
        checkpoint, report = _normalize_checkpoint_for_diffusers(checkpoint, target_dtype=dtype)
        if args.verify_nvfp4:
            _verify_nvfp4_decode(checkpoint, report)
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
