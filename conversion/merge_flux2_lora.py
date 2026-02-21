#!/usr/bin/env python3
"""
Merge a FLUX adapter into a copied Diffusers transformer directory.

Supported adapter formats:
- LoRA (lora_A/lora_B or lora_down/lora_up)
- LoKR (lokr_w1/lokr_w2 and decomposed variants)

This script ports the key-matching behavior from flux2.c's C LoRA loader:
- Suffix-based module matching with unknown key roots.
- Fallback aliases for non-standard FLUX module names.
- Fallback split for combined qkv adapters used by some trainers.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import Flux2Transformer2DModel
from safetensors.torch import load_file


@dataclass(frozen=True)
class LoraCandidate:
    key: str
    root: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class LoraPair:
    key_a: str
    key_b: str
    a: torch.Tensor
    b: torch.Tensor


@dataclass
class MergeStats:
    modules_applied: int = 0
    source_tensors_used: int = 0
    source_tensors_total: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse a LoRA/LoKR adapter into a FLUX transformer and save merged weights."
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model root (contains transformer/) or transformer directory directly.",
    )
    p.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="Path to adapter .safetensors file (LoRA or LoKR)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output merged transformer directory")
    p.add_argument("--lora-scale", type=float, default=1.0, help="Adapter scale for fusion")
    p.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Dtype used to load, fuse, and save transformer weights",
    )
    p.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size passed to save_pretrained (default: 10GB)",
    )
    return p.parse_args()


def resolve_transformer_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "config.json").exists():
        return path
    candidate = path / "transformer"
    if (candidate / "config.json").exists():
        return candidate
    raise FileNotFoundError(f"Could not find transformer/config.json under: {path}")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def collect_lora_candidates(
    state_dict: dict[str, torch.Tensor],
    module: str,
    suffixes: tuple[str, ...],
) -> list[LoraCandidate]:
    cands: list[LoraCandidate] = []
    for key, tensor in state_dict.items():
        for suffix in suffixes:
            target = f"{module}.{suffix}"
            if not key.endswith(target):
                continue
            root = key[: -len(target)]
            if root and not root.endswith("."):
                continue
            cands.append(LoraCandidate(key=key, root=root, tensor=tensor))
            break
    return cands


def resolve_lora_pair(
    module: str,
    format_label: str,
    cands_a: list[LoraCandidate],
    cands_b: list[LoraCandidate],
) -> LoraPair | None:
    if not cands_a and not cands_b:
        return None
    if not cands_a or not cands_b:
        raise ValueError(
            f"LoRA pair incomplete for module {module} (expected {format_label})"
        )

    matches: list[tuple[LoraCandidate, LoraCandidate]] = []
    for a in cands_a:
        for b in cands_b:
            if a.root == b.root:
                matches.append((a, b))

    if len(matches) == 1:
        a, b = matches[0]
        return LoraPair(key_a=a.key, key_b=b.key, a=a.tensor, b=b.tensor)

    bare = [m for m in matches if m[0].root == "" and m[1].root == ""]
    if len(matches) > 1 and len(bare) == 1:
        a, b = bare[0]
        return LoraPair(key_a=a.key, key_b=b.key, a=a.tensor, b=b.tensor)

    if not matches:
        raise ValueError(f"LoRA roots mismatch for module {module} ({format_label})")
    raise ValueError(f"LoRA pair ambiguous for module {module} ({format_label})")


def find_lora_pair(
    state_dict: dict[str, torch.Tensor],
    module: str,
) -> LoraPair | None:
    cands_a = collect_lora_candidates(
        state_dict, module, ("lora_A.weight", "lora_A.default.weight")
    )
    cands_b = collect_lora_candidates(
        state_dict, module, ("lora_B.weight", "lora_B.default.weight")
    )
    pair = resolve_lora_pair(module, "lora_A/lora_B", cands_a, cands_b)
    if pair is not None:
        return pair

    cands_a = collect_lora_candidates(
        state_dict, module, ("lora_down.weight", "lora_down.default.weight")
    )
    cands_b = collect_lora_candidates(
        state_dict, module, ("lora_up.weight", "lora_up.default.weight")
    )
    return resolve_lora_pair(module, "lora_down/lora_up", cands_a, cands_b)


def validate_pair_shapes(pair: LoraPair, module: str) -> None:
    if pair.a.ndim != 2 or pair.b.ndim != 2:
        raise ValueError(
            f"LoRA tensors for {module} must be rank-2, got A={tuple(pair.a.shape)} B={tuple(pair.b.shape)}"
        )
    if pair.b.shape[1] != pair.a.shape[0]:
        raise ValueError(
            f"LoRA rank mismatch for {module}: A={tuple(pair.a.shape)} B={tuple(pair.b.shape)}"
        )


def normalize_lora_state_dict(
    raw: dict[str, torch.Tensor],
    num_double_layers: int,
    num_single_layers: int,
) -> tuple[dict[str, torch.Tensor], MergeStats]:
    out: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    stats = MergeStats(source_tensors_total=len(raw))

    def add_pair(dest_module: str, pair: LoraPair) -> None:
        validate_pair_shapes(pair, dest_module)
        key_a = f"transformer.{dest_module}.lora_A.weight"
        key_b = f"transformer.{dest_module}.lora_B.weight"
        if key_a in out or key_b in out:
            raise ValueError(f"Duplicate normalized LoRA destination for module {dest_module}")
        out[key_a] = pair.a.contiguous()
        out[key_b] = pair.b.contiguous()
        used_keys.add(pair.key_a)
        used_keys.add(pair.key_b)
        stats.modules_applied += 1

    def add_from_source(dest_module: str, source_module: str) -> bool:
        pair = find_lora_pair(raw, source_module)
        if pair is None:
            return False
        add_pair(dest_module, pair)
        return True

    def add_qkv_fallback(source_module: str, dest_modules: tuple[str, str, str]) -> bool:
        pair = find_lora_pair(raw, source_module)
        if pair is None:
            return False
        validate_pair_shapes(pair, source_module)
        out_rows = int(pair.b.shape[0])
        if out_rows % 3 != 0:
            raise ValueError(
                f"Expected LoRA B rows divisible by 3 for {source_module}, got {tuple(pair.b.shape)}"
            )
        split = out_rows // 3
        for i, dest in enumerate(dest_modules):
            part = pair.b[i * split : (i + 1) * split, :].contiguous()
            add_pair(
                dest,
                LoraPair(
                    key_a=pair.key_a,
                    key_b=pair.key_b,
                    a=pair.a.contiguous(),
                    b=part,
                ),
            )
        return True

    # Shared modules.
    for module in (
        "x_embedder",
        "context_embedder",
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
        "proj_out",
    ):
        add_from_source(module, module)

    # Double-stream blocks with fallback aliases.
    for i in range(num_double_layers):
        q = add_from_source(
            f"transformer_blocks.{i}.attn.to_q",
            f"transformer_blocks.{i}.attn.to_q",
        )
        k = add_from_source(
            f"transformer_blocks.{i}.attn.to_k",
            f"transformer_blocks.{i}.attn.to_k",
        )
        v = add_from_source(
            f"transformer_blocks.{i}.attn.to_v",
            f"transformer_blocks.{i}.attn.to_v",
        )
        if not (q or k or v):
            add_qkv_fallback(
                f"double_blocks.{i}.img_attn.qkv",
                (
                    f"transformer_blocks.{i}.attn.to_q",
                    f"transformer_blocks.{i}.attn.to_k",
                    f"transformer_blocks.{i}.attn.to_v",
                ),
            )

        if not add_from_source(
            f"transformer_blocks.{i}.attn.to_out.0",
            f"transformer_blocks.{i}.attn.to_out.0",
        ):
            add_from_source(
                f"transformer_blocks.{i}.attn.to_out.0",
                f"double_blocks.{i}.img_attn.proj",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff.linear_in",
            f"transformer_blocks.{i}.ff.linear_in",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff.linear_in",
                f"double_blocks.{i}.img_mlp.0",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff.linear_out",
            f"transformer_blocks.{i}.ff.linear_out",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff.linear_out",
                f"double_blocks.{i}.img_mlp.2",
            )

        q = add_from_source(
            f"transformer_blocks.{i}.attn.add_q_proj",
            f"transformer_blocks.{i}.attn.add_q_proj",
        )
        k = add_from_source(
            f"transformer_blocks.{i}.attn.add_k_proj",
            f"transformer_blocks.{i}.attn.add_k_proj",
        )
        v = add_from_source(
            f"transformer_blocks.{i}.attn.add_v_proj",
            f"transformer_blocks.{i}.attn.add_v_proj",
        )
        if not (q or k or v):
            add_qkv_fallback(
                f"double_blocks.{i}.txt_attn.qkv",
                (
                    f"transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer_blocks.{i}.attn.add_v_proj",
                ),
            )

        if not add_from_source(
            f"transformer_blocks.{i}.attn.to_add_out",
            f"transformer_blocks.{i}.attn.to_add_out",
        ):
            add_from_source(
                f"transformer_blocks.{i}.attn.to_add_out",
                f"double_blocks.{i}.txt_attn.proj",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff_context.linear_in",
            f"transformer_blocks.{i}.ff_context.linear_in",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff_context.linear_in",
                f"double_blocks.{i}.txt_mlp.0",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff_context.linear_out",
            f"transformer_blocks.{i}.ff_context.linear_out",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff_context.linear_out",
                f"double_blocks.{i}.txt_mlp.2",
            )

    # Single-stream blocks with fallback aliases.
    for i in range(num_single_layers):
        if not add_from_source(
            f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
            f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
        ):
            add_from_source(
                f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
                f"single_blocks.{i}.linear1",
            )

        if not add_from_source(
            f"single_transformer_blocks.{i}.attn.to_out",
            f"single_transformer_blocks.{i}.attn.to_out",
        ):
            add_from_source(
                f"single_transformer_blocks.{i}.attn.to_out",
                f"single_blocks.{i}.linear2",
            )

    stats.source_tensors_used = len(used_keys)
    return out, stats


def detect_adapter_format(raw: dict[str, torch.Tensor]) -> str:
    keys = raw.keys()
    if any(".lora_A" in k or ".lora_B" in k or ".lora_down" in k or ".lora_up" in k for k in keys):
        return "lora"
    if any(".lokr_w1" in k or ".lokr_w2" in k for k in keys):
        return "lokr"
    if any(".hada_w1_a" in k or ".hada_w1_b" in k or ".hada_w2_a" in k or ".hada_w2_b" in k for k in keys):
        return "loha"
    return "unknown"


LOKR_SUFFIXES = (
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t1",
    "lokr_t2",
    "alpha",
)


@dataclass(frozen=True)
class LokrModule:
    module: str
    keys: dict[str, str]
    tensors: dict[str, torch.Tensor | None]


def find_lokr_module(
    state_dict: dict[str, torch.Tensor],
    module: str,
) -> LokrModule | None:
    by_root: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state_dict.items():
        for suffix in LOKR_SUFFIXES:
            target = f"{module}.{suffix}"
            if not key.endswith(target):
                continue
            root = key[: -len(target)]
            if root and not root.endswith("."):
                continue
            root_bucket = by_root.setdefault(root, {})
            if suffix in root_bucket:
                raise ValueError(
                    f"Duplicate LoKR tensor for module {module} and suffix {suffix} under root {root!r}"
                )
            root_bucket[suffix] = (key, tensor)
            break

    if not by_root:
        return None

    def root_is_valid(bucket: dict[str, tuple[str, torch.Tensor]]) -> bool:
        has_w1 = ("lokr_w1" in bucket) or (
            "lokr_w1_a" in bucket and "lokr_w1_b" in bucket
        )
        has_w2 = ("lokr_w2" in bucket) or (
            "lokr_w2_a" in bucket and "lokr_w2_b" in bucket
        )
        return has_w1 and has_w2

    valid_roots = [root for root, bucket in by_root.items() if root_is_valid(bucket)]
    if not valid_roots:
        details = ", ".join(
            f"{root!r}:{sorted(bucket.keys())}" for root, bucket in sorted(by_root.items())
        )
        raise ValueError(
            f"LoKR module {module} has incomplete tensors across roots: {details}"
        )

    if len(valid_roots) == 1:
        chosen_root = valid_roots[0]
    elif "" in valid_roots:
        chosen_root = ""
    else:
        details = ", ".join(repr(root) for root in valid_roots)
        raise ValueError(
            f"LoKR module {module} is ambiguous across roots: {details}"
        )

    bucket = by_root[chosen_root]
    keys: dict[str, str] = {}
    tensors: dict[str, torch.Tensor | None] = {}
    for suffix in LOKR_SUFFIXES:
        if suffix in bucket:
            key, tensor = bucket[suffix]
            keys[suffix] = key
            tensors[suffix] = tensor
        else:
            tensors[suffix] = None

    return LokrModule(module=module, keys=keys, tensors=tensors)


def cp_weight(wa: torch.Tensor, wb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 4:
        raise ValueError(
            f"Unsupported LoKR tensor rank for CP rebuild: expected 4-D, got {tuple(t.shape)}"
        )
    temp = torch.einsum("i j k l, j r -> i r k l", t, wb)
    return torch.einsum("i j k l, i r -> r j k l", temp, wa)


def rebuild_lokr_delta(
    lokr: LokrModule,
    target_shape: tuple[int, ...],
    base_scale: float,
) -> torch.Tensor:
    tensors = lokr.tensors
    w1 = tensors["lokr_w1"]
    w1a = tensors["lokr_w1_a"]
    w1b = tensors["lokr_w1_b"]
    w2 = tensors["lokr_w2"]
    w2a = tensors["lokr_w2_a"]
    w2b = tensors["lokr_w2_b"]
    t1 = tensors["lokr_t1"]
    t2 = tensors["lokr_t2"]
    alpha = tensors["alpha"]

    if w1 is None:
        if w1a is None or w1b is None:
            raise ValueError(f"LoKR module {lokr.module} is missing w1 tensors")
        if t1 is not None:
            w1 = cp_weight(w1a.float(), w1b.float(), t1.float())
        else:
            w1 = w1a.float() @ w1b.float()
    else:
        w1 = w1.float()

    if w2 is None:
        if w2a is None or w2b is None:
            raise ValueError(f"LoKR module {lokr.module} is missing w2 tensors")
        if t2 is not None:
            w2 = cp_weight(w2a.float(), w2b.float(), t2.float())
        else:
            w2 = w2a.float() @ w2b.float()
    else:
        w2 = w2.float()

    scale = base_scale
    if alpha is not None and (w1b is not None or w2b is not None):
        if alpha.numel() != 1:
            raise ValueError(
                f"LoKR alpha for {lokr.module} must be scalar, got {tuple(alpha.shape)}"
            )
        rank = int(w1b.shape[0]) if w1b is not None else int(w2b.shape[0])
        scale *= float(alpha.reshape(-1).float()[0].item()) / float(rank)

    rebuild = torch.kron(w1, w2)
    expected = math.prod(target_shape)
    if rebuild.numel() != expected:
        raise ValueError(
            f"LoKR shape mismatch for {lokr.module}: kron({tuple(w1.shape)}, {tuple(w2.shape)}) "
            f"has {rebuild.numel()} elements, expected {expected} for target {target_shape}"
        )
    return rebuild.reshape(target_shape) * scale


def merge_lokr_into_transformer(
    model: Flux2Transformer2DModel,
    raw: dict[str, torch.Tensor],
    num_double_layers: int,
    num_single_layers: int,
    scale: float,
) -> MergeStats:
    params = dict(model.named_parameters())
    used_keys: set[str] = set()
    stats = MergeStats(source_tensors_total=len(raw))

    def apply_delta(param_name: str, delta: torch.Tensor) -> None:
        w = params[param_name]
        if tuple(w.shape) != tuple(delta.shape):
            raise ValueError(
                f"Update shape mismatch for {param_name}: {tuple(delta.shape)} vs {tuple(w.shape)}"
            )
        merged = w.data.float() + delta.float()
        w.data.copy_(merged.to(dtype=w.dtype))

    def add_from_source(dest_module: str, source_module: str) -> bool:
        lokr = find_lokr_module(raw, source_module)
        if lokr is None:
            return False
        param_name = f"{dest_module}.weight"
        if param_name not in params:
            raise ValueError(f"Transformer parameter not found: {param_name}")
        delta = rebuild_lokr_delta(lokr, tuple(params[param_name].shape), scale)
        apply_delta(param_name, delta)
        used_keys.update(lokr.keys.values())
        stats.modules_applied += 1
        return True

    def add_qkv_fallback(source_module: str, dest_modules: tuple[str, str, str]) -> bool:
        lokr = find_lokr_module(raw, source_module)
        if lokr is None:
            return False

        q_name = f"{dest_modules[0]}.weight"
        k_name = f"{dest_modules[1]}.weight"
        v_name = f"{dest_modules[2]}.weight"
        if q_name not in params or k_name not in params or v_name not in params:
            raise ValueError(
                f"Transformer qkv destination missing for source {source_module}: "
                f"{q_name}, {k_name}, {v_name}"
            )
        q_shape = tuple(params[q_name].shape)
        k_shape = tuple(params[k_name].shape)
        v_shape = tuple(params[v_name].shape)
        if q_shape != k_shape or q_shape != v_shape:
            raise ValueError(
                f"Expected matching q/k/v shapes for {source_module}, got {q_shape} {k_shape} {v_shape}"
            )

        full_shape = (q_shape[0] * 3, q_shape[1])
        delta = rebuild_lokr_delta(lokr, full_shape, scale)
        split = q_shape[0]
        q_delta = delta[0:split, :]
        k_delta = delta[split : split * 2, :]
        v_delta = delta[split * 2 : split * 3, :]

        for dest_name, part in ((q_name, q_delta), (k_name, k_delta), (v_name, v_delta)):
            apply_delta(dest_name, part)
            stats.modules_applied += 1

        used_keys.update(lokr.keys.values())
        return True

    # Shared modules.
    for module in (
        "x_embedder",
        "context_embedder",
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
        "proj_out",
    ):
        add_from_source(module, module)

    for i in range(num_double_layers):
        q = add_from_source(
            f"transformer_blocks.{i}.attn.to_q",
            f"transformer_blocks.{i}.attn.to_q",
        )
        k = add_from_source(
            f"transformer_blocks.{i}.attn.to_k",
            f"transformer_blocks.{i}.attn.to_k",
        )
        v = add_from_source(
            f"transformer_blocks.{i}.attn.to_v",
            f"transformer_blocks.{i}.attn.to_v",
        )
        if not (q or k or v):
            add_qkv_fallback(
                f"double_blocks.{i}.img_attn.qkv",
                (
                    f"transformer_blocks.{i}.attn.to_q",
                    f"transformer_blocks.{i}.attn.to_k",
                    f"transformer_blocks.{i}.attn.to_v",
                ),
            )

        if not add_from_source(
            f"transformer_blocks.{i}.attn.to_out.0",
            f"transformer_blocks.{i}.attn.to_out.0",
        ):
            add_from_source(
                f"transformer_blocks.{i}.attn.to_out.0",
                f"double_blocks.{i}.img_attn.proj",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff.linear_in",
            f"transformer_blocks.{i}.ff.linear_in",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff.linear_in",
                f"double_blocks.{i}.img_mlp.0",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff.linear_out",
            f"transformer_blocks.{i}.ff.linear_out",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff.linear_out",
                f"double_blocks.{i}.img_mlp.2",
            )

        q = add_from_source(
            f"transformer_blocks.{i}.attn.add_q_proj",
            f"transformer_blocks.{i}.attn.add_q_proj",
        )
        k = add_from_source(
            f"transformer_blocks.{i}.attn.add_k_proj",
            f"transformer_blocks.{i}.attn.add_k_proj",
        )
        v = add_from_source(
            f"transformer_blocks.{i}.attn.add_v_proj",
            f"transformer_blocks.{i}.attn.add_v_proj",
        )
        if not (q or k or v):
            add_qkv_fallback(
                f"double_blocks.{i}.txt_attn.qkv",
                (
                    f"transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer_blocks.{i}.attn.add_v_proj",
                ),
            )

        if not add_from_source(
            f"transformer_blocks.{i}.attn.to_add_out",
            f"transformer_blocks.{i}.attn.to_add_out",
        ):
            add_from_source(
                f"transformer_blocks.{i}.attn.to_add_out",
                f"double_blocks.{i}.txt_attn.proj",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff_context.linear_in",
            f"transformer_blocks.{i}.ff_context.linear_in",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff_context.linear_in",
                f"double_blocks.{i}.txt_mlp.0",
            )

        if not add_from_source(
            f"transformer_blocks.{i}.ff_context.linear_out",
            f"transformer_blocks.{i}.ff_context.linear_out",
        ):
            add_from_source(
                f"transformer_blocks.{i}.ff_context.linear_out",
                f"double_blocks.{i}.txt_mlp.2",
            )

    for i in range(num_single_layers):
        if not add_from_source(
            f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
            f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
        ):
            add_from_source(
                f"single_transformer_blocks.{i}.attn.to_qkv_mlp_proj",
                f"single_blocks.{i}.linear1",
            )

        if not add_from_source(
            f"single_transformer_blocks.{i}.attn.to_out",
            f"single_transformer_blocks.{i}.attn.to_out",
        ):
            add_from_source(
                f"single_transformer_blocks.{i}.attn.to_out",
                f"single_blocks.{i}.linear2",
            )

    stats.source_tensors_used = len(used_keys)
    return stats


def merge_lora_into_transformer(
    model: Flux2Transformer2DModel,
    normalized_lora: dict[str, torch.Tensor],
    scale: float,
) -> int:
    params = dict(model.named_parameters())
    modules: list[str] = []
    prefix = "transformer."
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"

    for key in normalized_lora.keys():
        if key.startswith(prefix) and key.endswith(suffix_a):
            modules.append(key[len(prefix) : -len(suffix_a)])

    merged_modules = 0
    with torch.no_grad():
        for module in sorted(modules):
            key_a = f"{prefix}{module}{suffix_a}"
            key_b = f"{prefix}{module}{suffix_b}"
            if key_b not in normalized_lora:
                raise ValueError(f"Missing LoRA B tensor for module {module}")

            a = normalized_lora[key_a].float()
            b = normalized_lora[key_b].float()
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError(
                    f"LoRA tensors for {module} must be rank-2, got A={tuple(a.shape)} B={tuple(b.shape)}"
                )
            if b.shape[1] != a.shape[0]:
                raise ValueError(
                    f"LoRA rank mismatch for {module}: A={tuple(a.shape)} B={tuple(b.shape)}"
                )

            param_name = f"{module}.weight"
            if param_name not in params:
                raise ValueError(f"Transformer parameter not found for module {module} ({param_name})")

            w = params[param_name]
            expected = (int(b.shape[0]), int(a.shape[1]))
            if tuple(w.shape) != expected:
                raise ValueError(
                    f"LoRA shape mismatch for {module}: got A={tuple(a.shape)} B={tuple(b.shape)}, "
                    f"expected weight shape {tuple(w.shape)}"
                )

            merged = w.data.float()
            merged.addmm_(b, a, beta=1.0, alpha=scale)
            w.data.copy_(merged.to(dtype=w.dtype))
            merged_modules += 1

    return merged_modules


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    lora_path = args.lora.resolve()
    out_dir = args.out.resolve()

    if args.lora_scale <= 0:
        raise ValueError("--lora-scale must be > 0")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    transformer_dir = resolve_transformer_dir(model_dir)
    dtype = dtype_from_name(args.dtype)

    print(f"Transformer: {transformer_dir}")
    print(f"LoRA:        {lora_path}")
    print(f"Scale:       {args.lora_scale:.3f}")
    print(f"DType:       {dtype}")

    print("Loading transformer...")
    model = Flux2Transformer2DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    num_double = len(model.transformer_blocks)
    num_single = len(model.single_transformer_blocks)
    print(f"Transformer blocks: double={num_double} single={num_single}")

    print("Reading adapter tensors...")
    raw_lora = load_file(str(lora_path))
    adapter_format = detect_adapter_format(raw_lora)
    print(f"Detected adapter format: {adapter_format}")

    if adapter_format == "lora":
        normalized_lora, stats = normalize_lora_state_dict(
            raw_lora,
            num_double_layers=num_double,
            num_single_layers=num_single,
        )
        if not normalized_lora:
            raise RuntimeError("LoRA applied to 0 modules (no matching keys found)")

        print(
            "Normalized LoRA: "
            f"{len(normalized_lora)} tensors for {stats.modules_applied} modules "
            f"(used {stats.source_tensors_used}/{stats.source_tensors_total} source tensors)"
        )
        print("Merging LoRA into transformer weights...")
        merged_count = merge_lora_into_transformer(
            model=model,
            normalized_lora=normalized_lora,
            scale=args.lora_scale,
        )
        print(f"Merged LoRA into {merged_count} transformer modules.")
    elif adapter_format == "lokr":
        print("Merging LoKR into transformer weights...")
        stats = merge_lokr_into_transformer(
            model=model,
            raw=raw_lora,
            num_double_layers=num_double,
            num_single_layers=num_single,
            scale=args.lora_scale,
        )
        if stats.modules_applied == 0:
            raise RuntimeError("LoKR applied to 0 modules (no matching keys found)")
        print(
            "Resolved LoKR: "
            f"{stats.modules_applied} modules "
            f"(used {stats.source_tensors_used}/{stats.source_tensors_total} source tensors)"
        )
        print(f"Merged LoKR into {stats.modules_applied} transformer modules.")
    elif adapter_format == "loha":
        raise RuntimeError("LoHa adapters are not supported by this tool yet")
    else:
        raise RuntimeError(
            "Adapter format not recognized. Supported formats: LoRA and LoKR."
        )

    print("Saving merged transformer...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(out_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    print(f"Merged transformer saved to: {out_dir}")
    print("Tip: run check_flux2c_keys.py on the output directory before swapping it into flux2.c.")


if __name__ == "__main__":
    main()
