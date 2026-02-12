#!/usr/bin/env python3
"""
Run FLUX.2-klein with a local LoRA file using Diffusers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_model_dir = repo_root / "flux-klein-model"
    default_lora = Path(__file__).resolve().parent / "pytorch_lora_weights.safetensors"
    default_output = Path(__file__).resolve().parent / "knight.png"

    parser = argparse.ArgumentParser(description="Run local FLUX.2-klein + LoRA")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help=f"Path to local model directory (default: {default_model_dir})",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=default_lora,
        help=f"Path to LoRA safetensors file (default: {default_lora})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="pixel art sprite, a brave knight in shining armor, game asset, transparent background",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--dump-embeddings",
        type=Path,
        default=None,
        help="Write prompt embeddings as float32 .bin ([seq, dim]).",
    )
    parser.add_argument(
        "--dump-noise",
        type=Path,
        default=None,
        help="Write initial noise as float32 .bin ([channels, h, w]).",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Skip loading/applying LoRA (true baseline run).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output image path (default: {default_output})",
    )
    return parser.parse_args()


def prepare_diffusers_model_dir(model_dir: Path) -> Path:
    model_index = model_dir / "model_index.json"
    scheduler_config = model_dir / "scheduler" / "scheduler_config.json"
    if model_index.exists() and scheduler_config.exists():
        return model_dir

    temp_dir = Path(tempfile.mkdtemp(prefix="flux2-klein-local-"))
    for sub in ("tokenizer", "text_encoder", "transformer", "vae"):
        src = model_dir / sub
        if not src.exists():
            raise FileNotFoundError(f"Expected subfolder not found: {src}")
        os.symlink(src, temp_dir / sub, target_is_directory=True)

    if model_index.exists():
        shutil.copy2(model_index, temp_dir / "model_index.json")
    else:
        cached_model_index = Path(
            hf_hub_download(
                repo_id="black-forest-labs/FLUX.2-klein-4B",
                filename="model_index.json",
            )
        )
        shutil.copy2(cached_model_index, temp_dir / "model_index.json")

    (temp_dir / "scheduler").mkdir(parents=True, exist_ok=True)
    if scheduler_config.exists():
        shutil.copy2(scheduler_config, temp_dir / "scheduler" / "scheduler_config.json")
    else:
        cached_sched = Path(
            hf_hub_download(
                repo_id="black-forest-labs/FLUX.2-klein-4B",
                filename="scheduler/scheduler_config.json",
            )
        )
        shutil.copy2(cached_sched, temp_dir / "scheduler" / "scheduler_config.json")

    return temp_dir


def normalize_lora_state_dict(lora_path: Path) -> tuple[dict[str, torch.Tensor], int]:
    """Normalize local LoRA keys to Diffusers Flux2 transformer namespace.

    Local files can use keys like:
      - base_model.model.transformer_blocks.0....
      - transformer_blocks.0....
    Diffusers Flux2 pipeline expects:
      - transformer.transformer_blocks.0....
    """
    raw = load_file(str(lora_path))
    normalized: dict[str, torch.Tensor] = {}
    remapped = 0

    for key, value in raw.items():
        new_key = key
        for prefix in ("base_model.model.", "model.diffusion_model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        if not new_key.startswith("transformer."):
            new_key = f"transformer.{new_key}"
        if new_key != key:
            remapped += 1
        normalized[new_key] = value

    return normalized, remapped


def normalize_image_dim(v: int) -> int:
    return max(64, (int(v) // 16) * 16)


def dump_f32_tensor(path: Path, tensor: torch.Tensor) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor.detach().float().cpu().contiguous().numpy().tofile(path)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    lora_path = args.lora.resolve()
    out_path = args.output.resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not args.skip_lora and not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    device, dtype = pick_device_and_dtype()
    print(f"Device: {device}")
    print(f"DType: {dtype}")
    print(f"Model: {model_dir}")
    if args.skip_lora:
        print("LoRA:  (skipped)")
    else:
        print(f"LoRA:  {lora_path}")

    load_dir = prepare_diffusers_model_dir(model_dir)
    if load_dir != model_dir:
        print(f"Using staged model dir: {load_dir}")

    with torch.inference_mode():
        pipe = Flux2KleinPipeline.from_pretrained(
            str(load_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        if not args.skip_lora:
            state_dict, remapped = normalize_lora_state_dict(lora_path)
            print(f"LoRA tensors: {len(state_dict)} (remapped keys: {remapped})")
            pipe.load_lora_weights(state_dict, adapter_name="local_lora")

            adapters = pipe.get_list_adapters()
            transformer_adapters = adapters.get("transformer", [])
            if "local_lora" not in transformer_adapters:
                raise RuntimeError(
                    f"LoRA failed to attach to transformer adapters. Adapter map: {adapters}"
                )
            print(f"Loaded adapters: {adapters}")

            # Fuse for execution speed and deterministic single-adapter behavior.
            pipe.fuse_lora(
                components=["transformer"],
                lora_scale=args.lora_scale,
                adapter_names=["local_lora"],
            )
            print(f"Fused LoRA at scale {args.lora_scale:.3f}")

        pipe.to(device)

        height = normalize_image_dim(args.height)
        width = normalize_image_dim(args.width)
        if height != args.height or width != args.width:
            print(f"Adjusted size to {width}x{height} (multiples of 16, min 64).")

        generator = torch.Generator().manual_seed(args.seed)
        export_inputs = args.dump_embeddings is not None or args.dump_noise is not None

        if export_inputs:
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=args.prompt,
                device=torch.device(device),
                num_images_per_prompt=1,
                max_sequence_length=pipe.tokenizer_max_length,
            )
            latent_h = height // 16
            latent_w = width // 16
            # MPS expects an MPS generator for device sampling. Generate on CPU
            # with the seeded CPU generator, then move to pipeline device/dtype.
            latents = torch.randn(
                (1, 128, latent_h, latent_w),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            latents = latents.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

            if args.dump_embeddings is not None:
                dump_f32_tensor(args.dump_embeddings, prompt_embeds[0])
                print(f"Saved embeddings: {args.dump_embeddings.resolve()} shape={tuple(prompt_embeds[0].shape)}")

            if args.dump_noise is not None:
                dump_f32_tensor(args.dump_noise, latents[0])
                print(f"Saved noise: {args.dump_noise.resolve()} shape={tuple(latents[0].shape)}")

            image = pipe(
                prompt_embeds=prompt_embeds,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=height,
                width=width,
                latents=latents,
            ).images[0]
        else:
            image = pipe(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=height,
                width=width,
                generator=generator,
            ).images[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
