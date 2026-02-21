# FLUX2 Conversion Utilities

This folder is a standalone `uv` project for converting FLUX-style single-file
checkpoints (for example Civitai files) into Diffusers transformer format, then
checking key compatibility with `flux2.c`.

## Setup

```bash
cd flux2.c/conversion
uv sync
```

If your default Python is too old, use:

```bash
uv sync -p 3.11
```

## Convert a Civitai single-file checkpoint

```bash
uv run python convert_flux2_transformer.py \
  --src /absolute/path/to/unstable.safetensors \
  --out /absolute/path/to/converted/transformer
```

Optional flags:

- `--config` (default: `auto`)
- `--subfolder` (default: `transformer`)
- `--dtype` (`float32`, `float16`, `bfloat16`; default: `bfloat16`)
- `--verify-nvfp4` (strict sanity checks for decoded NVFP4 weights)

`--config auto` inspects checkpoint tensor shapes and picks 4B vs 9B automatically
(preferring local `flux-klein-model` / `flux-klein-9b` folders when available).

## Check `flux2.c` key compatibility

```bash
uv run python check_flux2c_keys.py \
  /absolute/path/to/converted/transformer
```

You can pass any of:
- `.../transformer/diffusion_pytorch_model.safetensors` (single file)
- `.../transformer/diffusion_pytorch_model.safetensors.index.json` (sharded)
- `.../transformer` directory (auto-detects either)

Exit codes:

- `0`: required keys found
- `2`: one or more required keys missing

## Merge a LoRA/LoKR adapter into a copied transformer (offline bake)

```bash
uv run python merge_flux2_lora.py \
  --model-dir /absolute/path/to/flux-klein-model \
  --lora /absolute/path/to/adapter.safetensors \
  --out /absolute/path/to/merged/transformer \
  --lora-scale 1.0
```

Notes:

- `--model-dir` accepts either the model root (`.../transformer` inside) or the
  transformer directory itself.
- Output is a standalone Diffusers transformer folder that can replace the
  original `transformer/` folder for `flux2.c`.
- The merge script follows `flux2.c` LoRA key-matching behavior:
  unknown key roots, `lora_A/B` and `lora_down/up`, and fallback aliases like
  `double_blocks.*` and `single_blocks.*`.
- LoKR adapters (`lokr_w1`/`lokr_w2` style, including decomposed variants) are
  also supported for offline baking.
- For maximum merge precision use `--dtype float32`; for smaller output and
  mmap-friendly runtime use `--dtype bfloat16` (default).

## Notes

- This uses `diffusers` from GitHub (`main`) to ensure `Flux2Transformer2DModel`
  is available.
- Large outputs (for example 9B) may be saved as sharded safetensors with
  `diffusion_pytorch_model.safetensors.index.json` plus multiple shard files.
- Comfy FP8 checkpoints (with keys like `.comfy_quant`, `.input_scale`,
  `.weight_scale`, `.weight_scale_2`) are normalized automatically during conversion.
- `--verify-nvfp4` prints decoded-layer stats and fails fast if NVFP4 decode
  distributions look suspicious.
- If conversion succeeds but key check still fails, the checkpoint likely uses a
  variant schema that still needs a post-conversion remap.
