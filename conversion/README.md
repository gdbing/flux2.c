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

- `--config` (default: `black-forest-labs/FLUX.2-klein-4B`)
- `--subfolder` (default: `transformer`)
- `--dtype` (`float32`, `float16`, `bfloat16`; default: `bfloat16`)

## Check `flux2.c` key compatibility

```bash
uv run python check_flux2c_keys.py \
  /absolute/path/to/converted/transformer/diffusion_pytorch_model.safetensors
```

Exit codes:

- `0`: required keys found
- `2`: one or more required keys missing

## Notes

- This uses `diffusers` from GitHub (`main`) to ensure `Flux2Transformer2DModel`
  is available.
- If conversion succeeds but key check still fails, the checkpoint likely uses a
  variant schema that still needs a post-conversion remap.
