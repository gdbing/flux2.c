#!/usr/bin/env python3
"""
Download FLUX.2-klein model files from HuggingFace.

Usage:
    python download_model.py [--output-dir DIR] [--base] [--9b] [--token TOKEN]

Requirements:
    pip install huggingface_hub

This downloads the VAE, transformer, and Qwen3 text encoder needed for inference.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Download FLUX.2-klein model files from HuggingFace'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto based on model type)'
    )
    parser.add_argument(
        '--base', action='store_true',
        help='Download the undistilled base model instead of the distilled model'
    )
    parser.add_argument(
        '--9b', dest='nine_b', action='store_true',
        help='Download the 9B model (non-commercial license, requires auth)'
    )
    parser.add_argument(
        '--token', '-t',
        default=None,
        help='HuggingFace authentication token (for gated models like 9B)'
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return 1

    # Determine token: CLI arg > env var > hf_token.txt file
    token = args.token
    if not token:
        import os
        token = os.environ.get('HF_TOKEN')
    if not token:
        token_file = Path('hf_token.txt')
        if token_file.exists():
            token = token_file.read_text().strip()

    if args.nine_b:
        repo_id = "black-forest-labs/FLUX.2-klein-9B"
        default_dir = "./flux-klein-9b-model"
    elif args.base:
        repo_id = "black-forest-labs/FLUX.2-klein-base-4B"
        default_dir = "./flux-klein-base-model"
    else:
        repo_id = "black-forest-labs/FLUX.2-klein-4B"
        default_dir = "./flux-klein-model"

    output_dir = Path(args.output_dir if args.output_dir else default_dir)

    print(f"FLUX.2 Model Downloader")
    print("================================")
    print()
    print(f"Repository: {repo_id}")
    print(f"Output dir: {output_dir}")
    if token:
        print(f"Auth: using token")
    print()

    # Files to download - VAE, transformer, Qwen3 text encoder, and model_index.json
    patterns = [
        "model_index.json",
        "vae/*.safetensors",
        "vae/*.json",
        "transformer/*.safetensors",
        "transformer/*.json",
        "text_encoder/*",
        "tokenizer/*",
    ]

    print("Downloading files...")
    print("(This may take a while depending on your connection)")
    print()

    try:
        model_dir = snapshot_download(
            repo_id,
            local_dir=str(output_dir),
            allow_patterns=patterns,
            ignore_patterns=["*.bin", "*.pt", "*.pth"],  # Skip pytorch format
            token=token,
        )
        print()
        print("Download complete!")
        print(f"Model saved to: {model_dir}")
        print()

        # Show file sizes
        vae_path = output_dir / "vae" / "diffusion_pytorch_model.safetensors"
        tf_path = output_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        te_path = output_dir / "text_encoder"

        total_size = 0
        if vae_path.exists():
            vae_size = vae_path.stat().st_size
            total_size += vae_size
            print(f"  VAE:          {vae_size / 1024 / 1024:.1f} MB")
        if tf_path.exists():
            tf_size = tf_path.stat().st_size
            total_size += tf_size
            print(f"  Transformer:  {tf_size / 1024 / 1024 / 1024:.2f} GB")
        if te_path.exists():
            te_size = sum(f.stat().st_size for f in te_path.rglob("*") if f.is_file())
            total_size += te_size
            print(f"  Text encoder: {te_size / 1024 / 1024 / 1024:.2f} GB")

        if total_size > 0:
            print(f"  Total:        {total_size / 1024 / 1024 / 1024:.2f} GB")
        print()
        print("Usage:")
        print(f"  ./flux -d {output_dir} -p \"your prompt\" -o output.png")
        print()

    except Exception as e:
        error_msg = str(e)
        print(f"Error downloading: {e}")
        print()
        if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
            print("Authentication required. For gated models (like 9B):")
            print("  1. Accept the license at https://huggingface.co/black-forest-labs/" +
                  repo_id.split('/')[-1])
            print("  2. Get your token from https://huggingface.co/settings/tokens")
            print("  3. Run: python download_model.py --token YOUR_TOKEN")
            print("  Or set HF_TOKEN env var, or save token to hf_token.txt")
        else:
            print("If you need to authenticate, run:")
            print("  huggingface-cli login")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
