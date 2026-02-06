# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Testing
- **Quick iteration**: use 256x256 with `--seed 42 -v` for timing measurements
- **Before committing**: run `make test` to verify no regressions
- **Benchmark command**:
  ```bash
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 256 -H 256 -v --seed 42
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 512 -H 512 -v --seed 42
  ```

## Pipeline
```
1. Text Encoding:    prompt -> Qwen3 4B (36 layers) -> [512, 7680] embeddings
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop (4 steps):
   per step: 5 double blocks -> 20 single blocks -> final layer -> velocity
4. VAE Decode:       latents -> VAE decoder -> RGB image
```

## Current Baseline (2026-02-06 / MacBook Pro M3 Max 40-core GPU, 128 GB, 400 GB/s)

### 256x256 (seq=256+512=768 tokens)
- Text encoding: 1.9s (Qwen3, cached on 2nd run) — 11.8s cold start
- Denoising total: 2172 ms (4 steps)
  - Step 1: 636 ms, Steps 2-4: ~512 ms each
  - Double blocks: ~520 ms (25%), Single blocks: ~1560 ms (75%)
- VAE decode: 0.4s
- Transformer loading: 1.3s (includes bf16 weight cache warmup)
- **Total: ~6.0s (cold text encoder), ~4.4s (warm)**

### 512x512 (seq=1024+512=1536 tokens)
- Text encoding: 1.9s
- Denoising total: 4146 ms (4 steps)
  - Step 1: 1129 ms, Steps 2-4: ~1006 ms each
  - Double blocks: ~930 ms (23%), Single blocks: ~3120 ms (77%)
- VAE decode: 1.6s
- **Total: ~9.3s**

### Key observations
- Step 1 is ~1.2x slower than subsequent steps (residual MPS warmup)
- Single blocks dominate (75-77% of denoising time)
- 20 single blocks vs 5 double blocks, so per-block: single ~78ms, double ~26ms (256x256)
- Each block does: batch_begin → ~12 GPU ops → batch_end → tensor_read (CPU sync)
- 25 blocks × 4 steps = 100 command buffer round-trips per generation

## Already Optimized
- Batched GPU ops within each block (batch_begin/batch_end)
- Fused QKV+MLP projection in single blocks
- Fused bf16 attention kernel (seq <= 1024)
- bf16 MPS attention fallback (seq > 1024)
- Pre-warm bf16->f16 weight cache
- Persistent GPU tensors
- SwiGLU fused on GPU

## Optimization Attempts

### Attempt 1: Pre-warm bf16 weight buffer cache (SUCCESS)
- In mmap mode, first denoising step paid ~800ms overhead to copy ~7GB of bf16 weight data
  from mmap'd safetensors to Metal GPU buffers (via `get_cached_bf16_buffer`)
- Moved cache population to model loading (`warmup_mmap_bf16_buffers()`)
- Loads each block's bf16 mmap pointers, copies weight data to Metal buffers, frees f32 weights
- 113 cache entries: 5 double blocks × 14 weights + 20 single blocks × 2 weights + 3 input/output
- Loading time: 0.2s → 1.3s (+1.1s for weight cache warmup)
- **Result: 256x256 denoising 2822 → 2172ms (23% faster), 512x512 4420 → 4146ms (6% faster)**
- Step 1 overhead: 256x256 781ms → 124ms (84% less), 512x512 354ms → 123ms (65% less)

### Attempt 1b: MPSGraph JIT pre-warming (FAILED)
- Tried pre-warming MPSGraph JIT compilation by running dummy matmuls with all dimension tuples
- Created graphs for 9 linear ops + 3 SDPA ops per resolution, allocated dummy Metal buffers
- Total JIT warmup: only ~80ms (MPSGraph compiles fast on M3 Max)
- **Result: no improvement — JIT compilation was not the bottleneck. Reverted.**

## Credits attribution rules
- Ideas / kernels / approaches should be only taken from BSD / MIT licensed code.
- If any optimization ideas or kernel code are taken from some other project,
  proper credits must be added to both the README and the relevant source file.
