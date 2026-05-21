# Strategy Comparison Synthesis

Status: synthesis from existing AWS EC2 reports plus the next benchmark hooks.

## Question

Which optimization strategy should the project pursue next after memory,
fusion, and Tensor Core matmul evidence on A10G?

## Existing Evidence

- Launch tuning and reduction strategy work established the measurement loop,
  but the simple memory primitives did not show a durable custom-kernel win
  large enough to justify wider sweeps without profiler evidence.
- Fusion is the strongest positive result so far. RMSNorm beat PyTorch by about
  5.5x for `float16` and 3.25x for `float32` in the 4096x4096 runs, with SwiGLU
  also showing a clear fused elementwise win.
- Matmul tiling is now useful as a learning track rather than a finished
  replacement for cuBLAS. The best Triton float16 row in the current A10G report
  reached roughly 25.6 TFLOP/s versus PyTorch/cuBLAS around 31 TFLOP/s, and the
  Nsight note shows Tensor Core use with occupancy/register/shared-memory
  tradeoffs still visible.

## Decision

Keep the default AWS workflow stable and add only focused optional evidence:

- `--include-matmul-sweep` should continue using `--matmul-input-precision tf32`
  in the live script so Tensor Core behavior is the intended question.
- `--include-rmsnorm-shape-sweep` should test whether the strongest fusion win
  survives smaller batches and larger hidden sizes.
- `--include-attention-baseline` should capture a PyTorch contiguous-KV decode
  baseline before a custom attention kernel exists.

## Next Command

```bash
./scripts/up
./scripts/benchmark \
  --run-id 2026-05-21-strategy-next \
  --include-matmul-sweep \
  --include-rmsnorm-shape-sweep \
  --include-attention-baseline \
  --with-profiling
./scripts/down
```

## Next Question

Does the RMSNorm fusion win persist across shape changes, and how large is the
contiguous-KV decode attention baseline before adding a Triton/CUDA kernel?
