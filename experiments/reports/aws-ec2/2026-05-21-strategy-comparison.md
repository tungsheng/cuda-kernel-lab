# Strategy Comparison Synthesis

Status: synthesis from existing AWS EC2 reports plus the
`2026-05-21-strategy-next` AWS validation run.

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

## Validation Run

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

Artifacts:

- Benchmark report:
  `experiments/reports/aws-ec2/2026-05-21-strategy-next.md`
- Profile summaries:
  `profiling/reports/2026-05-21-strategy-next/`

Key results:

- 115 benchmark rows loaded from 10 result files, with all correctness checks
  passing.
- RMSNorm shape sweep kept the Triton fusion win across the tested float16
  shapes: 2.367x at 512x1024, 2.876x at 1024x2048, 4.76x at 2048x4096, 5.599x
  at 4096x4096, and 5.901x at 4096x8192. The two smallest Triton rows were
  noisy enough to deserve reruns before precise claims.
- The first contiguous-KV decode-attention baseline was PyTorch-only by design:
  0.2273 ms p50 for `seq_len=2048`, `num_heads=16`, `head_dim=128`,
  `float16`.
- The best Triton matmul sweep row reached 25.74 TFLOP/s
  (`block_m=128`, `block_n=64`, `block_k=32`, `num_warps=8`,
  `num_stages=3`, `input_precision=tf32`) while PyTorch/cuBLAS stayed near
  30-31 TFLOP/s.
- Nsight confirmed RMSNorm is the cleaner current win: 90.91% DRAM throughput
  and 93.12% occupancy. The profiled matmul tile showed 45.19% Tensor Core
  utilization, 22.49% occupancy, 80 registers/thread, and 16 KiB dynamic shared
  memory per block.

## Next Question

Can a custom decode-attention kernel beat the 0.2273 ms PyTorch contiguous-KV
baseline without adding paged-cache indirection yet?
