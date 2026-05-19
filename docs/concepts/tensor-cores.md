# Tensor Cores

Tensor Cores accelerate matrix operations when the kernel uses compatible data
types, layouts, and tile shapes.

## When They Matter

Tensor Core work belongs in the matmul milestones. The comparison should show
the progression:

1. PyTorch/cuBLAS baseline
2. naive custom matmul
3. shared-memory tiled matmul
4. Triton matmul
5. Tensor Core-aware matmul

## What To Record

For Tensor Core experiments, record:

- M, N, K shape
- dtype and accumulation dtype
- tile shape
- layout assumptions
- p50/p95/p99 latency
- TFLOP/s
- profiler Tensor Core utilization when available

## Common Mistake

Do not claim Tensor Core usage from high TFLOP/s alone. Use profiler evidence
when the milestone is specifically about Tensor Core utilization.
