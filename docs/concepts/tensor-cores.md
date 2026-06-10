# Tensor Cores

Tensor Cores accelerate matrix operations when the kernel uses compatible data
types, layouts, and tile shapes.

## When They Matter

Tensor Core work belongs in the matmul milestones. In this repo, the current
comparison path is:

1. PyTorch/cuBLAS baseline
2. Triton tiled `tl.dot` matmul
3. tile-shape, launch-configuration, and persistent-schedule sweeps
4. profiler-backed Tensor Core validation

Native CUDA C++ matmul variants such as naive or shared-memory tiled kernels are
useful comparison points, but they are not part of the current implemented
kernel tree.

## What To Record

For Tensor Core experiments, record:

- M, N, K shape
- dtype and accumulation dtype
- tile shape
- `num_warps`, `num_stages`, and input precision
- layout assumptions
- p50/p95/p99 latency
- TFLOP/s
- profiler Tensor Core utilization when available

## Common Mistake

Do not claim Tensor Core usage from high TFLOP/s alone. Use profiler evidence
when the milestone is specifically about Tensor Core utilization.
