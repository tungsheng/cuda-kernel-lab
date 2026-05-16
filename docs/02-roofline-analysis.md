# Roofline Analysis

Roofline analysis relates arithmetic intensity to hardware limits.

```text
arithmetic intensity = FLOPs / bytes moved from high bandwidth memory
```

Low arithmetic intensity kernels are memory-bound. High arithmetic intensity
kernels may become compute-bound if data reuse is strong enough.

## Milestone 1 Primitives

Approximate memory traffic:

| Kernel | Reads | Writes | FLOPs | Expected Bottleneck |
| --- | ---: | ---: | ---: | --- |
| copy | 1 tensor | 1 tensor | 0 | memory bandwidth |
| scale | 1 tensor | 1 tensor | 1 per element | memory bandwidth |
| vector_add | 2 tensors | 1 tensor | 1 per element | memory bandwidth |
| reduction_sum | 1 tensor | 1 scalar | n - 1 | memory bandwidth plus reduction overhead |

## Benchmark Discipline

Each benchmark should report:

- p50, p95, and p99 latency
- estimated bytes moved
- effective GB/s
- estimated FLOPs
- effective TFLOP/s

The first pass uses simple analytical estimates. Profiler reports should later
replace or validate these estimates with measured memory transactions.

## Notes

Record device-specific observations here after running:

```bash
uv run python -m benchmarks.memory_bandwidth --op all --numel 16777216 --dtype float32
```
