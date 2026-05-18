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

When numbers are worth keeping, append JSONL records so later reports can be
traced back to the command, code revision, package versions, device metadata,
and raw latencies:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

## Notes

Record device-specific observations here after running:

```bash
uv run python -m inference_kernel_lab.benchmarks.memory_bandwidth --backend all --device cuda --op all --numel 16777216 --dtype float32
```

## PyTorch vs Triton Comparison

Use `--backend all` on a CUDA host to compare the PyTorch baseline with the
Triton implementation for the same memory traffic model:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
uv run python -m inference_kernel_lab.benchmarks.memory_bandwidth --backend all --device cuda --op all
```

Record one row per backend and primitive:

| Backend | Kernel | Shape | dtype | p50 ms | GB/s | TFLOP/s | Interpretation |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| torch | copy | TBD | float32 | TBD | TBD | 0 | memory-bound baseline |
| triton | copy | TBD | float32 | TBD | TBD | 0 | memory-bound custom kernel |
| torch | vector_add | TBD | float32 | TBD | TBD | TBD | memory-bound baseline |
| triton | vector_add | TBD | float32 | TBD | TBD | TBD | memory-bound custom kernel |

The important comparison is not just latency. For these primitives, the useful
number is effective bandwidth:

```text
effective GB/s = estimated bytes moved / p50 latency
```

If the kernel has very low arithmetic intensity and reaches a meaningful
fraction of peak HBM bandwidth, the roofline model says additional scalar FLOP
optimization is unlikely to move the result much. The next questions become
coalescing, launch overhead, vectorization, occupancy, and whether fusion can
remove reads or writes entirely.

## Softmax Fusion

Softmax adds reductions and transcendental math, but the first-order roofline
lesson is still memory movement. A naive two-kernel implementation moves about
four tensors through HBM:

```text
read input -> write intermediate -> read intermediate -> write output
```

The fused Triton kernel moves the idealized lower bound:

```text
read input -> store output
```

Run:

```bash
uv run python -m inference_kernel_lab.benchmarks.softmax --backend all --device cuda --rows 4096 --cols 1024
```

Record:

| Backend | Shape | dtype | Traffic Model | p50 ms | GB/s | Interpretation |
| --- | ---: | --- | --- | ---: | ---: | --- |
| torch | 4096x1024 | float32 | fused | TBD | TBD | library baseline |
| triton | 4096x1024 | float32 | fused | TBD | TBD | fused custom kernel |

## Normalization

RMSNorm and LayerNorm have low arithmetic intensity for typical transformer
hidden sizes because each row is read, reduced, scaled, and written. The fused
Triton kernels avoid extra intermediate tensors, but each output element still
depends on a row-level reduction.

Run:

```bash
uv run python -m inference_kernel_lab.benchmarks.norms --backend all --device cuda --op all --rows 4096 --cols 4096
```

Record:

| Backend | Kernel | Shape | dtype | p50 ms | GB/s | TFLOP/s | Interpretation |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| torch | RMSNorm | 4096x4096 | float32 | TBD | TBD | TBD | baseline |
| triton | RMSNorm | 4096x4096 | float32 | TBD | TBD | TBD | fused custom kernel |
| torch | LayerNorm | 4096x4096 | float32 | TBD | TBD | TBD | baseline |
| triton | LayerNorm | 4096x4096 | float32 | TBD | TBD | TBD | fused custom kernel |

For FP16/BF16 runs, include the epsilon value and correctness tolerance in the
report. Normalization kernels can look fast while still being numerically wrong
if accumulation dtype and epsilon handling are not explicit.
