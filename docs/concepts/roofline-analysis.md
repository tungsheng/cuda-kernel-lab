# Roofline Analysis

Roofline analysis relates arithmetic intensity to hardware limits.

```text
arithmetic intensity = FLOPs / bytes moved from high bandwidth memory
```

Low arithmetic intensity kernels are usually memory-bound. High arithmetic
intensity kernels may become compute-bound when data reuse is strong enough.
Use the roofline model to choose an optimization strategy before tuning blindly.

## Accounting Models

The benchmark output starts with analytical estimates. Profiler counters should
later validate or refine them.

| Primitive | HBM Traffic Model | FLOP Model | Expected Bottleneck |
| --- | --- | --- | --- |
| copy | read x, write out | 0 | memory bandwidth |
| scale | read x, write out | 1 per element | memory bandwidth |
| vector_add | read a and b, write out | 1 per element | memory bandwidth |
| reduction_sum | read x, write scalar | n - 1 | memory bandwidth plus reduction overhead |
| softmax fused | read row, write row | reductions, exp, divide | memory traffic plus transcendental/reduction cost |
| RMSNorm fused | read x and weight, write out | row reduction plus scale | memory traffic plus row reduction |
| LayerNorm fused | read x, weight, bias, write out | two row reductions plus affine | memory traffic plus row reduction |
| SwiGLU fused | read gate and up, write out | sigmoid and multiplies | memory traffic plus transcendental cost |
| matmul | read A and B, write C | 2 * M * N * K | compute throughput when tile reuse is effective |

Normalization traffic intentionally counts the affine parameter vector as if it
is read for every row. Real cache behavior can be better, so profiler notes
should record whether parameter loads are visible for the tested shape.

## Reporting Fields

Use the benchmark workflow for commands and JSONL output. When summarizing a
roofline result, keep these fields together:

- shape, dtype, backend, and device
- p50, p95, and p99 latency
- estimated bytes moved and effective GB/s
- estimated FLOPs and effective TFLOP/s
- command and run metadata when the result is worth keeping

`benchmark-report` emits a roofline section automatically when it recognizes
the GPU peak spec from provider metadata or CUDA device metadata. The current
named suite for this path is:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-roofline --with-profiling
```

For FP16/BF16 normalization runs, include the epsilon value and correctness
tolerance in the report. Normalization kernels can look fast while still being
numerically wrong if accumulation dtype and epsilon handling are not explicit.

## Interpreting Results

For low arithmetic intensity primitives, effective bandwidth is often more
useful than latency alone:

```text
effective GB/s = estimated bytes moved / p50 latency
```

If a kernel reaches a meaningful fraction of practical HBM bandwidth, additional
scalar FLOP optimization is unlikely to move the result much. The next questions
become coalescing, launch overhead, vectorization, occupancy, cache reuse, and
whether fusion can remove reads or writes entirely.

For fused kernels, compare both latency and the traffic denominator. A fused
softmax run with the `naive` traffic model does not change the kernel; it shows
the memory traffic a two-kernel implementation would have paid.
