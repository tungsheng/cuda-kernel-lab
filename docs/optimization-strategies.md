# Optimization Strategies

This repo studies CUDA optimization strategies with LLM-shaped primitives as
repeatable workloads. The goal is not just to write a faster kernel. The goal is
to show which strategy moved which metric, and why.

## Strategy Ladder

Use this ladder as a menu, not a checklist:

1. PyTorch baseline
2. naive custom kernel
3. coalesced memory access
4. vectorized loads and stores
5. warp-level or block-level reduction
6. shared-memory reuse when it reduces global traffic
7. fusion to remove intermediate tensors
8. launch/config tuning
9. profiler validation
10. final comparison table

## Evidence To Capture

Every strategy comparison should include:

- operation, shape, dtype, backend, and device
- method family, technique name, hypothesis, changed knobs, and expected
  profiler signal
- p50, p95, and p99 latency
- estimated bytes moved and effective GB/s
- estimated FLOPs and effective TFLOP/s
- raw result JSONL path
- profiler confirmation when available

Profiler-backed reports should add counters that explain the result, such as
memory throughput, occupancy, registers per thread, shared memory per block, or
Tensor Core utilization for matmul validation.

Use [Optimization Techniques](optimization-techniques.md) as the method catalog
and [Benchmark Workflow](benchmark-workflow.md) for commands.

## Comparison Pattern

Write comparisons in this order:

1. baseline result
2. change being tested
3. metric that moved
4. likely bottleneck explanation
5. profiler evidence, if collected
6. next strategy to try

For scope boundaries, use [Kernel/System Boundary](concepts/kernel-system-boundary.md).

## Track Guide

Memory variant track: use `copy`, `scale`, and `vector_add` to study
coalescing, block-size tuning, vectorization, and DRAM throughput.

Reduction track: compare `reduction_sum` variants with fixed shape, dtype,
backend, and device. Interpret first-pass profiler health separately from
end-to-end launch and finalization cost.

Fusion track: use softmax, RMSNorm, LayerNorm, and SwiGLU to test whether
removing intermediate tensors or framework launches improves p50 latency and
effective bandwidth.

Matmul track: move from memory-dominated kernels into tiled `tl.dot` reuse,
launch configuration, TFLOP/s, and Tensor Core validation. Use profiler counters
before claiming Tensor Core utilization.

Attention track: start with the PyTorch contiguous-KV baseline, vary sequence
length/head shape, and compare analytical traffic against the fused decode
target before adding custom addressing or fusion.

CUDA Graph replay track: use synthetic `decode_step` rows when the question is
launch overhead, graph reuse, dynamic-shape buckets, padding waste, or hot-loop
timing. Treat these rows as kernel-path evidence, not service-level serving
results.

Saved A10G decode evidence from `2026-05-22-round12-kv-active-views` showed the
same-stream dynamic piecewise graph path around `0.155-0.158 ms` p50 and
`0.228-0.232 ms` p95 across three tail seeds with dense buckets, zero padding,
and all correctness checks passing.
