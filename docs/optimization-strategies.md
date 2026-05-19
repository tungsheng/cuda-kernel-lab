# Optimization Strategies

This repo studies CUDA optimization strategies with LLM-shaped primitives as
repeatable workloads. The goal is not just to write a faster kernel. The goal is
to show which strategy moved which metric, and why.

## Strategy Ladder

Use this ladder as a menu, not a checklist. Skip a step when the experiment note
explains why it does not apply.

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
- p50, p95, and p99 latency
- estimated bytes moved and effective GB/s
- estimated FLOPs and effective TFLOP/s
- raw result JSONL path
- profiler confirmation when available

Profiler-backed reports should add the counters that explain the result, such as
memory throughput, occupancy, registers per thread, shared memory per block, or
Tensor Core utilization for matmul milestones.

## Comparison Pattern

Write comparisons in this order:

1. baseline result
2. change being tested
3. metric that moved
4. likely bottleneck explanation
5. profiler evidence, if collected
6. next strategy to try

For scope boundaries, use [Kernel/System Boundary](concepts/kernel-system-boundary.md).

## First Variant Track

Start with `vector_add` block-size tuning:

1. run the AWS EC2 baseline matrix
2. generate `experiments/aws-ec2-first-gpu-run.md` with `benchmark-report`
3. run `benchmark-matrix --include-vector-add-sweep` to capture the first
   PyTorch-vs-Triton `vector_add` block-size variants
4. profile the best candidate with Nsight Compute
5. compare against the PyTorch and default Triton baseline
