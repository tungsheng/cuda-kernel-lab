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
- method family, technique name, hypothesis, changed knobs, and expected profiler signal
- p50, p95, and p99 latency
- estimated bytes moved and effective GB/s
- estimated FLOPs and effective TFLOP/s
- raw result JSONL path
- profiler confirmation when available

Profiler-backed reports should add the counters that explain the result, such as
memory throughput, occupancy, registers per thread, shared memory per block, or
Tensor Core utilization for matmul milestones.

Use [Optimization Techniques](optimization-techniques.md) as the catalog for
method names and experiment wording. The generated benchmark report also emits
an "Optimization Techniques Tested" section from the JSONL metadata.

## Comparison Pattern

Write comparisons in this order:

1. baseline result
2. change being tested
3. metric that moved
4. likely bottleneck explanation
5. profiler evidence, if collected
6. next strategy to try

For scope boundaries, use [Kernel/System Boundary](concepts/kernel-system-boundary.md).

## Memory Variant Track

Start with `vector_add` block-size tuning:

1. run `./scripts/live-benchmark --run-id <run-id> --with-profiling`
2. read `experiments/reports/aws-ec2/<run-id>.md`
3. profile the Triton memory bottleneck with Nsight Compute
4. compare against the PyTorch and default Triton baseline
5. decide whether a narrower launch/config sweep is justified

## Second Variant Track

Use `reduction_sum` to study reduction strategy tradeoffs:

1. compare iterative Triton reduction against the two-pass variant
2. keep block size, shape, dtype, and device fixed
3. inspect whether extra partial reads/writes change effective bandwidth
4. profile the faster strategy to confirm occupancy and memory behavior

## Next Fusion Track

Use `swiglu` to study elementwise fusion without reduction complexity:

1. compare PyTorch against the fused Triton kernel
2. keep rows, columns, dtype, and block size fixed
3. inspect whether removing intermediate activation tensors improves effective bandwidth
4. profile the fused kernel before moving into matmul tiling

## Matmul Progression Track

Use `matmul` to move from memory-dominated kernels into reuse and tile shape:

1. compare PyTorch against the tiled Triton `tl.dot` implementation
2. run `--include-matmul-sweep` to compare focused float16 tile shapes
3. separate p50 latency from achieved TFLOP/s and tail noise
4. use Nsight Compute to confirm occupancy, registers, shared memory, and Tensor Core utilization

Recommended live command:

```bash
./scripts/live-benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```
