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

1. run `./scripts/benchmark --run-id <run-id> --with-profiling` on a Pod started with `./scripts/up`
2. read `experiments/reports/runpod/<run-id>.md`
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
2. run `--include-matmul-sweep` to compare focused float16 tile shapes and
   launch configurations
3. separate p50 latency from achieved TFLOP/s and tail noise
4. use Nsight Compute to confirm occupancy, registers, shared memory, pipeline
   staging, and Tensor Core utilization

Recommended live command:

```bash
./scripts/benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

## Attention Baseline Track

Use `attention` to bridge from dense matmul reuse into KV-cache dominated decode
work:

1. capture the PyTorch contiguous-KV baseline with `--include-attention-baseline`
2. vary sequence length, head count, and head dimension with standalone
   `benchmark-attention` commands
3. compare effective traffic against the fused target model before writing a
   custom kernel
4. only then add paged-cache addressing or Triton/CUDA decode fusion

Recommended live command for the next evidence bundle:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --include-matmul-sweep \
  --include-rmsnorm-shape-sweep \
  --include-attention-baseline \
  --with-profiling
```

## CUDA Graph Replay Track

Use `decode_step` when the question is launch overhead rather than a single
kernel implementation:

1. run the staged matrix: `naive-eager`, `fused-eager`, `naive-graph`,
   `fused-graph`, `fused-piecewise-graph`
2. compare host latency, CUDA event latency, synthetic tokens/sec, CPU
   utilization, and launch-overhead estimates
3. replay the dynamic trace to measure graph hit rate, padding waste,
   scheduler decision latency, host step CPU time, queue wait, and batch
   occupancy
4. use Nsight Systems when launch ordering or CPU scheduling needs proof
5. use Nsight Compute on individual fused kernels for occupancy and HBM
   throughput

Recommended standalone command:

```bash
uv run benchmark-decode-step --mode all --device cuda --dtype float16
```

Recommended live command:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --only-decode-step \
  --include-decode-bucket-sweep \
  --include-decode-tail-sweep \
  --decode-attention-backend sdpa-head-major \
  --decode-dynamic-copy-mode resident \
  --decode-piecewise-post-mode eager \
  --decode-orchestration-timing off \
  --decode-tail-buckets '1,2,3,4,5,6,7,8'
```

This command matches the latest saved decode optimization path. It uses
head-major resident KV views for SDPA, same-stream piecewise CUDA Graph replay,
an eager post-attention add, dense graph buckets, and hot-loop timing without
per-region probes.

Saved A10G evidence from `2026-05-22-round12-kv-active-views`:

- fixed-shape `fused-piecewise-graph-same-stream`: `0.1375 ms` p50
- dynamic dense-bucket same-stream graph tail seeds: `0.155-0.158 ms` p50 and
  `0.228-0.232 ms` p95
- dense `1,2,3,4,5,6,7,8` buckets had zero padding; coarser policies reduced
  captures but introduced padding at skipped active batch sizes
- all correctness checks passed in the saved report

Use `--decode-tail-buckets '1,2,4,8;1,2,3,4,6,8'` when the question is bucket
policy tradeoff. Use `--decode-dynamic-copy-mode x-only` when the current
activation should still be staged while KV cache is modeled as resident. Turn
orchestration timing back on when the next bottleneck needs a per-region host
breakdown.

Recommended dynamic trace command:

```bash
uv run benchmark-decode-step --dynamic-trace --mode all --device cuda --dtype float16
```

Use `dynamic-piecewise-graph-same-stream` as the primary dynamic graph target
and keep `dynamic-piecewise-graph` as the ordered-stream comparison.
