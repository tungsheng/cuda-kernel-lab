# Research Notes

These notes are the lab notebook for the kernel work. Keep them focused on
questions, measurements, and interpretation rather than generic GPU reference
material.

## Reading Path

1. [GPU Execution Model](01-gpu-execution-model.md): vocabulary for mapping work
   to threads, warps, blocks, and grids.
2. [Roofline Analysis](02-roofline-analysis.md): how to relate bytes, FLOPs,
   latency, and bandwidth.
3. [Memory Hierarchy](03-memory-hierarchy.md): what to inspect in memory-bound
   kernels.
4. [Kernel Fusion](04-kernel-fusion.md): how to reason about eliminated reads,
   writes, and intermediate tensors.
5. [KV Cache Layout](05-kv-cache-layout.md): why layout choices matter for
   decode and batching.
6. [Inference System Lessons](06-inference-system-lessons.md): how kernel wins
   interact with batching, scheduling, and tail latency.

## Documentation Workflow

For each milestone:

- State the bottleneck or hypothesis before adding benchmark numbers.
- Capture benchmark output as JSONL under `experiments/results/`.
- Summarize the smallest useful result table in the relevant note.
- Add profiler observations only after recording the command, shape, dtype,
  hardware, and interpretation.

Use `profiling/reports/` for compact profiler writeups and reserve these docs
for the lessons that should remain true across individual runs.
