# Milestones

The project moves from simple memory traffic toward increasingly realistic CUDA
optimization strategies. LLM-shaped primitives provide useful workloads, but
the lesson is always the optimization strategy and the benchmark evidence.

| Milestone | Focus | Status |
| --- | --- | --- |
| 0 | benchmark discipline, device info, result metadata | implemented |
| 1 | memory bandwidth primitives | implemented |
| 2 | reduction strategies | strategy variants added |
| 3 | softmax fusion | implemented |
| 4 | normalization fusion | implemented |
| 5 | SwiGLU elementwise fusion | implemented |
| 6 | matmul tiling progression | tile-shape sweep added |
| 7 | Tensor Core matmul | launch sweep active |
| 8 | attention microkernel optimization | PyTorch contiguous-KV baseline added |
| 9 | CUDA Graph replay and dynamic decode scheduling | resident same-stream piecewise graph path benchmarked |

## Success Criteria

Each milestone should leave behind enough code and notes to explain:

- why a primitive is memory-bound or compute-bound
- which optimization strategy changed the bottleneck
- how much latency, bandwidth, or throughput moved
- whether profiler counters confirm the benchmark interpretation
- what strategy should be tried next

Use [Optimization Strategies](optimization-strategies.md) for the comparison
pattern.

## Saved Decode Evidence

The saved A10G decode evidence in
`experiments/reports/aws-ec2/2026-05-22-round12-kv-active-views.md`. It
benchmarks resident head-major KV views, same-stream piecewise CUDA Graph
replay, eager post-add, dense batch buckets, and hot-loop timing with
orchestration probes off. Treat it as the synthetic resident-KV upper
bound for milestone 9.
