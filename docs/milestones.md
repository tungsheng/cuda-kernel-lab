# Milestones

The project moves from isolated primitives toward decode-time serving behavior.

| Milestone | Focus | Status |
| --- | --- | --- |
| 0 | setup, tests, benchmark discipline, GPU info | implemented |
| 1 | copy, scale, vector add, reduction sum | implemented |
| 2 | fused row-wise softmax | implemented |
| 3 | RMSNorm and LayerNorm forward kernels | implemented |
| 4 | SwiGLU elementwise fusion | planned |
| 5 | matmul progression and tiling | planned |
| 6 | contiguous and paged KV cache layout | planned |
| 7 | decode attention microkernel | planned |
| 8 | mini inference scheduler simulator | planned |
| 9 | integration demo with dashboard/report | planned |

## Success Criteria

The code and notes should make it easy to explain:

- why decode is memory-bound
- why kernel fusion helps
- why tiled matmul improves reuse
- why KV cache layout affects serving throughput
- why faster kernels do not automatically solve tail latency

## Milestone Note Checklist

For each milestone, record:

- question or bottleneck
- implemented backends
- benchmark commands
- result summary
- profiler observations when available
- next question
