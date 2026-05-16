# Memory Hierarchy

This note captures how each kernel uses the GPU memory hierarchy.

## Levels To Track

- Registers: fastest storage, private to each thread.
- Shared memory: block-local storage used for cooperation and data reuse.
- L2 cache: shared cache across streaming multiprocessors.
- High bandwidth memory: large global memory with high latency and high
  throughput.

## Kernel Checklist

For each implementation, document:

- global memory reads and writes
- whether loads are coalesced
- whether vectorized loads are used
- shared memory footprint per block
- register pressure if visible in profiler output
- cache hit behavior if relevant

