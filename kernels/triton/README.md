# Triton Kernels

This directory contains Triton implementations of each primitive.

Current implementation:

1. Memory bandwidth primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
2. Fused row-wise softmax

Planned order:

1. RMSNorm / LayerNorm
2. SwiGLU elementwise fusion
3. Matmul progression
4. Paged KV lookup
5. Decode attention microkernel

Run the memory primitive comparison on a CUDA host:

```bash
uv run python -m benchmarks.memory_bandwidth --backend all --device cuda --op all
```

Run the softmax comparison on a CUDA host:

```bash
uv run python -m benchmarks.softmax --backend all --device cuda --rows 4096 --cols 1024
```
