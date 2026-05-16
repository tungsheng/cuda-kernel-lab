# Triton Kernels

This directory contains Triton implementations of each primitive.

Current implementation:

1. Memory bandwidth primitives: `copy`, `scale`, `vector_add`, `reduction_sum`

Planned order:

1. Fused softmax
2. RMSNorm / LayerNorm
3. SwiGLU elementwise fusion
4. Matmul progression
5. Paged KV lookup
6. Decode attention microkernel

Run the memory primitive comparison on a CUDA host:

```bash
uv run python -m benchmarks.memory_bandwidth --backend all --device cuda --op all
```
