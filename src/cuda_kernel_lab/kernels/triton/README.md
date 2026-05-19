# Triton Kernels

This directory contains Triton implementations of each primitive.

Current implementation:

1. Memory bandwidth primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
2. Fused row-wise softmax
3. Row-wise RMSNorm and LayerNorm forward kernels

Planned order:

1. SwiGLU elementwise fusion
2. Matmul progression
3. Paged KV lookup
4. Decode attention microkernel

Use `--backend triton` in the benchmark commands when you want to isolate these
implementations from PyTorch baselines.
