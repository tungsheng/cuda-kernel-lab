# Triton Kernels

This directory contains Triton implementations of each primitive.

Current implementation:

1. Memory bandwidth primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
2. Fused row-wise softmax
3. Row-wise RMSNorm and LayerNorm forward kernels
4. Fused SwiGLU elementwise activation
5. Tiled matmul progression kernel

Planned order:

1. Tensor Core matmul validation
2. Paged KV lookup
3. Decode attention microkernel

Use `--backend triton` in the benchmark commands when you want to isolate these
implementations from PyTorch baselines.
