# Triton Kernels

This directory contains Triton implementations used by the benchmark workflows.

Implemented tracks:

1. Memory bandwidth primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
2. Fused row-wise softmax
3. Row-wise RMSNorm and LayerNorm forward kernels
4. Fused SwiGLU elementwise activation
5. Tiled matmul progression with tile and launch sweeps

Active validation tracks:

1. Tensor Core matmul validation with profiler counters
2. Contiguous KV-cache attention baseline context
3. Synthetic decode-step graph replay around fused Triton kernels

Use `--backend triton` when a benchmark should isolate these implementations
from PyTorch baselines.
