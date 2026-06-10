# Triton Kernels

This directory contains Triton implementations used by the benchmark workflows.

Implemented tracks:

1. Memory bandwidth primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
2. Fused row-wise softmax
3. Row-wise RMSNorm and LayerNorm forward kernels
4. Fused SwiGLU elementwise activation
5. Tiled matmul progression with tile, launch, grouped-order, and persistent
   schedule sweeps

Active validation tracks:

1. Tensor Core matmul validation with profiler counters
2. PyTorch contiguous KV-cache attention baseline context
3. Synthetic decode-step graph replay using Triton RMSNorm/SwiGLU regions
   around PyTorch/SDPA attention

Use `--backend triton` when a benchmark supports isolating these
implementations from PyTorch baselines. The standalone attention benchmark is
currently PyTorch-only, and `decode_step` selects Triton fused components
internally when they are available.
