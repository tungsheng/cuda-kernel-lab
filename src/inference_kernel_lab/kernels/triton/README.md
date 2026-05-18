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

Run the memory primitive comparison on a CUDA host:

```bash
uv run python -m inference_kernel_lab.benchmarks.memory_bandwidth --backend all --device cuda --op all
```

Run the softmax comparison on a CUDA host:

```bash
uv run python -m inference_kernel_lab.benchmarks.softmax --backend all --device cuda --rows 4096 --cols 1024
```

Run the normalization comparison on a CUDA host:

```bash
uv run python -m inference_kernel_lab.benchmarks.norms --backend all --device cuda --op all
```
