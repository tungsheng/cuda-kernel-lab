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

## Normalization Kernels

RMSNorm and LayerNorm are row-wise reduction plus broadcast kernels. The Triton
implementations use one program per row:

```text
load row -> reduce in FP32 -> compute scale -> load affine params -> store row
```

Approximate fused HBM traffic:

| Kernel | Input Reads | Param Reads | Output Writes | Estimated Traffic |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm | x | weight | output | 3 tensors |
| LayerNorm | x | weight + bias | output | 4 tensors |

This estimate intentionally counts the weight and bias vector as if it is read
for every row. Real cache behavior can be better, especially when the parameter
vector is small enough to stay hot in L2. Profiler notes should record whether
parameter loads appear to matter for the tested shape.

Precision notes:

- reductions accumulate in FP32 in the Triton kernels
- outputs are stored back to the input dtype
- FP16/BF16 tests should use looser tolerances than FP32
- epsilon values are part of the numerical contract and should be reported

Run on a CUDA host:

```bash
uv run python -m inference_kernel_lab.benchmarks.norms --backend all --device cuda --op all --rows 4096 --cols 4096
```
