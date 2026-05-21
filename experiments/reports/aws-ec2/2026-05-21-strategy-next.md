# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-21-strategy-next/attention.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/matmul-tile-shape.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/matmul.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/rmsnorm-shape-sweep.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-21-strategy-next/vector-add-block-size.jsonl`

## Environment

- Git commit: `3f16c1a0f9bce168a1208d571292b81f53d4af22`
- Git dirty: `False`
- Python: `3.10.12`
- Platform: `Linux-6.8.0-1055-aws-x86_64-with-glibc2.35`
- PyTorch: `2.12.0`
- Triton: `3.7.0`
- CUDA devices: `NVIDIA A10G (22.06 GiB)`

## Optimization Techniques Tested

| Family | Technique | Used By | Hypothesis |
| --- | --- | --- | --- |
| baseline | PyTorch reference baseline | torch controls | Establish the latency, bandwidth, and correctness baseline for comparison. |
| launch tuning | Coalesced block-size tuning | memory copy, memory scale, memory vector_add | Varying Triton block size for contiguous streaming kernels can improve occupancy and memory throughput. |
| reduction | Iterative block reduction | memory reduction_sum | Repeated Triton block reductions over FP32 partial sums should stream memory efficiently, while repeated launches expose orchestration overhead. |
| reduction | Two-pass block reduction | memory reduction_sum | Reducing to FP32 partial sums with Triton and finalizing in a second step can cut repeated launches, but may pay partial-traffic or framework cleanup cost. |
| fusion | Elementwise SwiGLU fusion | swiglu swiglu | Fusing sigmoid, SiLU gating, multiply, and store should avoid materialized activation intermediates, lowering memory traffic and launch overhead. |
| fusion | Row-wise LayerNorm fusion | norms layernorm | Fusing row reductions, normalization, parameter loads, and affine writeback should remove framework overhead and avoid intermediate normalization tensors. |
| fusion | Row-wise RMSNorm fusion | norms rmsnorm | Fusing row reductions, normalization, parameter loads, and affine writeback should remove framework overhead and avoid intermediate normalization tensors. |
| fusion | Row-wise softmax fusion | softmax softmax | Keeping row max, subtract, exp, sum, divide, and store inside one kernel should reduce global-memory traffic and launch overhead versus a naive multi-kernel path. |
| tiling | Tiled dot-product reuse | matmul matmul | Triton tile-shape and launch-configuration sweeps with `tl.dot` can increase arithmetic intensity and Tensor Core utilization, but may trade off occupancy, pipeline depth, and register pressure. |

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| attention | decode_attention | float16 | 2048x16x128 | seq_len=2048, num_heads=16, head_dim=128, scale=0.0883883 | torch | PyTorch reference baseline | 0.2273 | 73.84 | 0.07452 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06966 | 90.31 | 30.83 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07011 | 89.73 | 30.63 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07061 | 89.1 | 30.41 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.0707 | 88.98 | 30.37 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07064 | 89.06 | 30.4 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1505 | 445.8 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2888 | 464.8 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08602 | 391.6 | 0.195 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 449.8 | 0.1122 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1526 | 439.8 | 0.11 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2929 | 458.3 | 0.05729 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2212 | 455.1 | 0.07585 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4301 | 468.1 | 0.03901 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.172 | 780.2 | 0.7801 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3133 | 856.7 | 0.4283 |
| norms | rmsnorm | float16 | 512x1024 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.05018 | 62.69 | 0.05223 |
| norms | rmsnorm | float16 | 1024x2048 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.05018 | 250.8 | 0.209 |
| norms | rmsnorm | float16 | 2048x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.1024 | 491.5 | 0.4096 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.169 | 595.7 | 0.4964 |
| norms | rmsnorm | float16 | 4096x8192 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3103 | 648.9 | 0.5407 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3098 | 649.9 | 0.2708 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.05018 | 334.4 | 0.4178 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.08192 | 409.6 | 0.2559 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2437 | 413 | 0.3442 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4547 | 442.8 | 0.1845 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| attention | decode_attention | float16 | 2048x16x128 | seq_len=2048, num_heads=16, head_dim=128, scale=0.0883883 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2273 | 0.251 | 0.257 | 73.84 | 0.07452 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07475 | 0.08814 | 90.35 | 30.84 | 1 | 1.074 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.1015 | 0.1065 | 72.28 | 24.67 | 0.8 | 1.166 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07378 | 0.08604 | 90.35 | 30.84 | 1 | 1.06 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08806 | 0.1044 | 0.1088 | 71.44 | 24.39 | 0.7907 | 1.186 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07784 | 0.08808 | 89.04 | 30.39 | 1 | 1.102 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09114 | 0.1076 | 0.1116 | 69.03 | 23.56 | 0.7753 | 1.181 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06966 | 0.07685 | 0.08707 | 90.31 | 30.83 | 1 | 1.103 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09011 | 0.1096 | 0.1127 | 69.82 | 23.83 | 0.7731 | 1.216 noisy |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.08031 | 0.08913 | 89.04 | 30.39 | 1 | 1.137 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.09738 | 0.1055 | 72.28 | 24.67 | 0.8118 | 1.119 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.0789 | 0.08921 | 89.04 | 30.39 | 1 | 1.117 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.09564 | 0.1096 | 72.28 | 24.67 | 0.8118 | 1.099 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07011 | 0.08213 | 0.08821 | 89.73 | 30.63 | 1 | 1.171 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08344 | 0.09016 | 0.1014 | 75.4 | 25.74 | 0.8403 | 1.081 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0789 | 0.08603 | 90.35 | 30.84 | 1 | 1.133 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08397 | 0.08909 | 0.1015 | 74.93 | 25.58 | 0.8293 | 1.061 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07475 | 0.08107 | 89.04 | 30.39 | 1 | 1.058 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1884 | 0.2059 | 0.2079 | 33.4 | 11.4 | 0.3751 | 1.093 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07286 | 0.08502 | 90.35 | 30.84 | 1 | 1.046 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.126 | 0.1444 | 0.1538 | 49.95 | 17.05 | 0.5528 | 1.146 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.08095 | 0.0891 | 89.04 | 30.39 | 1 | 1.146 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.127 | 0.1434 | 0.1454 | 49.55 | 16.91 | 0.5565 | 1.13 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07491 | 0.08092 | 89.04 | 30.39 | 1 | 1.06 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1802 | 0.1916 | 0.199 | 34.91 | 11.92 | 0.392 | 1.063 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07578 | 0.0891 | 90.35 | 30.84 | 1 | 1.088 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1812 | 0.1967 | 0.1996 | 34.71 | 11.85 | 0.3842 | 1.085 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07383 | 0.07894 | 90.35 | 30.84 | 1 | 1.06 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1352 | 0.1455 | 0.1516 | 46.55 | 15.89 | 0.5152 | 1.076 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07061 | 0.07706 | 0.08909 | 89.1 | 30.41 | 1 | 1.091 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1352 | 0.1516 | 0.1557 | 46.55 | 15.89 | 0.5224 | 1.122 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07583 | 0.085 | 90.35 | 30.84 | 1 | 1.089 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1833 | 0.1977 | 0.2028 | 34.32 | 11.72 | 0.3799 | 1.079 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0727 | 0.08919 | 90.35 | 30.84 | 1 | 1.044 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1874 | 0.1977 | 0.214 | 33.57 | 11.46 | 0.3716 | 1.055 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07685 | 0.08706 | 89.04 | 30.39 | 1 | 1.088 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1034 | 0.1178 | 0.1219 | 60.83 | 20.76 | 0.6832 | 1.139 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07583 | 0.0891 | 90.35 | 30.84 | 1 | 1.089 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1024 | 0.1219 | 0.1281 | 61.44 | 20.97 | 0.68 | 1.19 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07281 | 0.08709 | 91.7 | 31.3 | 1 | 1.061 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1147 | 0.1312 | 0.1342 | 54.86 | 18.72 | 0.5982 | 1.144 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0809 | 0.0881 | 90.35 | 30.84 | 1 | 1.162 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1167 | 0.1352 | 0.1526 | 53.89 | 18.4 | 0.5965 | 1.158 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.0707 | 0.07885 | 0.09012 | 88.98 | 30.37 | 1 | 1.115 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1065 | 0.1229 | 0.1301 | 59.08 | 20.16 | 0.6639 | 1.154 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07572 | 0.08911 | 90.35 | 30.84 | 1 | 1.087 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1065 | 0.1199 | 0.127 | 59.08 | 20.16 | 0.6538 | 1.125 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07378 | 0.07993 | 90.35 | 30.84 | 1 | 1.06 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1065 | 0.1249 | 0.1312 | 59.08 | 20.16 | 0.6538 | 1.173 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.0768 | 0.08814 | 89.04 | 30.39 | 1 | 1.087 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1065 | 0.1179 | 0.124 | 59.08 | 20.16 | 0.6635 | 1.107 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07788 | 0.08605 | 90.35 | 30.84 | 1 | 1.118 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.1014 | 0.1096 | 74.02 | 25.27 | 0.8193 | 1.193 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07064 | 0.07987 | 0.09012 | 89.06 | 30.4 | 1 | 1.131 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08909 | 0.1055 | 0.1117 | 70.62 | 24.11 | 0.7929 | 1.184 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07685 | 0.08706 | 89.04 | 30.39 | 1 | 1.088 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.09836 | 0.1129 | 72.28 | 24.67 | 0.8118 | 1.13 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0768 | 0.08807 | 90.35 | 30.84 | 1 | 1.103 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.1034 | 0.1118 | 72.28 | 24.67 | 0.8 | 1.188 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07782 | 0.08706 | 90.35 | 30.84 | 1 | 1.118 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09318 | 0.1137 | 74.02 | 25.27 | 0.8193 | 1.096 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07578 | 0.08808 | 90.35 | 30.84 | 1 | 1.088 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08602 | 0.1026 | 0.1138 | 73.14 | 24.97 | 0.8095 | 1.192 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07506 | 0.08806 | 90.35 | 30.84 | 1 | 1.078 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09728 | 0.1056 | 74.02 | 25.27 | 0.8193 | 1.145 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07895 | 0.08704 | 89.04 | 30.39 | 1 | 1.117 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09315 | 0.1045 | 74.02 | 25.27 | 0.8313 | 1.096 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1505 | 0.1546 | 0.1659 | 445.8 | 0 | 1 | 1.027 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.17 | 0.1773 | 0.1884 | 394.8 | 0 | 0.8855 | 1.043 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2888 | 0.2929 | 0.297 | 464.8 | 0 | 1 | 1.014 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3154 | 0.3237 | 0.34 | 425.6 | 0 | 0.9156 | 1.026 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08602 | 0.09114 | 0.09328 | 391.6 | 0.195 | 1 | 1.06 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.126 | 0.1496 | 0.1527 | 267.4 | 0.1332 | 0.6829 | 1.187 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1537 | 0.1628 | 452.9 | 0.113 | 1 | 1.035 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1761 | 0.1844 | 0.1977 | 381.8 | 0.09526 | 0.843 | 1.047 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1567 | 0.1608 | 449.8 | 0.1122 | 1 | 1.048 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1792 | 0.1937 | 0.2026 | 375.2 | 0.09362 | 0.8343 | 1.081 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1526 | 0.1567 | 0.1619 | 439.8 | 0.11 | 1 | 1.027 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.171 | 0.1813 | 0.1874 | 392.4 | 0.09811 | 0.8922 | 1.06 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2929 | 0.298 | 0.3012 | 458.3 | 0.05729 | 1 | 1.017 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3154 | 0.3267 | 0.3328 | 425.6 | 0.05319 | 0.9286 | 1.036 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2212 | 0.2263 | 0.2273 | 455.1 | 0.07585 | 1 | 1.023 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2437 | 0.2499 | 0.255 | 413 | 0.06884 | 0.9076 | 1.025 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4352 | 0.4362 | 467 | 0.03892 | 1 | 1.01 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4552 | 0.4701 | 0.4741 | 442.3 | 0.03686 | 0.9472 | 1.033 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4301 | 0.4352 | 0.4424 | 468.1 | 0.03901 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4593 | 0.4731 | 0.4793 | 438.4 | 0.03653 | 0.9365 | 1.03 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4353 | 0.4435 | 467 | 0.03892 | 1 | 1.01 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4547 | 0.469 | 0.4813 | 442.8 | 0.0369 | 0.9482 | 1.032 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2181 | 0.2286 | 0.2355 | 615.4 | 0.6153 | 1 | 1.048 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.172 | 0.1824 | 0.1905 | 780.2 | 0.7801 | 1.268 | 1.06 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4321 | 0.4383 | 0.4475 | 621.2 | 0.3106 | 1 | 1.014 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3133 | 0.3229 | 0.3359 | 856.7 | 0.4283 | 1.379 | 1.03 |
| norms | rmsnorm | float16 | 512x1024 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1188 | 0.1393 | 0.1413 | 26.48 | 0.02206 | 1 | 1.173 |
| norms | rmsnorm | float16 | 512x1024 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.05018 | 0.06973 | 0.08822 | 62.69 | 0.05223 | 2.367 | 1.39 noisy |
| norms | rmsnorm | float16 | 1024x2048 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1443 | 0.1475 | 0.1537 | 87.19 | 0.07265 | 1 | 1.022 |
| norms | rmsnorm | float16 | 1024x2048 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.05018 | 0.06083 | 0.08914 | 250.8 | 0.209 | 2.876 | 1.212 noisy |
| norms | rmsnorm | float16 | 2048x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4874 | 0.4926 | 0.5018 | 103.3 | 0.08605 | 1 | 1.011 |
| norms | rmsnorm | float16 | 2048x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.1024 | 0.1169 | 0.1431 | 491.5 | 0.4096 | 4.76 | 1.142 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9472 | 0.9585 | 0.9646 | 106.3 | 0.08856 | 0.9989 | 1.012 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9462 | 0.9585 | 0.9689 | 106.4 | 0.08865 | 1 | 1.013 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.171 | 0.1823 | 0.1885 | 588.6 | 0.4905 | 5.533 | 1.066 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.169 | 0.1741 | 0.1762 | 595.7 | 0.4964 | 5.599 | 1.03 |
| norms | rmsnorm | float16 | 4096x8192 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.831 | 1.836 | 1.839 | 110 | 0.09163 | 1 | 1.003 |
| norms | rmsnorm | float16 | 4096x8192 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3103 | 0.3216 | 0.3287 | 648.9 | 0.5407 | 5.901 | 1.036 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.011 | 1.017 | 1.025 | 199.2 | 0.08299 | 1 | 1.006 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3098 | 0.3216 | 0.3268 | 649.9 | 0.2708 | 3.263 | 1.038 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.05018 | 0.05227 | 0.0594 | 334.4 | 0.4178 | 1 | 1.042 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06554 | 0.07583 | 0.09426 | 256 | 0.3199 | 0.7656 | 1.157 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.08192 | 0.08412 | 0.09638 | 409.6 | 0.2559 | 1 | 1.027 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.09933 | 0.1158 | 0.1229 | 337.8 | 0.2111 | 0.8247 | 1.165 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7158 | 0.7209 | 0.723 | 140.6 | 0.1172 | 1 | 1.007 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2437 | 0.254 | 0.2571 | 413 | 0.3442 | 2.937 | 1.042 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.418 | 1.424 | 1.43 | 142 | 0.05915 | 1 | 1.004 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4547 | 0.467 | 0.472 | 442.8 | 0.1845 | 3.119 | 1.027 |

## Observation

- Loaded 115 benchmark rows from 10 result files.
- Fastest backend split: torch 47, triton 10.
- All 115 correctness checks passed.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.901x); norms rmsnorm float16 eps=1e-06 (5.599x); norms rmsnorm float16 eps=1e-06 (5.533x).
- Noisy rows at p95/p50 >= 1.2: norms rmsnorm float16 eps=1e-06 (2.367x); matmul matmul float16 block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 (1.216 noise); norms rmsnorm float16 eps=1e-06 (2.876x).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.901x); norms rmsnorm float16 eps=1e-06 (5.599x); norms rmsnorm float16 eps=1e-06 (5.533x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.
- Tiled matmul rows should be judged by TFLOP/s and Tensor Core counters; the current best Triton tile/launch config is matmul matmul float16 block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 at 25.74 TFLOP/s.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
