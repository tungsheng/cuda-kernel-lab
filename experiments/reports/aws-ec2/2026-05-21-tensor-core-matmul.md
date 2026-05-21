# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/matmul-tile-shape.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/matmul.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul/vector-add-block-size.jsonl`

## Environment

- Git commit: `c72bdc2d1a473729541935c83f671266f4096e49`
- Git dirty: `True`
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
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06864 | 91.66 | 31.29 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07014 | 89.69 | 30.62 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.0696 | 90.39 | 30.85 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.07061 | 89.1 | 30.41 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | PyTorch reference baseline | 0.06954 | 90.48 | 30.88 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 448.9 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2888 | 464.8 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08499 | 396.3 | 0.1974 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 449.8 | 0.1122 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1516 | 442.8 | 0.1107 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2939 | 456.7 | 0.05709 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2202 | 457.2 | 0.0762 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4301 | 468.1 | 0.03901 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4306 | 467.6 | 0.03897 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.1731 | 775.6 | 0.7755 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3133 | 856.7 | 0.4283 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.171 | 588.6 | 0.4905 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3108 | 647.8 | 0.2699 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.05018 | 334.4 | 0.4178 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.08192 | 409.6 | 0.2559 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2417 | 416.5 | 0.3471 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4577 | 439.8 | 0.1833 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07788 | 0.08914 | 90.35 | 30.84 | 1 | 1.118 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08602 | 0.09523 | 0.1097 | 73.14 | 24.97 | 0.8095 | 1.107 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07788 | 0.08606 | 89.04 | 30.39 | 1 | 1.102 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.1037 | 0.1127 | 72.28 | 24.67 | 0.8118 | 1.191 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07485 | 0.08814 | 89.04 | 30.39 | 1 | 1.059 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09011 | 0.1016 | 0.1106 | 69.82 | 23.83 | 0.7841 | 1.128 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07378 | 0.08607 | 91.7 | 31.3 | 1 | 1.075 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08806 | 0.1006 | 0.1119 | 71.44 | 24.39 | 0.7791 | 1.142 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06864 | 0.07187 | 0.08814 | 91.66 | 31.29 | 1 | 1.047 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08902 | 0.1035 | 0.1128 | 70.67 | 24.12 | 0.771 | 1.162 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07613 | 0.08816 | 91.7 | 31.3 | 1 | 1.11 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08806 | 0.1035 | 0.1127 | 71.44 | 24.39 | 0.7791 | 1.176 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07071 | 0.07384 | 91.7 | 31.3 | 1 | 1.031 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09539 | 0.1046 | 74.02 | 25.27 | 0.8072 | 1.122 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07885 | 0.08706 | 90.35 | 30.84 | 1 | 1.132 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09436 | 0.1045 | 74.02 | 25.27 | 0.8193 | 1.11 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07281 | 0.08914 | 91.7 | 31.3 | 1 | 1.061 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1874 | 0.2028 | 0.2069 | 33.57 | 11.46 | 0.3661 | 1.082 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07798 | 0.0902 | 89.04 | 30.39 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.127 | 0.1454 | 0.1526 | 49.55 | 16.91 | 0.5565 | 1.145 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07588 | 0.08816 | 90.35 | 30.84 | 1 | 1.09 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1249 | 0.1372 | 0.1434 | 50.36 | 17.19 | 0.5574 | 1.098 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0769 | 0.0881 | 90.35 | 30.84 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1792 | 0.1936 | 0.2062 | 35.11 | 11.98 | 0.3886 | 1.081 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07014 | 0.07885 | 0.09009 | 89.69 | 30.62 | 1 | 1.124 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1823 | 0.1976 | 0.2028 | 34.52 | 11.78 | 0.3848 | 1.084 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07588 | 0.08704 | 91.7 | 31.3 | 1 | 1.106 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1352 | 0.1516 | 0.1537 | 46.55 | 15.89 | 0.5076 | 1.122 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07685 | 0.08501 | 91.7 | 31.3 | 1 | 1.12 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1362 | 0.1475 | 0.1546 | 46.2 | 15.77 | 0.5038 | 1.083 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07578 | 0.08712 | 91.7 | 31.3 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1843 | 0.2007 | 0.2029 | 34.13 | 11.65 | 0.3722 | 1.089 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07798 | 0.08807 | 89.04 | 30.39 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1864 | 0.1935 | 0.2019 | 33.76 | 11.52 | 0.3791 | 1.038 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07895 | 0.08807 | 90.35 | 30.84 | 1 | 1.134 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1034 | 0.1129 | 0.126 | 60.87 | 20.78 | 0.6737 | 1.092 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0748 | 0.08096 | 90.35 | 30.84 | 1 | 1.074 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1024 | 0.1188 | 0.1251 | 61.44 | 20.97 | 0.68 | 1.161 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07795 | 0.08813 | 90.35 | 30.84 | 1 | 1.119 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1147 | 0.1199 | 0.126 | 54.86 | 18.72 | 0.6071 | 1.045 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07286 | 0.08301 | 91.7 | 31.3 | 1 | 1.062 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1167 | 0.124 | 0.1311 | 53.89 | 18.4 | 0.5877 | 1.062 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.0696 | 0.07608 | 0.08502 | 90.39 | 30.85 | 1 | 1.093 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1055 | 0.1229 | 0.1251 | 59.65 | 20.36 | 0.6599 | 1.165 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07383 | 0.08813 | 90.35 | 30.84 | 1 | 1.06 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1065 | 0.124 | 0.1383 | 59.08 | 20.16 | 0.6538 | 1.164 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07583 | 0.08197 | 89.04 | 30.39 | 1 | 1.073 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1055 | 0.1219 | 0.126 | 59.65 | 20.36 | 0.6699 | 1.155 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07281 | 0.08296 | 91.7 | 31.3 | 1 | 1.061 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1044 | 0.1199 | 0.1252 | 60.24 | 20.56 | 0.6569 | 1.148 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07788 | 0.08911 | 90.35 | 30.84 | 1 | 1.118 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08397 | 0.09841 | 0.1086 | 74.93 | 25.58 | 0.8293 | 1.172 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07583 | 0.09016 | 89.04 | 30.39 | 1 | 1.073 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.09329 | 0.1045 | 72.28 | 24.67 | 0.8118 | 1.072 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07685 | 0.08704 | 90.35 | 30.84 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08602 | 0.09431 | 0.1127 | 73.14 | 24.97 | 0.8095 | 1.096 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07695 | 0.08806 | 90.35 | 30.84 | 1 | 1.105 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=128, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08602 | 0.09933 | 0.1147 | 73.14 | 24.97 | 0.8095 | 1.155 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07061 | 0.07685 | 0.08807 | 89.1 | 30.41 | 1 | 1.088 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08397 | 0.09329 | 0.1045 | 74.93 | 25.58 | 0.8409 | 1.111 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07485 | 0.08503 | 90.35 | 30.84 | 1 | 1.075 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09549 | 0.1168 | 74.02 | 25.27 | 0.8193 | 1.123 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07895 | 0.08821 | 90.35 | 30.84 | 1 | 1.134 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=3, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08499 | 0.09139 | 0.1056 | 74.02 | 25.27 | 0.8193 | 1.075 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06954 | 0.07183 | 0.07899 | 90.48 | 30.88 | 1 | 1.033 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=64, block_n=64, block_k=32, num_warps=8, num_stages=4, input_precision=tf32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.08704 | 0.09758 | 0.1098 | 72.28 | 24.67 | 0.7989 | 1.121 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1526 | 0.17 | 448.9 | 0 | 1 | 1.021 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1731 | 0.1844 | 0.1926 | 387.8 | 0 | 0.8639 | 1.065 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2888 | 0.2939 | 0.298 | 464.8 | 0 | 1 | 1.018 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3228 | 0.3318 | 426.9 | 0 | 0.9186 | 1.027 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08499 | 0.08811 | 0.09337 | 396.3 | 0.1974 | 1 | 1.037 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.128 | 0.1722 | 0.1875 | 263.2 | 0.1311 | 0.664 | 1.345 noisy |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1577 | 0.1639 | 452.9 | 0.113 | 1 | 1.062 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1772 | 0.1854 | 0.1957 | 379.6 | 0.09471 | 0.8382 | 1.047 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1517 | 0.1649 | 449.8 | 0.1122 | 1 | 1.015 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1792 | 0.1946 | 0.1988 | 375.2 | 0.09362 | 0.8343 | 1.086 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1516 | 0.1577 | 0.1669 | 442.8 | 0.1107 | 1 | 1.041 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1731 | 0.1803 | 0.1936 | 387.8 | 0.09695 | 0.8757 | 1.042 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2939 | 0.3 | 0.3021 | 456.7 | 0.05709 | 1 | 1.021 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3246 | 0.3288 | 426.9 | 0.05337 | 0.9349 | 1.033 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2202 | 0.2253 | 0.2264 | 457.2 | 0.0762 | 1 | 1.023 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2427 | 0.2571 | 0.2765 | 414.8 | 0.06913 | 0.9072 | 1.059 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4301 | 0.4352 | 0.4373 | 468.1 | 0.03901 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4557 | 0.47 | 0.4711 | 441.8 | 0.03682 | 0.9438 | 1.031 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4374 | 0.4424 | 467 | 0.03892 | 1 | 1.015 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4588 | 0.4722 | 0.4782 | 438.9 | 0.03657 | 0.9397 | 1.029 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4306 | 0.4352 | 0.4434 | 467.6 | 0.03897 | 1 | 1.011 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4536 | 0.4661 | 0.4721 | 443.8 | 0.03698 | 0.9491 | 1.028 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2181 | 0.2232 | 0.2243 | 615.4 | 0.6153 | 1 | 1.023 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.1731 | 0.1813 | 0.1895 | 775.6 | 0.7755 | 1.26 | 1.048 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4342 | 0.4435 | 0.4557 | 618.3 | 0.3091 | 1 | 1.021 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3133 | 0.3277 | 0.3297 | 856.7 | 0.4283 | 1.386 | 1.046 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9462 | 0.9523 | 0.9605 | 106.4 | 0.08865 | 1 | 1.006 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.171 | 0.1784 | 0.1854 | 588.6 | 0.4905 | 5.533 | 1.043 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.013 | 1.019 | 1.026 | 198.8 | 0.08283 | 1 | 1.006 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3108 | 0.3196 | 0.3267 | 647.8 | 0.2699 | 3.259 | 1.028 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.05018 | 0.05432 | 0.05942 | 334.4 | 0.4178 | 1 | 1.083 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06554 | 0.07685 | 0.09242 | 256 | 0.3199 | 0.7656 | 1.173 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.08192 | 0.08499 | 0.09021 | 409.6 | 0.2559 | 1 | 1.038 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.09933 | 0.1047 | 0.1168 | 337.8 | 0.2111 | 0.8247 | 1.054 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7168 | 0.7229 | 0.726 | 140.4 | 0.117 | 1 | 1.009 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2417 | 0.2531 | 0.2591 | 416.5 | 0.3471 | 2.966 | 1.047 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.42 | 1.425 | 1.427 | 141.8 | 0.05906 | 1 | 1.004 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4577 | 0.4721 | 0.4742 | 439.8 | 0.1833 | 3.103 | 1.031 |

## Observation

- Loaded 104 benchmark rows from 8 result files.
- Fastest backend split: torch 46, triton 6.
- All 104 correctness checks passed.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.533x); norms rmsnorm float32 eps=1e-06 (3.259x); swiglu swiglu float32 block_size=1024 (3.103x).
- Noisy rows at p95/p50 >= 1.2: memory reduction_sum float16 reduction_strategy=iterative, block_size=1024 (1.345 noise).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.533x); norms rmsnorm float32 eps=1e-06 (3.259x); swiglu swiglu float32 block_size=1024 (3.103x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.
- Tiled matmul rows should be judged by TFLOP/s and Tensor Core counters; the current best Triton tile/launch config is matmul matmul float16 block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, input_precision=tf32 at 25.58 TFLOP/s.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
