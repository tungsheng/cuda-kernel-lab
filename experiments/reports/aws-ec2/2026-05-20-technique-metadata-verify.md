# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/matmul-tile-shape.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/matmul.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/vector-add-block-size.jsonl`

## Environment

- Git commit: `725e35d0e7249f1dceb158cce8b8716994703d5d`
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
| tiling | Tiled dot-product reuse | matmul matmul | Triton tile-shape sweeps with `tl.dot` can increase arithmetic intensity and Tensor Core utilization, but may trade off occupancy and register pressure. |

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32 | torch | PyTorch reference baseline | 0.07066 | 89.04 | 30.39 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32 | torch | PyTorch reference baseline | 0.06963 | 90.35 | 30.84 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64 | torch | PyTorch reference baseline | 0.06861 | 91.7 | 31.3 |
| matmul | matmul | float32 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32 | torch | PyTorch reference baseline | 0.1587 | 79.28 | 13.53 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 448.9 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2888 | 464.8 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08602 | 391.6 | 0.195 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 449.8 | 0.1122 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 449.8 | 0.1122 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1526 | 439.8 | 0.11 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2939 | 456.7 | 0.05709 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2202 | 457.2 | 0.0762 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4321 | 465.9 | 0.03882 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.172 | 780.2 | 0.7801 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3133 | 856.7 | 0.4283 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.17 | 592.2 | 0.4935 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3103 | 648.9 | 0.2703 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.04915 | 341.3 | 0.4265 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.08192 | 409.6 | 0.2559 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2457 | 409.7 | 0.3414 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4567 | 440.8 | 0.1837 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07583 | 0.08812 | 90.35 | 30.84 | 1 | 1.089 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1874 | 0.2028 | 0.2131 | 33.57 | 11.46 | 0.3716 | 1.082 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.07066 | 0.07788 | 0.08903 | 89.04 | 30.39 | 1 | 1.102 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=16, block_n=32, block_k=32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1239 | 0.1423 | 0.1466 | 50.78 | 17.33 | 0.5702 | 1.149 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.0769 | 0.08503 | 90.35 | 30.84 | 1 | 1.104 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=16, block_k=32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1321 | 0.1435 | 0.1536 | 47.63 | 16.26 | 0.5271 | 1.086 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06963 | 0.07583 | 0.08907 | 90.35 | 30.84 | 1 | 1.089 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=32 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1024 | 0.1261 | 0.1322 | 61.44 | 20.97 | 0.68 | 1.231 noisy |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06861 | 0.07286 | 0.08501 | 91.7 | 31.3 | 1 | 1.062 |
| matmul | matmul | float16 | 1024x1024x1024 | block_m=32, block_n=32, block_k=64 | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.1034 | 0.1219 | 0.1342 | 60.83 | 20.76 | 0.6634 | 1.179 |
| matmul | matmul | float32 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1587 | 0.1659 | 0.1763 | 79.28 | 13.53 | 1 | 1.045 |
| matmul | matmul | float32 | 1024x1024x1024 | block_m=16, block_n=16, block_k=32 | triton | triton-tiled-dot | Tiled dot-product reuse | fail | 0.2755 | 0.2878 | 0.299 | 45.68 | 7.796 | 0.5762 | 1.045 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1546 | 0.1598 | 448.9 | 0 | 1 | 1.034 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1731 | 0.1802 | 0.1925 | 387.8 | 0 | 0.8639 | 1.041 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2888 | 0.295 | 0.3011 | 464.8 | 0 | 1 | 1.021 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3258 | 0.3369 | 426.9 | 0 | 0.9186 | 1.036 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08602 | 0.08924 | 0.09421 | 391.6 | 0.195 | 1 | 1.038 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.129 | 0.1526 | 0.1557 | 261.1 | 0.13 | 0.6667 | 1.183 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1547 | 0.1568 | 449.8 | 0.1122 | 1 | 1.035 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1761 | 0.1905 | 0.2028 | 381.8 | 0.09526 | 0.8488 | 1.082 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1569 | 0.169 | 449.8 | 0.1122 | 1 | 1.049 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1792 | 0.1938 | 0.2059 | 375.2 | 0.09362 | 0.8343 | 1.082 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1526 | 0.1587 | 0.1679 | 439.8 | 0.11 | 1 | 1.04 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1731 | 0.1782 | 0.1813 | 387.8 | 0.09695 | 0.8817 | 1.03 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2939 | 0.3001 | 0.3072 | 456.7 | 0.05709 | 1 | 1.021 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3139 | 0.3277 | 0.3369 | 427.6 | 0.05345 | 0.9363 | 1.044 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2202 | 0.2254 | 0.2285 | 457.2 | 0.0762 | 1 | 1.024 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2447 | 0.258 | 0.2622 | 411.3 | 0.06855 | 0.8996 | 1.054 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4363 | 0.4485 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4572 | 0.4701 | 0.4772 | 440.3 | 0.03669 | 0.9429 | 1.028 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4321 | 0.4374 | 0.4413 | 465.9 | 0.03882 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4598 | 0.4742 | 0.4783 | 437.9 | 0.03649 | 0.9399 | 1.031 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4362 | 0.4405 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4536 | 0.4702 | 0.4813 | 443.8 | 0.03698 | 0.9503 | 1.037 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2171 | 0.2305 | 0.2346 | 618.3 | 0.6182 | 1 | 1.062 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.172 | 0.1782 | 0.1875 | 780.2 | 0.7801 | 1.262 | 1.036 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4331 | 0.4403 | 0.4475 | 619.7 | 0.3099 | 1 | 1.017 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3133 | 0.3278 | 0.337 | 856.7 | 0.4283 | 1.382 | 1.046 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9452 | 0.9513 | 0.9606 | 106.5 | 0.08875 | 1 | 1.006 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.17 | 0.1793 | 0.1875 | 592.2 | 0.4935 | 5.56 | 1.055 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.011 | 1.017 | 1.026 | 199.2 | 0.08299 | 1 | 1.006 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3103 | 0.3196 | 0.3278 | 648.9 | 0.2703 | 3.257 | 1.03 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.04915 | 0.05335 | 0.06562 | 341.3 | 0.4265 | 1 | 1.085 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06554 | 0.0789 | 0.09523 | 256 | 0.3199 | 0.75 | 1.204 noisy |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.08192 | 0.08607 | 0.0911 | 409.6 | 0.2559 | 1 | 1.051 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.09779 | 0.1025 | 0.1086 | 343.1 | 0.2144 | 0.8377 | 1.048 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7168 | 0.723 | 0.726 | 140.4 | 0.117 | 1 | 1.009 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2457 | 0.2591 | 0.2694 | 409.7 | 0.3414 | 2.917 | 1.054 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.42 | 1.426 | 1.432 | 141.8 | 0.05906 | 1 | 1.004 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4567 | 0.471 | 0.4793 | 440.8 | 0.1837 | 3.11 | 1.031 |

## Observation

- Loaded 50 benchmark rows from 8 result files.
- Fastest backend split: torch 19, triton 6.
- Correctness summary: fail 1, pass 49.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.56x); norms rmsnorm float32 eps=1e-06 (3.257x); swiglu swiglu float32 block_size=1024 (3.11x).
- Noisy rows at p95/p50 >= 1.2: matmul matmul float16 block_m=32, block_n=32, block_k=32 (1.231 noise); softmax softmax float16 traffic_model=fused (1.204 noise).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.56x); norms rmsnorm float32 eps=1e-06 (3.257x); swiglu swiglu float32 block_size=1024 (3.11x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.
- Tiled matmul rows should be judged by TFLOP/s and Tensor Core counters; the current best Triton tile is matmul matmul float16 block_m=32, block_n=32, block_k=32 (1.231 noise) at 20.97 TFLOP/s.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
