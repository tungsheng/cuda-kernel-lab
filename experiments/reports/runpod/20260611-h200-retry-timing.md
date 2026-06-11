# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/runpod/20260611-h200-retry-timing/memory.jsonl`
- `experiments/results/runpod/20260611-h200-retry-timing/norms.jsonl`
- `experiments/results/runpod/20260611-h200-retry-timing/reduction-strategy.jsonl`
- `experiments/results/runpod/20260611-h200-retry-timing/softmax.jsonl`
- `experiments/results/runpod/20260611-h200-retry-timing/swiglu.jsonl`
- `experiments/results/runpod/20260611-h200-retry-timing/vector-add-block-size.jsonl`

## Environment

- Git commit: `a113a646e6a18ce3796f812de6976dbb452f923f`
- Git dirty: `False`
- Provider: `runpod`
- Provider id: `99alnql81pfet0`
- Provider GPU: `NVIDIA H200`
- Provider cloud: `SECURE`
- Provider image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Python: `3.12.3`
- Platform: `Linux-6.8.0-83-generic-x86_64-with-glibc2.39`
- PyTorch: `2.9.1`
- Triton: `3.5.1`
- CUDA devices: `NVIDIA H200 (139.81 GiB)`

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

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.02323 | 2889 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.03869 | 3469 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.01998 | 1686 | 0.8395 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.02982 | 2255 | 0.5625 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.03043 | 2210 | 0.5513 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.02435 | 2756 | 0.6889 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.03976 | 3376 | 0.422 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.03117 | 3230 | 0.5383 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.05373 | 3747 | 0.3123 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.05413 | 3719 | 0.31 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.05414 | 3718 | 0.3099 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.03784 | 3547 | 3.547 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.05445 | 4930 | 2.465 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.03499 | 2877 | 2.397 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.05192 | 3878 | 1.616 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.01594 | 1053 | 1.315 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.01547 | 2169 | 1.355 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.04218 | 2387 | 1.989 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.06619 | 3042 | 1.267 |

## Roofline Summary

- Spec: `NVIDIA H200 SXM` (NVIDIA H200 published peak specs).
- Peak HBM bandwidth: `4800 GB/s`.

| Primitive | Operation | Dtype | Shape | Backend | Strategy | Intensity FLOP/B | Achieved GB/s | HBM Peak % | Achieved TFLOP/s | Math Peak % | Bound |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| norms | layernorm | float32 | 4096x4096 | triton | triton-fused-layernorm | 0.5 | 4930 | 102.7 | 2.465 | 0.2492 | memory |
| norms | rmsnorm | float32 | 4096x4096 | triton | triton-fused-rmsnorm | 0.4166 | 3878 | 80.78 | 1.616 | 0.1634 | memory |
| memory | vector_add | float32 | 16777216 | torch | torch-baseline | 0.08333 | 3747 | 78.07 | 0.3123 | 0.03157 | memory |
| memory | vector_add | float32 | 16777216 | torch | torch-baseline | 0.08333 | 3719 | 77.49 | 0.31 | 0.03134 | memory |
| memory | vector_add | float32 | 16777216 | torch | torch-baseline | 0.08333 | 3718 | 77.47 | 0.3099 | 0.03133 | memory |
| norms | layernorm | float16 | 4096x4096 | triton | triton-fused-layernorm | 0.9999 | 3547 | 73.9 | 3.547 | 0.1792 | memory |
| memory | copy | float32 | 16777216 | torch | torch-baseline |  | 3469 | 72.28 | 0 | 0 | memory |
| memory | scale | float32 | 16777216 | torch | torch-baseline | 0.125 | 3376 | 70.33 | 0.422 | 0.04267 | memory |
| memory | vector_add | float16 | 16777216 | torch | torch-baseline | 0.1667 | 3230 | 67.29 | 0.5383 | 0.0272 | memory |
| swiglu | swiglu | float32 | 4096x4096 | triton | triton-fused-swiglu | 0.4167 | 3042 | 63.37 | 1.267 | 0.1281 | memory |
| memory | copy | float16 | 16777216 | torch | torch-baseline |  | 2889 | 60.18 | 0 | 0 | memory |
| norms | rmsnorm | float16 | 4096x4096 | triton | triton-fused-rmsnorm | 0.8333 | 2877 | 59.93 | 2.397 | 0.1211 | memory |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.02323 | 0.02481 | 0.02558 | 2889 | 0 | 1 | 1.068 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.034 | 0.03812 | 0.04077 | 1974 | 0 | 0.6833 | 1.121 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.03869 | 0.04032 | 0.04132 | 3469 | 0 | 1 | 1.042 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.05018 | 0.05258 | 0.05424 | 2675 | 0 | 0.771 | 1.048 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.01998 | 0.02048 | 0.0216 | 1686 | 0.8395 | 1 | 1.025 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.06101 | 0.06954 | 0.0739 | 552.2 | 0.275 | 0.3276 | 1.14 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.02982 | 0.03133 | 0.03155 | 2255 | 0.5625 | 1 | 1.05 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.05779 | 0.06462 | 0.07084 | 1163 | 0.2903 | 0.5161 | 1.118 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.03043 | 0.03213 | 0.03409 | 2210 | 0.5513 | 1 | 1.056 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.04064 | 0.0442 | 0.0461 | 1655 | 0.4128 | 0.7488 | 1.087 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.02435 | 0.02608 | 0.02707 | 2756 | 0.6889 | 1 | 1.071 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.03419 | 0.03697 | 0.03969 | 1963 | 0.4907 | 0.7122 | 1.081 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.03976 | 0.04151 | 0.04179 | 3376 | 0.422 | 1 | 1.044 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.04998 | 0.05242 | 0.06112 | 2685 | 0.3357 | 0.7955 | 1.049 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.03117 | 0.03332 | 0.03576 | 3230 | 0.5383 | 1 | 1.069 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.04421 | 0.04839 | 0.05185 | 2277 | 0.3795 | 0.705 | 1.095 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.05373 | 0.05517 | 0.0563 | 3747 | 0.3123 | 1 | 1.027 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.0665 | 0.06863 | 0.07049 | 3028 | 0.2523 | 0.808 | 1.032 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.05413 | 0.05558 | 0.05735 | 3719 | 0.31 | 1 | 1.027 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.06717 | 0.07036 | 0.07448 | 2997 | 0.2498 | 0.8059 | 1.047 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.05414 | 0.05556 | 0.05683 | 3718 | 0.3099 | 1 | 1.026 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.06624 | 0.06854 | 0.07169 | 3039 | 0.2533 | 0.8174 | 1.035 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.0409 | 0.04263 | 0.04421 | 3282 | 3.282 | 1 | 1.042 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.03784 | 0.04034 | 0.04586 | 3547 | 3.547 | 1.081 | 1.066 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.06074 | 0.06237 | 0.06381 | 4420 | 2.21 | 1 | 1.027 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.05445 | 0.05716 | 0.0634 | 4930 | 2.465 | 1.115 | 1.05 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2135 | 0.2153 | 0.2181 | 471.4 | 0.3929 | 1 | 1.008 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.03499 | 0.038 | 0.0418 | 2877 | 2.397 | 6.102 | 1.086 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1744 | 0.1756 | 0.1774 | 1155 | 0.4811 | 1 | 1.007 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.05192 | 0.0545 | 0.05893 | 3878 | 1.616 | 3.358 | 1.05 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.01594 | 0.01668 | 0.02052 | 1053 | 1.315 | 1 | 1.047 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.02366 | 0.02636 | 0.02901 | 709 | 0.8859 | 0.6734 | 1.114 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.01547 | 0.01632 | 0.0183 | 2169 | 1.355 | 1 | 1.055 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.0223 | 0.02531 | 0.02863 | 1504 | 0.9399 | 0.6937 | 1.135 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.09589 | 0.09687 | 0.09719 | 1050 | 0.8748 | 1 | 1.01 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.04218 | 0.04485 | 0.04887 | 2387 | 1.989 | 2.274 | 1.064 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1735 | 0.1748 | 0.1772 | 1160 | 0.4835 | 1 | 1.007 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.06619 | 0.06868 | 0.07207 | 3042 | 1.267 | 2.621 | 1.038 |

## Observation

- Loaded 38 benchmark rows from 6 result files.
- Fastest backend split: torch 13, triton 6.
- All 38 correctness checks passed.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (6.102x); norms rmsnorm float32 eps=1e-06 (3.358x); swiglu swiglu float32 block_size=1024 (2.621x).
- No rows exceeded the 1.2 p95/p50 noise threshold.

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (6.102x); norms rmsnorm float32 eps=1e-06 (3.358x); swiglu swiglu float32 block_size=1024 (2.621x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.

## Next Question

What does Nsight Compute show for the Triton memory primitive bottleneck?
