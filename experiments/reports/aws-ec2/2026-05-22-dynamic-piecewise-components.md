# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/decode-step.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-components/vector-add-block-size.jsonl`

## Environment

- Git commit: `fa306ecf6d2abe28239afe1e0bc68bdeb2561f20`
- Git dirty: `True`
- Python: `3.10.12`
- Platform: `Linux-6.8.0-1055-aws-x86_64-with-glibc2.35`
- PyTorch: `2.12.0`
- Triton: `3.7.0`
- CUDA devices: `NVIDIA A10G (22.06 GiB)`

## Optimization Techniques Tested

| Family | Technique | Used By | Hypothesis |
| --- | --- | --- | --- |
| baseline | Naive eager decode step | decode_step decode_step | A decomposed PyTorch decode step establishes the end-to-end launch and intermediate-allocation baseline for one synthetic token. |
| baseline | PyTorch reference baseline | torch controls | Establish the latency, bandwidth, and correctness baseline for comparison. |
| launch tuning | Coalesced block-size tuning | memory copy, memory scale, memory vector_add | Varying Triton block size for contiguous streaming kernels can improve occupancy and memory throughput. |
| reduction | Iterative block reduction | memory reduction_sum | Repeated Triton block reductions over FP32 partial sums should stream memory efficiently, while repeated launches expose orchestration overhead. |
| reduction | Two-pass block reduction | memory reduction_sum | Reducing to FP32 partial sums with Triton and finalizing in a second step can cut repeated launches, but may pay partial-traffic or framework cleanup cost. |
| fusion | Fused eager decode step | decode_step decode_step | Replacing decomposed normalization and activation work with fused kernels should reduce kernel count and intermediate memory traffic before graph replay. |
| fusion | Elementwise SwiGLU fusion | swiglu swiglu | Fusing sigmoid, SiLU gating, multiply, and store should avoid materialized activation intermediates, lowering memory traffic and launch overhead. |
| fusion | Row-wise LayerNorm fusion | norms layernorm | Fusing row reductions, normalization, parameter loads, and affine writeback should remove framework overhead and avoid intermediate normalization tensors. |
| fusion | Row-wise RMSNorm fusion | norms rmsnorm | Fusing row reductions, normalization, parameter loads, and affine writeback should remove framework overhead and avoid intermediate normalization tensors. |
| fusion | Row-wise softmax fusion | softmax softmax | Keeping row max, subtract, exp, sum, divide, and store inside one kernel should reduce global-memory traffic and launch overhead versus a naive multi-kernel path. |
| launch replay | Fused CUDA Graph replay | decode_step decode_step | Combining fused kernels with CUDA Graph replay should reduce both intermediate traffic and per-token launch overhead. |
| launch replay | Fused piecewise CUDA Graph replay | decode_step decode_step | Capturing the static fused pre/post-attention regions while leaving attention eager should keep graph benefits when batch and sequence shapes vary. |
| launch replay | Naive CUDA Graph replay | decode_step decode_step | Replaying the decomposed decode step inside a CUDA Graph should reduce Python and driver launch overhead without changing the kernels themselves. |

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4588 | 59.57 | 0.05984 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1512 | 180.7 | 0.1815 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.3915 | 69.8 | 0.07012 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4702 | 58.13 | 0.05838 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.1627 | 168 | 0.1687 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5698 | 67.26 | 0.173 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.571 | 67.12 | 0.1726 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1505 | 445.8 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2888 | 464.8 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08602 | 391.6 | 0.195 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1526 | 439.8 | 0.11 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2929 | 458.3 | 0.05729 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2212 | 455.1 | 0.07585 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4301 | 468.1 | 0.03901 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4301 | 468.1 | 0.03901 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.172 | 780.2 | 0.7801 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3133 | 856.7 | 0.4283 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.171 | 588.6 | 0.4905 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3113 | 646.7 | 0.2695 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.05018 | 334.4 | 0.4178 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.08192 | 409.6 | 0.2559 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2447 | 411.3 | 0.3428 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4567 | 440.8 | 0.1837 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4588 | 0.4896 | 0.495 | 59.57 | 0.05984 |  | 1.067 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1512 | 0.1554 | 0.169 | 180.7 | 0.1815 |  | 1.028 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise_graph | Fused piecewise CUDA Graph replay | pass | 0.3915 | 0.4108 | 0.4131 | 69.8 | 0.07012 |  | 1.049 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4702 | 0.4856 | 0.5188 | 58.13 | 0.05838 |  | 1.033 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1627 | 0.1756 | 0.1797 | 168 | 0.1687 |  | 1.079 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.5698 | 0.9123 | 1.05 | 67.26 | 0.173 |  | 1.601 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.571 | 1.053 | 1.239 | 67.12 | 0.1726 |  | 1.844 noisy |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1505 | 0.1537 | 0.1557 | 445.8 | 0 | 1 | 1.021 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1731 | 0.1772 | 0.1843 | 387.8 | 0 | 0.8698 | 1.024 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2888 | 0.2929 | 0.3041 | 464.8 | 0 | 1 | 1.014 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3139 | 0.3208 | 0.3328 | 427.6 | 0 | 0.9201 | 1.022 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08602 | 0.09111 | 0.09531 | 391.6 | 0.195 | 1 | 1.059 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1229 | 0.1444 | 0.1498 | 274.1 | 0.1365 | 0.6999 | 1.175 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1537 | 0.1659 | 452.9 | 0.113 | 1 | 1.035 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1731 | 0.1751 | 0.1772 | 388.5 | 0.09695 | 0.858 | 1.012 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1506 | 0.1628 | 452.9 | 0.113 | 1 | 1.015 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1802 | 0.1917 | 0.2069 | 373.1 | 0.09309 | 0.8239 | 1.064 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1526 | 0.1588 | 0.1679 | 439.8 | 0.11 | 1 | 1.041 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.173 | 0.1761 | 0.1824 | 387.9 | 0.09697 | 0.8819 | 1.018 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2929 | 0.2959 | 0.3072 | 458.3 | 0.05729 | 1 | 1.01 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3194 | 0.3298 | 426.9 | 0.05337 | 0.9316 | 1.016 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2212 | 0.2253 | 0.2274 | 455.1 | 0.07585 | 1 | 1.019 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2427 | 0.2489 | 0.2591 | 414.8 | 0.06913 | 0.9114 | 1.026 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4301 | 0.4342 | 0.4352 | 468.1 | 0.03901 | 1 | 1.01 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4547 | 0.462 | 0.4701 | 442.8 | 0.0369 | 0.9459 | 1.016 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4362 | 0.4405 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4608 | 0.4742 | 0.4813 | 436.9 | 0.03641 | 0.9356 | 1.029 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4301 | 0.4353 | 0.4373 | 468.1 | 0.03901 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4506 | 0.4639 | 0.468 | 446.8 | 0.03724 | 0.9545 | 1.03 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2176 | 0.2253 | 0.2335 | 616.8 | 0.6168 | 1 | 1.035 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.172 | 0.1783 | 0.1895 | 780.2 | 0.7801 | 1.265 | 1.036 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4342 | 0.4403 | 0.4455 | 618.3 | 0.3091 | 1 | 1.014 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3133 | 0.3236 | 0.3277 | 856.7 | 0.4283 | 1.386 | 1.033 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9492 | 0.9647 | 0.9687 | 106 | 0.08837 | 1 | 1.016 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.171 | 0.1772 | 0.1813 | 588.6 | 0.4905 | 5.551 | 1.036 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.013 | 1.019 | 1.025 | 198.8 | 0.08283 | 1 | 1.006 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3113 | 0.3205 | 0.3258 | 646.7 | 0.2695 | 3.253 | 1.03 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.05018 | 0.05222 | 0.05846 | 334.4 | 0.4178 | 1 | 1.041 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06654 | 0.07905 | 0.08608 | 252.1 | 0.315 | 0.754 | 1.188 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.08192 | 0.08504 | 0.08702 | 409.6 | 0.2559 | 1 | 1.038 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.09933 | 0.1106 | 0.124 | 337.8 | 0.2111 | 0.8247 | 1.113 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7168 | 0.7219 | 0.726 | 140.4 | 0.117 | 1 | 1.007 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2447 | 0.2571 | 0.2611 | 411.3 | 0.3428 | 2.929 | 1.05 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.42 | 1.426 | 1.437 | 141.8 | 0.05906 | 1 | 1.004 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4567 | 0.4732 | 0.4782 | 440.8 | 0.1837 | 3.11 | 1.036 |

## Observation

- Loaded 45 benchmark rows from 8 result files.
- Fastest backend split: dynamic-eager 1, dynamic-piecewise-graph 1, fused 3, naive 2, torch 13, triton 6.
- Correctness summary: not checked 1, pass 44.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.551x); norms rmsnorm float32 eps=1e-06 (3.253x); swiglu swiglu float32 block_size=1024 (3.11x).
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.844 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.601 noise).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.551x); norms rmsnorm float32 eps=1e-06 (3.253x); swiglu swiglu float32 block_size=1024 (3.11x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
