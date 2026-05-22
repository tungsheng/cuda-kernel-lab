# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/decode-step.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-stream-fix/vector-add-block-size.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4785 | 57.11 | 0.05737 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1519 | 179.9 | 0.1807 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.4105 | 66.58 | 0.06688 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4637 | 58.94 | 0.0592 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.1632 | 167.4 | 0.1682 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.576 | 66.54 | 0.1711 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5719 | 67.01 | 0.1723 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 448.9 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2898 | 463.2 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08602 | 391.6 | 0.195 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 449.8 | 0.1122 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1526 | 439.8 | 0.11 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2929 | 458.3 | 0.05729 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2212 | 455.1 | 0.07585 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4301 | 468.1 | 0.03901 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.172 | 780.2 | 0.7801 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3133 | 856.7 | 0.4283 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.171 | 588.8 | 0.4906 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3103 | 648.9 | 0.2703 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.04915 | 341.3 | 0.4265 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.08293 | 404.6 | 0.2528 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2447 | 411.4 | 0.3428 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4562 | 441.3 | 0.1839 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4785 | 0.5085 | 0.5204 | 57.11 | 0.05737 |  | 1.063 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1519 | 0.166 | 0.1743 | 179.9 | 0.1807 |  | 1.093 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise_graph | Fused piecewise CUDA Graph replay | pass | 0.4105 | 0.4393 | 0.4587 | 66.58 | 0.06688 |  | 1.07 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4637 | 0.4973 | 0.5166 | 58.94 | 0.0592 |  | 1.072 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1632 | 0.1783 | 0.1918 | 167.4 | 0.1682 |  | 1.092 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.576 | 0.9087 | 1.051 | 66.54 | 0.1711 |  | 1.578 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | fail | 0.5719 | 1.048 | 1.239 | 67.01 | 0.1723 |  | 1.832 noisy |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1536 | 0.1598 | 448.9 | 0 | 1 | 1.027 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1725 | 0.1823 | 0.1915 | 389 | 0 | 0.8666 | 1.057 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2898 | 0.2939 | 0.3065 | 463.2 | 0 | 1 | 1.014 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.334 | 0.34 | 426.9 | 0 | 0.9218 | 1.063 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08602 | 0.08908 | 0.09526 | 391.6 | 0.195 | 1 | 1.036 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.13 | 0.1537 | 0.166 | 259 | 0.129 | 0.6614 | 1.182 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1505 | 0.1547 | 452.9 | 0.113 | 1 | 1.014 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1772 | 0.1906 | 0.1978 | 379.6 | 0.09471 | 0.8382 | 1.076 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1576 | 0.1578 | 449.8 | 0.1122 | 1 | 1.054 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1802 | 0.1997 | 0.2069 | 373.1 | 0.09309 | 0.8295 | 1.108 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1526 | 0.1588 | 0.1629 | 439.8 | 0.11 | 1 | 1.041 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.173 | 0.1794 | 0.1865 | 387.9 | 0.09698 | 0.882 | 1.037 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2929 | 0.297 | 0.3072 | 458.3 | 0.05729 | 1 | 1.014 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3154 | 0.3258 | 0.333 | 425.6 | 0.05319 | 0.9286 | 1.033 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2212 | 0.2253 | 0.2366 | 455.1 | 0.07585 | 1 | 1.019 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2447 | 0.2489 | 0.2509 | 411.4 | 0.06856 | 0.9039 | 1.017 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4301 | 0.4332 | 0.4373 | 468.1 | 0.03901 | 1 | 1.007 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4567 | 0.469 | 0.4711 | 440.8 | 0.03674 | 0.9417 | 1.027 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4352 | 0.4424 | 467 | 0.03892 | 1 | 1.01 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4608 | 0.4772 | 0.4877 | 436.9 | 0.03641 | 0.9356 | 1.036 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4363 | 0.4383 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4536 | 0.4639 | 0.467 | 443.8 | 0.03698 | 0.9503 | 1.023 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2186 | 0.2243 | 0.2294 | 614 | 0.614 | 1 | 1.026 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.172 | 0.1792 | 0.1823 | 780.2 | 0.7801 | 1.271 | 1.042 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4332 | 0.4393 | 0.4495 | 619.7 | 0.3098 | 1 | 1.014 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3133 | 0.3236 | 0.3359 | 856.7 | 0.4283 | 1.382 | 1.033 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9482 | 0.9635 | 0.9687 | 106.2 | 0.08846 | 1 | 1.016 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.171 | 0.1761 | 0.1813 | 588.8 | 0.4906 | 5.546 | 1.03 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.015 | 1.026 | 1.036 | 198.4 | 0.08266 | 1 | 1.011 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3103 | 0.3217 | 0.3267 | 648.9 | 0.2703 | 3.271 | 1.037 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.04915 | 0.05335 | 0.06558 | 341.3 | 0.4265 | 1 | 1.085 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06451 | 0.07281 | 0.08127 | 260.1 | 0.325 | 0.7619 | 1.129 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.08293 | 0.09011 | 0.1035 | 404.6 | 0.2528 | 1 | 1.087 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.09933 | 0.1106 | 0.1198 | 337.8 | 0.2111 | 0.8349 | 1.113 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7168 | 0.7219 | 0.7271 | 140.4 | 0.117 | 1 | 1.007 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2447 | 0.2571 | 0.2653 | 411.4 | 0.3428 | 2.929 | 1.051 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.418 | 1.422 | 1.448 | 142 | 0.05915 | 1 | 1.003 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4562 | 0.4671 | 0.4752 | 441.3 | 0.1839 | 3.109 | 1.024 |

## Observation

- Loaded 45 benchmark rows from 8 result files.
- Fastest backend split: dynamic-eager 1, dynamic-piecewise-graph 1, fused 3, naive 2, torch 13, triton 6.
- Correctness summary: fail 1, not checked 1, pass 43.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.546x); norms rmsnorm float32 eps=1e-06 (3.271x); swiglu swiglu float32 block_size=1024 (3.109x).
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.832 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.578 noise).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.546x); norms rmsnorm float32 eps=1e-06 (3.271x); swiglu swiglu float32 block_size=1024 (3.109x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
