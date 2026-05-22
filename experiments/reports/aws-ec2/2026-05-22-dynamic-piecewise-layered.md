# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/decode-step.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-layered/vector-add-block-size.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4784 | 57.13 | 0.05739 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1525 | 179.2 | 0.18 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.418 | 65.38 | 0.06567 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4556 | 59.99 | 0.06026 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.164 | 166.7 | 0.1674 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5629 | 68.08 | 0.1751 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5471 | 70.05 | 0.1801 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1495 | 448.9 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2888 | 464.8 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08499 | 396.3 | 0.1974 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.8 | 0.113 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1516 | 442.8 | 0.1107 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2934 | 457.5 | 0.05719 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2202 | 457.2 | 0.0762 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4306 | 467.6 | 0.03896 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.171 | 784.9 | 0.7848 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3133 | 856.7 | 0.4283 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.171 | 588.7 | 0.4906 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3113 | 646.7 | 0.2695 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.05018 | 334.4 | 0.4178 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.0809 | 414.8 | 0.2591 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2427 | 414.8 | 0.3457 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4577 | 439.8 | 0.1833 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4784 | 0.5012 | 0.5051 | 57.13 | 0.05739 |  | 1.048 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1525 | 0.1598 | 0.172 | 179.2 | 0.18 |  | 1.047 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise_graph | Fused piecewise CUDA Graph replay | pass | 0.418 | 0.4353 | 0.4438 | 65.38 | 0.06567 |  | 1.041 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4556 | 0.4751 | 0.5097 | 59.99 | 0.06026 |  | 1.043 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.164 | 0.1771 | 0.1795 | 166.7 | 0.1674 |  | 1.08 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.5629 | 0.9098 | 1.049 | 68.08 | 0.1751 |  | 1.616 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | fail | 0.5471 | 1.054 | 1.253 | 70.05 | 0.1801 |  | 1.926 noisy |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1495 | 0.1517 | 0.1649 | 448.9 | 0 | 1 | 1.015 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.172 | 0.1793 | 0.1852 | 390.1 | 0 | 0.869 | 1.042 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2888 | 0.2939 | 0.2949 | 464.8 | 0 | 1 | 1.018 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3216 | 0.3308 | 426.9 | 0 | 0.9186 | 1.023 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08499 | 0.08712 | 0.09319 | 396.3 | 0.1974 | 1 | 1.025 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.129 | 0.1548 | 0.1773 | 261.1 | 0.13 | 0.6587 | 1.2 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1597 | 0.1669 | 452.9 | 0.113 | 1 | 1.075 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1761 | 0.1805 | 0.1949 | 381.8 | 0.09526 | 0.843 | 1.025 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1557 | 0.1649 | 452.8 | 0.113 | 1 | 1.049 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1792 | 0.1917 | 0.209 | 375.2 | 0.09362 | 0.8287 | 1.07 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1516 | 0.1538 | 0.1628 | 442.8 | 0.1107 | 1 | 1.015 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.172 | 0.1782 | 0.1915 | 390.1 | 0.09752 | 0.881 | 1.036 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2934 | 0.299 | 0.3042 | 457.5 | 0.05719 | 1 | 1.019 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3197 | 0.3277 | 426.9 | 0.05337 | 0.9332 | 1.017 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2202 | 0.2252 | 0.2304 | 457.2 | 0.0762 | 1 | 1.023 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2437 | 0.2591 | 0.2663 | 413 | 0.06884 | 0.9033 | 1.063 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4362 | 0.4374 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4577 | 0.47 | 0.4782 | 439.8 | 0.03665 | 0.9418 | 1.027 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4362 | 0.4373 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4588 | 0.4721 | 0.4742 | 438.9 | 0.03657 | 0.9397 | 1.029 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4306 | 0.4372 | 0.4383 | 467.6 | 0.03896 | 1 | 1.015 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4516 | 0.4679 | 0.4721 | 445.8 | 0.03715 | 0.9535 | 1.036 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2191 | 0.2275 | 0.2365 | 612.5 | 0.6124 | 1 | 1.038 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.171 | 0.1762 | 0.1792 | 784.9 | 0.7848 | 1.281 | 1.03 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4352 | 0.4424 | 0.4444 | 616.8 | 0.3084 | 1 | 1.016 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3133 | 0.3237 | 0.3369 | 856.7 | 0.4283 | 1.389 | 1.033 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9477 | 0.9534 | 0.9626 | 106.2 | 0.08851 | 1 | 1.006 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.171 | 0.1752 | 0.1833 | 588.7 | 0.4906 | 5.542 | 1.024 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.014 | 1.019 | 1.02 | 198.6 | 0.08275 | 1 | 1.005 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3113 | 0.3258 | 0.3359 | 646.7 | 0.2695 | 3.256 | 1.047 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.05018 | 0.05222 | 0.06048 | 334.4 | 0.4178 | 1 | 1.041 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06656 | 0.07895 | 0.08604 | 252.1 | 0.315 | 0.7538 | 1.186 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.0809 | 0.08504 | 0.09219 | 414.8 | 0.2591 | 1 | 1.051 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.09928 | 0.1086 | 0.1199 | 338 | 0.2112 | 0.8148 | 1.094 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7158 | 0.7219 | 0.7301 | 140.6 | 0.1172 | 1 | 1.009 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2427 | 0.2541 | 0.2591 | 414.8 | 0.3457 | 2.949 | 1.047 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.42 | 1.425 | 1.427 | 141.8 | 0.05908 | 1 | 1.004 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4577 | 0.4742 | 0.4793 | 439.8 | 0.1833 | 3.102 | 1.036 |

## Observation

- Loaded 45 benchmark rows from 8 result files.
- Fastest backend split: dynamic-eager 1, dynamic-piecewise-graph 1, fused 3, naive 2, torch 13, triton 6.
- Correctness summary: fail 1, not checked 1, pass 43.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.542x); norms rmsnorm float32 eps=1e-06 (3.256x); swiglu swiglu float32 block_size=1024 (3.102x).
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.926 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.616 noise).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.542x); norms rmsnorm float32 eps=1e-06 (3.256x); swiglu swiglu float32 block_size=1024 (3.102x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
