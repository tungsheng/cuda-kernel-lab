# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/decode-step.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/memory.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/norms.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/reduction-strategy.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/softmax.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/swiglu.jsonl`
- `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode/vector-add-block-size.jsonl`

## Environment

- Git commit: `fa306ecf6d2abe28239afe1e0bc68bdeb2561f20`
- Git dirty: `False`
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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4842 | 56.45 | 0.0567 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1507 | 181.4 | 0.1822 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.322 | 84.88 | 0.08526 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4737 | 57.7 | 0.05795 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.1618 | 168.9 | 0.1697 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5663 | 67.67 | 0.174 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.4566 | 83.93 | 0.2158 |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1505 | 445.8 | 0 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2888 | 464.8 | 0 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.08704 | 387 | 0.1928 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | PyTorch reference baseline | 0.1485 | 452.9 | 0.113 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | PyTorch reference baseline | 0.1505 | 446.7 | 0.1115 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.1536 | 436.9 | 0.1092 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2939 | 456.7 | 0.05709 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.2212 | 455.1 | 0.07585 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | PyTorch reference baseline | 0.4311 | 467 | 0.03892 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | PyTorch reference baseline | 0.4321 | 465.9 | 0.03882 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.171 | 784.9 | 0.7848 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | Row-wise LayerNorm fusion | 0.3123 | 859.5 | 0.4297 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.17 | 592.2 | 0.4935 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | Row-wise RMSNorm fusion | 0.3113 | 646.7 | 0.2695 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.04915 | 341.3 | 0.4265 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | PyTorch reference baseline | 0.08294 | 404.5 | 0.2527 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.2447 | 411.3 | 0.3428 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | Elementwise SwiGLU fusion | 0.4557 | 441.8 | 0.1841 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4842 | 0.5081 | 0.5167 | 56.45 | 0.0567 |  | 1.049 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1507 | 0.1586 | 0.1758 | 181.4 | 0.1822 |  | 1.052 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise_graph | Fused piecewise CUDA Graph replay | pass | 0.322 | 0.3483 | 0.361 | 84.88 | 0.08526 |  | 1.082 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4737 | 0.4898 | 0.5001 | 57.7 | 0.05795 |  | 1.034 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1618 | 0.1765 | 0.1799 | 168.9 | 0.1697 |  | 1.091 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.5663 | 0.905 | 1.049 | 67.67 | 0.174 |  | 1.598 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | fail | 0.4566 | 1.044 | 1.243 | 83.93 | 0.2158 |  | 2.287 noisy |
| memory | copy | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1505 | 0.1536 | 0.1578 | 445.8 | 0 | 1 | 1.02 |
| memory | copy | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.174 | 0.1833 | 0.1916 | 385.6 | 0 | 0.865 | 1.054 |
| memory | copy | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2888 | 0.2929 | 0.296 | 464.8 | 0 | 1 | 1.014 |
| memory | copy | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3154 | 0.3277 | 0.3359 | 425.6 | 0 | 0.9156 | 1.039 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.08704 | 0.09221 | 0.1045 | 387 | 0.1928 | 1 | 1.059 |
| memory | reduction_sum | float16 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1311 | 0.1556 | 0.166 | 257 | 0.128 | 0.6641 | 1.187 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1485 | 0.1536 | 0.1638 | 452.9 | 0.113 | 1 | 1.034 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=iterative, block_size=1024 | triton | triton-reduction-iterative | Iterative block reduction | pass | 0.1772 | 0.1876 | 0.2007 | 379.6 | 0.09471 | 0.8382 | 1.059 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1505 | 0.1599 | 0.1679 | 446.7 | 0.1115 | 1 | 1.063 |
| memory | reduction_sum | float32 | 16777216 | reduction_strategy=two_pass, block_size=1024 | triton | triton-reduction-two-pass | Two-pass block reduction | pass | 0.1802 | 0.1967 | 0.2007 | 373.1 | 0.09309 | 0.8352 | 1.091 |
| memory | scale | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.1536 | 0.1567 | 0.1588 | 436.9 | 0.1092 | 1 | 1.02 |
| memory | scale | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.1731 | 0.1823 | 0.1946 | 387.8 | 0.09695 | 0.8876 | 1.054 |
| memory | scale | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2939 | 0.297 | 0.3052 | 456.7 | 0.05709 | 1 | 1.011 |
| memory | scale | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.3144 | 0.3247 | 0.3287 | 426.9 | 0.05336 | 0.9348 | 1.033 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2212 | 0.2274 | 0.2294 | 455.1 | 0.07585 | 1 | 1.028 |
| memory | vector_add | float16 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.2457 | 0.2521 | 0.2611 | 409.7 | 0.06828 | 0.9003 | 1.026 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4352 | 0.4373 | 467 | 0.03892 | 1 | 1.01 |
| memory | vector_add | float32 | 16777216 | block_size=1024 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4577 | 0.4783 | 0.4856 | 439.8 | 0.03665 | 0.9418 | 1.045 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4311 | 0.4362 | 0.4393 | 467 | 0.03892 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=2048 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4598 | 0.4741 | 0.4813 | 437.9 | 0.03649 | 0.9376 | 1.031 |
| memory | vector_add | float32 | 16777216 | block_size=512 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4321 | 0.4372 | 0.4383 | 465.9 | 0.03882 | 1 | 1.012 |
| memory | vector_add | float32 | 16777216 | block_size=512 | triton | triton-block-size | Coalesced block-size tuning | pass | 0.4536 | 0.4669 | 0.4823 | 443.8 | 0.03698 | 0.9526 | 1.029 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.2171 | 0.2226 | 0.2356 | 618.3 | 0.6182 | 1 | 1.025 |
| norms | layernorm | float16 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.171 | 0.1782 | 0.1833 | 784.9 | 0.7848 | 1.269 | 1.042 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | torch | torch-baseline | PyTorch reference baseline | pass | 0.4321 | 0.4373 | 0.4424 | 621.2 | 0.3106 | 1 | 1.012 |
| norms | layernorm | float32 | 4096x4096 | eps=1e-05 | triton | triton-fused-layernorm | Row-wise LayerNorm fusion | pass | 0.3123 | 0.3195 | 0.3247 | 859.5 | 0.4297 | 1.384 | 1.023 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 0.9462 | 0.9513 | 0.9585 | 106.4 | 0.08865 | 1 | 1.005 |
| norms | rmsnorm | float16 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.17 | 0.1772 | 0.1864 | 592.2 | 0.4935 | 5.566 | 1.042 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | torch | torch-baseline | PyTorch reference baseline | pass | 1.011 | 1.019 | 1.026 | 199.2 | 0.08299 | 1 | 1.008 |
| norms | rmsnorm | float32 | 4096x4096 | eps=1e-06 | triton | triton-fused-rmsnorm | Row-wise RMSNorm fusion | pass | 0.3113 | 0.3175 | 0.3247 | 646.7 | 0.2695 | 3.247 | 1.02 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.04915 | 0.05233 | 0.0564 | 341.3 | 0.4265 | 1 | 1.065 |
| softmax | softmax | float16 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.06554 | 0.08003 | 0.09729 | 256 | 0.3199 | 0.75 | 1.221 noisy |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | torch | torch-baseline | PyTorch reference baseline | pass | 0.08294 | 0.08709 | 0.0973 | 404.5 | 0.2527 | 1 | 1.05 |
| softmax | softmax | float32 | 4096x1024 | traffic_model=fused | triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | 0.1004 | 0.1064 | 0.1169 | 334.4 | 0.2089 | 0.8265 | 1.061 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 0.7178 | 0.722 | 0.725 | 140.2 | 0.1169 | 1 | 1.006 |
| swiglu | swiglu | float16 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.2447 | 0.2581 | 0.2611 | 411.3 | 0.3428 | 2.933 | 1.055 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | torch | torch-baseline | PyTorch reference baseline | pass | 1.419 | 1.426 | 1.43 | 141.9 | 0.05911 | 1 | 1.005 |
| swiglu | swiglu | float32 | 4096x4096 | block_size=1024 | triton | triton-fused-swiglu | Elementwise SwiGLU fusion | pass | 0.4557 | 0.4691 | 0.4772 | 441.8 | 0.1841 | 3.115 | 1.03 |

## Observation

- Loaded 45 benchmark rows from 8 result files.
- Fastest backend split: dynamic-eager 1, dynamic-piecewise-graph 1, fused 3, naive 2, torch 13, triton 6.
- Correctness summary: fail 1, not checked 1, pass 43.
- Largest Triton wins vs torch: norms rmsnorm float16 eps=1e-06 (5.566x); norms rmsnorm float32 eps=1e-06 (3.247x); swiglu swiglu float32 block_size=1024 (3.115x).
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (2.287 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (1.598 noise); softmax softmax float16 traffic_model=fused (1.221 noise).

## Technique Takeaways

- Fusion techniques produced the strongest Triton wins by removing intermediate traffic or launch overhead: norms rmsnorm float16 eps=1e-06 (5.566x); norms rmsnorm float32 eps=1e-06 (3.247x); swiglu swiglu float32 block_size=1024 (3.115x).
- Launch tuning for simple coalesced memory kernels did not beat PyTorch; compare GB/s and profiler DRAM throughput before adding wider block-size sweeps.
- Reduction-strategy rows separate first-pass streaming bandwidth from end-to-end launch and finalization cost.

## Interpretation

- Triton is strongest where a fused kernel removes framework overhead or intermediate memory traffic.
- Memory primitive baselines still favor PyTorch; profile before adding another broad launch-parameter sweep.
- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
