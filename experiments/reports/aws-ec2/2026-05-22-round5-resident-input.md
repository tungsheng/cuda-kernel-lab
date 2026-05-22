# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round5-resident-input/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round5-resident-input/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round5-resident-input/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round5-resident-input/decode-step.jsonl`

## Environment

- Git commit: `681bd1605a9089994205a2ce54e7a26b2bd4d5ba`
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
| fusion | Fused eager decode step | decode_step decode_step | Replacing decomposed normalization and activation work with fused kernels should reduce kernel count and intermediate memory traffic before graph replay. |
| launch replay | Fused CUDA Graph replay | decode_step decode_step | Combining fused kernels with CUDA Graph replay should reduce both intermediate traffic and per-token launch overhead. |
| launch replay | Fused piecewise CUDA Graph replay | decode_step decode_step | Capturing the static fused pre/post-attention regions while leaving attention eager should keep graph benefits when batch and sequence shapes vary. |
| launch replay | Fused same-stream piecewise CUDA Graph replay | decode_step decode_step | Replaying captured fused pre/post-attention regions on the caller stream should preserve dynamic-shape graph reuse while removing explicit stream handoff cost. |
| launch replay | Naive CUDA Graph replay | decode_step decode_step | Replaying the decomposed decode step inside a CUDA Graph should reduce Python and driver launch overhead without changing the kernels themselves. |

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused eager decode step | 0.3228 | 84.66 | 0.08504 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused CUDA Graph replay | 0.1104 | 247.5 | 0.2486 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused piecewise CUDA Graph replay | 0.2176 | 125.6 | 0.1262 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1679 | 162.8 | 0.1635 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | Naive eager decode step | 0.324 | 84.37 | 0.08474 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | Naive CUDA Graph replay | 0.1211 | 225.7 | 0.2268 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | Fused eager decode step | 0.3666 | 104.5 | 0.2688 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | Fused eager decode step | 0.3661 | 104.7 | 0.2692 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | Fused eager decode step | 0.3601 | 106.4 | 0.2737 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | Fused eager decode step | 0.3872 | 98.98 | 0.2545 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | Fused eager decode step | 0.3646 | 105.1 | 0.2703 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2244 | 170.7 | 0.4391 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2292 | 167.2 | 0.43 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2265 | 169.2 | 0.4351 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2334 | 164.2 | 0.4222 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2328 | 164.6 | 0.4232 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1776 | 215.8 | 0.5549 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1806 | 212.2 | 0.5457 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1786 | 214.6 | 0.5518 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1834 | 209 | 0.5375 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1821 | 210.5 | 0.5412 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.3228 | 0.3896 | 0.4588 | 84.66 | 0.08504 |  | 1.207 noisy |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1104 | 0.114 | 0.1221 | 247.5 | 0.2486 |  | 1.032 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2176 | 0.2498 | 0.251 | 125.6 | 0.1262 |  | 1.148 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1679 | 0.1937 | 0.204 | 162.8 | 0.1635 |  | 1.154 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.324 | 0.3584 | 0.374 | 84.37 | 0.08474 |  | 1.106 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1211 | 0.1366 | 0.1428 | 225.7 | 0.2268 |  | 1.128 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3666 | 0.4085 | 0.4143 | 104.5 | 0.2688 |  | 1.114 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3661 | 0.4039 | 0.4273 | 104.7 | 0.2692 |  | 1.103 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3601 | 0.4019 | 0.4137 | 106.4 | 0.2737 |  | 1.116 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3872 | 0.5494 | 0.5801 | 98.98 | 0.2545 |  | 1.419 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3707 | 0.4086 | 0.4156 | 103.4 | 0.2658 |  | 1.102 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3646 | 0.3947 | 0.4268 | 105.1 | 0.2703 |  | 1.083 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2244 | 0.2811 | 0.3271 | 170.7 | 0.4391 |  | 1.252 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2292 | 0.2879 | 0.3245 | 167.2 | 0.43 |  | 1.256 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2265 | 0.2834 | 0.3245 | 169.2 | 0.4351 |  | 1.251 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2334 | 0.2833 | 0.3286 | 164.2 | 0.4222 |  | 1.213 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2328 | 0.2971 | 0.3264 | 164.6 | 0.4232 |  | 1.276 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2346 | 0.2879 | 0.3271 | 163.3 | 0.42 |  | 1.227 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1776 | 0.2497 | 0.2795 | 215.8 | 0.5549 |  | 1.406 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1832 | 0.2482 | 0.2933 | 213.7 | 0.5717 |  | 1.355 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1803 | 0.2462 | 0.2835 | 215.2 | 0.5765 |  | 1.366 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1819 | 0.2482 | 0.291 | 217 | 0.5844 |  | 1.364 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1806 | 0.2417 | 0.2815 | 212.2 | 0.5457 |  | 1.339 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1786 | 0.2393 | 0.2802 | 214.6 | 0.5518 |  | 1.34 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1834 | 0.2405 | 0.2797 | 209 | 0.5375 |  | 1.311 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1821 | 0.2497 | 0.2812 | 210.5 | 0.5412 |  | 1.371 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1832 | 0.2395 | 0.2819 | 209.2 | 0.538 |  | 1.307 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.1818 | 0.2475 | 0.2893 | 2.324e+04 | 1.75e+04 | 0 | 0.3022 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.1832 | 0.2482 | 0.2933 | 1.355 | 2.3e+04 | 1.612e+04 | 1.06 | 0 | 8 (p95 0.3141 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.1803 | 0.2462 | 0.2835 | 1.366 | 2.317e+04 | 1.624e+04 | 1.11 | 0 | 8 (p95 0.2945 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.1819 | 0.2482 | 0.291 | 1.364 | 2.356e+04 | 2.015e+04 | 0.97 | 0 | 8 (p95 0.2979 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 1 | 10 | 0.3812 | 0.5589 | 0.5658 | 1.466 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3728 | 0.428 | 0.4412 | 1.148 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.3705 | 0.4131 | 0.4176 | 1.115 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.3887 | 0.4127 | 0.4138 | 1.062 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3721 | 0.4116 | 0.4154 | 1.106 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3682 | 0.4026 | 0.4394 | 1.093 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2564 | 0.3288 | 0.3334 | 1.283 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.2655 | 0.3279 | 0.3284 | 1.235 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.2608 | 0.3247 | 0.3287 | 1.245 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2553 | 0.3246 | 0.3262 | 1.271 | 6.875 |

### Host Orchestration

| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | p99 ms | Total ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_copy_host_ms | 500 | 0 | 0 | 0 | 0 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_attention_host_ms | 500 | 0.06787 | 0.08312 | 0.09712 | 34.13 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_post_eager_host_ms | 500 | 0.02397 | 0.02877 | 0.04345 | 12.39 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.00855 | 0.009473 | 0.01104 | 4.406 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | scheduler_decision_host_ms | 500 | 0.00084 | 0.00106 | 0.00119 | 0.4388 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_copy_host_ms | 500 | 0 | 0 | 0 | 0 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_attention_host_ms | 500 | 0.06798 | 0.08554 | 0.09414 | 34.25 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_post_eager_host_ms | 500 | 0.02408 | 0.02818 | 0.04885 | 12.42 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.008575 | 0.009511 | 0.02558 | 4.441 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | scheduler_decision_host_ms | 500 | 0.00088 | 0.00111 | 0.001351 | 0.4611 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_copy_host_ms | 500 | 0 | 0 | 0 | 0 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_attention_host_ms | 500 | 0.06819 | 0.08799 | 0.09679 | 34.35 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_post_eager_host_ms | 500 | 0.02428 | 0.02729 | 0.04334 | 12.45 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.008495 | 0.00923 | 0.01121 | 4.339 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | scheduler_decision_host_ms | 500 | 0.00083 | 0.00097 | 0.0011 | 0.418 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager (1.419 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager (1.406 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager (1.371 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
