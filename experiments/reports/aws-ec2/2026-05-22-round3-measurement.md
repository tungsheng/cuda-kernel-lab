# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round3-measurement/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round3-measurement/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round3-measurement/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round3-measurement/decode-step.jsonl`

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
| fusion | Fused eager decode step | decode_step decode_step | Replacing decomposed normalization and activation work with fused kernels should reduce kernel count and intermediate memory traffic before graph replay. |
| launch replay | Fused CUDA Graph replay | decode_step decode_step | Combining fused kernels with CUDA Graph replay should reduce both intermediate traffic and per-token launch overhead. |
| launch replay | Fused piecewise CUDA Graph replay | decode_step decode_step | Capturing the static fused pre/post-attention regions while leaving attention eager should keep graph benefits when batch and sequence shapes vary. |
| launch replay | Fused same-stream piecewise CUDA Graph replay | decode_step decode_step | Replaying captured fused pre/post-attention regions on the caller stream should preserve dynamic-shape graph reuse while removing explicit stream handoff cost. |
| launch replay | Naive CUDA Graph replay | decode_step decode_step | Replaying the decomposed decode step inside a CUDA Graph should reduce Python and driver launch overhead without changing the kernels themselves. |

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused eager decode step | 0.3267 | 83.64 | 0.08402 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused CUDA Graph replay | 0.1104 | 247.5 | 0.2486 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused piecewise CUDA Graph replay | 0.2791 | 97.91 | 0.09834 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused same-stream piecewise CUDA Graph replay | 0.1768 | 154.6 | 0.1553 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | Naive eager decode step | 0.3207 | 85.23 | 0.08561 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | Naive CUDA Graph replay | 0.1219 | 224.2 | 0.2252 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3685 | 104 | 0.2674 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3623 | 105.8 | 0.272 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3686 | 104 | 0.2674 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3677 | 104.2 | 0.2681 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3656 | 104.8 | 0.2695 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3124 | 122.7 | 0.3155 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3132 | 122.4 | 0.3147 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3277 | 117 | 0.3008 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3319 | 115.5 | 0.2969 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3343 | 114.7 | 0.2948 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2088 | 183.5 | 0.4719 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2124 | 180.5 | 0.4641 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.217 | 176.6 | 0.4542 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2355 | 162.7 | 0.4185 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.234 | 163.8 | 0.4212 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-eager | Fused eager decode step | pass | 0.3267 | 0.3552 | 0.3764 | 83.64 | 0.08402 |  | 1.087 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1104 | 0.1169 | 0.1232 | 247.5 | 0.2486 |  | 1.059 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2791 | 0.2989 | 0.3149 | 97.91 | 0.09834 |  | 1.071 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1768 | 0.1991 | 0.2131 | 154.6 | 0.1553 |  | 1.126 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | naive-eager | Naive eager decode step | pass | 0.3207 | 0.3596 | 0.3632 | 85.23 | 0.08561 |  | 1.121 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1219 | 0.1302 | 0.1403 | 224.2 | 0.2252 |  | 1.068 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3685 | 0.4085 | 0.4359 | 104 | 0.2674 |  | 1.108 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3623 | 0.4008 | 0.4087 | 105.8 | 0.272 |  | 1.106 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3686 | 0.4013 | 0.4178 | 104 | 0.2674 |  | 1.089 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3677 | 0.4118 | 0.4164 | 104.2 | 0.2681 |  | 1.12 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3737 | 0.4051 | 0.4202 | 102.5 | 0.2637 |  | 1.084 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3656 | 0.3994 | 0.4231 | 104.8 | 0.2695 |  | 1.092 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3124 | 0.3435 | 0.3538 | 122.7 | 0.3155 |  | 1.1 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3132 | 0.3619 | 0.3757 | 122.4 | 0.3147 |  | 1.155 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3277 | 0.3643 | 0.3675 | 117 | 0.3008 |  | 1.112 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3319 | 0.3699 | 0.383 | 115.5 | 0.2969 |  | 1.114 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3343 | 0.3685 | 0.386 | 114.7 | 0.2948 |  | 1.102 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3374 | 0.369 | 0.3812 | 113.6 | 0.2921 |  | 1.094 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2088 | 0.2664 | 0.3055 | 183.5 | 0.4719 |  | 1.276 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.215 | 0.2765 | 0.3162 | 182.1 | 0.4872 |  | 1.286 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2123 | 0.2727 | 0.3087 | 182.8 | 0.4895 |  | 1.284 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2182 | 0.2771 | 0.3122 | 180.9 | 0.4873 |  | 1.27 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2124 | 0.2848 | 0.3106 | 180.5 | 0.4641 |  | 1.341 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.217 | 0.2814 | 0.3082 | 176.6 | 0.4542 |  | 1.297 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2355 | 0.2828 | 0.3092 | 162.7 | 0.4185 |  | 1.201 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2388 | 0.2846 | 0.313 | 160.5 | 0.4127 |  | 1.192 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.234 | 0.2799 | 0.3077 | 163.8 | 0.4212 |  | 1.196 |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.2152 | 0.2754 | 0.3124 | 2.007e+04 | 1.573e+04 | 0 | 0.3174 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.215 | 0.2765 | 0.3162 | 1.286 | 2.002e+04 | 1.447e+04 | 1.051 | 0 | 8 (p95 0.3197 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.2123 | 0.2727 | 0.3087 | 1.284 | 2.005e+04 | 1.467e+04 | 1.01 | 0 | 8 (p95 0.3128 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.2182 | 0.2771 | 0.3122 | 1.27 | 2.015e+04 | 1.804e+04 | 1.35 | 0 | 8 (p95 0.3196 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.3871 | 0.4253 | 0.4336 | 1.099 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.366 | 0.4205 | 0.428 | 1.149 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.3804 | 0.4147 | 0.4155 | 1.09 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3755 | 0.4126 | 0.4361 | 1.099 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.3841 | 0.4105 | 0.4181 | 1.069 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3726 | 0.4025 | 0.4167 | 1.08 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 6 | 23 | 0.3421 | 0.3786 | 0.3842 | 1.107 | 10.87 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3484 | 0.3764 | 0.391 | 1.08 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3446 | 0.3738 | 0.3874 | 1.085 | 20.93 |
| dynamic-piecewise-graph | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3458 | 0.3725 | 0.3841 | 1.077 | 20.93 |

### Host Orchestration

| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | p99 ms | Total ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_copy_host_ms | 500 | 0.023 | 0.02983 | 0.04176 | 12.02 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_x_copy_host_ms | 500 | 0.023 | 0.02983 | 0.04176 | 12.02 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_attention_host_ms | 500 | 0.09409 | 0.115 | 0.1183 | 47.77 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.008631 | 0.009721 | 0.02573 | 4.5 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.00753 | 0.00877 | 0.01054 | 3.902 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | scheduler_decision_host_ms | 500 | 0.0008605 | 0.001051 | 0.0012 | 0.4343 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_copy_host_ms | 500 | 0.02309 | 0.02847 | 0.04138 | 11.96 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_x_copy_host_ms | 500 | 0.02309 | 0.02847 | 0.04138 | 11.96 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_attention_host_ms | 500 | 0.09321 | 0.1136 | 0.1151 | 47.28 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.008695 | 0.009652 | 0.01243 | 4.455 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.007755 | 0.008745 | 0.0101 | 3.969 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | scheduler_decision_host_ms | 500 | 0.00086 | 0.00101 | 0.00108 | 0.4311 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_copy_host_ms | 500 | 0.0234 | 0.02964 | 0.04077 | 12.14 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_x_copy_host_ms | 500 | 0.0234 | 0.02964 | 0.04077 | 12.14 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_attention_host_ms | 500 | 0.09378 | 0.1146 | 0.1166 | 47.6 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.00922 | 0.01037 | 0.02702 | 4.782 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.00781 | 0.00887 | 0.01063 | 3.986 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | scheduler_decision_host_ms | 500 | 0.0009 | 0.00135 | 0.00144 | 0.4864 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only (1.341 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only (1.297 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only (1.286 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
