# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round4-eager-post/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round4-eager-post/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round4-eager-post/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round4-eager-post/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused eager decode step | 0.325 | 84.1 | 0.08447 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused CUDA Graph replay | 0.1099 | 248.8 | 0.2499 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused piecewise CUDA Graph replay | 0.2143 | 127.5 | 0.1281 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1668 | 163.9 | 0.1646 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | Naive eager decode step | 0.3172 | 86.15 | 0.08654 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | Naive CUDA Graph replay | 0.1212 | 225.5 | 0.2265 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | Fused eager decode step | 0.3683 | 104 | 0.2676 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | Fused eager decode step | 0.3688 | 103.9 | 0.2672 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | Fused eager decode step | 0.3546 | 108.1 | 0.2779 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | Fused eager decode step | 0.3525 | 108.7 | 0.2796 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | Fused eager decode step | 0.371 | 103.3 | 0.2657 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2634 | 145.5 | 0.3742 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2587 | 148.2 | 0.381 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2775 | 138.1 | 0.3551 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.268 | 143 | 0.3678 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2766 | 138.5 | 0.3562 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2085 | 183.8 | 0.4727 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2102 | 182.3 | 0.4687 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2167 | 176.8 | 0.4547 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2156 | 177.7 | 0.457 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2232 | 171.7 | 0.4415 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.325 | 0.3548 | 0.3739 | 84.1 | 0.08447 |  | 1.092 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1099 | 0.1128 | 0.1201 | 248.8 | 0.2499 |  | 1.026 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2143 | 0.2428 | 0.2543 | 127.5 | 0.1281 |  | 1.133 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1668 | 0.192 | 0.2078 | 163.9 | 0.1646 |  | 1.151 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.3172 | 0.3487 | 0.3728 | 86.15 | 0.08654 |  | 1.099 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1212 | 0.1376 | 0.1515 | 225.5 | 0.2265 |  | 1.136 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3683 | 0.3969 | 0.4124 | 104 | 0.2676 |  | 1.078 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3688 | 0.3967 | 0.4134 | 103.9 | 0.2672 |  | 1.076 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3546 | 0.3853 | 0.4054 | 108.1 | 0.2779 |  | 1.087 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3525 | 0.3826 | 0.4232 | 108.7 | 0.2796 |  | 1.085 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.372 | 0.4081 | 0.418 | 103 | 0.2649 |  | 1.097 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.371 | 0.4019 | 0.4135 | 103.3 | 0.2657 |  | 1.084 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2634 | 0.3281 | 0.358 | 145.5 | 0.3742 |  | 1.246 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2587 | 0.3323 | 0.3582 | 148.2 | 0.381 |  | 1.285 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2775 | 0.3313 | 0.3616 | 138.1 | 0.3551 |  | 1.194 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.268 | 0.3241 | 0.357 | 143 | 0.3678 |  | 1.21 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2766 | 0.3417 | 0.3602 | 138.5 | 0.3562 |  | 1.235 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2812 | 0.3333 | 0.3595 | 136.3 | 0.3505 |  | 1.185 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2085 | 0.269 | 0.305 | 183.8 | 0.4727 |  | 1.29 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2118 | 0.2773 | 0.3227 | 184.8 | 0.4945 |  | 1.309 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.211 | 0.2698 | 0.3082 | 183.9 | 0.4927 |  | 1.279 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2097 | 0.2814 | 0.3172 | 188.3 | 0.507 |  | 1.342 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2102 | 0.2791 | 0.3081 | 182.3 | 0.4687 |  | 1.328 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2167 | 0.2792 | 0.3155 | 176.8 | 0.4547 |  | 1.288 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2156 | 0.2714 | 0.3027 | 177.7 | 0.457 |  | 1.259 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2235 | 0.2867 | 0.3107 | 171.5 | 0.4409 |  | 1.283 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only, post=eager | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2232 | 0.2851 | 0.3084 | 171.7 | 0.4415 |  | 1.277 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.2108 | 0.2762 | 0.316 | 2.03e+04 | 1.567e+04 | 0 | 0.3193 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.2118 | 0.2773 | 0.3227 | 1.309 | 2.013e+04 | 1.443e+04 | 1.04 | 0 | 8 (p95 0.3242 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.211 | 0.2698 | 0.3082 | 1.279 | 2.018e+04 | 1.482e+04 | 1.151 | 0 | 8 (p95 0.3134 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.2097 | 0.2814 | 0.3172 | 1.342 | 2.058e+04 | 1.777e+04 | 1.01 | 0 | 8 (p95 0.3202 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.3684 | 0.4233 | 0.4262 | 1.149 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 7 | 11 | 0.3727 | 0.4216 | 0.4362 | 1.131 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3786 | 0.4162 | 0.4243 | 1.099 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3717 | 0.4126 | 0.4392 | 1.11 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3765 | 0.405 | 0.4114 | 1.076 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.371 | 0.4011 | 0.4061 | 1.081 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2999 | 0.3616 | 0.3618 | 1.206 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3068 | 0.3592 | 0.3786 | 1.171 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.3181 | 0.3583 | 0.3584 | 1.126 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2902 | 0.3573 | 0.363 | 1.231 | 6.875 |

### Host Orchestration

| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | p99 ms | Total ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_copy_host_ms | 500 | 0.02484 | 0.03228 | 0.04294 | 12.93 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_x_copy_host_ms | 500 | 0.02484 | 0.03228 | 0.04294 | 12.93 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_attention_host_ms | 500 | 0.0696 | 0.08183 | 0.0963 | 34.89 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_post_eager_host_ms | 500 | 0.02427 | 0.027 | 0.04328 | 12.4 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.0091 | 0.009912 | 0.01753 | 4.686 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | scheduler_decision_host_ms | 500 | 0.000865 | 0.00104 | 0.00134 | 0.4355 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_copy_host_ms | 500 | 0.02498 | 0.03105 | 0.04474 | 12.98 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_x_copy_host_ms | 500 | 0.02498 | 0.03105 | 0.04474 | 12.98 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_attention_host_ms | 500 | 0.06962 | 0.08619 | 0.09672 | 35.03 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_post_eager_host_ms | 500 | 0.02394 | 0.02664 | 0.04405 | 12.31 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.008905 | 0.009852 | 0.02491 | 4.616 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | scheduler_decision_host_ms | 500 | 0.00087 | 0.001151 | 0.001271 | 0.4489 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_copy_host_ms | 500 | 0.02438 | 0.02991 | 0.0432 | 12.68 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_x_copy_host_ms | 500 | 0.02438 | 0.02991 | 0.0432 | 12.68 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_attention_host_ms | 500 | 0.06918 | 0.08683 | 0.09721 | 34.85 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_post_eager_host_ms | 500 | 0.02367 | 0.03193 | 0.04297 | 12.29 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.009435 | 0.01013 | 0.02541 | 4.826 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | scheduler_decision_host_ms | 500 | 0.00086 | 0.00101 | 0.00118 | 0.431 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager (1.342 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only, post=eager (1.328 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only, post=eager (1.309 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
