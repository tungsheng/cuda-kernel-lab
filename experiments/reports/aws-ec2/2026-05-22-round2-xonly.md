# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round2-xonly/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round2-xonly/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round2-xonly/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round2-xonly/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused eager decode step | 0.3446 | 79.31 | 0.07966 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused CUDA Graph replay | 0.1162 | 235.2 | 0.2362 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused piecewise CUDA Graph replay | 0.2826 | 96.72 | 0.09715 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | Fused same-stream piecewise CUDA Graph replay | 0.1854 | 147.4 | 0.1481 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | Naive eager decode step | 0.3403 | 80.32 | 0.08067 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | Naive CUDA Graph replay | 0.1276 | 214.2 | 0.2151 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.374 | 102.5 | 0.2635 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3776 | 101.5 | 0.261 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3819 | 100.4 | 0.2581 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3871 | 99.01 | 0.2546 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-eager | Fused eager decode step | 0.3818 | 100.4 | 0.2581 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3058 | 125.3 | 0.3222 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3198 | 119.8 | 0.3081 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3312 | 115.7 | 0.2976 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3419 | 112.1 | 0.2883 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.3393 | 113 | 0.2905 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2142 | 178.9 | 0.4601 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2213 | 173.2 | 0.4454 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2307 | 166.1 | 0.4273 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2409 | 159.1 | 0.409 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.2412 | 158.9 | 0.4085 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-eager | Fused eager decode step | pass | 0.3446 | 0.3771 | 0.3897 | 79.31 | 0.07966 |  | 1.094 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1162 | 0.1236 | 0.1308 | 235.2 | 0.2362 |  | 1.063 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2826 | 0.3247 | 0.3316 | 96.72 | 0.09715 |  | 1.149 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1854 | 0.2092 | 0.2111 | 147.4 | 0.1481 |  | 1.129 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | naive-eager | Naive eager decode step | pass | 0.3403 | 0.3702 | 0.385 | 80.32 | 0.08067 |  | 1.088 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1276 | 0.1407 | 0.1491 | 214.2 | 0.2151 |  | 1.103 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.374 | 0.4103 | 0.4266 | 102.5 | 0.2635 |  | 1.097 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3776 | 0.4162 | 0.4342 | 101.5 | 0.261 |  | 1.102 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3819 | 0.4197 | 0.4278 | 100.4 | 0.2581 |  | 1.099 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3871 | 0.4197 | 0.43 | 99.01 | 0.2546 |  | 1.084 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3818 | 0.4213 | 0.4303 | 100.4 | 0.2581 |  | 1.103 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3826 | 0.4038 | 0.4162 | 100.2 | 0.2576 |  | 1.055 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3058 | 0.3378 | 0.3576 | 125.3 | 0.3222 |  | 1.104 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3198 | 0.364 | 0.373 | 119.8 | 0.3081 |  | 1.138 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3312 | 0.3688 | 0.3783 | 115.7 | 0.2976 |  | 1.114 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3419 | 0.3824 | 0.4014 | 112.1 | 0.2883 |  | 1.119 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3393 | 0.3742 | 0.3819 | 113 | 0.2905 |  | 1.103 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.3439 | 0.3772 | 0.3868 | 111.4 | 0.2865 |  | 1.097 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2142 | 0.2721 | 0.3125 | 178.9 | 0.4601 |  | 1.27 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2258 | 0.2862 | 0.3338 | 173.3 | 0.4638 |  | 1.267 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2206 | 0.2806 | 0.3187 | 175.9 | 0.4712 |  | 1.272 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2216 | 0.2835 | 0.3257 | 178.2 | 0.4798 |  | 1.279 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2213 | 0.2858 | 0.3127 | 173.2 | 0.4454 |  | 1.292 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2307 | 0.2877 | 0.3183 | 166.1 | 0.4273 |  | 1.247 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2409 | 0.2887 | 0.3333 | 159.1 | 0.409 |  | 1.198 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2412 | 0.2873 | 0.3319 | 158.9 | 0.4085 |  | 1.191 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=x-only | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.2432 | 0.292 | 0.3152 | 157.6 | 0.4053 |  | 1.201 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.2227 | 0.2834 | 0.3261 | 1.937e+04 | 1.529e+04 | 0 | 0.3326 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.2258 | 0.2862 | 0.3338 | 1.267 | 1.911e+04 | 1.398e+04 | 287.3 | 0 | 8 (p95 0.3342 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.2206 | 0.2806 | 0.3187 | 1.272 | 1.935e+04 | 1.425e+04 | 281.7 | 0 | 8 (p95 0.334 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.2216 | 0.2835 | 0.3257 | 1.279 | 1.964e+04 | 1.764e+04 | 284.7 | 0 | 8 (p95 0.3296 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.3916 | 0.4393 | 0.4462 | 1.122 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3899 | 0.4295 | 0.4324 | 1.102 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 4 | 14 | 0.3838 | 0.4259 | 0.4271 | 1.11 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3788 | 0.4241 | 0.4321 | 1.12 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.4002 | 0.4237 | 0.4301 | 1.059 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.376 | 0.4048 | 0.4191 | 1.076 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 4 | 28 | 0.3495 | 0.3858 | 0.3971 | 1.104 | 12.5 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3494 | 0.38 | 0.3837 | 1.088 | 20.93 |
| dynamic-piecewise-graph | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 4 | 28 | 0.3411 | 0.3789 | 0.3818 | 1.111 | 12.5 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3439 | 0.3732 | 0.3767 | 1.085 | 6.875 |

### Host Orchestration

| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | p99 ms | Total ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_copy_host_ms | 500 | 0.02434 | 0.03195 | 0.04381 | 12.79 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_x_copy_host_ms | 500 | 0.02434 | 0.03195 | 0.04381 | 12.79 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_attention_host_ms | 500 | 0.09356 | 0.1139 | 0.1202 | 47.4 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.00915 | 0.01041 | 0.02618 | 4.734 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.00829 | 0.009762 | 0.01254 | 4.261 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_copy_host_ms | 500 | 0.02345 | 0.02696 | 0.04301 | 12.13 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_x_copy_host_ms | 500 | 0.02345 | 0.02696 | 0.04301 | 12.13 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_attention_host_ms | 500 | 0.09275 | 0.1134 | 0.1161 | 47.07 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.0091 | 0.01005 | 0.01176 | 4.63 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.00835 | 0.009615 | 0.02511 | 4.319 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_copy_host_ms | 500 | 0.02377 | 0.03241 | 0.04257 | 12.47 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_x_copy_host_ms | 500 | 0.02377 | 0.03241 | 0.04257 | 12.47 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_attention_host_ms | 500 | 0.09253 | 0.1129 | 0.1179 | 46.83 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.00911 | 0.01015 | 0.01182 | 4.653 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.008345 | 0.009293 | 0.01009 | 4.234 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=x-only (1.292 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only (1.279 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=x-only (1.272 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
