# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round6-timing-off/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round6-timing-off/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round6-timing-off/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round6-timing-off/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused eager decode step | 0.3275 | 83.45 | 0.08382 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused CUDA Graph replay | 0.1096 | 249.4 | 0.2506 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused piecewise CUDA Graph replay | 0.2141 | 127.7 | 0.1282 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1649 | 165.7 | 0.1664 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | Naive eager decode step | 0.3226 | 84.71 | 0.08509 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | Naive CUDA Graph replay | 0.1211 | 225.7 | 0.2268 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.3497 | 109.6 | 0.2818 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.365 | 105 | 0.27 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.364 | 105.3 | 0.2707 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.3485 | 110 | 0.2828 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.3679 | 104.2 | 0.2679 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.215 | 178.2 | 0.4583 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2267 | 169.1 | 0.4348 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2284 | 167.8 | 0.4315 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2259 | 169.6 | 0.4362 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2293 | 167.2 | 0.4298 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1682 | 227.8 | 0.5858 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1793 | 213.7 | 0.5496 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1795 | 213.5 | 0.5491 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1777 | 215.7 | 0.5547 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1789 | 214.2 | 0.5508 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.3275 | 0.3642 | 0.3683 | 83.45 | 0.08382 |  | 1.112 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1096 | 0.119 | 0.1291 | 249.4 | 0.2506 |  | 1.086 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2141 | 0.2415 | 0.2542 | 127.7 | 0.1282 |  | 1.128 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1649 | 0.1988 | 0.2067 | 165.7 | 0.1664 |  | 1.205 noisy |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.3226 | 0.35 | 0.3581 | 84.71 | 0.08509 |  | 1.085 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1211 | 0.1356 | 0.1414 | 225.7 | 0.2268 |  | 1.12 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3497 | 0.3814 | 0.401 | 109.6 | 0.2818 |  | 1.091 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.365 | 0.3954 | 0.4086 | 105 | 0.27 |  | 1.083 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.364 | 0.4015 | 0.4085 | 105.3 | 0.2707 |  | 1.103 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3485 | 0.3835 | 0.4008 | 110 | 0.2828 |  | 1.1 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3684 | 0.3968 | 0.413 | 104 | 0.2675 |  | 1.077 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.3679 | 0.4032 | 0.4149 | 104.2 | 0.2679 |  | 1.096 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.215 | 0.2755 | 0.3202 | 178.2 | 0.4583 |  | 1.281 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2267 | 0.289 | 0.3259 | 169.1 | 0.4348 |  | 1.275 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2284 | 0.2865 | 0.3271 | 167.8 | 0.4315 |  | 1.255 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2259 | 0.2845 | 0.3236 | 169.6 | 0.4362 |  | 1.259 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2293 | 0.2936 | 0.3275 | 167.2 | 0.4298 |  | 1.28 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2293 | 0.3061 | 0.3259 | 167.1 | 0.4298 |  | 1.335 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1682 | 0.2469 | 0.276 | 227.8 | 0.5858 |  | 1.468 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1771 | 0.2462 | 0.292 | 221 | 0.5913 |  | 1.39 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1777 | 0.245 | 0.2827 | 218.4 | 0.5849 |  | 1.379 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1781 | 0.2501 | 0.284 | 221.7 | 0.5971 |  | 1.404 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1793 | 0.2385 | 0.2802 | 213.7 | 0.5496 |  | 1.33 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1795 | 0.2371 | 0.2791 | 213.5 | 0.5491 |  | 1.321 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1777 | 0.238 | 0.2792 | 215.7 | 0.5547 |  | 1.34 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1789 | 0.2378 | 0.2804 | 214.2 | 0.5508 |  | 1.329 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1836 | 0.2392 | 0.2823 | 208.7 | 0.5367 |  | 1.303 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.1776 | 0.2471 | 0.2862 | 2.364e+04 | 1.752e+04 | 0 | 0.293 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.1771 | 0.2462 | 0.292 | 1.39 | 2.353e+04 | 1.625e+04 | 247.8 | 0 | 8 (p95 0.293 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.1777 | 0.245 | 0.2827 | 1.379 | 2.352e+04 | 1.633e+04 | 246.5 | 0 | 8 (p95 0.2932 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.1781 | 0.2501 | 0.284 | 1.404 | 2.388e+04 | 2e+04 | 251.9 | 0 | 8 (p95 0.2929 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3807 | 0.4123 | 0.4178 | 1.083 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 2 | 19 | 0.3616 | 0.4086 | 0.4095 | 1.13 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.3809 | 0.4076 | 0.4212 | 1.07 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.3659 | 0.4045 | 0.4174 | 1.105 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.3507 | 0.4017 | 0.4176 | 1.145 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.3676 | 0.4008 | 0.4009 | 1.09 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.2539 | 0.3306 | 0.3363 | 1.302 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2538 | 0.3272 | 0.3293 | 1.289 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.2524 | 0.326 | 0.3277 | 1.291 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2539 | 0.3237 | 0.3269 | 1.275 | 6.875 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off (1.468 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off (1.404 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa, copy=resident, post=eager, orchestration_timing=off (1.39 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
