# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round8-skip-resident-copy/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round8-skip-resident-copy/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round8-skip-resident-copy/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round8-skip-resident-copy/decode-step.jsonl`

## Environment

- Git commit: `d79e3e90a2344247f29aa8c2808cc86d31b7bf78`
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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused eager decode step | 0.3608 | 75.74 | 0.07608 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused CUDA Graph replay | 0.1475 | 185.3 | 0.1861 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused piecewise CUDA Graph replay | 0.2096 | 130.4 | 0.131 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1605 | 170.3 | 0.171 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive eager decode step | 0.3395 | 80.5 | 0.08086 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive CUDA Graph replay | 0.159 | 171.9 | 0.1726 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.414 | 92.57 | 0.238 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4144 | 92.49 | 0.2378 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4138 | 92.61 | 0.2382 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4096 | 93.55 | 0.2406 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.416 | 92.12 | 0.2369 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2201 | 174.1 | 0.4478 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2209 | 173.5 | 0.446 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2192 | 174.8 | 0.4496 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2217 | 172.9 | 0.4445 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2117 | 181.1 | 0.4656 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1711 | 228.8 | 0.6122 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1714 | 223.6 | 0.5749 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.173 | 221.5 | 0.5696 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1733 | 221.1 | 0.5685 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1716 | 223.3 | 0.5741 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.3608 | 0.3978 | 0.4052 | 75.74 | 0.07608 |  | 1.103 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1475 | 0.1563 | 0.1656 | 185.3 | 0.1861 |  | 1.06 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2096 | 0.2362 | 0.2442 | 130.4 | 0.131 |  | 1.127 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1605 | 0.1909 | 0.1938 | 170.3 | 0.171 |  | 1.189 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.3395 | 0.3643 | 0.3703 | 80.5 | 0.08086 |  | 1.073 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.159 | 0.1722 | 0.1769 | 171.9 | 0.1726 |  | 1.083 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.414 | 0.5736 | 0.652 | 92.57 | 0.238 |  | 1.386 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4144 | 0.5754 | 0.6562 | 92.49 | 0.2378 |  | 1.389 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4138 | 0.5723 | 0.6531 | 92.61 | 0.2382 |  | 1.383 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4096 | 0.572 | 0.6528 | 93.55 | 0.2406 |  | 1.396 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.416 | 0.569 | 0.6527 | 92.12 | 0.2369 |  | 1.368 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4182 | 0.5743 | 0.6572 | 91.65 | 0.2357 |  | 1.373 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2201 | 0.2679 | 0.3383 | 174.1 | 0.4478 |  | 1.217 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2209 | 0.275 | 0.3199 | 173.5 | 0.446 |  | 1.245 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2192 | 0.2692 | 0.3165 | 174.8 | 0.4496 |  | 1.228 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2217 | 0.2695 | 0.3167 | 172.9 | 0.4445 |  | 1.215 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2117 | 0.2678 | 0.3131 | 181.1 | 0.4656 |  | 1.265 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2178 | 0.2682 | 0.3152 | 175.9 | 0.4525 |  | 1.231 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1723 | 0.2215 | 0.2709 | 222.4 | 0.572 |  | 1.286 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1711 | 0.2411 | 0.2824 | 228.8 | 0.6122 |  | 1.41 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1723 | 0.2301 | 0.2715 | 225.2 | 0.6033 |  | 1.336 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1732 | 0.2333 | 0.2819 | 228 | 0.614 |  | 1.348 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1714 | 0.2255 | 0.273 | 223.6 | 0.5749 |  | 1.316 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.173 | 0.2249 | 0.2711 | 221.5 | 0.5696 |  | 1.3 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1733 | 0.2371 | 0.2714 | 221.1 | 0.5685 |  | 1.368 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1742 | 0.2387 | 0.2777 | 220 | 0.5658 |  | 1.37 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1716 | 0.2258 | 0.2691 | 223.3 | 0.5741 |  | 1.315 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.1722 | 0.2349 | 0.2786 | 2.453e+04 | 1.847e+04 | 0 | 0.2824 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.1711 | 0.2411 | 0.2824 | 1.41 | 2.438e+04 | 1.659e+04 | 242.1 | 0 | 8 (p95 0.2876 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.1723 | 0.2301 | 0.2715 | 1.336 | 2.451e+04 | 1.738e+04 | 231.1 | 0 | 8 (p95 0.2749 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.1732 | 0.2333 | 0.2819 | 1.348 | 2.47e+04 | 2.143e+04 | 234.4 | 0 | 8 (p95 0.2846 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.4843 | 0.6565 | 0.6611 | 1.355 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.4736 | 0.6547 | 0.6561 | 1.382 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.4892 | 0.6533 | 0.657 | 1.336 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.485 | 0.653 | 0.6562 | 1.347 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4291 | 0.6173 | 0.6713 | 1.438 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4426 | 0.5805 | 0.6555 | 1.312 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.2535 | 0.3393 | 0.3399 | 1.339 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.2503 | 0.32 | 0.3209 | 1.279 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2486 | 0.3167 | 0.3171 | 1.274 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2454 | 0.3165 | 0.317 | 1.29 | 6.875 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.41 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.396 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.389 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
