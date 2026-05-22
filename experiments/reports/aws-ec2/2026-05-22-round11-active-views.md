# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round11-active-views/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round11-active-views/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round11-active-views/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round11-active-views/decode-step.jsonl`

## Environment

- Git commit: `5e4d7aa94145427a918621929b9bc757e057c838`
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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused eager decode step | 0.3637 | 75.15 | 0.07549 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused CUDA Graph replay | 0.148 | 184.7 | 0.1855 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused piecewise CUDA Graph replay | 0.1882 | 145.2 | 0.1459 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1374 | 198.9 | 0.1998 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive eager decode step | 0.3528 | 77.47 | 0.07781 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive CUDA Graph replay | 0.1587 | 172.2 | 0.173 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4126 | 92.88 | 0.2388 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4108 | 93.29 | 0.2399 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4072 | 94.12 | 0.242 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4152 | 92.3 | 0.2373 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4106 | 93.33 | 0.24 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.1973 | 194.3 | 0.4996 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2 | 191.6 | 0.4927 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.1991 | 192.5 | 0.495 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2 | 191.6 | 0.4928 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.202 | 189.7 | 0.4878 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1544 | 251.4 | 0.6734 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1559 | 245.9 | 0.6323 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1553 | 246.8 | 0.6347 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1568 | 244.4 | 0.6286 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1577 | 243.1 | 0.625 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.3637 | 0.4049 | 0.4151 | 75.15 | 0.07549 |  | 1.114 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.148 | 0.158 | 0.168 | 184.7 | 0.1855 |  | 1.068 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1882 | 0.2149 | 0.2235 | 145.2 | 0.1459 |  | 1.142 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1374 | 0.1608 | 0.1692 | 198.9 | 0.1998 |  | 1.17 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.3528 | 0.3893 | 0.4175 | 77.47 | 0.07781 |  | 1.104 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1587 | 0.1676 | 0.1774 | 172.2 | 0.173 |  | 1.056 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4126 | 0.5712 | 0.6517 | 92.88 | 0.2388 |  | 1.384 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4108 | 0.5795 | 0.6525 | 93.29 | 0.2399 |  | 1.411 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4072 | 0.5711 | 0.6649 | 94.12 | 0.242 |  | 1.403 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4152 | 0.5718 | 0.652 | 92.3 | 0.2373 |  | 1.377 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4106 | 0.5718 | 0.6502 | 93.33 | 0.24 |  | 1.393 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4111 | 0.573 | 0.6576 | 93.23 | 0.2397 |  | 1.394 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1973 | 0.2638 | 0.3066 | 194.3 | 0.4996 |  | 1.337 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2 | 0.261 | 0.3075 | 191.6 | 0.4927 |  | 1.305 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1991 | 0.2627 | 0.306 | 192.5 | 0.495 |  | 1.32 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2 | 0.2602 | 0.3068 | 191.6 | 0.4928 |  | 1.301 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2023 | 0.2596 | 0.3121 | 189.5 | 0.4872 |  | 1.283 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.202 | 0.2764 | 0.3102 | 189.7 | 0.4878 |  | 1.368 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1567 | 0.2181 | 0.2676 | 244.5 | 0.6287 |  | 1.391 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1584 | 0.232 | 0.28 | 247.1 | 0.6612 |  | 1.464 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1544 | 0.227 | 0.2698 | 251.4 | 0.6734 |  | 1.47 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1581 | 0.2291 | 0.2791 | 249.7 | 0.6723 |  | 1.449 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1559 | 0.2343 | 0.2688 | 245.9 | 0.6323 |  | 1.503 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1553 | 0.2159 | 0.2681 | 246.8 | 0.6347 |  | 1.39 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1568 | 0.221 | 0.2684 | 244.4 | 0.6286 |  | 1.409 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1579 | 0.2164 | 0.2682 | 242.7 | 0.6242 |  | 1.371 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1577 | 0.2207 | 0.2721 | 243.1 | 0.625 |  | 1.4 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.157 | 0.2293 | 0.2763 | 2.695e+04 | 1.89e+04 | 0 | 0.2805 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.1584 | 0.232 | 0.28 | 1.464 | 2.676e+04 | 1.724e+04 | 235.5 | 0 | 8 (p95 0.2827 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.1544 | 0.227 | 0.2698 | 1.47 | 2.689e+04 | 1.762e+04 | 227.9 | 0 | 8 (p95 0.2726 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.1581 | 0.2291 | 0.2791 | 1.449 | 2.721e+04 | 2.182e+04 | 236.3 | 0 | 8 (p95 0.2861 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.4786 | 0.6654 | 0.6759 | 1.39 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.4756 | 0.659 | 0.663 | 1.386 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.4797 | 0.6528 | 0.6593 | 1.361 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.4957 | 0.6524 | 0.6601 | 1.316 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4213 | 0.5974 | 0.6545 | 1.418 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4259 | 0.5799 | 0.6607 | 1.361 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.2449 | 0.3172 | 0.323 | 1.295 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.2337 | 0.3077 | 0.3116 | 1.317 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2355 | 0.307 | 0.3107 | 1.304 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2347 | 0.3061 | 0.3075 | 1.304 | 6.875 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.503 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.47 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.464 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
