# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round7-head-major-v2/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round7-head-major-v2/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round7-head-major-v2/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round7-head-major-v2/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused eager decode step | 0.36 | 75.92 | 0.07626 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused CUDA Graph replay | 0.1476 | 185.2 | 0.186 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused piecewise CUDA Graph replay | 0.2093 | 130.6 | 0.1312 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1606 | 170.2 | 0.171 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive eager decode step | 0.3569 | 76.58 | 0.07692 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive CUDA Graph replay | 0.158 | 173 | 0.1737 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4095 | 93.59 | 0.2407 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4133 | 92.73 | 0.2384 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4055 | 94.52 | 0.2431 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.413 | 92.79 | 0.2386 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4164 | 92.03 | 0.2367 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.222 | 172.6 | 0.4439 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2196 | 174.5 | 0.4488 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2169 | 176.7 | 0.4543 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2221 | 172.6 | 0.4437 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2198 | 174.3 | 0.4483 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1706 | 231.4 | 0.6232 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1722 | 222.5 | 0.5722 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1715 | 223.5 | 0.5748 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1754 | 218.5 | 0.5618 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1732 | 221.3 | 0.569 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.36 | 0.3946 | 0.3961 | 75.92 | 0.07626 |  | 1.096 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1476 | 0.1549 | 0.1669 | 185.2 | 0.186 |  | 1.049 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2093 | 0.2349 | 0.2396 | 130.6 | 0.1312 |  | 1.123 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1606 | 0.1893 | 0.1972 | 170.2 | 0.171 |  | 1.179 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.3569 | 0.3911 | 0.4231 | 76.58 | 0.07692 |  | 1.096 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.158 | 0.1651 | 0.1745 | 173 | 0.1737 |  | 1.045 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4095 | 0.5746 | 0.6521 | 93.59 | 0.2407 |  | 1.403 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4133 | 0.5737 | 0.6567 | 92.73 | 0.2384 |  | 1.388 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4055 | 0.5739 | 0.6527 | 94.52 | 0.2431 |  | 1.415 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.413 | 0.5786 | 0.6565 | 92.79 | 0.2386 |  | 1.401 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4164 | 0.5714 | 0.655 | 92.03 | 0.2367 |  | 1.372 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4187 | 0.5793 | 0.6786 | 91.53 | 0.2354 |  | 1.384 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.222 | 0.2809 | 0.3169 | 172.6 | 0.4439 |  | 1.265 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2196 | 0.2759 | 0.3153 | 174.5 | 0.4488 |  | 1.257 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2169 | 0.2685 | 0.3165 | 176.7 | 0.4543 |  | 1.238 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2221 | 0.2686 | 0.3145 | 172.6 | 0.4437 |  | 1.209 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2205 | 0.2867 | 0.3183 | 173.8 | 0.4469 |  | 1.3 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2198 | 0.2732 | 0.3197 | 174.3 | 0.4483 |  | 1.243 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1726 | 0.2252 | 0.2708 | 222.1 | 0.5711 |  | 1.305 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1784 | 0.2394 | 0.2842 | 219.4 | 0.587 |  | 1.342 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1732 | 0.2301 | 0.2813 | 224.1 | 0.6002 |  | 1.329 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1706 | 0.2385 | 0.2794 | 231.4 | 0.6232 |  | 1.398 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1722 | 0.2258 | 0.2751 | 222.5 | 0.5722 |  | 1.311 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1715 | 0.2295 | 0.2706 | 223.5 | 0.5748 |  | 1.339 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1754 | 0.2337 | 0.2767 | 218.5 | 0.5618 |  | 1.333 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1757 | 0.2311 | 0.2755 | 218.1 | 0.5608 |  | 1.315 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1732 | 0.2291 | 0.2713 | 221.3 | 0.569 |  | 1.323 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.1741 | 0.236 | 0.2816 | 2.427e+04 | 1.835e+04 | 0 | 0.2873 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.1784 | 0.2394 | 0.2842 | 1.342 | 2.368e+04 | 1.671e+04 | 241.4 | 0 | 8 (p95 0.2881 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.1732 | 0.2301 | 0.2813 | 1.329 | 2.421e+04 | 1.738e+04 | 231.1 | 0 | 8 (p95 0.2901 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.1706 | 0.2385 | 0.2794 | 1.398 | 2.49e+04 | 2.096e+04 | 239.5 | 0 | 8 (p95 0.2838 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.4958 | 0.6647 | 0.6715 | 1.341 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.4781 | 0.6573 | 0.6725 | 1.375 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.4799 | 0.6571 | 0.6633 | 1.369 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.4862 | 0.6538 | 0.6747 | 1.345 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4281 | 0.6181 | 0.6579 | 1.444 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4362 | 0.6086 | 0.6806 | 1.395 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.2534 | 0.3175 | 0.3179 | 1.253 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2427 | 0.3172 | 0.3301 | 1.307 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.2579 | 0.3154 | 0.316 | 1.223 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2438 | 0.3146 | 0.3164 | 1.291 | 6.875 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.415 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.403 noise); decode_step decode_step float16 mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.401 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
