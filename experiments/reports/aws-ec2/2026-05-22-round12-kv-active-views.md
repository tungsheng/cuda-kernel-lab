# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-round12-kv-active-views/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round12-kv-active-views/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round12-kv-active-views/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-round12-kv-active-views/decode-step.jsonl`

## Environment

- Git commit: `125559e1c053d27dceec53aaf66d727ff0640929`
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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused eager decode step | 0.3666 | 74.54 | 0.07488 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused CUDA Graph replay | 0.1482 | 184.4 | 0.1852 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused piecewise CUDA Graph replay | 0.187 | 146.2 | 0.1468 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | Fused same-stream piecewise CUDA Graph replay | 0.1375 | 198.7 | 0.1996 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive eager decode step | 0.3395 | 80.5 | 0.08086 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | Naive CUDA Graph replay | 0.1594 | 171.4 | 0.1722 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.399 | 96.04 | 0.247 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4071 | 94.14 | 0.2421 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.415 | 92.36 | 0.2375 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4184 | 91.59 | 0.2355 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | Fused eager decode step | 0.4136 | 92.65 | 0.2383 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.1955 | 196 | 0.5041 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.1962 | 195.3 | 0.5022 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.2005 | 191.2 | 0.4916 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.198 | 193.5 | 0.4976 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.1968 | 194.7 | 0.5007 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1549 | 252.6 | 0.676 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1537 | 249.3 | 0.641 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1587 | 241.5 | 0.6209 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1579 | 242.7 | 0.6241 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.1554 | 246.6 | 0.6341 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-eager | Fused eager decode step | pass | 0.3666 | 0.401 | 0.4104 | 74.54 | 0.07488 |  | 1.094 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1482 | 0.1589 | 0.1663 | 184.4 | 0.1852 |  | 1.072 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.187 | 0.2168 | 0.2265 | 146.2 | 0.1468 |  | 1.16 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1375 | 0.1681 | 0.1731 | 198.7 | 0.1996 |  | 1.222 noisy |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-eager | Naive eager decode step | pass | 0.3395 | 0.3668 | 0.394 | 80.5 | 0.08086 |  | 1.08 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096, attention=sdpa-head-major, post=eager | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1594 | 0.1679 | 0.1765 | 171.4 | 0.1722 |  | 1.053 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.399 | 0.5728 | 0.6538 | 96.04 | 0.247 |  | 1.436 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4071 | 0.5735 | 0.6532 | 94.14 | 0.2421 |  | 1.409 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.415 | 0.5781 | 0.6548 | 92.36 | 0.2375 |  | 1.393 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4184 | 0.5738 | 0.6543 | 91.59 | 0.2355 |  | 1.371 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4136 | 0.581 | 0.6549 | 92.65 | 0.2383 |  | 1.405 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.4138 | 0.5723 | 0.6552 | 92.61 | 0.2382 |  | 1.383 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1955 | 0.2629 | 0.3086 | 196 | 0.5041 |  | 1.345 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1962 | 0.2604 | 0.3072 | 195.3 | 0.5022 |  | 1.327 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.2005 | 0.2669 | 0.3128 | 191.2 | 0.4916 |  | 1.331 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.198 | 0.2593 | 0.3029 | 193.5 | 0.4976 |  | 1.309 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1977 | 0.2609 | 0.3099 | 193.9 | 0.4985 |  | 1.319 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.1968 | 0.2608 | 0.3062 | 194.7 | 0.5007 |  | 1.325 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1563 | 0.2187 | 0.2689 | 245.1 | 0.6304 |  | 1.399 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1549 | 0.2318 | 0.2841 | 252.6 | 0.676 |  | 1.497 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1583 | 0.2287 | 0.2685 | 245.1 | 0.6566 |  | 1.445 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1565 | 0.2282 | 0.2799 | 252.3 | 0.6793 |  | 1.458 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1537 | 0.2309 | 0.2684 | 249.3 | 0.641 |  | 1.502 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1587 | 0.2303 | 0.2691 | 241.5 | 0.6209 |  | 1.451 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1579 | 0.2312 | 0.267 | 242.7 | 0.6241 |  | 1.464 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1556 | 0.2168 | 0.2689 | 246.3 | 0.6333 |  | 1.393 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.1554 | 0.2199 | 0.2688 | 246.6 | 0.6341 |  | 1.415 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.1566 | 0.2296 | 0.2775 | 2.705e+04 | 1.888e+04 | 0 | 0.2803 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.1549 | 0.2318 | 0.2841 | 1.497 | 2.688e+04 | 1.725e+04 | 232.9 | 0 | 8 (p95 0.2848 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.1583 | 0.2287 | 0.2685 | 1.445 | 2.689e+04 | 1.749e+04 | 229.7 | 0 | 8 (p95 0.2742 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.1565 | 0.2282 | 0.2799 | 1.458 | 2.736e+04 | 2.191e+04 | 229.2 | 0 | 8 (p95 0.2817 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.4694 | 0.6619 | 0.6663 | 1.41 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.486 | 0.656 | 0.6884 | 1.35 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.484 | 0.655 | 0.6598 | 1.353 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.4804 | 0.6534 | 0.6564 | 1.36 | 0 |
| dynamic-eager | `decode-step-dynamic.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4331 | 0.6166 | 0.6668 | 1.424 | 0 |
| dynamic-eager | `decode-step-dynamic-buckets.jsonl` | 1,2,4,8 | 0 | 8 | 43 | 0.4194 | 0.6028 | 0.6563 | 1.437 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.2385 | 0.3166 | 0.3209 | 1.327 | 0 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.2331 | 0.3135 | 0.3268 | 1.345 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.2342 | 0.3072 | 0.3076 | 1.312 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.2379 | 0.3031 | 0.3066 | 1.274 | 6.875 |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.502 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.497 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8, attention=sdpa-head-major, copy=resident, post=eager, orchestration_timing=off (1.464 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
