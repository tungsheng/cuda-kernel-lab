# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-tail-policy-live/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-policy-live/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-policy-live/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-policy-live/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4816 | 56.74 | 0.057 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1519 | 179.9 | 0.1807 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.4242 | 64.43 | 0.06472 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused same-stream piecewise CUDA Graph replay | 0.3298 | 82.86 | 0.08323 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4747 | 57.57 | 0.05783 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.1627 | 168 | 0.1688 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-eager | Fused eager decode step | 0.5748 | 66.67 | 0.1714 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-eager | Fused eager decode step | 0.5716 | 67.05 | 0.1724 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | Fused eager decode step | 0.5773 | 66.39 | 0.1707 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | Fused eager decode step | 0.5551 | 69.03 | 0.1775 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5757 | 66.56 | 0.1712 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5576 | 68.73 | 0.1767 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5596 | 68.48 | 0.1761 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5685 | 67.42 | 0.1734 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.582 | 65.85 | 0.1693 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.585 | 65.51 | 0.1685 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4726 | 82.82 | 0.2216 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.464 | 82.59 | 0.2124 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4752 | 80.64 | 0.2074 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4938 | 77.61 | 0.1996 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4878 | 78.56 | 0.202 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4816 | 0.5089 | 0.5158 | 56.74 | 0.057 |  | 1.057 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1519 | 0.1672 | 0.1727 | 179.9 | 0.1807 |  | 1.1 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.4242 | 0.4527 | 0.465 | 64.43 | 0.06472 |  | 1.067 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.3298 | 0.3494 | 0.3669 | 82.86 | 0.08323 |  | 1.059 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4747 | 0.5027 | 0.5124 | 57.57 | 0.05783 |  | 1.059 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1627 | 0.1796 | 0.1866 | 168 | 0.1688 |  | 1.104 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5748 | 0.902 | 1.062 | 66.67 | 0.1714 |  | 1.569 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5716 | 0.9131 | 1.064 | 67.05 | 0.1724 |  | 1.597 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5773 | 0.9202 | 1.059 | 66.39 | 0.1707 |  | 1.594 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5551 | 0.8959 | 1.044 | 69.03 | 0.1775 |  | 1.614 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5757 | 0.9065 | 1.075 | 66.56 | 0.1712 |  | 1.575 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5769 | 0.906 | 1.049 | 66.43 | 0.1708 |  | 1.571 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5576 | 1.037 | 1.233 | 68.73 | 0.1767 |  | 1.859 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5596 | 1.053 | 1.23 | 68.48 | 0.1761 |  | 1.881 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5685 | 1.048 | 1.243 | 67.42 | 0.1734 |  | 1.843 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.582 | 1.035 | 1.231 | 65.85 | 0.1693 |  | 1.779 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5878 | 1.048 | 1.239 | 65.2 | 0.1677 |  | 1.783 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.585 | 1.05 | 1.24 | 65.51 | 0.1685 |  | 1.795 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4735 | 1.053 | 1.23 | 80.94 | 0.2081 |  | 2.223 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4726 | 1.123 | 1.307 | 82.82 | 0.2216 |  | 2.376 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.48 | 1.098 | 1.246 | 80.84 | 0.2165 |  | 2.287 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4793 | 1.116 | 1.28 | 82.37 | 0.2218 |  | 2.328 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.464 | 1.036 | 1.229 | 82.59 | 0.2124 |  | 2.233 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4752 | 1.041 | 1.233 | 80.64 | 0.2074 |  | 2.191 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4876 | 1.127 | 1.305 | 80.26 | 0.2148 |  | 2.311 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.485 | 1.114 | 1.244 | 80 | 0.2143 |  | 2.296 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.5015 | 1.112 | 1.281 | 78.72 | 0.212 |  | 2.217 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4938 | 1.062 | 1.249 | 77.61 | 0.1996 |  | 2.15 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4878 | 1.041 | 1.236 | 78.56 | 0.202 |  | 2.135 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.5061 | 1.041 | 1.228 | 75.72 | 0.1947 |  | 2.056 noisy |

## Dynamic Trace Detail

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.4726 | 1.123 | 1.307 | 2.376 | 7444 | 3562 | 1124 | 0 | 8 (p95 1.325 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.48 | 1.098 | 1.246 | 2.287 | 7449 | 3643 | 1099 | 0 | 8 (p95 1.276 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.4793 | 1.116 | 1.28 | 2.328 | 7552 | 4480 | 1117 | 0 | 8 (p95 1.318 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | 0.4876 | 1.127 | 1.305 | 2.311 | 7357 | 3549 | 1128 | 5.504 | 8 (p95 1.259 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | 0.485 | 1.114 | 1.244 | 2.296 | 7437 | 3591 | 1115 | 6.012 | 8 (p95 1.243 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | 0.5015 | 1.112 | 1.281 | 2.217 | 7445 | 4497 | 1113 | 5.071 | 8 (p95 1.266 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 61 | 0.7622 | 1.325 | 1.369 | 1.739 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,7,8 | 2 | 8 | 65 | 0.8284 | 1.318 | 1.354 | 1.591 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,7,8 | 1 | 8 | 49 | 0.7097 | 1.276 | 1.348 | 1.798 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,6,8 | 2 | 8 | 128 | 0.769 | 1.266 | 1.341 | 1.646 | 6.152 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 129 | 0.8264 | 1.259 | 1.358 | 1.524 | 6.589 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-buckets.jsonl` | 1,2,4,6,8 | 0 | 8 | 20 | 0.8216 | 1.249 | 1.256 | 1.52 | 6.875 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,6,8 | 1 | 8 | 115 | 0.7591 | 1.243 | 1.335 | 1.638 | 7.174 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.8328 | 1.243 | 1.248 | 1.492 | 6.875 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.7806 | 1.24 | 1.245 | 1.589 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 9 | 0.7837 | 1.237 | 1.241 | 1.579 | 0 |

### Host Orchestration

| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | p99 ms | Total ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_copy_host_ms | 500 | 0.06768 | 0.08636 | 0.08989 | 34.85 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_attention_host_ms | 500 | 0.2684 | 0.2968 | 0.3096 | 136.2 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.01059 | 0.01128 | 0.02773 | 5.438 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.00913 | 0.01042 | 0.02636 | 4.744 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_copy_host_ms | 500 | 0.07042 | 0.08964 | 0.09293 | 36.22 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_attention_host_ms | 500 | 0.2708 | 0.3016 | 0.3155 | 137.9 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.01046 | 0.01149 | 0.02843 | 5.382 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.00892 | 0.009932 | 0.01851 | 4.574 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_copy_host_ms | 500 | 0.06826 | 0.08506 | 0.08903 | 34.8 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_attention_host_ms | 500 | 0.2712 | 0.2999 | 0.3081 | 136.6 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.01003 | 0.01091 | 0.02762 | 5.16 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.00861 | 0.009775 | 0.01149 | 4.4 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | input_copy_host_ms | 500 | 0.07142 | 0.09561 | 0.1083 | 38.36 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | piecewise_attention_host_ms | 500 | 0.2742 | 0.3093 | 0.3217 | 138.8 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.01022 | 0.01093 | 0.0134 | 5.205 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.00879 | 0.009991 | 0.02591 | 4.587 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | input_copy_host_ms | 500 | 0.06874 | 0.09383 | 0.1048 | 37.33 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | piecewise_attention_host_ms | 500 | 0.2754 | 0.3068 | 0.3196 | 138.4 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.01056 | 0.01126 | 0.02819 | 5.404 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.009051 | 0.01002 | 0.01145 | 4.603 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | input_copy_host_ms | 500 | 0.06954 | 0.09108 | 0.1055 | 37.1 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | piecewise_attention_host_ms | 500 | 0.2789 | 0.3147 | 0.3273 | 140.3 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.01058 | 0.01182 | 0.0282 | 5.472 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.00925 | 0.01054 | 0.01944 | 4.752 |

## Observation

- Loaded 30 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 30 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 (2.376 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 (2.328 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 (2.311 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
