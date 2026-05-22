# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-tail-policy-3way-live/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-policy-3way-live/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-policy-3way-live/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-policy-3way-live/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4747 | 57.58 | 0.05784 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1522 | 179.6 | 0.1804 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.4249 | 64.32 | 0.06461 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused same-stream piecewise CUDA Graph replay | 0.3242 | 84.3 | 0.08467 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.47 | 58.15 | 0.05841 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.1623 | 168.4 | 0.1691 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-eager | Fused eager decode step | 0.5811 | 65.95 | 0.1696 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-eager | Fused eager decode step | 0.5785 | 66.24 | 0.1704 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | Fused eager decode step | 0.5758 | 66.55 | 0.1711 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | Fused eager decode step | 0.5734 | 66.84 | 0.1719 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5454 | 70.26 | 0.1807 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5644 | 67.9 | 0.1746 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.565 | 67.83 | 0.1744 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5764 | 66.49 | 0.171 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5901 | 64.94 | 0.167 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5578 | 68.71 | 0.1767 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4741 | 80.84 | 0.2079 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4804 | 81.47 | 0.218 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4809 | 79.68 | 0.2049 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4873 | 78.64 | 0.2022 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.464 | 82.59 | 0.2124 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4747 | 0.4974 | 0.5031 | 57.58 | 0.05784 |  | 1.048 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1522 | 0.164 | 0.17 | 179.6 | 0.1804 |  | 1.078 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.4249 | 0.4496 | 0.4637 | 64.32 | 0.06461 |  | 1.058 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.3242 | 0.3545 | 0.3568 | 84.3 | 0.08467 |  | 1.093 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.47 | 0.4951 | 0.5171 | 58.15 | 0.05841 |  | 1.053 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1623 | 0.1781 | 0.1817 | 168.4 | 0.1691 |  | 1.098 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5811 | 0.9056 | 1.053 | 65.95 | 0.1696 |  | 1.559 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5785 | 0.9043 | 1.058 | 66.24 | 0.1704 |  | 1.563 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5758 | 0.9072 | 1.05 | 66.55 | 0.1711 |  | 1.575 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5734 | 0.9007 | 1.065 | 66.84 | 0.1719 |  | 1.571 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5454 | 0.8975 | 1.04 | 70.26 | 0.1807 |  | 1.646 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.574 | 0.9044 | 1.052 | 66.76 | 0.1717 |  | 1.576 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5644 | 1.029 | 1.232 | 67.9 | 0.1746 |  | 1.823 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.565 | 1.039 | 1.247 | 67.83 | 0.1744 |  | 1.839 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5764 | 1.048 | 1.242 | 66.49 | 0.171 |  | 1.818 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5901 | 1.036 | 1.233 | 64.94 | 0.167 |  | 1.756 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5578 | 1.045 | 1.238 | 68.71 | 0.1767 |  | 1.873 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5835 | 1.051 | 1.245 | 65.67 | 0.1689 |  | 1.801 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4741 | 1.039 | 1.232 | 80.84 | 0.2079 |  | 2.192 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4792 | 1.122 | 1.305 | 81.67 | 0.2185 |  | 2.34 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4774 | 1.098 | 1.242 | 81.28 | 0.2177 |  | 2.301 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4791 | 1.121 | 1.283 | 82.41 | 0.2219 |  | 2.339 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4835 | 1.042 | 1.231 | 79.26 | 0.2038 |  | 2.155 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4804 | 1.138 | 1.304 | 81.47 | 0.218 |  | 2.369 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4804 | 1.11 | 1.262 | 80.77 | 0.2163 |  | 2.31 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4934 | 1.124 | 1.292 | 80.01 | 0.2155 |  | 2.279 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4809 | 1.038 | 1.247 | 79.68 | 0.2049 |  | 2.158 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4871 | 1.124 | 1.304 | 80.35 | 0.215 |  | 2.308 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.483 | 1.112 | 1.244 | 80.34 | 0.2152 |  | 2.302 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4903 | 1.109 | 1.291 | 80.52 | 0.2168 |  | 2.261 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4873 | 1.038 | 1.23 | 78.64 | 0.2022 |  | 2.13 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.464 | 1.035 | 1.227 | 82.59 | 0.2124 |  | 2.231 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4869 | 1.039 | 1.231 | 78.71 | 0.2024 |  | 2.133 noisy |

## Dynamic Trace Detail

### Tail Policy Summary

| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,2,3,4,5,6,7,8 | 3 | 0.4786 | 1.114 | 1.277 | 7461 | 3890 | 0 | 1.323 |
| 1,2,3,4,6,8 | 3 | 0.4868 | 1.115 | 1.279 | 7447 | 3888 | 5.529 | 1.26 |
| 1,2,3,4,5,6,8 | 3 | 0.4848 | 1.124 | 1.286 | 7439 | 3855 | 2.85 | 1.265 |

### Tail Sweep

| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | 0.4792 | 1.122 | 1.305 | 2.34 | 7387 | 3566 | 1123 | 0 | 8 (p95 1.336 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | 0.4774 | 1.098 | 1.242 | 2.301 | 7470 | 3642 | 1099 | 0 | 8 (p95 1.314 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | 0.4791 | 1.121 | 1.283 | 2.339 | 7525 | 4461 | 1122 | 0 | 8 (p95 1.32 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 0 | 0.4804 | 1.138 | 1.304 | 2.369 | 7367 | 3515 | 1142 | 2.957 | 8 (p95 1.263 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 1 | 0.4804 | 1.11 | 1.262 | 2.31 | 7471 | 3604 | 1111 | 2.887 | 8 (p95 1.259 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 2 | 0.4934 | 1.124 | 1.292 | 2.279 | 7481 | 4447 | 1126 | 2.706 | 8 (p95 1.273 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | 0.4871 | 1.124 | 1.304 | 2.308 | 7352 | 3557 | 1126 | 5.504 | 8 (p95 1.262 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | 0.483 | 1.112 | 1.244 | 2.302 | 7458 | 3597 | 1113 | 6.012 | 8 (p95 1.244 ms) |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | 0.4903 | 1.109 | 1.291 | 2.261 | 7530 | 4511 | 1110 | 5.071 | 8 (p95 1.273 ms) |

### Worst Dynamic Buckets

| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | p99 ms | p95/p50 | Pad % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,7,8 | 0 | 8 | 61 | 0.7673 | 1.336 | 1.373 | 1.742 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,7,8 | 2 | 8 | 65 | 0.8271 | 1.32 | 1.357 | 1.595 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,7,8 | 1 | 8 | 49 | 0.7074 | 1.314 | 1.352 | 1.858 | 0 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,6,8 | 2 | 8 | 128 | 0.7694 | 1.273 | 1.34 | 1.655 | 6.152 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,8 | 2 | 8 | 128 | 0.7742 | 1.273 | 1.339 | 1.644 | 6.152 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 129 | 0.829 | 1.263 | 1.35 | 1.524 | 6.589 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 129 | 0.8278 | 1.262 | 1.346 | 1.524 | 6.589 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` | 1,2,3,4,5,6,8 | 1 | 8 | 115 | 0.7582 | 1.259 | 1.333 | 1.66 | 7.174 |
| dynamic-piecewise-graph | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,5,6,8 | 0 | 8 | 20 | 0.8292 | 1.247 | 1.251 | 1.504 | 6.875 |
| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-buckets.jsonl` | 1,2,3,4,6,8 | 0 | 8 | 20 | 0.8222 | 1.247 | 1.247 | 1.516 | 6.875 |

### Host Orchestration

| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | p99 ms | Total ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | input_copy_host_ms | 500 | 0.06973 | 0.08647 | 0.09063 | 35.64 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_attention_host_ms | 500 | 0.2743 | 0.3056 | 0.3237 | 139 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.01017 | 0.0112 | 0.01888 | 5.232 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.00867 | 0.01009 | 0.02579 | 4.559 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | input_copy_host_ms | 500 | 0.06876 | 0.08749 | 0.09551 | 35.36 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_attention_host_ms | 500 | 0.2704 | 0.2994 | 0.3226 | 138 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.01007 | 0.01164 | 0.0281 | 5.247 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.00871 | 0.01024 | 0.01117 | 4.48 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | input_copy_host_ms | 500 | 0.06983 | 0.089 | 0.09626 | 35.9 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_attention_host_ms | 500 | 0.2713 | 0.2988 | 0.3109 | 136.8 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.01038 | 0.01103 | 0.01297 | 5.272 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,7,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.00882 | 0.009851 | 0.01736 | 4.542 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 0 | input_copy_host_ms | 500 | 0.07137 | 0.09403 | 0.1096 | 37.93 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 0 | piecewise_attention_host_ms | 500 | 0.2713 | 0.3051 | 0.3217 | 138.1 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.0105 | 0.01123 | 0.02777 | 5.386 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.008861 | 0.009941 | 0.02564 | 4.585 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 1 | input_copy_host_ms | 500 | 0.06796 | 0.08848 | 0.1025 | 35.9 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 1 | piecewise_attention_host_ms | 500 | 0.2699 | 0.3086 | 0.3222 | 137.7 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.01064 | 0.01142 | 0.01392 | 5.405 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.009221 | 0.0104 | 0.01191 | 4.68 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 2 | input_copy_host_ms | 500 | 0.06997 | 0.08941 | 0.1067 | 36.82 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 2 | piecewise_attention_host_ms | 500 | 0.2757 | 0.3109 | 0.3186 | 139.3 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.0106 | 0.01129 | 0.01294 | 5.396 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,5,6,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.00881 | 0.009861 | 0.01163 | 4.521 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | input_copy_host_ms | 500 | 0.07021 | 0.09219 | 0.1081 | 37.62 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | piecewise_attention_host_ms | 500 | 0.2749 | 0.3103 | 0.3239 | 139.3 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | piecewise_post_graph_host_ms | 500 | 0.01039 | 0.01117 | 0.02811 | 5.377 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 0 | piecewise_pre_graph_host_ms | 500 | 0.009025 | 0.00987 | 0.01044 | 4.579 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | input_copy_host_ms | 500 | 0.06808 | 0.08795 | 0.1032 | 36.72 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | piecewise_attention_host_ms | 500 | 0.2706 | 0.3085 | 0.329 | 137.5 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | piecewise_post_graph_host_ms | 500 | 0.01055 | 0.0114 | 0.0177 | 5.363 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 1 | piecewise_pre_graph_host_ms | 500 | 0.00907 | 0.01007 | 0.01866 | 4.646 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | input_copy_host_ms | 500 | 0.06795 | 0.08928 | 0.1039 | 36.4 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | piecewise_attention_host_ms | 500 | 0.2674 | 0.3066 | 0.3157 | 136.3 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | piecewise_post_graph_host_ms | 500 | 0.01053 | 0.01128 | 0.0283 | 5.452 |
| dynamic-piecewise-graph-same-stream | 1,2,3,4,6,8 | 2 | piecewise_pre_graph_host_ms | 500 | 0.00913 | 0.01014 | 0.0117 | 4.621 |

## Observation

- Loaded 33 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 33 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 (2.369 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 (2.34 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 (2.339 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
