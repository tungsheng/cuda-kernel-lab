# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-tail-sweep-live/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-sweep-live/decode-step-dynamic-tail.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-sweep-live/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-tail-sweep-live/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4736 | 57.71 | 0.05796 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1506 | 181.4 | 0.1822 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.4142 | 65.98 | 0.06627 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused same-stream piecewise CUDA Graph replay | 0.317 | 86.22 | 0.0866 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4599 | 59.43 | 0.05969 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.1612 | 169.5 | 0.1703 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-eager | Fused eager decode step | 0.5637 | 67.98 | 0.1748 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-eager | Fused eager decode step | 0.5692 | 67.32 | 0.1731 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | Fused eager decode step | 0.5579 | 68.69 | 0.1766 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | Fused eager decode step | 0.5662 | 67.68 | 0.174 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5394 | 71.05 | 0.1827 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5457 | 70.23 | 0.1806 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5542 | 69.15 | 0.1778 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5545 | 69.11 | 0.1777 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5666 | 67.64 | 0.1739 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5592 | 68.53 | 0.1762 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4537 | 84.48 | 0.2172 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4746 | 80.74 | 0.2076 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4713 | 83.04 | 0.2222 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4676 | 81.96 | 0.2108 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4587 | 83.54 | 0.2148 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4736 | 0.507 | 0.5189 | 57.71 | 0.05796 |  | 1.07 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1506 | 0.1656 | 0.1695 | 181.4 | 0.1822 |  | 1.1 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.4142 | 0.4443 | 0.4554 | 65.98 | 0.06627 |  | 1.073 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.317 | 0.3385 | 0.3575 | 86.22 | 0.0866 |  | 1.068 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4599 | 0.4873 | 0.5039 | 59.43 | 0.05969 |  | 1.06 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.1612 | 0.1777 | 0.1789 | 169.5 | 0.1703 |  | 1.102 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5637 | 0.8993 | 1.067 | 67.98 | 0.1748 |  | 1.595 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5692 | 0.9028 | 1.055 | 67.32 | 0.1731 |  | 1.586 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5579 | 0.9091 | 1.04 | 68.69 | 0.1766 |  | 1.629 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5662 | 0.9229 | 1.053 | 67.68 | 0.174 |  | 1.63 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5394 | 0.9133 | 1.04 | 71.05 | 0.1827 |  | 1.693 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | pass | 0.5685 | 0.9037 | 1.048 | 67.41 | 0.1733 |  | 1.59 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5457 | 1.027 | 1.23 | 70.23 | 0.1806 |  | 1.882 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5542 | 1.034 | 1.229 | 69.15 | 0.1778 |  | 1.866 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5545 | 1.045 | 1.242 | 69.11 | 0.1777 |  | 1.885 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5666 | 1.035 | 1.23 | 67.64 | 0.1739 |  | 1.826 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5592 | 1.047 | 1.248 | 68.53 | 0.1762 |  | 1.873 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5788 | 1.06 | 1.241 | 66.21 | 0.1703 |  | 1.832 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,7,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4537 | 1.035 | 1.227 | 84.48 | 0.2172 |  | 2.282 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,5,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4746 | 1.037 | 1.228 | 80.74 | 0.2076 |  | 2.185 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4723 | 1.038 | 1.23 | 81.15 | 0.2087 |  | 2.198 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4713 | 1.122 | 1.304 | 83.04 | 0.2222 |  | 2.381 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4734 | 1.107 | 1.245 | 81.97 | 0.2196 |  | 2.339 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4879 | 1.11 | 1.278 | 80.92 | 0.2179 |  | 2.275 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4676 | 1.062 | 1.23 | 81.96 | 0.2108 |  | 2.27 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4587 | 1.05 | 1.225 | 83.54 | 0.2148 |  | 2.289 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4788 | 1.038 | 1.23 | 80.04 | 0.2058 |  | 2.169 noisy |

## Observation

- Loaded 27 benchmark rows from 4 result files.
- Fastest backend split: dynamic-eager 5, dynamic-piecewise-graph 5, dynamic-piecewise-graph-same-stream 5, fused 4, naive 2.
- All 27 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 (2.381 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 (2.339 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (2.289 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
