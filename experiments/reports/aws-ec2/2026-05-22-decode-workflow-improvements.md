# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements/decode-step-dynamic-buckets.jsonl`
- `experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements/decode-step-dynamic.jsonl`
- `experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements/decode-step.jsonl`

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
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused eager decode step | 0.4786 | 57.1 | 0.05736 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused CUDA Graph replay | 0.1499 | 182.4 | 0.1832 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused piecewise CUDA Graph replay | 0.4174 | 65.47 | 0.06577 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | Fused same-stream piecewise CUDA Graph replay | 0.3223 | 84.8 | 0.08518 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive eager decode step | 0.4863 | 56.21 | 0.05646 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | Naive CUDA Graph replay | 0.161 | 169.8 | 0.1705 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | Fused eager decode step | 0.5601 | 68.42 | 0.1759 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | Fused eager decode step | 0.5691 | 67.35 | 0.1732 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | Fused eager decode step | 0.5671 | 67.58 | 0.1738 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5556 | 68.97 | 0.1774 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.5749 | 66.66 | 0.1714 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | 0.576 | 66.53 | 0.1711 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4677 | 81.94 | 0.2107 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4813 | 79.63 | 0.2048 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | 0.4796 | 79.91 | 0.2055 |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-eager | Fused eager decode step | pass | 0.4786 | 0.507 | 0.5165 | 57.1 | 0.05736 |  | 1.059 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-graph | Fused CUDA Graph replay | pass | 0.1499 | 0.158 | 0.1727 | 182.4 | 0.1832 |  | 1.054 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.4174 | 0.4428 | 0.4594 | 65.47 | 0.06577 |  | 1.061 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=fused-piecewise-graph-same-stream, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | fused | fused-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.3223 | 0.3463 | 0.3602 | 84.8 | 0.08518 |  | 1.075 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-eager, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-eager | Naive eager decode step | pass | 0.4863 | 0.5463 | 0.5605 | 56.21 | 0.05646 |  | 1.123 |
| decode_step | decode_step | float16 | 1x2048x16x64x4096 | mode=naive-graph, batch_size=1, seq_len=2048, hidden_dim=1024, intermediate_dim=4096 | naive | naive-graph | Naive CUDA Graph replay | pass | 0.161 | 0.17 | 0.1793 | 169.8 | 0.1705 |  | 1.056 |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.5601 | 0.901 | 1.047 | 68.42 | 0.1759 |  | 1.609 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.5691 | 0.9007 | 1.046 | 67.35 | 0.1732 |  | 1.583 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-eager, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-eager | dynamic-eager | Fused eager decode step | not checked | 0.5671 | 0.9127 | 1.048 | 67.58 | 0.1738 |  | 1.609 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5556 | 1.047 | 1.242 | 68.97 | 0.1774 |  | 1.884 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.5749 | 1.034 | 1.232 | 66.66 | 0.1714 |  | 1.799 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph | dynamic-piecewise-graph | Fused piecewise CUDA Graph replay | pass | 0.576 | 1.046 | 1.252 | 66.53 | 0.1711 |  | 1.816 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4677 | 1.036 | 1.227 | 81.94 | 0.2107 |  | 2.216 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4813 | 1.036 | 1.243 | 79.63 | 0.2048 |  | 2.153 noisy |
| decode_step | decode_step | float16 | 8x2048x16x64x4096 | mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 | dynamic-piecewise-graph-same-stream | dynamic-piecewise-graph-same-stream | Fused same-stream piecewise CUDA Graph replay | pass | 0.4796 | 1.036 | 1.229 | 79.91 | 0.2055 |  | 2.16 noisy |

## Observation

- Loaded 15 benchmark rows from 3 result files.
- Fastest backend split: dynamic-eager 3, dynamic-piecewise-graph 3, dynamic-piecewise-graph-same-stream 3, fused 4, naive 2.
- Correctness summary: not checked 3, pass 12.
- No Triton rows beat the matching torch baseline in this result set.
- Noisy rows at p95/p50 >= 1.2: decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,3,4,6,8 (2.216 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,8 (2.16 noise); decode_step decode_step float16 mode=dynamic-piecewise-graph-same-stream, max_batch_size=8, seq_len=2048, buckets=1,2,4,6,8 (2.153 noise).

## Technique Takeaways

- Fusion rows should be read as tests of intermediate-traffic removal, not as generic Triton-vs-PyTorch comparisons.

## Interpretation

- Noisy rows should be profiled or rerun before treating their p50 latency as stable.

## Next Question

What does Nsight Compute show for the noisy Triton rows and the largest fused win?
