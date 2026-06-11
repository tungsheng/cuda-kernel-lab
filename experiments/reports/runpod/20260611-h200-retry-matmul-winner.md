# GPU Benchmark Report

Status: generated from benchmark JSONL

## Question

What are the baseline PyTorch and Triton measurements for this CUDA
Kernel Lab benchmark run?

## Result Files

- `experiments/results/runpod/20260611-h200-retry-matmul-winner/matmul-autotune.jsonl`

## Environment

- Git commit: `a113a646e6a18ce3796f812de6976dbb452f923f`
- Git dirty: `True`
- Provider: `runpod`
- Provider id: `99alnql81pfet0`
- Provider GPU: `NVIDIA H200`
- Provider cloud: `SECURE`
- Provider image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Python: `3.12.3`
- Platform: `Linux-6.8.0-83-generic-x86_64-with-glibc2.39`
- PyTorch: `2.9.1`
- Triton: `3.5.1`
- CUDA devices: `NVIDIA H200 (139.81 GiB)`

## Optimization Techniques Tested

| Family | Technique | Used By | Hypothesis |
| --- | --- | --- | --- |
| baseline | PyTorch reference baseline | torch controls | Establish the latency, bandwidth, and correctness baseline for comparison. |
| tiling | Tiled dot-product reuse | matmul matmul | Triton tile-shape and launch-configuration sweeps with `tl.dot` can increase arithmetic intensity and Tensor Core utilization, but may trade off occupancy, pipeline depth, and register pressure. |

## Fastest By Operation

| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | Technique | p50 ms | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | PyTorch reference baseline | 0.08656 | 1220 | 533.4 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | PyTorch reference baseline | 0.08861 | 1192 | 521.1 |

## Roofline Summary

- Spec: `NVIDIA H200 SXM` (NVIDIA H200 published peak specs).
- Peak HBM bandwidth: `4800 GB/s`.

| Primitive | Operation | Dtype | Shape | Backend | Strategy | Intensity FLOP/B | Achieved GB/s | HBM Peak % | Achieved TFLOP/s | Math Peak % | Bound |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| matmul | matmul | bfloat16 | 512x11008x4096 | torch | torch-baseline | 437 | 1220 | 25.43 | 533.4 | 26.95 | compute |
| matmul | matmul | float16 | 512x11008x4096 | torch | torch-baseline | 437 | 1192 | 24.84 | 521.1 | 26.33 | compute |

## Matmul Gap Summary

| Dtype | Shape | Best Triton Variant | Triton TFLOP/s | Torch TFLOP/s | Triton/Torch % | Triton Peak % | Next Action |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | 474.3 | 533.4 | 88.92 | 23.97 | profile Tensor Core utilization |
| float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | 464.9 | 521.1 | 89.22 | 23.49 | profile Tensor Core utilization |

## Backend Detail

| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | torch-baseline | PyTorch reference baseline | pass | 0.08725 | 0.08863 | 0.09177 | 1211 | 529.2 | 0.9978 | 1.016 |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | torch-baseline | PyTorch reference baseline | pass | 0.08656 | 0.08824 | 0.0904 | 1220 | 533.4 | 1.006 | 1.019 |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | torch-baseline | PyTorch reference baseline | pass | 0.08706 | 0.08817 | 0.09166 | 1214 | 530.4 | 1 | 1.013 |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09926 | 0.105 | 0.1088 | 1064 | 465.1 | 0.877 | 1.058 |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09885 | 0.102 | 0.1065 | 1069 | 467.1 | 0.8807 | 1.032 |
| matmul | matmul | bfloat16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09734 | 0.1005 | 0.1035 | 1085 | 474.3 | 0.8943 | 1.033 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | torch-baseline | PyTorch reference baseline | pass | 0.08866 | 0.08981 | 0.09442 | 1192 | 520.8 | 0.9995 | 1.013 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | torch-baseline | PyTorch reference baseline | pass | 0.08885 | 0.09021 | 0.09443 | 1189 | 519.7 | 0.9973 | 1.015 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | torch | torch-baseline | PyTorch reference baseline | pass | 0.08861 | 0.08967 | 0.09394 | 1192 | 521.1 | 1 | 1.012 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09931 | 0.1017 | 0.109 | 1064 | 464.9 | 0.8922 | 1.024 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.09981 | 0.1034 | 0.107 | 1058 | 462.6 | 0.8878 | 1.036 |
| matmul | matmul | float16 | 512x11008x4096 | block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard | triton | triton-tiled-dot | Tiled dot-product reuse | pass | 0.101 | 0.1034 | 0.1055 | 1046 | 457 | 0.877 | 1.023 |

## Observation

- Loaded 12 benchmark rows from 1 result file.
- Fastest backend split: torch 2.
- All 12 correctness checks passed.
- No Triton rows beat the matching torch baseline in this result set.
- No rows exceeded the 1.2 p95/p50 noise threshold.

## Technique Takeaways

- Tiled matmul rows should be judged by TFLOP/s and Tensor Core counters; the current best Triton tile/launch config is matmul matmul bfloat16 block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=5, input_precision=tf32, group_m=8, schedule=standard at 474.3 TFLOP/s.

## Interpretation

- Use the fastest-by-operation table to choose the next profiler target.

## Next Question

What do Nsight Compute Tensor Core counters show for the largest Triton matmul gaps?
