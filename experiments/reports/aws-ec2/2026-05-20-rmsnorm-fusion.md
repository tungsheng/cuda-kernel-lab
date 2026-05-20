# RMSNorm Fusion Experiment

## Question

Does row-wise RMSNorm fusion explain a real Triton speedup on the A10G benchmark run?

## Optimization Technique

- Method family: fusion
- Technique: Row-wise RMSNorm fusion
- Hypothesis: fusing row reductions, normalization, parameter loads, and affine writeback should remove framework overhead and avoid intermediate normalization tensors.
- Control: PyTorch reference baseline for `rmsnorm`
- Knobs changed: backend changed from PyTorch baseline to Triton fused RMSNorm; shape, dtype, warmup, iterations, and epsilon stayed fixed.
- Expected profiler signal: high DRAM throughput, no large intermediate writes, and acceptable occupancy despite the per-row reduction register footprint.

## Command

```bash
./scripts/live-benchmark --run-id 2026-05-20-technique-metadata-verify --include-matmul-sweep --with-profiling --ingress-cidr 107.115.224.0/20
```

Focused benchmark command from the run:

```bash
uv run benchmark-norms --backend all --device cuda --op all --rows 4096 --cols 4096 --dtype float16 --warmup 25 --iterations 100 --output experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/norms.jsonl
```

## Environment

- Device: NVIDIA A10G, 22.06 GiB
- Driver/CUDA: CUDA-capable AWS Deep Learning AMI; exact driver version was not captured in benchmark JSONL.
- Python: 3.10.12
- PyTorch: 2.12.0
- Triton: 3.7.0
- Git commit: `725e35d0e7249f1dceb158cce8b8716994703d5d`

## Shape

- Operation: RMSNorm
- Backend: PyTorch baseline vs Triton fused RMSNorm
- Shape: 4096x4096
- Dtype: float16
- Warmup: 25
- Iterations: 100

## Result

- Result file: `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify/norms.jsonl`
- PyTorch p50: 0.9452 ms
- Triton p50: 0.1700 ms
- Triton p95: 0.1793 ms
- Triton p99: 0.1875 ms
- Triton GB/s: 592.2
- Triton TFLOP/s: 0.4935
- Speedup vs PyTorch: 5.56x
- Correctness: pass

Float32 showed the same direction with a smaller but still clear win:

| Backend | Dtype | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch | float32 | 1.011 | 1.017 | 1.026 | 199.2 | 0.0830 | pass |
| Triton fused | float32 | 0.3103 | 0.3196 | 0.3278 | 648.9 | 0.2703 | pass |

## Profiler Notes

Full profiler note: `profiling/reports/2026-05-20-technique-metadata-verify/norms-rmsnorm-float16.md`

- Key counters: DRAM throughput 91.42% of peak sustained elapsed; occupancy 92.80%; registers per thread 40; dynamic shared memory 32 bytes per block; Tensor Core utilization 0%.
- Confirmation or surprise: the counters match a bandwidth-oriented row-wise fusion rather than a Tensor Core optimization. High DRAM throughput and high occupancy support the claim that the fused Triton kernel is removing overhead and keeping memory traffic efficient.

## Observation

RMSNorm is the strongest positive fusion win in the fresh benchmark report. Triton fused RMSNorm beat the PyTorch reference baseline by 5.56x for float16 and 3.257x for float32, with correctness passing in both rows.

## Interpretation

The result fits the optimization hypothesis. RMSNorm has enough per-row work and memory traffic that fusing reduction, normalization, weight load, and output writeback into one Triton kernel materially reduces overhead. The profiler counters also point to a memory-throughput win, not a Tensor Core path.

## Next Question

Does the RMSNorm fusion win hold across smaller hidden sizes and batch counts, and where does PyTorch regain enough library efficiency to close the gap?
