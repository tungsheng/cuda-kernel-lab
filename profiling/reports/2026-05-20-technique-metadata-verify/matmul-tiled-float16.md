# A10G matmul float16 tiled-dot profile

## Context

- Benchmark command: `uv run benchmark-matmul --backend triton --device cuda --m 1024 --n 1024 --k 1024 --dtype float16 --block-m 32 --block-n 32 --block-k 32 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-20-technique-metadata-verify-profiled/matmul-tiled-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify-profiled/matmul-tiled-float16.jsonl`
- Operation: `matmul`
- Strategy label: `triton-tiled-dot-block-32x32x32`
- Method family: `tiling`
- Optimization technique: `Tiled dot-product reuse`
- Hypothesis: Triton tile-shape sweeps with `tl.dot` can increase arithmetic intensity and Tensor Core utilization, but may trade off occupancy and register pressure.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 81344 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 17.98 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 6219136 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 2542336 | byte | `dram__bytes_write.sum` |
| Occupancy | 77.78 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 40 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 8192 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 27.28 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 27.28 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
