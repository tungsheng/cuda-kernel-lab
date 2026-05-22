# A10G matmul float16 tiled-dot profile

## Context

- Benchmark command: `uv run benchmark-matmul --backend triton --device cuda --m 1024 --n 1024 --k 1024 --dtype float16 --block-m 64 --block-n 64 --block-k 32 --num-warps 4 --num-stages 3 --input-precision tf32 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode-profiled/matmul-tiled-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode-profiled/matmul-tiled-float16.jsonl`
- Operation: `matmul`
- Strategy label: `triton-tiled-dot-block-64x64x32-warps-4-stages-3`
- Method family: `tiling`
- Optimization technique: `Tiled dot-product reuse`
- Hypothesis: Triton tile-shape and launch-configuration sweeps with `tl.dot` can increase arithmetic intensity and Tensor Core utilization, but may trade off occupancy, pipeline depth, and register pressure.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 56992 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 23.81 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 5739008 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 2393088 | byte | `dram__bytes_write.sum` |
| Occupancy | 22.46 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 80 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 16384 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 45.27 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 45.27 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
