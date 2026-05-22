# A10G dynamic decode step piecewise CUDA Graph float16 profile

## Context

- Benchmark command: `uv run benchmark-decode-step --dynamic-trace --mode dynamic-piecewise-graph --device cuda --dtype float16 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements-profiled/decode-step-dynamic-piecewise-graph-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements-profiled/decode-step-dynamic-piecewise-graph-float16.jsonl`
- Operation: `decode_step`
- Strategy label: `dynamic-piecewise-graph`
- Method family: `launch replay`
- Optimization technique: `Fused piecewise CUDA Graph replay`
- Hypothesis: Capturing the static fused pre/post-attention regions while leaving attention eager should keep graph benefits when batch and sequence shapes vary.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 34560 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 86.12 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 9310976 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 8510208 | byte | `dram__bytes_write.sum` |
| Occupancy | 89.84 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 30 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
