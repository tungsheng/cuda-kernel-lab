# A10G decode step fused piecewise CUDA Graph float16 profile

## Context

- Benchmark command: `uv run benchmark-decode-step --mode fused-piecewise-graph --device cuda --dtype float16 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements-profiled/decode-step-fused-piecewise-graph-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-22-decode-workflow-improvements-profiled/decode-step-fused-piecewise-graph-float16.jsonl`
- Operation: `decode_step`
- Strategy label: `fused-piecewise-graph`
- Method family: `launch replay`
- Optimization technique: `Fused piecewise CUDA Graph replay`
- Hypothesis: Capturing the static fused pre/post-attention regions while leaving attention eager should keep graph benefits when batch and sequence shapes vary.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 34784 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 85.86 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 9303168 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 8583680 | byte | `dram__bytes_write.sum` |
| Occupancy | 90.81 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 30 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
