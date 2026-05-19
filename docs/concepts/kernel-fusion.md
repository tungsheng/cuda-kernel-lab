# Kernel Fusion

Kernel fusion combines operations so intermediate values can stay in registers or
shared memory instead of being written to and read back from high bandwidth
memory.

## What To Measure

For each fused kernel, compare:

- separate-kernel latency
- fused-kernel latency
- estimated memory traffic before and after fusion
- profiler-observed memory throughput
- whether fusion increased register pressure or reduced occupancy

## First Targets

- softmax: combine max, exponentiation, sum, and normalization
- SwiGLU elementwise path: combine SiLU activation and multiply
- normalization: combine reduction, scale, and bias where applicable

## Softmax

Softmax is the first fusion target because the unfused version creates expensive
round trips through high bandwidth memory.

| Model | Reads | Writes | Estimated Traffic |
| --- | ---: | ---: | ---: |
| naive two-kernel softmax | input + intermediate | intermediate + output | 4 tensors |
| fused row-wise softmax | input | output | 2 tensors |

The Triton softmax performs the row max, exponential, row sum, and final
normalization in one program per row:

```text
load row -> subtract max -> exp -> reduce sum -> normalize -> store row
```

Run the standard comparison from the benchmark workflow. The default softmax
traffic model is `fused`, which reports idealized HBM movement for both PyTorch
and Triton. To show the traffic a naive two-kernel implementation would pay,
rerun with:

```bash
uv run benchmark-softmax --backend triton --device cuda --traffic-model naive
```

That flag changes only the GB/s denominator. It does not change the kernel.
