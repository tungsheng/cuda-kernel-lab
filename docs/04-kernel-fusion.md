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

The Triton implementation performs the row max, exponential, row sum, and final
normalization in one program per row:

```text
load row -> subtract max -> exp -> reduce sum -> normalize -> store row
```

Run the comparison on a CUDA host:

```bash
uv run python -m inference_kernel_lab.benchmarks.softmax --backend all --device cuda --rows 4096 --cols 1024
```

The default benchmark traffic model is `fused`, which reports the idealized
lower-bound HBM movement for PyTorch and Triton. To see the memory traffic a
naive implementation would pay, rerun with:

```bash
uv run python -m inference_kernel_lab.benchmarks.softmax --backend triton --device cuda --traffic-model naive
```

That second command does not make the Triton kernel naive; it changes the
denominator used for the GB/s estimate so the cost of the avoided intermediate
tensor is explicit.
