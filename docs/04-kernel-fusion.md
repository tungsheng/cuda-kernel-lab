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

