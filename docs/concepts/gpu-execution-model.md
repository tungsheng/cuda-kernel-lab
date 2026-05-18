# GPU Execution Model

This note tracks the execution concepts needed to reason about kernels in this
lab. Keep entries tied to implementation choices and profiler observations.

## Working Vocabulary

- Thread: one program instance executing kernel code.
- Warp: a group of threads scheduled together.
- Block: a group of threads that can cooperate through shared memory and
  synchronization.
- Grid: all blocks launched for one kernel.
- Occupancy: how many warps can be resident on a streaming multiprocessor.
- Coalescing: whether neighboring lanes access neighboring memory addresses.
- Launch overhead: fixed CPU/runtime cost paid before GPU work starts.

## Kernel Questions

For each kernel, record:

- how work maps to threads and blocks
- whether memory access is coalesced
- whether shared memory is used
- what limits occupancy
- what the dominant bottleneck appears to be
- which benchmark command and shape produced the observation

## First Lesson

For simple copy, scale, vector add, and reduction kernels, the useful question is
usually not "how many FLOPs did this do?" The useful question is "how many bytes
had to cross high bandwidth memory, and what fraction of peak bandwidth did the
kernel sustain?"
