# CUDA C++ Kernels

This directory is reserved for CUDA C++ implementations loaded from Python with
`torch.utils.cpp_extension`.

Each kernel should include:

- a correctness test against the PyTorch baseline
- a benchmark entry
- a memory traffic estimate
- profiler notes once inspected with Nsight Compute or Nsight Systems

