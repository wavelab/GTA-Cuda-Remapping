#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::stringstream ss;
        ss << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line;
        if (abort) throw std::runtime_error(ss.str());
    }
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define LAUNCH(kernel) kernel<<<(numElem + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>
#define SINGLE_ARG(...) __VA_ARGS__