// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#include "gpu_search.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void k_stub(GpuWinner* w) {
  // does nothing yet
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    w->found = 0;
  }
}

bool gpu_search_stub(const PrefixSpec&, GpuWinner& outWinner) {
  GpuWinner* dW = nullptr;
  cudaMalloc(&dW, sizeof(GpuWinner));
  cudaMemset(dW, 0, sizeof(GpuWinner));

  k_stub<<<1,1>>>(dW);
  cudaDeviceSynchronize();

  cudaMemcpy(&outWinner, dW, sizeof(GpuWinner), cudaMemcpyDeviceToHost);
  cudaFree(dW);
  return outWinner.found != 0;
}
