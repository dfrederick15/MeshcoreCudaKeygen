// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#include "seedpool.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

static uint8_t* g_dSeeds = nullptr;
static size_t   g_seedCount = 0;

extern "C" void seedpool_free() {
  if (g_dSeeds) cudaFree(g_dSeeds);
  g_dSeeds = nullptr;
  g_seedCount = 0;
}

__global__ void seedpool_fill_kernel(uint8_t* seeds, size_t count, uint64_t baseCounter) {
  size_t i = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;

  uint64_t x = baseCounter + (uint64_t)i * 0x9E3779B97F4A7C15ULL;
  auto next64 = [&]() {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
  };

  uint8_t* out = &seeds[i * 32];
  uint64_t a = next64();
  uint64_t b = next64();
  uint64_t c = next64();
  uint64_t d = next64();

  *reinterpret_cast<uint64_t*>(out +  0) = a;
  *reinterpret_cast<uint64_t*>(out +  8) = b;
  *reinterpret_cast<uint64_t*>(out + 16) = c;
  *reinterpret_cast<uint64_t*>(out + 24) = d;
}

extern "C" bool seedpool_init_bytes(size_t poolBytes) {
  poolBytes = (poolBytes / 32) * 32;
  if (poolBytes < 32) return false;

  seedpool_free();

  g_seedCount = poolBytes / 32;
  if (cudaMalloc(&g_dSeeds, g_seedCount * 32) != cudaSuccess) {
    g_dSeeds = nullptr;
    g_seedCount = 0;
    return false;
  }
  return true;
}

extern "C" bool seedpool_generate(uint64_t baseCounter, cudaStream_t stream) {
  if (!g_dSeeds || g_seedCount == 0) return false;

  const int threads = 256;
  const int blocks = (int)((g_seedCount + (size_t)threads - 1) / (size_t)threads);
  seedpool_fill_kernel<<<blocks, threads, 0, stream>>>(g_dSeeds, g_seedCount, baseCounter);

  return cudaGetLastError() == cudaSuccess;
}

extern "C" size_t seedpool_seed_count() { return g_seedCount; }
extern "C" const uint8_t* seedpool_device_ptr() { return g_dSeeds; }
