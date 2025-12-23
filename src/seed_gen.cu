#include "seed_gen.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

__global__ void k_generate_seeds(uint8_t* seeds, size_t count, uint64_t baseCounter) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= count) return;

  uint64_t s = baseCounter + (uint64_t)i;
  uint8_t* out = seeds + i * 32;

  uint64_t a = splitmix64(s);
  uint64_t b = splitmix64(s);
  uint64_t c = splitmix64(s);
  uint64_t d = splitmix64(s);

  #pragma unroll
  for (int k = 0; k < 8; k++) out[k]       = (uint8_t)(a >> (8*k));
  #pragma unroll
  for (int k = 0; k < 8; k++) out[8 + k]   = (uint8_t)(b >> (8*k));
  #pragma unroll
  for (int k = 0; k < 8; k++) out[16 + k]  = (uint8_t)(c >> (8*k));
  #pragma unroll
  for (int k = 0; k < 8; k++) out[24 + k]  = (uint8_t)(d >> (8*k));
}

void gpu_generate_seeds(uint8_t* d_seeds, size_t count, uint64_t baseCounter, cudaStream_t stream) {
  const int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);
  k_generate_seeds<<<blocks, threads, 0, stream>>>(d_seeds, count, baseCounter);
}
