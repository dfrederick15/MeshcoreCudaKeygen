#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

// Result returned from GPU search
struct GpuHit {
  bool found = false;
  unsigned char seed[32]{};
  unsigned char pub[32]{};
  unsigned char priv64[64]{};
};

// ---------------- Seed pool API ----------------
// Allocate pool in VRAM. Size rounds down to multiple of 32 bytes.
bool   gpu_seedpool_init_bytes(size_t bytes);
bool   gpu_seedpool_init_seeds(size_t seedCount);
void   gpu_seedpool_free();
size_t gpu_seedpool_seed_count();

// Fill the VRAM pool with deterministic pseudo-random seeds derived from baseCounter.
bool gpu_seedpool_generate(uint64_t baseCounter, cudaStream_t stream = 0);

// ---------------- Search APIs ----------------
// Legacy helper (kept for CLI usage / benchmark): internally uses a temp pool equal to batch.
GpuHit gpu_find_pubkey_prefix_ed25519_cuda(const std::string& prefix, size_t batch);

// New: Search using the VRAM seed pool (must be initialized + generated first).
// startIndex advances (wraps) so you can scan continuously.
GpuHit gpu_find_pubkey_prefix_ed25519_cuda_seedpool(
  const std::string& prefix,
  size_t batch,
  size_t* startIndex,
  cudaStream_t stream = 0
);
