// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
// src/gpu_search_ed25519_cuda.cu

#include "seedpool.cuh"
#include "ed25519_pubonly.cuh"
#include "gpu_search_ed25519_cuda.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "ed25519.cuh"          // external: ed25519_kernel_create_keypair_batch


// ---------------- Prefix packing ----------------
static inline int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

struct PackedPrefix {
  uint8_t bytes[32]{};
  int fullBytes = 0;
  int hasNibble = 0;     // 0/1
  uint8_t nibbleHi = 0;  // 0..15 (high nibble)
};

static PackedPrefix pack_prefix(const std::string& raw) {
  std::string s;
  s.reserve(raw.size());
  for (char c : raw) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
    s.push_back(c);
  }

  PackedPrefix p{};
  if (s.empty()) return p;

  const int nHex = (int)s.size();
  p.fullBytes = nHex / 2;
  p.hasNibble = (nHex % 2) ? 1 : 0;

  for (int i = 0; i < p.fullBytes && i < 32; i++) {
    int hi = hexval(s[2 * i]);
    int lo = hexval(s[2 * i + 1]);
    if (hi < 0 || lo < 0) { p.fullBytes = 0; p.hasNibble = 0; return p; }
    p.bytes[i] = (uint8_t)((hi << 4) | lo);
  }

  if (p.hasNibble && p.fullBytes < 32) {
    int hi = hexval(s[2 * p.fullBytes]);
    if (hi < 0) { p.fullBytes = 0; p.hasNibble = 0; return p; }
    p.nibbleHi = (uint8_t)hi;
  }
  return p;
}

// ---------------- Device prefix storage ----------------
__constant__ uint8_t c_prefix_bytes[32];
__constant__ int     c_prefix_fullBytes;
__constant__ int     c_prefix_hasNibble;
__constant__ uint8_t c_prefix_nibbleHi;

static bool upload_prefix_to_device_async(const PackedPrefix& p, cudaStream_t stream) {
  cudaError_t st = cudaSuccess;

  st = cudaMemcpyToSymbolAsync(c_prefix_bytes, p.bytes, sizeof(p.bytes), 0, cudaMemcpyHostToDevice, stream);
  if (st != cudaSuccess) return false;

  st = cudaMemcpyToSymbolAsync(c_prefix_fullBytes, &p.fullBytes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  if (st != cudaSuccess) return false;

  st = cudaMemcpyToSymbolAsync(c_prefix_hasNibble, &p.hasNibble, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  if (st != cudaSuccess) return false;

  st = cudaMemcpyToSymbolAsync(c_prefix_nibbleHi, &p.nibbleHi, sizeof(uint8_t), 0, cudaMemcpyHostToDevice, stream);
  if (st != cudaSuccess) return false;

  return true;
}

// ---------------- GPU scan kernel ----------------
__device__ __forceinline__ bool match_prefix_pub32(const uint8_t* pub) {
  for (int i = 0; i < c_prefix_fullBytes; i++) {
    if (pub[i] != c_prefix_bytes[i]) return false;
  }
  if (c_prefix_hasNibble) {
    uint8_t want = (uint8_t)(c_prefix_nibbleHi << 4);
    if ( (pub[c_prefix_fullBytes] & 0xF0) != want ) return false;
  }
  return true;
}

__global__ void scan_pubkeys_for_prefix(const uint8_t* pubkeys, int count, int* foundIndex) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= count) return;

  // atomic read to avoid any weird caching behavior
  if (atomicAdd(foundIndex, 0) >= 0) return;

  const uint8_t* pub = &pubkeys[idx * 32];
  if (match_prefix_pub32(pub)) {
    atomicCAS(foundIndex, -1, idx);
  }
}

// ---------------- Seedpool wrappers ----------------
bool gpu_seedpool_init_bytes(size_t poolBytes) { return seedpool_init_bytes(poolBytes); }
bool gpu_seedpool_generate(uint64_t baseCounter, cudaStream_t stream) { return seedpool_generate(baseCounter, stream); }
size_t gpu_seedpool_seed_count() { return seedpool_seed_count(); }
const uint8_t* gpu_seedpool_device_ptr() { return seedpool_device_ptr(); }
void gpu_seedpool_free() { seedpool_free(); }

// ---------------- Helper for error checking ----------------
static inline bool ck(cudaError_t st) {
  return st == cudaSuccess;
}

static inline bool ck_last() {
  return cudaGetLastError() == cudaSuccess;
}

// ---------------- Core search (pub-only for batch, priv only on hit) ----------------
GpuHit gpu_find_pubkey_prefix_ed25519_cuda_seedpool(
  const std::string& hexPrefix,
  size_t batch,
  size_t* ioCursor,
  cudaStream_t stream
) {
  GpuHit out{};
  out.found = false;
  if (batch == 0) return out;

  const uint8_t* dSeedsAll = gpu_seedpool_device_ptr();
  const size_t poolSeeds = gpu_seedpool_seed_count();
  if (!dSeedsAll || poolSeeds == 0) return out;

  size_t cursor = (ioCursor) ? (*ioCursor % poolSeeds) : 0;
  // ensure contiguous window
  if (cursor + batch > poolSeeds) cursor = 0;

  if (ioCursor) {
    *ioCursor = cursor + batch;
    if (*ioCursor >= poolSeeds) *ioCursor = 0;
  }

  const uint8_t* dSeeds = dSeedsAll + cursor * 32;

  // Upload prefix constants on same stream as kernels
  PackedPrefix pp = pack_prefix(hexPrefix);
  if (!upload_prefix_to_device_async(pp, stream)) {
    return out;
  }

  // Allocate batch pubkeys + found index
  uint8_t* dPub = nullptr;
  int* dFound = nullptr;

  if (!ck(cudaMalloc(&dPub, batch * 32))) return out;
  if (!ck(cudaMalloc(&dFound, sizeof(int)))) { cudaFree(dPub); return out; }

  // Set foundIndex = -1 reliably (0xFFFFFFFF)
  if (!ck(cudaMemsetAsync(dFound, 0xFF, sizeof(int), stream))) {
    cudaFree(dPub); cudaFree(dFound); return out;
  }

  const int threads = 256;
  const int blocks = (int)((batch + (size_t)threads - 1) / (size_t)threads);

  // 1) pub-only keygen
  ed25519_kernel_create_pubkey_batch<<<blocks, threads, 0, stream>>>(
    (unsigned char*)dPub,
    (const unsigned char*)dSeeds,
    (int)batch
  );
  if (!ck_last()) { cudaFree(dPub); cudaFree(dFound); return out; }

  // 2) scan
  scan_pubkeys_for_prefix<<<blocks, threads, 0, stream>>>(dPub, (int)batch, dFound);
  if (!ck_last()) { cudaFree(dPub); cudaFree(dFound); return out; }

  // 3) read found index
  int hFound = -1;
  if (!ck(cudaMemcpyAsync(&hFound, dFound, sizeof(int), cudaMemcpyDeviceToHost, stream))) {
    cudaFree(dPub); cudaFree(dFound); return out;
  }
  cudaError_t stSync = cudaStreamSynchronize(stream);
  if (!ck(stSync)) { cudaFree(dPub); cudaFree(dFound); return out; }

  if (hFound >= 0 && (size_t)hFound < batch) {
    out.found = true;

    // copy seed + pub
    ck(cudaMemcpyAsync(out.seed, dSeeds + (size_t)hFound * 32, 32, cudaMemcpyDeviceToHost, stream));
    ck(cudaMemcpyAsync(out.pub,  dPub   + (size_t)hFound * 32, 32, cudaMemcpyDeviceToHost, stream));

    // compute full keypair ONLY for the winner (limit=1)
    uint8_t* dPub1 = nullptr;
    uint8_t* dPriv1 = nullptr;
    ck(cudaMalloc(&dPub1, 32));
    ck(cudaMalloc(&dPriv1, 64));

    const uint8_t* dSeed1 = dSeeds + (size_t)hFound * 32;
    ed25519_kernel_create_keypair_batch<<<1, 1, 0, stream>>>(
      (unsigned char*)dPub1,
      (unsigned char*)dPriv1,
      (const unsigned char*)dSeed1,
      1
    );
    if (ck_last()) {
      ck(cudaMemcpyAsync(out.priv64, dPriv1, 64, cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);
    }

    if (dPub1) cudaFree(dPub1);
    if (dPriv1) cudaFree(dPriv1);
  }

  cudaFree(dPub);
  cudaFree(dFound);
  return out;
}

// Single-shot helper (used by benchmark/CLI paths)
GpuHit gpu_find_pubkey_prefix_ed25519_cuda(const std::string& hexPrefix, size_t batch) {
  GpuHit out{};
  out.found = false;
  if (batch == 0) return out;

  if (!gpu_seedpool_init_bytes(batch * 32)) return out;

  cudaStream_t s{};
  cudaStreamCreate(&s);

  // seedpool_generate should fill seeds in VRAM
  gpu_seedpool_generate(0xABCDEF1234567890ULL, s);
  cudaStreamSynchronize(s);

  size_t cursor = 0;
  out = gpu_find_pubkey_prefix_ed25519_cuda_seedpool(hexPrefix, batch, &cursor, s);

  cudaStreamDestroy(s);
  gpu_seedpool_free();
  return out;
}
