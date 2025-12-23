#include "gpu_search_ed25519_cuda.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <chrono>
#include <mutex>
#include <vector>


// external ed25519_cuda header (in include path via CMake target_include_directories)
#include "ed25519.cuh"

// =====================================================================================
// Seed pool in VRAM
// =====================================================================================
static uint8_t* g_dSeedPool = nullptr;
static size_t   g_seedPoolSeeds = 0;

// SplitMix64 for fast deterministic pseudo-random
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Fill VRAM pool with 32-byte seeds
__global__ void kFillSeedPool(uint8_t* outSeeds, size_t seedCount, uint64_t baseCounter) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= seedCount) return;

  uint64_t x0 = splitmix64(baseCounter + (uint64_t)i * 4ULL + 0ULL);
  uint64_t x1 = splitmix64(baseCounter + (uint64_t)i * 4ULL + 1ULL);
  uint64_t x2 = splitmix64(baseCounter + (uint64_t)i * 4ULL + 2ULL);
  uint64_t x3 = splitmix64(baseCounter + (uint64_t)i * 4ULL + 3ULL);

  uint8_t* p = outSeeds + i * 32;

#pragma unroll
  for (int b = 0; b < 8; b++) p[b +  0] = (uint8_t)(x0 >> (8 * b));
#pragma unroll
  for (int b = 0; b < 8; b++) p[b +  8] = (uint8_t)(x1 >> (8 * b));
#pragma unroll
  for (int b = 0; b < 8; b++) p[b + 16] = (uint8_t)(x2 >> (8 * b));
#pragma unroll
  for (int b = 0; b < 8; b++) p[b + 24] = (uint8_t)(x3 >> (8 * b));
}

bool gpu_seedpool_init_bytes(size_t bytes) {
  gpu_seedpool_free();
  bytes = (bytes / 32) * 32;
  if (bytes < 32) return false;
  return gpu_seedpool_init_seeds(bytes / 32);
}

bool gpu_seedpool_init_seeds(size_t seedCount) {
  gpu_seedpool_free();
  if (seedCount == 0) return false;

  uint8_t* d = nullptr;
  cudaError_t st = cudaMalloc((void**)&d, seedCount * 32);
  if (st != cudaSuccess) return false;

  g_dSeedPool = d;
  g_seedPoolSeeds = seedCount;
  return true;
}

void gpu_seedpool_free() {
  if (g_dSeedPool) cudaFree(g_dSeedPool);
  g_dSeedPool = nullptr;
  g_seedPoolSeeds = 0;
}

size_t gpu_seedpool_seed_count() {
  return g_seedPoolSeeds;
}

bool gpu_seedpool_generate(uint64_t baseCounter, cudaStream_t stream) {
  if (!g_dSeedPool || g_seedPoolSeeds == 0) return false;
  const int threads = 256;
  const int blocks = (int)((g_seedPoolSeeds + (size_t)threads - 1) / (size_t)threads);
  kFillSeedPool<<<blocks, threads, 0, stream>>>(g_dSeedPool, g_seedPoolSeeds, baseCounter);
  return (cudaGetLastError() == cudaSuccess);
}

// =====================================================================================
// Prefix parsing (host -> device struct)
// =====================================================================================
struct PrefixSpec {
  uint8_t bytes[32]{};
  int fullBytes = 0;
  int hasNibble = 0;
  uint8_t nibbleHi = 0;
};

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

static PrefixSpec parse_prefix(const std::string& inRaw) {
  std::string s;
  s.reserve(inRaw.size());
  for (char c : inRaw) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
    s.push_back(c);
  }

  PrefixSpec ps{};
  if (s.empty()) return ps;

  int n = (int)s.size();
  ps.fullBytes = n / 2;
  ps.hasNibble = (n % 2) ? 1 : 0;

  for (int i = 0; i < ps.fullBytes && i < 32; i++) {
    int hi = hexval(s[2 * i]);
    int lo = hexval(s[2 * i + 1]);
    if (hi < 0 || lo < 0) {
      ps.fullBytes = 0;
      ps.hasNibble = 0;
      return ps;
    }
    ps.bytes[i] = (uint8_t)((hi << 4) | lo);
  }

  if (ps.hasNibble && ps.fullBytes < 32) {
    int hi = hexval(s[2 * ps.fullBytes]);
    if (hi < 0) {
      ps.fullBytes = 0;
      ps.hasNibble = 0;
      return ps;
    }
    ps.nibbleHi = (uint8_t)hi;
  }
  return ps;
}

__device__ __forceinline__ bool match_prefix_pub(const unsigned char* pub, const PrefixSpec& ps) {
  for (int i = 0; i < ps.fullBytes; i++) {
    if (pub[i] != ps.bytes[i]) return false;
  }
  if (ps.hasNibble) {
    unsigned char want = (unsigned char)(ps.nibbleHi << 4);
    if ((pub[ps.fullBytes] & 0xF0) != want) return false;
  }
  return true;
}

// =====================================================================================
// Seed window copy (pool -> contiguous) + scan kernel
// =====================================================================================
__global__ void kCopySeedsFromPool(
  const uint8_t* pool, size_t poolSeeds,
  size_t startIndex,
  uint8_t* outSeeds, size_t batch
) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= batch) return;

  size_t poolIdx = (startIndex + i) % poolSeeds;
  const uint8_t* src = pool + poolIdx * 32;
  uint8_t* dst = outSeeds + i * 32;

#pragma unroll
  for (int k = 0; k < 32; k++) dst[k] = src[k];
}

struct DeviceResult {
  int found; // 0/1
  int index; // winner index within batch
};

__global__ void kScanPubkeys(
  const uint8_t* pubKeys, // batch*32
  size_t batch,
  PrefixSpec ps,
  DeviceResult* out
) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= batch) return;

  if (atomicAdd(&out->found, 0) != 0) return;

  const unsigned char* pub = (const unsigned char*)(pubKeys + i * 32);
  if (match_prefix_pub(pub, ps)) {
    if (atomicCAS(&out->found, 0, 1) == 0) {
      out->index = (int)i;
    }
  }
}

// =====================================================================================
// Workspace reuse + CUDA Graph (fused launch sequence)
// =====================================================================================
struct GpuWorkspace {
  size_t batch = 0;

  uint8_t* dSeeds = nullptr;  // batch*32
  uint8_t* dPub   = nullptr;  // batch*32
  uint8_t* dPriv  = nullptr;  // batch*64
  DeviceResult* dRes = nullptr;

  DeviceResult* hResPinned = nullptr; // pinned host copy of DeviceResult

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;

  cudaGraphNode_t nCopy = nullptr;
  cudaGraphNode_t nKeyp = nullptr;
  cudaGraphNode_t nScan = nullptr;
  cudaGraphNode_t nMemcpy = nullptr;

  // cached params used to update graph nodes
  size_t lastStartIndex = 0;
  PrefixSpec lastPs{};
  bool hasLastPs = false;
};

static std::mutex g_wsMu;
static GpuWorkspace g_ws;

static void ws_free(GpuWorkspace& ws) {
  if (ws.graphExec) { cudaGraphExecDestroy(ws.graphExec); ws.graphExec = nullptr; }
  if (ws.graph)     { cudaGraphDestroy(ws.graph); ws.graph = nullptr; }

  if (ws.hResPinned) { cudaFreeHost(ws.hResPinned); ws.hResPinned = nullptr; }

  if (ws.dRes)   { cudaFree(ws.dRes); ws.dRes = nullptr; }
  if (ws.dPriv)  { cudaFree(ws.dPriv); ws.dPriv = nullptr; }
  if (ws.dPub)   { cudaFree(ws.dPub); ws.dPub = nullptr; }
  if (ws.dSeeds) { cudaFree(ws.dSeeds); ws.dSeeds = nullptr; }

  ws.batch = 0;
  ws.nCopy = ws.nKeyp = ws.nScan = ws.nMemcpy = nullptr;
  ws.lastStartIndex = 0;
  ws.hasLastPs = false;
}

static bool ws_ensure(size_t batch, cudaStream_t stream) {
  if (g_ws.batch == batch && g_ws.graphExec != nullptr) return true;

  // rebuild
  ws_free(g_ws);
  g_ws.batch = batch;

  cudaError_t st;

  st = cudaMalloc((void**)&g_ws.dSeeds, batch * 32); if (st != cudaSuccess) return false;
  st = cudaMalloc((void**)&g_ws.dPub,   batch * 32); if (st != cudaSuccess) return false;
  st = cudaMalloc((void**)&g_ws.dPriv,  batch * 64); if (st != cudaSuccess) return false;
  st = cudaMalloc((void**)&g_ws.dRes, sizeof(DeviceResult)); if (st != cudaSuccess) return false;

  st = cudaMallocHost((void**)&g_ws.hResPinned, sizeof(DeviceResult)); if (st != cudaSuccess) return false;
  std::memset(g_ws.hResPinned, 0, sizeof(DeviceResult));

  // Capture graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  // node 1: memset dRes
  cudaMemsetAsync(g_ws.dRes, 0, sizeof(DeviceResult), stream);

  // node 2: copy seeds window (params updated each call)
  {
    const int threads = 256;
    const int blocks = (int)((batch + (size_t)threads - 1) / (size_t)threads);
    kCopySeedsFromPool<<<blocks, threads, 0, stream>>>(g_dSeedPool, g_seedPoolSeeds, 0, g_ws.dSeeds, batch);
  }

  // node 3: keypair batch kernel (must be launched)
  {
    const int threads = 256;
    const int blocks  = (int)((batch + (size_t)threads - 1) / (size_t)threads);
    ed25519_kernel_create_keypair_batch<<<blocks, threads, 0, stream>>>(
      (unsigned char*)g_ws.dPub,
      (unsigned char*)g_ws.dPriv,
      (const unsigned char*)g_ws.dSeeds,
      (int)batch
    );
  }

  // node 4: scan pubkeys (ps updated each call)
  {
    const int threads = 256;
    const int blocks = (int)((batch + (size_t)threads - 1) / (size_t)threads);
    PrefixSpec ps{};
    kScanPubkeys<<<blocks, threads, 0, stream>>>(g_ws.dPub, batch, ps, g_ws.dRes);
  }

  // node 5: memcpy result to pinned host
  cudaMemcpyAsync(g_ws.hResPinned, g_ws.dRes, sizeof(DeviceResult), cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &g_ws.graph);

  if (!g_ws.graph) return false;

  st = cudaGraphInstantiate(&g_ws.graphExec, g_ws.graph, nullptr, nullptr, 0);
  if (st != cudaSuccess) {
    ws_free(g_ws);
    return false;
  }

  // Find nodes we want to update (copy + scan) by walking graph nodes.
  // Easiest: just get all nodes and pick by type + index order.
  size_t numNodes = 0;
  cudaGraphGetNodes(g_ws.graph, nullptr, &numNodes);
  std::vector<cudaGraphNode_t> nodes(numNodes);
  cudaGraphGetNodes(g_ws.graph, nodes.data(), &numNodes);

  // Heuristic: in capture order, nodes will include kernel nodes in order.
  // We’ll identify the two kernel nodes we need to update by function pointer.
  for (auto n : nodes) {
    cudaGraphNodeType t{};
    cudaGraphNodeGetType(n, &t);
    if (t == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams kp{};
      cudaGraphKernelNodeGetParams(n, &kp);
      if (kp.func == (void*)kCopySeedsFromPool) g_ws.nCopy = n;
      else if (kp.func == (void*)kScanPubkeys) g_ws.nScan = n;
      else if (kp.func == (void*)ed25519_kernel_create_keypair_batch) g_ws.nKeyp = n;
    } else if (t == cudaGraphNodeTypeMemcpy) {
      // This will match the dRes->host memcpy
      g_ws.nMemcpy = n;
    }
  }

  // It's OK if nKeyp isn't found (some toolchains wrap it), but copy+scan must exist.
  if (!g_ws.nCopy || !g_ws.nScan) {
    ws_free(g_ws);
    return false;
  }

  return true;
}

static bool ws_update_params(size_t startIndex, const PrefixSpec& ps) {
  // Update copy kernel params
  {
    cudaKernelNodeParams kp{};
    cudaGraphKernelNodeGetParams(g_ws.nCopy, &kp);

    // args match: (pool, poolSeeds, startIndex, outSeeds, batch)
    void* args[] = {
      (void*)&g_dSeedPool,
      (void*)&g_seedPoolSeeds,
      (void*)&startIndex,
      (void*)&g_ws.dSeeds,
      (void*)&g_ws.batch
    };
    kp.kernelParams = args;

    cudaError_t st = cudaGraphExecKernelNodeSetParams(g_ws.graphExec, g_ws.nCopy, &kp);
    if (st != cudaSuccess) return false;
  }

  // Update scan kernel params (ps is passed by value)
  {
    cudaKernelNodeParams kp{};
    cudaGraphKernelNodeGetParams(g_ws.nScan, &kp);

    // args match: (pubKeys, batch, ps, out)
    void* args[] = {
      (void*)&g_ws.dPub,
      (void*)&g_ws.batch,
      (void*)&ps,
      (void*)&g_ws.dRes
    };
    kp.kernelParams = args;

    cudaError_t st = cudaGraphExecKernelNodeSetParams(g_ws.graphExec, g_ws.nScan, &kp);
    if (st != cudaSuccess) return false;
  }

  return true;
}

// =====================================================================================
// Public API - seedpool search using workspace + graph
// =====================================================================================
GpuHit gpu_find_pubkey_prefix_ed25519_cuda_seedpool(
  const std::string& prefix,
  size_t batch,
  size_t* startIndex,
  cudaStream_t stream
) {
  GpuHit ret{};
  ret.found = false;

  if (!g_dSeedPool || g_seedPoolSeeds == 0 || !startIndex || batch == 0) return ret;

  PrefixSpec ps = parse_prefix(prefix);

  {
    std::lock_guard<std::mutex> lk(g_wsMu);

    if (!ws_ensure(batch, stream)) {
      return ret;
    }
    if (!ws_update_params(*startIndex, ps)) {
      return ret;
    }

    // Launch the captured sequence as one "thing"
    cudaError_t st = cudaGraphLaunch(g_ws.graphExec, stream);
    if (st != cudaSuccess) return ret;

    cudaStreamSynchronize(stream);

    DeviceResult hRes = *g_ws.hResPinned;

    if (hRes.found != 0 && hRes.index >= 0 && (size_t)hRes.index < batch) {
      const size_t idx = (size_t)hRes.index;

      // Copy winning seed/pub/priv only (tiny)
      cudaMemcpyAsync(ret.seed,   g_ws.dSeeds + idx * 32, 32, cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(ret.pub,    g_ws.dPub   + idx * 32, 32, cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(ret.priv64, g_ws.dPriv  + idx * 64, 64, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);

      ret.found = true;
    }
  }

  // advance cursor (outside mutex)
  *startIndex = (*startIndex + batch) % g_seedPoolSeeds;
  return ret;
}

// =====================================================================================
// Legacy helper: temp pool == batch; fill; scan once.
// Used by benchmark / CLI fallback.
// =====================================================================================
GpuHit gpu_find_pubkey_prefix_ed25519_cuda(const std::string& prefix, size_t batch) {
  GpuHit ret{};
  ret.found = false;
  if (batch == 0) return ret;

  cudaStream_t s{};
  cudaStreamCreate(&s);

  if (!gpu_seedpool_init_seeds(batch)) {
    cudaStreamDestroy(s);
    return ret;
  }

  // Host-safe baseCounter
  const uint64_t baseCounter =
    (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();

  gpu_seedpool_generate(baseCounter, s);
  cudaStreamSynchronize(s);

  size_t idx = 0;
  ret = gpu_find_pubkey_prefix_ed25519_cuda_seedpool(prefix, batch, &idx, s);

  gpu_seedpool_free();
  cudaStreamDestroy(s);
  return ret;
}
