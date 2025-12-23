// src/main.cpp
#include <cstdio>
#include <string>
#include <cctype>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <cstring>

#include <cuda_runtime.h>

#include "cpu_ref.h"
#include "seed_gen.cuh"
#include "gpu_search_ed25519_cuda.h"

void launch_test();

// ---------------- Prefix matcher (byte/nibble, no hex conversion) ----------------
static bool matches_hex_prefix(const std::array<unsigned char,32>& pub, const std::string& hexPrefixRaw) {
  std::string hexPrefix;
  hexPrefix.reserve(hexPrefixRaw.size());
  for (char c : hexPrefixRaw) {
    if (!std::isspace((unsigned char)c)) hexPrefix.push_back((char)std::tolower((unsigned char)c));
  }
  if (hexPrefix.empty()) return true;

  size_t fullBytes = hexPrefix.size() / 2;
  bool hasNibble = (hexPrefix.size() % 2) != 0;

  auto hexval = [](char h)->int{
    if (h>='0' && h<='9') return h-'0';
    if (h>='a' && h<='f') return 10 + (h-'a');
    return -1;
  };

  for (size_t i = 0; i < fullBytes; i++) {
    int hi = hexval(hexPrefix[2*i]);
    int lo = hexval(hexPrefix[2*i+1]);
    if (hi<0 || lo<0) return false;
    unsigned char b = (unsigned char)((hi<<4) | lo);
    if (pub[i] != b) return false;
  }
  if (hasNibble) {
    int hi = hexval(hexPrefix[2*fullBytes]);
    if (hi<0) return false;
    unsigned char want = (unsigned char)(hi << 4);
    if ( (pub[fullBytes] & 0xF0) != want ) return false;
  }
  return true;
}

// ---------------- Parallel CPU verify over a pinned buffer ----------------
static void verify_batch_parallel(
  const uint8_t* h_seeds, size_t count,
  const std::string& prefix,
  unsigned int threads,
  std::atomic<bool>& found,
  std::mutex& foundMu,
  KeypairResult& foundKp,
  std::atomic<uint64_t>& totalTries
) {
  std::vector<std::thread> workers;
  workers.reserve(threads);

  auto worker = [&](size_t startIdx, size_t endIdx) {
    for (size_t i = startIdx; i < endIdx; i++) {
      if (found.load(std::memory_order_relaxed)) return;

      std::array<unsigned char,32> seed{};
      const uint8_t* p = &h_seeds[i * 32];
      for (int k = 0; k < 32; k++) seed[k] = (unsigned char)p[k];

      auto kp = make_keypair_from_seed(seed);
      uint64_t triesNow = totalTries.fetch_add(1, std::memory_order_relaxed) + 1;

      if (matches_hex_prefix(kp.pub, prefix)) {
        bool expected = false;
        if (found.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> lg(foundMu);
          foundKp = kp;
          std::printf("\nFOUND after %llu tries\n", (unsigned long long)triesNow);
        }
        return;
      }
    }
  };

  size_t chunk = (count + threads - 1) / threads;
  for (unsigned int t = 0; t < threads; t++) {
    size_t s = (size_t)t * chunk;
    size_t e = s + chunk;
    if (s >= count) break;
    if (e > count) e = count;
    workers.emplace_back(worker, s, e);
  }

  for (auto& th : workers) th.join();
}

static void print_usage(const char* exe) {
  std::printf(
    "Usage:\n"
    "  %s <prefixHex> <batch> [cpuThreads]           (Mode A: GPU seedgen + CPU verify)\n"
    "  %s --gpu <prefixHex> <batch>                 (Mode C: GPU pubkey+scan + GPU priv64)\n"
    "\n"
    "Examples:\n"
    "  %s 146880 500000\n"
    "  %s --gpu 146880 500000\n",
    exe, exe, exe, exe
  );
}

int main(int argc, char** argv) {
  launch_test();

  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  bool useGpuPubScan = false;
  int argi = 1;

  if (std::string(argv[argi]) == "--gpu") {
    useGpuPubScan = true;
    argi++;
    if (argc - argi < 2) {
      print_usage(argv[0]);
      return 1;
    }
  }

  // ---------------- Mode C: GPU pubkey generation + GPU scan + GPU priv64 ----------------
  if (useGpuPubScan) {
    std::string prefix = argv[argi++];
    int batch = std::stoi(argv[argi++]);
    if (batch <= 0) {
      std::printf("batch must be > 0\n");
      return 1;
    }

    std::printf("Mode C (--gpu): GPU pubkey+scan for prefix '%s' with batch=%d\n", prefix.c_str(), batch);

    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto lastPrint = t0;

    while (true) {
      GpuHit hit;
      try {
        hit = gpu_find_pubkey_prefix_ed25519_cuda(prefix, batch);
      } catch (const std::exception& e) {
        std::printf("GPU search error: %s\n", e.what());
        return 1;
      }

      total += (uint64_t)batch;

      auto now = std::chrono::steady_clock::now();
      if (now - lastPrint >= std::chrono::seconds(2)) {
        lastPrint = now;
        double secs = std::chrono::duration<double>(now - t0).count();
        double rate = (secs > 0.0) ? (total / secs) : 0.0;
        std::printf("...%llu tries (%.0f keys/sec)\n", (unsigned long long)total, rate);
      }

      if (hit.found) {
        // Correctness check: CPU recompute pub from seed and compare to GPU pub
        {
          std::array<unsigned char,32> seedArr{};
          for (int i = 0; i < 32; i++) seedArr[i] = hit.seed[i];
          auto cpuKp = make_keypair_from_seed(seedArr);
          if (memcmp(cpuKp.pub.data(), hit.pub, 32) != 0) {
            std::printf("ERROR: GPU pubkey does not match CPU pubkey for the winning seed.\n");
            std::printf("GPU pub: %s\n", hex_of(hit.pub, 32).c_str());
            std::printf("CPU pub: %s\n", hex_of(cpuKp.pub.data(), 32).c_str());
            return 1;
          }
        }

        std::string pubHex  = hex_of(hit.pub, 32);
        std::string privHex = hex_of(hit.priv64, 64);

        std::printf("\nFOUND after %llu tries\n", (unsigned long long)total);
        std::printf("{\n  \"public_key\": \"%s\",\n  \"private_key\": \"%s\"\n}\n", pubHex.c_str(), privHex.c_str());
        return 0;
      }
    }
  }

  // ---------------- Mode A (default): pipelined GPU seedgen + CPU verify ----------------
  std::string prefix = (argc >= 2) ? argv[1] : "";
  const size_t BATCH = (argc >= 3) ? (size_t)std::stoull(argv[2]) : (size_t)200000;

  unsigned int hw = std::thread::hardware_concurrency();
  unsigned int THREADS = (argc >= 4) ? (unsigned)std::stoul(argv[3]) : (hw > 1 ? (hw - 1) : 1);
  if (THREADS < 1) THREADS = 1;

  std::printf("Mode A: Searching for pubkey hex prefix: '%s'\n", prefix.c_str());
  std::printf("Batch=%llu seeds, CPU threads=%u\n", (unsigned long long)BATCH, THREADS);

  // --- Double-buffer: 2 streams, 2 device buffers, 2 pinned host buffers ---
  cudaStream_t s0{}, s1{};
  cudaStreamCreate(&s0);
  cudaStreamCreate(&s1);

  uint8_t *d0=nullptr, *d1=nullptr;
  cudaMalloc((void**)&d0, BATCH * 32);
  cudaMalloc((void**)&d1, BATCH * 32);

  uint8_t *h0=nullptr, *h1=nullptr;
  cudaMallocHost((void**)&h0, BATCH * 32); // pinned
  cudaMallocHost((void**)&h1, BATCH * 32); // pinned

  std::atomic<bool> found(false);
  std::mutex foundMu;
  KeypairResult foundKp{};
  std::atomic<uint64_t> totalTries(0);

  auto t0 = std::chrono::steady_clock::now();
  auto lastPrint = t0;

  uint64_t baseCounter = 0x12345678ABCDEF00ULL;

  // Kick off first batch on stream 0 (d0->h0)
  gpu_generate_seeds(d0, BATCH, baseCounter, s0);
  cudaMemcpyAsync(h0, d0, BATCH * 32, cudaMemcpyDeviceToHost, s0);
  baseCounter += (uint64_t)BATCH;

  bool nextIs1 = true; // next launch uses buffer 1; while we verify 0 after sync

  while (!found.load(std::memory_order_relaxed)) {
    if (nextIs1) {
      // Launch next batch into (d1->h1) asynchronously
      gpu_generate_seeds(d1, BATCH, baseCounter, s1);
      cudaMemcpyAsync(h1, d1, BATCH * 32, cudaMemcpyDeviceToHost, s1);
      baseCounter += (uint64_t)BATCH;

      // Wait for previous copy (stream 0), then verify h0
      cudaStreamSynchronize(s0);
      verify_batch_parallel(h0, BATCH, prefix, THREADS, found, foundMu, foundKp, totalTries);
    } else {
      gpu_generate_seeds(d0, BATCH, baseCounter, s0);
      cudaMemcpyAsync(h0, d0, BATCH * 32, cudaMemcpyDeviceToHost, s0);
      baseCounter += (uint64_t)BATCH;

      cudaStreamSynchronize(s1);
      verify_batch_parallel(h1, BATCH, prefix, THREADS, found, foundMu, foundKp, totalTries);
    }

    nextIs1 = !nextIs1;

    auto now = std::chrono::steady_clock::now();
    if (now - lastPrint >= std::chrono::seconds(2)) {
      lastPrint = now;
      double secs = std::chrono::duration<double>(now - t0).count();
      uint64_t tries = totalTries.load(std::memory_order_relaxed);
      double rate = (secs > 0.0) ? (tries / secs) : 0.0;
      std::printf("...%llu tries (%.0f keys/sec)\n", (unsigned long long)tries, rate);
    }
  }

  // If we found mid-verify, we may still have an in-flight copy; that's fine—just print.
  {
    std::lock_guard<std::mutex> lg(foundMu);
    std::string pubHex  = hex_of(foundKp.pub.data(), foundKp.pub.size());
    std::string privHex = hex_of(foundKp.priv64.data(), foundKp.priv64.size());
    std::printf("{\n  \"public_key\": \"%s\",\n  \"private_key\": \"%s\"\n}\n", pubHex.c_str(), privHex.c_str());
  }

  // Cleanup
  cudaFreeHost(h0);
  cudaFreeHost(h1);
  cudaFree(d0);
  cudaFree(d1);
  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);

  return 0;
}
