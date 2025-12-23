# MeshCore CUDA Keygen - AI Assistant Instructions

## Project Overview

**Purpose**: CUDA-accelerated Ed25519 public key search utility that finds keys matching a given hex prefix.

**Core Architecture**: GPU-CPU hybrid pipeline where GPUs generate random seeds in bulk, CPUs verify against target prefix via OpenSSL/ed25519 crypto.

## Key Components & Data Flow

### 1. GPU Seed Generation (`src/seed_gen.cu`)
- **Function**: `gpu_generate_seeds()` → launches `k_generate_seeds` kernel
- **Purpose**: Generate 32-byte random seeds deterministically via splitmix64 PRNG
- **Parameters**: 
  - `d_seeds`: pre-allocated device memory (count * 32 bytes)
  - `count`: batch size (typically 200k)
  - `baseCounter`: deterministic offset ensuring unique seed sequences across batches
- **Grid Config**: 256 threads/block, dynamic block count based on `count`
- **Key Detail**: Each CUDA thread generates exactly one 32-byte seed using 4 × splitmix64 calls (little-endian byte packing)

### 2. CPU Keypair Verification (`src/cpu_ref.cpp`)
- **Function**: `make_keypair_from_seed()` - the performance bottleneck
- **Workflow**:
  1. SHA-512(seed) → 64-byte expanded secret
  2. Ed25519 scalar clamp per RFC 8032 (bits 0,1,2,7 manipulation)
  3. Ed25519 public key derivation from clamped secret (via orlp/ed25519 library)
- **Dependencies**: OpenSSL (EVP SHA-512), ed25519.lib (orlp/ed25519 fork)
- **Critical**: The ed25519_create_keypair expects 32-byte seed input, not expanded secret

### 3. Main Search Loop (`src/main.cpp`)
- **Algorithm**: Batch → GPU → CPU parallel verification → repeat until found
- **Concurrency Model**:
  - GPU fills batch while CPU works on previous batch (not async - synchronous cudaDeviceSynchronize)
  - CPU spawns `THREADS` worker threads (default: hardware_concurrency - 1)
  - Each worker processes assigned chunk via `matches_hex_prefix()` prefix matcher
- **CLI Args**: `[prefix] [batch_size] [thread_count]`
  - Example: `meshcore_cuda_keygen "1234ab" 500000 8`
- **Output**: JSON with found public/private keys on match

### 4. Prefix Matching (`src/main.cpp::matches_hex_prefix()`)
- Supports partial nibbles (e.g., "1a3" = 1 full byte + upper nibble)
- Case-insensitive hex parsing with whitespace tolerance
- Byte-by-byte comparison against public key buffer

## Build & Dependencies

### CMake Configuration
- **Toolchain**: vcpkg (hardcoded path: `C:/Users/devin/vcpkg/`)
- **Key Settings**:
  - CUDA Separable Compilation: ON (enables linking of .cu files)
  - CUDA fast math: enabled (`--use_fast_math`), lineinfo debug symbols added
  - C++17 standard for both CXX and CUDA
- **Dependencies**:
  - OpenSSL (vcpkg: for EVP/SHA512)
  - ed25519 (orlp/ed25519, installed via vcpkg)
- **Post-Build**: Copies OpenSSL DLLs (libcrypto-3-x64.dll, libssl-3-x64.dll) to output

### Build Commands
```bash
# Generate Visual Studio solution
cmake -G "Visual Studio 17 2022" -B build

# Build Release (optimized)
cmake --build build --config Release

# Build Debug (with symbols)
cmake --build build --config Debug
```

## Performance & Optimization Patterns

### CUDA-Specific
- **Splitmix64 PRNG**: Stateful counter-based RNG (not cryptographic, deterministic by thread)
- **Memory Layout**: Seeds stored consecutively in device memory for cache efficiency
- **Synchronization**: `cudaDeviceSynchronize()` after each batch (blocking, sequential)

### CPU-Specific
- **Atomic Operations**: `std::atomic<uint64_t> totalTries` for lock-free counter increments
- **Early Exit**: Worker threads check `found.load()` flag frequently (memory_order_relaxed)
- **Chunk Distribution**: Work chunked evenly across threads to minimize lock contention

### Why This Hybrid Approach
- GPU excels at bulk, low-complexity seed generation (high throughput, low memory footprint per seed)
- CPU required for cryptographic operations (Ed25519/SHA512 not easily parallelized on GPU, library constraints)
- Minimizes host-device transfers: only 32-byte results needed, not intermediate crypto states

## Development Workflows

### Adding Features
1. **GPU-side changes** (seed generation): Modify `k_generate_seeds` kernel in `seed_gen.cu`, test with launch_test() stub
2. **CPU crypto changes**: Update `make_keypair_from_seed()` in `cpu_ref.cpp`, maintain SHA-512→clamp→ed25519 pipeline
3. **CLI/main loop**: Edit argument parsing or batch strategy in `main.cpp`

### Debugging CUDA Code
- Add `cudaMemcpy` call to copy device results to host for inspection
- Use `-lineinfo` flag (already enabled) with cuda-gdb or NVIDIA Nsight for step debugging
- Check `cudaGetErrorString()` calls - all CUDA ops already wrapped

### Debugging CPU Code
- Prefix matcher is complex: trace hex parsing with test cases (whitespace, nibbles)
- Worker thread synchronization via mutex `foundMu` ensures clean output

## Critical Implementation Details

### Memory Management
- **Device**: `cudaMalloc` once per batch at startup, `cudaFree` on exit
- **Host**: `std::vector<uint8_t>` for seed staging, auto-freed
- **No async copies**: Synchronous `cudaMemcpy` (Device→Host) waits for GPU completion

### Determinism & Reproducibility
- **Seed generation**: Fully deterministic via `baseCounter` offset per batch
- **Prefix matching**: Exact byte comparison (no floating-point inexactness)
- **Recommendation**: Same CLI args + GPU will reproduce same key order, but NOT same keys across different GPU architectures

### OpenSSL/ed25519 Integration
- **EVP Context Management**: `EVP_MD_CTX_new/free` required (no auto-cleanup)
- **ed25519_create_keypair**: Expects 32-byte seed; internally computes SHA512 + clamp (redundant with CPU ref, but library design)
- **Vcpkg Paths**: Hardcoded in CMakeLists.txt - update if vcpkg location changes

## Common Patterns & Conventions

- **Error Handling**: Early returns with error strings to stderr, no exceptions except CPU_ref.cpp
- **Hex Encoding**: `hex_of()` utility in cpu_ref.cpp, ostringstream with iomanip
- **Threading**: Thread pool created per batch (not reused), joined before next batch
- **CUDA Sync**: Always check `cudaError_t` after async operations, no CUDA device queries

## File Organization

```
src/
  main.cpp         → CLI, batch loop, prefix matching
  seed_gen.cu      → CUDA kernel for seed generation
  seed_gen.cuh     → Public GPU API
  cpu_ref.cpp/h    → Ed25519 keypair generation (CPU reference)
  kernel.cu        → Stub for CUDA device visibility
```

## Next Steps for Extension

- **Algorithm Improvements**: Async GPU/CPU pipeline (requires ring buffer + streams)
- **Multi-GPU Support**: Distribute batches across multiple CUDA devices
- **Better PRNG**: Consider cryptographic PRNG (e.g., ChaCha20) for non-deterministic mode
- **Benchmarking**: Profile GPU vs CPU bottleneck (likely GPU seed gen at 256 threads/block)
