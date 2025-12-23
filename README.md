# Meshcore CUDA Key Generator

High-performance GPU-accelerated Ed25519 key generator for MeshCore, implemented
in C++ and CUDA. This project efficiently searches for Ed25519 public keys
matching a given hexadecimal prefix using NVIDIA GPUs.

The application supports both a command-line interface and a simple GUI, and is
optimized for modern CUDA-capable hardware.

---

## Features

- GPU-accelerated Ed25519 key generation using CUDA
- Prefix-based public key searching
- CPU and GPU verification paths
- Optional VRAM-backed seed pool
- Real-time performance metrics (keys/sec, runtime, ETA)
- Windows build support with CMake and Visual Studio

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support  
  (tested on RTX 3060 Ti)

### Software
- Windows 10 or newer
- NVIDIA CUDA Toolkit (tested with CUDA 13.x)
- Visual Studio 2022 (Community, Professional, or Build Tools)
- CMake 3.24 or newer
- vcpkg (for OpenSSL)

---

## Dependencies

- **OpenSSL** (via vcpkg)
- **ed25519_cuda** (vendored under `external/`)
- Standard C++17 library

---

## Build Instructions (Windows)

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd MeshcoreCudaKeygen

2. Configure with CMake

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

3. Build

cmake --build build --config Release

After a successful build, binaries will be located in:

build\Release\

Usage
Command-Line

meshcore_cuda_keygen.exe <hex_prefix> [batch_size]

Example:

meshcore_cuda_keygen.exe 146880 500000

    hex_prefix – Hexadecimal prefix to search for

    batch_size – Number of keys per GPU batch (optional)

GUI

Run:

mesh