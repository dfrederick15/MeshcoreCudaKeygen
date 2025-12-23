#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Generates `count` seeds (32 bytes each) into device memory `d_seeds`.
// Launches into `stream` (non-blocking).
void gpu_generate_seeds(uint8_t* d_seeds, size_t count, uint64_t baseCounter, cudaStream_t stream);
