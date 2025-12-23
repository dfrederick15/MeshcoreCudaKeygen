// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

bool seedpool_init_bytes(size_t poolBytes);
bool seedpool_generate(uint64_t baseCounter, cudaStream_t stream);

size_t seedpool_seed_count();
const uint8_t* seedpool_device_ptr();

void seedpool_free();

#ifdef __cplusplus
}
#endif
