// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ed25519_kernel_create_pubkey_batch(
  unsigned char* public_key,
  const unsigned char* seed,
  int limit
);

#ifdef __cplusplus
}
#endif
