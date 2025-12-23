// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#include "ed25519_pubonly.cuh"

#include "sha512.cuh"
#include "ge.cuh"

__device__ __forceinline__ void ed25519_kernel_create_pubkey(unsigned char* public_key, const unsigned char* seed) {
  ge_p3 A;

  unsigned char h[64];
  sha512(seed, 32, h);

  h[0]  &= 248;
  h[31] &= 63;
  h[31] |= 64;

  ge_scalarmult_base(&A, h);
  ge_p3_tobytes(public_key, &A);
}

__global__ void ed25519_kernel_create_pubkey_batch(
  unsigned char* public_key,
  const unsigned char* seed,
  int limit
) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= limit) return;
  ed25519_kernel_create_pubkey(&public_key[idx * 32], &seed[idx * 32]);
}
