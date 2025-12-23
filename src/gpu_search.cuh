// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#pragma once
#include <cstdint>
#include <cstddef>

struct GpuWinner {
  uint32_t found;      // 0/1
  uint32_t index;      // thread/global candidate index
  uint8_t  seed[32];
  uint8_t  pub[32];
  uint8_t  priv64[64];
};

// prefix is passed as bytes + optional nibble mask
struct PrefixSpec {
  uint8_t bytes[32];
  uint8_t fullBytes;   // number of full bytes to compare
  uint8_t hasNibble;   // 0/1
  uint8_t nibbleHigh;  // high nibble value (0..15) if hasNibble
};

bool gpu_search_stub(const PrefixSpec& prefix, GpuWinner& outWinner);
