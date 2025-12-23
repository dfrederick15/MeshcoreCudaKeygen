// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Devin Frederick
#pragma once
#include <array>
#include <string>

struct KeypairResult {
  std::array<unsigned char, 32> seed{};
  std::array<unsigned char, 32> pub{};
  std::array<unsigned char, 64> priv64{}; // sha512(seed) with clamp applied to first 32 bytes
};

KeypairResult make_keypair_from_seed(const std::array<unsigned char,32>& seed);
std::string hex_of(const unsigned char* p, size_t n);
