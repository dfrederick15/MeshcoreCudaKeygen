#include "cpu_ref.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <ed25519.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// ======================================================
// Ed25519 scalar clamp (RFC 8032)
// ======================================================
static void clamp_ed25519_scalar(unsigned char s[32]) {
  s[0]  &= 248;
  s[31] &= 127;
  s[31] |= 64;
}

// ======================================================
// Hex encoder
// ======================================================
std::string hex_of(const unsigned char* p, size_t n) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < n; i++) {
    oss << std::setw(2) << (unsigned)p[i];
  }
  return oss.str();
}

// ======================================================
// CPU reference keypair generation
//  - seed: 32 bytes
//  - priv64: SHA512(seed), clamped
//  - pub: Ed25519 basepoint multiply (from seed)
// ======================================================
KeypairResult make_keypair_from_seed(
  const std::array<unsigned char,32>& seed
) {
  KeypairResult r{};
  r.seed = seed;

  // --------------------------------------------------
  // SHA-512(seed) -> priv64
  // --------------------------------------------------
  unsigned int outLen = 0;

  EVP_MD_CTX* md = EVP_MD_CTX_new();
  if (!md) {
    throw std::runtime_error("EVP_MD_CTX_new failed");
  }

  if (EVP_DigestInit_ex(md, EVP_sha512(), nullptr) != 1 ||
      EVP_DigestUpdate(md, seed.data(), seed.size()) != 1 ||
      EVP_DigestFinal_ex(md, r.priv64.data(), &outLen) != 1) {
    EVP_MD_CTX_free(md);
    throw std::runtime_error("SHA-512 failed");
  }

  EVP_MD_CTX_free(md);

  if (outLen != 64) {
    throw std::runtime_error("SHA-512 returned wrong length");
  }

  // --------------------------------------------------
  // Clamp expanded secret (first 32 bytes)
  // --------------------------------------------------
  clamp_ed25519_scalar(r.priv64.data());

  // --------------------------------------------------
  // Public key from seed (orlp/ed25519)
  // --------------------------------------------------
  unsigned char tmp_sk[64]; // library private format (seed || pub)
  ed25519_create_keypair(
    r.pub.data(),     // out: public key (32)
    tmp_sk,           // out: private key (64, unused)
    seed.data()       // in: seed (32)
  );

  return r;
}
