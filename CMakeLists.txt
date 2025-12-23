cmake_minimum_required(VERSION 3.24)

set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "" FORCE)
set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "" FORCE)

project(meshcore_cuda_keygen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenSSL REQUIRED)

# ------------------------------------------------------------
# Shared ed25519_cuda sources (GPU only)
# ------------------------------------------------------------
set(ED25519_CUDA_SOURCES
  external/ed25519_cuda/add_scalar.cu
  external/ed25519_cuda/fe.cu
  external/ed25519_cuda/ge.cu
  external/ed25519_cuda/keccakf.cu
  external/ed25519_cuda/keypair.cu
  external/ed25519_cuda/key_exchange.cu
  external/ed25519_cuda/onion.cu
  external/ed25519_cuda/sc.cu
  external/ed25519_cuda/seed.cu
  external/ed25519_cuda/sha512.cu
  external/ed25519_cuda/sign.cu
  external/ed25519_cuda/util.cu
  external/ed25519_cuda/verify.cu
)

# ------------------------------------------------------------
# Console EXE
# ------------------------------------------------------------
add_executable(meshcore_cuda_keygen
  src/main.cpp
  src/kernel.cu
  src/seed_gen.cu
  src/cpu_ref.cpp
  src/gpu_search_ed25519_cuda.cu
  src/ed25519_pubonly.cu
  src/seedpool.cu
  src/gpu_search_ed25519_cuda.cu
  src/gpu_search_ed25519_cuda.h
  ${ED25519_CUDA_SOURCES}
)

target_include_directories(meshcore_cuda_keygen PRIVATE
  src
  external/ed25519_cuda
)

target_link_libraries(meshcore_cuda_keygen PRIVATE
  OpenSSL::Crypto
)

set_target_properties(meshcore_cuda_keygen PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------
# GUI EXE (Windows subsystem)
# ------------------------------------------------------------
add_executable(meshcore_cuda_keygen_gui WIN32
  src/gui_main.cpp
  src/cpu_ref.cpp
  src/gpu_search_ed25519_cuda.cu
  ${ED25519_CUDA_SOURCES}
)

target_include_directories(meshcore_cuda_keygen_gui PRIVATE
  src
  external/ed25519_cuda
)

target_link_libraries(meshcore_cuda_keygen_gui PRIVATE
  OpenSSL::Crypto
  comctl32
  bcrypt
)

set_target_properties(meshcore_cuda_keygen_gui PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

if (MSVC)
  target_link_options(meshcore_cuda_keygen_gui PRIVATE "/SUBSYSTEM:WINDOWS")
endif()

target_include_directories(meshcore_cuda_keygen_gui PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/ed25519_cuda
)

target_sources(meshcore_cuda_keygen_gui PRIVATE
  src/ed25519_pubonly.cu
  src/seedpool.cu
)
