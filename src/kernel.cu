#include <cuda_runtime.h>

__global__ void k_test() {}

void launch_test() {
  k_test<<<1,1>>>();
}
