#ifndef INCLUDE_UTILS_CUH_
#define INCLUDE_UTILS_CUH_

#include <stdio.h>

constexpr int kCudaThreadNum = 512;

inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadNum - 1) / kCudaThreadNum;
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define ERROR_CHECK(function) \
do { \
  cudaError_t error_code = function; \
  if (error_code != cudaSuccess) { \
    printf("CUDA error:\r\ncode = %d, name = %s, description = %s\r\nfile = %s, line%d\r\n", \
      error_code, cudaGetErrorName(error_code), \
      cudaGetErrorString(error_code), __FILE__, __LINE__); \
  } \
} while (0);

#endif  // INCLUDE_UTILS_CUH_
