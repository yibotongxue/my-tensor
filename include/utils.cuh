#ifndef MYTENSOR_INCLUDE_UTILS_H_
#define MYTENSOR_INCLUDE_UTILS_H_

#include <iostream>

constexpr int kCudaThreadNum = 512;

inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadNum - 1) / kCudaThreadNum;
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        std::cerr << "CUDA error:\r\ncode=" << error_code
          << ", name=" << cudaGetErrorName(error_code)
          << ", description=" << cudaGetErrorString(error_code)
          << "\r\nfile=" << filename
          << ", line" << lineNumber << "\r\n";
        return error_code;
    }
    return error_code;
}

#define ERROR_CHECK(function) \
do { \
  cudaError_t error_code = function; \
  if (error_code != cudaSuccess) { \
        std::cerr << "CUDA error:\r\ncode=" << error_code \
          << ", name=" << cudaGetErrorName(error_code) \
          << ", description=" << cudaGetErrorString(error_code) \
          << "\r\nfile=" << __FILE__ \
          << ", line" << __LINE__ << "\r\n"; \
  } \
} while(0);

#endif // MYTENSOR_INCLUDE_UTILS_H_
