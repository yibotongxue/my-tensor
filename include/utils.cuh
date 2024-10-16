#ifndef INCLUDE_UTILS_CUH_
#define INCLUDE_UTILS_CUH_

#include <stdio.h>

constexpr int kCudaThreadNum = 512;

inline int CudaGetBlocks(const int N)
{
  return (N + kCudaThreadNum - 1) / kCudaThreadNum;
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#ifdef DEBUG
#define ERROR_CHECK(function)                                                   \
  do                                                                            \
  {                                                                             \
    cudaError_t error_code = function;                                          \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
      std::cerr << "CUDA error:\r\ncode = " << error_code                       \
                << ", name = " << cudaGetErrorName(error_code)                  \
                << ", description = " << cudaGetErrorString(error_code)         \
                << "\r\nfile = " << __FILE__ << ", line" << __LINE__ << "\r\n"; \
    }                                                                           \
  } while (0);
#else
#define ERROR_CHECK(function) function
#endif // DEBUG

#define CHECK_GPU_AVAILABLE                                      \
  do                                                             \
  {                                                              \
    int device_count = 0;                                        \
    cudaError_t error = cudaGetDeviceCount(&device_count);       \
    if (error != cudaSuccess || device_count == 0)               \
    {                                                            \
      std::cerr << "No CUDA campatable GPU found!" << std::endl; \
      throw std::runtime_error("No CUDA campatable GPU found!"); \
    }                                                            \
  } while (0);

#define DISABLE_LAYER_COPY(layer_name)                      \
  layer_name(const layer_name<T> &) = delete;               \
  layer_name<T> &operator=(const layer_name<T> &) = delete; \
  layer_name(layer_name<T> &&) = delete;                    \
  layer_name<T> &operator=(layer_name<T> &&) = delete;

#endif // INCLUDE_UTILS_CUH_
