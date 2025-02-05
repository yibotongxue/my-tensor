// Copyright 2024 yibotongxue

#ifndef INCLUDE_UTILS_HPP_
#define INCLUDE_UTILS_HPP_

#include <algorithm>
#include <iostream>
#include <ranges>  // NOLINT

template <typename T>
concept Arithmetic = std::is_arithmetic<T>::value;

constexpr int kCudaThreadNum = 512;

inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadNum - 1) / kCudaThreadNum;
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_BLOCK_LOOP(i, n) \
  for (int i = threadIdx.x; i < (n); i += blockDim.x)

#ifdef DEBUG
#define CUDA_CHECK(function)                                            \
  do {                                                                  \
    cudaError_t error_code = function;                                  \
    if (error_code != cudaSuccess) {                                    \
      std::cerr << "CUDA error:\r\ncode = " << error_code               \
                << ", name = " << cudaGetErrorName(error_code)          \
                << ", description = " << cudaGetErrorString(error_code) \
                << "\r\nfile = " << __FILE__ << ", line" << __LINE__    \
                << "\r\n";                                              \
    }                                                                   \
  } while (0);
#else
#define CUDA_CHECK(function) function
#endif  // DEBUG

#ifdef DEBUG
#define CUBLAS_CHECK(function)                                         \
  do {                                                                 \
    cublasStatus_t status = function;                                  \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
      std::cerr << "CuBLAS error:\r\nstatus = " << status              \
                << ", name = " << cublasGetStatusName(status)          \
                << ", description = " << cublasGetStatusString(status) \
                << "\r\nfile = " << __FILE__ << ", line" << __LINE__   \
                << "\r\n";                                             \
    }                                                                  \
  } while (0);
#else
#define CUBLAS_CHECK(function) function
#endif  // DEBUG

#ifdef DEBUG
#define CURAND_CHECK(function)                                          \
  do {                                                                  \
    curandStatus_t status = function;                                   \
    if (status != CURAND_STATUS_SUCCESS) {                              \
      std::cerr << "cuRAND error:\r\nstatus = " << status               \
                << "\r\nfile = " << __FILE__ << ", line = " << __LINE__ \
                << "\r\n";                                              \
    }                                                                   \
  } while (0);
#else
#define CURAND_CHECK(function) function
#endif  // DEBUG

#define CHECK_GPU_AVAILABLE                                      \
  do {                                                           \
    int device_count = 0;                                        \
    cudaError_t error = cudaGetDeviceCount(&device_count);       \
    if (error != cudaSuccess || device_count == 0) {             \
      std::cerr << "No CUDA campatable GPU found!" << std::endl; \
      throw std::runtime_error("No CUDA campatable GPU found!"); \
    }                                                            \
  } while (0);

#define DISABLE_LAYER_COPY(layer_name)                      \
  layer_name(const layer_name<T> &) = delete;               \
  layer_name<T> &operator=(const layer_name<T> &) = delete; \
  layer_name(layer_name<T> &&) = delete;                    \
  layer_name<T> &operator=(layer_name<T> &&) = delete;

#define CHECK_SAME_SHAPE(tensor, another)            \
  do {                                               \
    if (tensor->GetShape() != another->GetShape()) { \
      throw ShapeError("Shape not match.");          \
    }                                                \
  } while (0);

#define BLAS_UNIMPLEMENTION throw BlasError("Unimplemention error.");

#define IM2COL_UNIMPLEMENTION throw Im2colError("Unimplemention error.");

#define CHECK_KERNEL_SHAPE                          \
  if ((kernel_h % 2 == 0) || (kernel_w % 2 == 0)) { \
    throw Im2colError("Kernel shape not be even."); \
  }

#define RAW_PTR(vec) thrust::raw_pointer_cast(vec.data())

#define SPAN_DATA(tensor, T) \
  std::span<T>(tensor->GetCPUDataPtr(), tensor->GetSize())

#define SPAN_DIFF(tensor, T) \
  std::span<T>(tensor->GetCPUDiffPtr(), tensor->GetSize())

#define PTR_CAST(raw_ptr) thrust::device_pointer_cast(raw_ptr)

#define PRINT_DATA(tensor, T)                                  \
  std::ranges::copy(SPAN_DATA(tensor, T),                      \
                    std::ostream_iterator<T>(std::cout, " ")); \
  std::cout << std::endl;

#define PRINT_DIFF(tensor, T)                                  \
  std::ranges::copy(SPAN_DIFF(tensor, T),                      \
                    std::ostream_iterator<T>(std::cout, " ")); \
  std::cout << std::endl;

#endif  // INCLUDE_UTILS_HPP_
