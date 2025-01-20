// Copyright 2024 yibotongxue

#ifndef INCLUDE_MEMORY_UTIL_HPP_
#define INCLUDE_MEMORY_UTIL_HPP_

#include "common.hpp"
#include "utils.hpp"

namespace my_tensor {

void MyMallocCPU(void** ptr, std::size_t size);

void MyMallocGPU(void** ptr, std::size_t size);

void MyMemcpyCPU2CPU(void* dst, const void* src, std::size_t count);

void MyMemcpyCPU2GPU(void* dst, const void* src, std::size_t count);

void MyMemcpyGPU2CPU(void* dst, const void* src, std::size_t count);

void MyMemcpyGPU2GPU(void* dst, const void* src, std::size_t count);

void MyMemFreeCPU(void* ptr);

void MyMemFreeGPU(void* ptr);

template <Arithmetic T>
inline T Visit_CPU(const T* const data, std::size_t index) {
  return data[index];
}

template <Arithmetic T>
inline T Visit_GPU(const T* const data, std::size_t index) {
  T result;
  MyMemcpyGPU2CPU(&result, data + index, sizeof(T));
  return result;
}

template <Arithmetic T>
inline void Fill_CPU(T* const data, std::size_t count, T value) {
  for (std::size_t i = 0; i < count; i++) {
    data[i] = value;
  }
}

template <Arithmetic T>
void Fill_GPU(T* const data, std::size_t count, T value);

extern template void Fill_GPU(float* const data, std::size_t count,
                              float value);
}  // namespace my_tensor

#endif  // INCLUDE_MEMORY_UTIL_HPP_
