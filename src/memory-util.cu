// Copyright 2024 yibotongxue

#include "memory-util.hpp"
#include "utils.hpp"

namespace my_tensor {
void MyMallocCPU(void** ptr, size_t size) { *ptr = malloc(size); }

void MyMallocGPU(void** ptr, size_t size) { CUDA_CHECK(cudaMalloc(ptr, size)); }

void MyMemcpyCPU2CPU(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void MyMemcpyCPU2GPU(void* dst, const void* src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void MyMemcpyGPU2CPU(void* dst, const void* src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void MyMemcpyGPU2GPU(void* dst, const void* src, size_t size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void MyMemFreeCPU(void* ptr) { free(ptr); }

void MyMemFreeGPU(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }
}  // namespace my_tensor
