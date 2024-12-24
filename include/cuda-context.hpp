// Copyright 2024 yibotongxue

#ifndef INCLUDE_CUDA_CONTEXT_HPP_
#define INCLUDE_CUDA_CONTEXT_HPP_

#include <cublas_v2.h>
#include <curand.h>

#include <memory>
#include <random>

namespace my_tensor {
class CudaContext {
 public:
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  static CudaContext& Get();
  inline static cublasHandle_t& cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t& curand_generator() {
    return Get().curand_generator_;
  }

 protected:
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;

 private:
  CudaContext();
};  // class CudaContext
}  // namespace my_tensor

#endif  // INCLUDE_CUDA_CONTEXT_HPP_
