// Copyright 2024 yibotongxue

#ifndef INCLUDE_FILLER_CUH_
#define INCLUDE_FILLER_CUH_

#include <curand_kernel.h>
#include <thrust/fill.h>

#include <cmath>

#include "error.h"
#include "filler-parameter.hpp"
#include "tensor.cuh"
#include "utils.cuh"

namespace my_tensor {

template <typename T = float>
class Filler {
 public:
  explicit Filler(FillerParameterPtr param) : filler_parameter_(param) {}

  virtual void Fill(TensorPtr<T> tensor) = 0;

  virtual ~Filler() = default;

 protected:
  FillerParameterPtr filler_parameter_;
};  // class Filler

template <typename T = float>
class ZeroFiller final : public Filler<T> {
 public:
  explicit ZeroFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto zero_param = std::dynamic_pointer_cast<ZeroFillerParameter>(param);
    assert(zero_param.get() != nullptr);
  }

  void Fill(TensorPtr<T> tensor) override {
    T *data = tensor->GetGPUDataPtr();
    ERROR_CHECK(cudaMemset(&data, 0, tensor->GetSize() * sizeof(T)));
  }
};  // class ZeroFiller

template <typename T = float>
class ConstantFiller final : public Filler<T> {
 public:
  explicit ConstantFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto con_param = std::dynamic_pointer_cast<ConstantFillerParameter>(param);
    assert(con_param.get() != nullptr);
    val_ = T(con_param->val_);
  }

  void Fill(TensorPtr<T> tensor) override {
    auto &tensor_data = tensor->GetGPUData();
    thrust::fill(tensor_data.begin(), tensor_data.end(), val_);
  }

 private:
  const T val_;
};  // class ConstantFiller

template <typename T = float>
class XavierFiller final : public Filler<T> {
 public:
  explicit XavierFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto xparam = std::dynamic_pointer_cast<XavierFillerParameter>(param);
    assert(xparam.get() != nullptr);
    n_in_ = xparam->n_in_;
    n_out_ = xparam->n_out_;
  }

  void Fill(TensorPtr<T> tensor) override {
    throw FillerError("Unimplemention error.");
  }

 private:
  int n_in_;
  int n_out_;
};  // class XavierFiller

__global__ static void XavierFillerKernel(float *data, float limit, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    curandState state;
    curand_init(1234 + i, i, 0, &state);
    data[i] = curand_uniform(&state) * 2 * limit - limit;
  }
}

template <>
void XavierFiller<>::Fill(TensorPtr<> tensor) {
  int n = tensor->GetSize();
  float limit = std::sqrt(6.0f / (n_in_ + n_out_));
  XavierFillerKernel<<<CudaGetBlocks(n), kCudaThreadNum>>>(
      tensor->GetGPUDataPtr(), limit, n);
}

template <typename T = float>
class HeFiller final : public Filler<T> {
 public:
  explicit HeFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto hparam = std::dynamic_pointer_cast<HeFillerParameter>(param);
    assert(hparam.get() != nullptr);
    n_ = hparam->n_;
  }

  void Fill(TensorPtr<T> tensor) override {
    throw FillerError("Unimplemention error.");
  }

 private:
  int n_;
};  // class HeFiller

__global__ static void HeFillKernel(float *data, float limit, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    curandState state;
    curand_init(1234 + i, i, 0, &state);
    data[i] = curand_normal(&state) * limit;
  }
}

template <>
void HeFiller<>::Fill(TensorPtr<> tensor) {
  float *data = tensor->GetGPUDataPtr();
  int n = tensor->GetSize();
  float limit = std::sqrt(2.0f / n_);
  HeFillKernel<<<CudaGetBlocks(n), kCudaThreadNum>>>(data, limit, n);
}

}  // namespace my_tensor

#endif  // INCLUDE_FILLER_CUH_
