// Copyright 2024 yibotongxue

#ifndef INCLUDE_FILLER_HPP_
#define INCLUDE_FILLER_HPP_

#ifndef CPU_ONLY
#include <curand_kernel.h>
#include <thrust/fill.h>
#endif  // CPU_ONLY

#include <cassert>
#include <cmath>
#include <memory>

#include "common.hpp"
#include "error.hpp"
#include "filler-parameter.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace my_tensor {

template <typename T = float>
class Filler {
 public:
  explicit Filler(FillerParameterPtr param) : filler_parameter_(param) {}

  virtual void Fill(TensorPtr<T> tensor) {
    if (MyTensorContext::on_cpu()) {
      FillCPU(tensor);
    } else {
      FillGPU(tensor);
    }
  }

  virtual ~Filler() = default;

 protected:
  FillerParameterPtr filler_parameter_;

  virtual void FillCPU(TensorPtr<T> tensor) = 0;
  virtual void FillGPU(TensorPtr<T> tensor) = 0;
};  // class Filler

template <typename T = float>
class ZeroFiller final : public Filler<T> {
 public:
  explicit ZeroFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto zero_param = std::dynamic_pointer_cast<ZeroFillerParameter>(param);
    assert(zero_param.get() != nullptr);
  }

 private:
  void FillCPU(TensorPtr<T> tensor) override;
  void FillGPU(TensorPtr<T> tensor) override;
};  // class ZeroFiller

template <typename T = float>
class ConstantFiller final : public Filler<T> {
 public:
  explicit ConstantFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto con_param = std::dynamic_pointer_cast<ConstantFillerParameter>(param);
    assert(con_param.get() != nullptr);
    val_ = T(con_param->val_);
  }

 private:
  void FillCPU(TensorPtr<T> tensor) override;
  void FillGPU(TensorPtr<T> tensor) override;

 private:
  T val_;
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

 private:
  void FillCPU(TensorPtr<T> tensor) override {
    throw FillerError("Unimplemention error.");
  }
  void FillGPU(TensorPtr<T> tensor) override {
    throw FillerError("Unimplemention error.");
  }

 private:
  int n_in_;
  int n_out_;
};  // class XavierFiller

template <typename T = float>
class HeFiller final : public Filler<T> {
 public:
  explicit HeFiller(FillerParameterPtr param) : Filler<T>(param) {
    auto hparam = std::dynamic_pointer_cast<HeFillerParameter>(param);
    assert(hparam.get() != nullptr);
    n_ = hparam->n_;
  }

 private:
  void FillCPU(TensorPtr<T> tensor) override {
    throw FillerError("Unimplemention error.");
  }
  void FillGPU(TensorPtr<T> tensor) override {
    throw FillerError("Unimplemention error.");
  }

 private:
  int n_;
};  // class HeFiller

template <typename T = float>
using FillerPtr = std::shared_ptr<Filler<T>>;

template <typename T = float>
inline FillerPtr<T> CreateFiller(FillerParameterPtr param) {
  auto mode = param->init_mode_;
  switch (mode) {
    case InitMode::kZero:
      return std::make_shared<ZeroFiller<T>>(param);
    case InitMode::kConstant:
      return std::make_shared<ConstantFiller<T>>(param);
    case InitMode::kXavier:
      return std::make_shared<XavierFiller<T>>(param);
    case InitMode::kHe:
      return std::make_shared<HeFiller<T>>(param);
    default:
      throw FillerError("Unimplemention error.");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_FILLER_HPP_
