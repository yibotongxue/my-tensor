// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_HPP_
#define INCLUDE_LAYER_HPP_

#include <spdlog/spdlog.h>

#include <memory>
#include <vector>

#include "common.hpp"
#include "layer-parameter.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace my_tensor {
// Layer abstract class.
template <typename T = float>
class Layer {
 public:
  // Default constructor.
  explicit Layer(LayerParameterPtr param) : layer_param_(param) {}

  void SetUp(const std::vector<TensorPtr<T>>& bottom,
             const std::vector<TensorPtr<T>>& top) {
    CheckTensorCount(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    spdlog::info("Layer {} setup done.", layer_param_->name_);
  }

  virtual void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                const std::vector<TensorPtr<T>>& top) const = 0;
  virtual void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) {}
  virtual void Reshape(const std::vector<TensorPtr<T>>& bottom,
                       const std::vector<TensorPtr<T>>& top) const = 0;

  virtual std::vector<TensorPtr<T>> GetLearnableParameters() { return {}; }

  // The layer can not be copied or moved.
  DISABLE_LAYER_COPY(Layer)

  virtual ~Layer() = default;

  void Forward(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) {
    if (MyTensorContext::on_cpu()) {
      ForwardCPU(bottom, top);
    } else {
      ForwardGPU(bottom, top);
    }
  }

  void Backward(const std::vector<TensorPtr<T>>& top,
                const std::vector<TensorPtr<T>>& bottom) {
    if (MyTensorContext::on_cpu()) {
      BackwardCPU(top, bottom);
    } else {
      BackwardGPU(top, bottom);
    }
  }

  // Pure virtual methods, forward and backward.
  // CPU
  virtual void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) = 0;
  virtual void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                           const std::vector<TensorPtr<T>>& bottom) = 0;
  // GPU
  virtual void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) = 0;
  virtual void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                           const std::vector<TensorPtr<T>>& bottom) = 0;

  void SetTrain() { is_train_ = true; }
  void SetTest() { is_train_ = false; }

 protected:
  LayerParameterPtr layer_param_;
  bool is_train_ = true;
};

// Layer pointer.
template <typename T = float>
using LayerPtr = std::shared_ptr<my_tensor::Layer<T>>;

extern template class my_tensor::Layer<>;
}  // namespace my_tensor

#endif  // INCLUDE_LAYER_HPP_
