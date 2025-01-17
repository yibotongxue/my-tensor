// Copyright 2025 yibotongxue

#include "batchnorm.hpp"

#include <memory>
#include <vector>

#include "error.hpp"

namespace my_tensor {

template <typename T>
void BatchNorm<T>::CheckTensorCount(
    const std::vector<TensorPtr<T>>& bottom,
    const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw BatchNormError(
        "The bottom of batchnorm layer should have one tensor.");
  }
  if (top.size() != 1) {
    throw BatchNormError("The top of batchnorm layer should have one tensor.");
  }
}

template <typename T>
void BatchNorm<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                           const std::vector<TensorPtr<T>>& top) const {
  top[0]->Resize(bottom[0]->GetShape());
}

template <typename T>
void BatchNorm<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                              const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() != 4) {
    throw BatchNormError(
        "Input of batchnorm layer should be four dimesion tensor.");
  }
  std::shared_ptr<BatchNormParameter> param =
      std::dynamic_pointer_cast<BatchNormParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  channels_ = param->channels_;
  std::vector<int> shape = {1, channels_, 1, 1};
  gama_ = std::make_shared<Tensor<T>>(shape);
  beta_ = std::make_shared<Tensor<T>>(shape);
  mean_cache_ = std::make_shared<Tensor<T>>(shape);
  variance_cache_ = std::make_shared<Tensor<T>>(shape);
  batch_cnt_ = 0;
}

}  // namespace my_tensor
