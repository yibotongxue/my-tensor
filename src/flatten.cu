// Copyright 2024 yibotongxue

#include <thrust/copy.h>

#include <vector>

#include "error.h"
#include "flatten.cuh"

namespace my_tensor {

template <typename T>
void Flatten<T>::CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                  const std::vector<TensorPtr<T>>& top) const {
  if (bottom.size() != 1) {
    throw FlattenError(
        "The bottom of flatten layer should have only one tensor.");
  }
  if (top.size() != 1) {
    throw FlattenError("The top of flatten layer should have only one tensor.");
  }
}

template <typename T>
void Flatten<T>::LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (bottom[0]->GetShape().size() < 2) {
    throw FlattenError(
        "The bottom of flatten layer should be an at least two dimension "
        "tensor.");
  }
  bottom_shape_ = bottom[0]->GetShape();
  top_shape_ = {
      bottom[0]->GetShape()[0],
      static_cast<int>(bottom[0]->GetSize() / bottom[0]->GetShape()[0])};
  auto param = std::dynamic_pointer_cast<FlattenParameter>(this->layer_param_);
  assert(param.get() != nullptr);
  inplace_ = param->inplace_;
}

template <typename T>
void Flatten<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  int expect_size = bottom[0]->GetSize();
  if (top[0]->GetSize() != expect_size) {
    throw FlattenError("The top size not match flatten layer.");
  }
  if (inplace_) {
    assert(bottom[0].get() == top[0].get());
  } else {
    top[0]->Reshape(top_shape_);
  }
}

template <typename T>
void Flatten<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (inplace_) {
    bottom[0]->Reshape(top_shape_);
  } else {
    thrust::copy(bottom[0]->GetCPUData().begin(), bottom[0]->GetCPUData().end(),
                 top[0]->GetCPUData().begin());
  }
}

template <typename T>
void Flatten<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  if (inplace_) {
    bottom[0]->Reshape(bottom_shape_);
  } else {
    thrust::copy(top[0]->GetCPUDiff().begin(), top[0]->GetCPUDiff().end(),
                 bottom[0]->GetCPUDiff().begin());
  }
}

template <typename T>
void Flatten<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (inplace_) {
    bottom[0]->Reshape(top_shape_);
  } else {
    thrust::copy(bottom[0]->GetGPUData().begin(), bottom[0]->GetGPUData().end(),
                 top[0]->GetGPUData().begin());
  }
}

template <typename T>
void Flatten<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  if (inplace_) {
    bottom[0]->Reshape(bottom_shape_);
  } else {
    thrust::copy(top[0]->GetGPUDiff().begin(), top[0]->GetGPUDiff().end(),
                 bottom[0]->GetGPUDiff().begin());
  }
}

template class Flatten<>;

}  // namespace my_tensor
