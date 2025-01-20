// Copyright 2024 yibotongxue

#include "flatten.hpp"

#include <thrust/copy.h>

#include <vector>

#include "error.hpp"
#include "memory-util.hpp"

namespace my_tensor {

template <Arithmetic T>
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

template <Arithmetic T>
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

template <Arithmetic T>
void Flatten<T>::Reshape(const std::vector<TensorPtr<T>>& bottom,
                         const std::vector<TensorPtr<T>>& top) const {
  if (inplace_) {
    assert(bottom[0].get() == top[0].get());
  } else {
    top[0]->Resize(top_shape_);
  }
}

template <Arithmetic T>
void Flatten<T>::ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (inplace_) {
    bottom[0]->Reshape(top_shape_);
  } else {
    MyMemcpyCPU2CPU(top[0]->GetCPUDataPtr(), bottom[0]->GetCPUDataPtr(),
                    bottom[0]->GetSize() * sizeof(T));
  }
}

template <Arithmetic T>
void Flatten<T>::BackwardCPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  if (inplace_) {
    bottom[0]->Reshape(bottom_shape_);
  } else {
    MyMemcpyCPU2CPU(bottom[0]->GetCPUDiffPtr(), top[0]->GetCPUDiffPtr(),
                    top[0]->GetSize() * sizeof(T));
  }
}

template <Arithmetic T>
void Flatten<T>::ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                            const std::vector<TensorPtr<T>>& top) {
  if (inplace_) {
    bottom[0]->Reshape(top_shape_);
  } else {
    MyMemcpyGPU2GPU(top[0]->GetGPUDataPtr(), bottom[0]->GetGPUDataPtr(),
                    bottom[0]->GetSize() * sizeof(T));
  }
}

template <Arithmetic T>
void Flatten<T>::BackwardGPU(const std::vector<TensorPtr<T>>& top,
                             const std::vector<TensorPtr<T>>& bottom) {
  if (inplace_) {
    bottom[0]->Reshape(bottom_shape_);
  } else {
    MyMemcpyGPU2GPU(bottom[0]->GetGPUDiffPtr(), top[0]->GetGPUDiffPtr(),
                    top[0]->GetSize() * sizeof(T));
  }
}

template class Flatten<>;

}  // namespace my_tensor
