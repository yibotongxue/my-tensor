#include "tensor-facade.cuh"

namespace my_tensor {
template <typename T>
TensorFacade<T>::TensorFacade(const std::vector<int>& shape) : tensor_(std::make_shared<Tensor<T>>(shape)) {}

template <typename T>
void TensorFacade<T>::Reshape(const std::vector<int>& shape) {
  tensor_->Reshape(shape);
}

template <typename T>
void TensorFacade<T>::SetData(const std::vector<T>& data) {
  tensor_->SetCPUData(data);
  tensor_->SetGPUData(data);
}
}
