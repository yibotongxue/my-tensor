// Copyright 2024 yibotongxue

#ifndef INCLUDE_TENSOR_HPP_
#define INCLUDE_TENSOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "synced-vector.hpp"

namespace my_tensor {

// Tensor class
template <typename T = float>
class Tensor {
 public:
  Tensor();

  // The explicit constructors.
  explicit Tensor(const std::vector<int>& shape);

  // The Tensor is copable.
  Tensor(const Tensor<T>& tensor);
  Tensor<T>& operator=(const Tensor<T>& tensor);
  Tensor(Tensor<T>&& tensor);
  Tensor<T>& operator=(Tensor<T>&& tensor);

  // The destructors, which will free the dynamic allocated memory.
  ~Tensor() = default;

  // Set methods.
  void SetCPUData(const T* const data, size_t size) {
    data_->SetCPUData(data, size);
  }
  void SetCPUDiff(const T* const diff, size_t size) {
    AllocateDiff();
    diff_->SetCPUData(diff, size);
  }
  void SetGPUData(const T* const data, size_t size) {
    data_->SetGPUData(data, size);
  }
  void SetGPUDiff(const T* const diff, size_t size) {
    AllocateDiff();
    diff_->SetGPUData(diff, size);
  }

  // Get methods.
  T GetCPUData(size_t index) const { return data_->host(index); }
  T GetCPUDiff(size_t index) const {
    AllocateDiff();
    return diff_->host(index);
  }
  T GetGPUData(size_t index) const { return data_->device(index); }
  T GetGPUDiff(size_t index) const {
    AllocateDiff();
    return diff_->device(index);
  }

  inline T GetData(size_t index) const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUData(index);
    } else {
      return GetGPUData(index);
    }
  }

  inline T GetDiff(size_t index) const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDiff(index);
    } else {
      return GetGPUDiff(index);
    }
  }

  // #endif  // CPU_ONLY
  // CPU
  const T* GetCPUDataPtr() const { return data_->GetCPUPtr(); }
  T* GetCPUDataPtr() { return data_->GetMutableCPUPtr(); }
  const T* GetCPUDiffPtr() const {
    AllocateDiff();
    return diff_->GetCPUPtr();
  }
  T* GetCPUDiffPtr() {
    AllocateDiff();
    return diff_->GetMutableCPUPtr();
  }
  // GPU
  const T* GetGPUDataPtr() const { return data_->GetGPUPtr(); }
  T* GetGPUDataPtr() { return data_->GetMutableGPUPtr(); }
  const T* GetGPUDiffPtr() const {
    AllocateDiff();
    return diff_->GetGPUPtr();
  }
  T* GetGPUDiffPtr() {
    AllocateDiff();
    return diff_->GetMutableGPUPtr();
  }

  const T* GetDataPtr() const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDataPtr();
    } else {
      return GetGPUDataPtr();
    }
  }

  T* GetDataPtr() {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDataPtr();
    } else {
      return GetGPUDataPtr();
    }
  }

  const T* GetDiffPtr() const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDiffPtr();
    } else {
      return GetGPUDiffPtr();
    }
  }

  T* GetDiffPtr() {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDiffPtr();
    } else {
      return GetGPUDiffPtr();
    }
  }

  const std::vector<int>& GetShape() const { return shape_; }

  int GetSize() const { return size_; }

  void Reshape(const std::vector<int>& shape);

  void Resize(const std::vector<int>& shape);

 private:
  std::vector<int> shape_;
  int size_;
  SyncedVectorPtr<T> data_;
  mutable SyncedVectorPtr<T> diff_;

  void AllocateDiff() const;
  void CheckShape() const;
};  // class Tensor

template <typename T = float>
using TensorPtr = std::shared_ptr<my_tensor::Tensor<T>>;

extern template class my_tensor::Tensor<>;
extern template class my_tensor::Tensor<int>;
}  // namespace my_tensor

#endif  // INCLUDE_TENSOR_HPP_
