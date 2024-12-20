// Copyright 2024 yibotongxue

#ifndef INCLUDE_TENSOR_HPP_
#define INCLUDE_TENSOR_HPP_

#ifndef CPU_ONLY
#include <thrust/device_vector.h>
#endif  // CPU_ONLY

#include <memory>
#include <string>
#include <vector>

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
  void SetCPUData(const std::vector<T>& data) { data_->SetCPUData(data); }
  void SetCPUDiff(const std::vector<T>& diff) { diff_->SetCPUData(diff); }
  void SetGPUData(const std::vector<T>& data) { data_->SetGPUData(data); }
  void SetGPUDiff(const std::vector<T>& diff) { diff_->SetGPUData(diff); }

  // Set by intetor
  template <typename Iter>
  void SetCPUData(const Iter begin, const Iter end) {
    data_->SetCPUData(begin, end);
  }
  template <typename Iter>
  void SetCPUDiff(const Iter begin, const Iter end) {
    diff_->SetCPUData(begin, end);
  }
  template <typename Iter>
  void SetGPUData(const Iter begin, const Iter end) {
    data_->SetGPUData(begin, end);
  }
  template <typename Iter>
  void SetGPUDiff(const Iter begin, const Iter end) {
    diff_->SetGPUData(begin, end);
  }

  // Get methods.
  // CPU
  const std::vector<T>& GetCPUData() const { return data_->GetCPUData(); }
  std::vector<T>& GetCPUData() { return data_->GetMutableCPUData(); }
  const std::vector<T>& GetCPUDiff() const { return diff_->GetCPUData(); }
  std::vector<T>& GetCPUDiff() { return diff_->GetMutableCPUData(); }
  // GPU
#ifndef CPU_ONLY
  const thrust::device_vector<T>& GetGPUData() const {
    return data_->GetGPUData();
  }
  thrust::device_vector<T>& GetGPUData() { return data_->GetMutableGPUData(); }
  const thrust::device_vector<T>& GetGPUDiff() const {
    return diff_->GetGPUData();
  }
  thrust::device_vector<T>& GetGPUDiff() { return diff_->GetMutableGPUData(); }
#endif  // CPU_ONLY
  // CPU
  const T* GetCPUDataPtr() const { return data_->GetCPUPtr(); }
  T* GetCPUDataPtr() { return data_->GetMutableCPUPtr(); }
  const T* GetCPUDiffPtr() const { return diff_->GetCPUPtr(); }
  T* GetCPUDiffPtr() { return diff_->GetMutableCPUPtr(); }
  // GPU
  const T* GetGPUDataPtr() const { return data_->GetGPUPtr(); }
  T* GetGPUDataPtr() { return data_->GetMutableGPUPtr(); }
  const T* GetGPUDiffPtr() const { return diff_->GetGPUPtr(); }
  T* GetGPUDiffPtr() { return diff_->GetMutableGPUPtr(); }

  const std::vector<int>& GetShape() const { return shape_; }

  int GetSize() const { return size_; }

  void Reshape(const std::vector<int>& shape);

  void Resize(const std::vector<int>& shape);

 private:
  std::vector<int> shape_;
  int size_;
  SyncedVectorPtr<T> data_;
  SyncedVectorPtr<T> diff_;

  void CheckShape() const;
};  // class Tensor

template <typename T = float>
using TensorPtr = std::shared_ptr<my_tensor::Tensor<T>>;

extern template class my_tensor::Tensor<>;
extern template class my_tensor::Tensor<int>;
}  // namespace my_tensor

#endif  // INCLUDE_TENSOR_HPP_
