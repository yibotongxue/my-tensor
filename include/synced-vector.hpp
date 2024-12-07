// Copyright 2024 yibotongxue

#ifndef INCLUDE_SYNCED_VECTOR_HPP_
#define INCLUDE_SYNCED_VECTOR_HPP_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>
#include <vector>

#include "error.hpp"

namespace my_tensor {

template <typename T = float>
class SyncedVector {
 public:
  SyncedVector();
  explicit SyncedVector(size_t size);
  ~SyncedVector() = default;

  SyncedVector(const SyncedVector<T>& vec);
  SyncedVector<T>& operator=(const SyncedVector<T>& vec);
  SyncedVector(SyncedVector<T>&& vec);
  SyncedVector<T>& operator=(SyncedVector<T>&& vec);

  enum VectorState { kUninitialized, kHeadAtCPU, kHeadAtGPU, kSynced };

  const std::vector<T>& GetCPUData();
  std::vector<T>& GetMutableCPUData();
  void SetCPUData(const std::vector<T>& data);
  const T* GetCPUPtr();
  T* GetMutableCPUPtr();
  const thrust::device_vector<T>& GetGPUData();
  thrust::device_vector<T>& GetMutableGPUData();
  void SetGPUData(const std::vector<T>& data);
  const T* GetGPUPtr();
  T* GetMutableGPUPtr();

  template <typename Iter>
  inline void SetCPUData(const Iter begin, const Iter end) {
    cpu_data_.resize(std::distance(begin, end));
    cpu_data_.assign(begin, end);
    size_ = cpu_data_.size();
    state_ = kHeadAtCPU;
  }

  template <typename Iter>
  inline void SetGPUData(const Iter begin, const Iter end) {
    gpu_data_.resize(std::distance(begin, end));
    gpu_data_.assign(begin, end);
    size_ = gpu_data_.size();
    state_ = kHeadAtGPU;
  }

  void ClearCPUData();
  void ClearGPUData();

  inline size_t size() const noexcept { return size_; }

  void Resize(size_t size) noexcept { size_ = size; }

 private:
  VectorState state_;
  size_t size_;
  std::vector<T> cpu_data_;
  thrust::device_vector<T> gpu_data_;

  void ToCPU();
  void ToGPU();
};

template <typename T>
using SyncedVectorPtr = std::shared_ptr<SyncedVector<T>>;

extern template class SyncedVector<>;
extern template class SyncedVector<int>;
}  // namespace my_tensor

#endif  // INCLUDE_SYNCED_VECTOR_HPP_
