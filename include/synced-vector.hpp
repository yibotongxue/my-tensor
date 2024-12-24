// Copyright 2024 yibotongxue

#ifndef INCLUDE_SYNCED_VECTOR_HPP_
#define INCLUDE_SYNCED_VECTOR_HPP_

#include <memory>
#include <span>  // NOLINT

#include "error.hpp"

namespace my_tensor {

template <typename T = float>
class SyncedVector {
 public:
  SyncedVector();
  explicit SyncedVector(size_t size);
  ~SyncedVector();

  SyncedVector(const SyncedVector<T>& vec);
  SyncedVector<T>& operator=(const SyncedVector<T>& vec);
  SyncedVector(SyncedVector<T>&& vec);
  SyncedVector<T>& operator=(SyncedVector<T>&& vec);

  enum VectorState { kUninitialized, kHeadAtCPU, kHeadAtGPU, kSynced };

  void SetCPUData(const T* const data, size_t size);
  const T* GetCPUPtr();
  T* GetMutableCPUPtr();
  void SetGPUData(const T* const data, size_t size);
  const T* GetGPUPtr();
  T* GetMutableGPUPtr();

  std::span<T> GetCPUSpan() {
    ToCPU();
    return std::span<T>(cpu_data_, size_);
  }

  T host(size_t index);
  T device(size_t index);

  inline size_t size() const noexcept { return size_; }

  void Resize(size_t size) noexcept { size_ = size; }

 private:
  VectorState state_;
  size_t size_;
  T* cpu_data_;
  T* gpu_data_;

  void ToCPU();
  void ToGPU();
};

template <typename T>
using SyncedVectorPtr = std::shared_ptr<SyncedVector<T>>;

extern template class SyncedVector<>;
extern template class SyncedVector<int>;
}  // namespace my_tensor

#endif  // INCLUDE_SYNCED_VECTOR_HPP_
