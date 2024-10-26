// Copyright 2024 yibotongxue

#ifndef INCLUDE_SYNCED_VECTOR_CUH_
#define INCLUDE_SYNCED_VECTOR_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>
#include <vector>

#include "error.h"

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

  const thrust::host_vector<T>& GetCPUData();
  thrust::host_vector<T>& GetMutableCPUData();
  const thrust::device_vector<T>& GetGPUData();
  thrust::device_vector<T>& GetMutableGPUData();
  void SetCPUData(const std::vector<T>& data);
  void SetGPUData(const std::vector<T>& data);
  const T* GetCPUPtr();
  T* GetMutableCPUPtr();
  const T* GetGPUPtr();
  T* GetMutableGPUPtr();

  void ClearCPUData();
  void ClearGPUData();

  inline size_t size() const { return size_; }

  // void Resize(size_t size);

 private:
  VectorState state_;
  size_t size_;
  thrust::host_vector<T> cpu_data_;
  thrust::device_vector<T> gpu_data_;

  void ToCPU();
  void ToGPU();
};

template <typename T>
using SyncedVectorPtr = std::shared_ptr<SyncedVector<T>>;

extern template class SyncedVector<>;
extern template class SyncedVector<double>;
}  // namespace my_tensor

#endif  // INCLUDE_SYNCED_VECTOR_CUH_
