// Copyright 2024 yibotongxue

#include "synced-vector.hpp"

#include <iterator>
#include <utility>
#include <vector>

#include "error.hpp"
#include "memory-util.hpp"

namespace my_tensor {

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>::SyncedVector() noexcept
    : state_(kUninitialized),
      size_(0),
      cpu_data_(nullptr),
      gpu_data_(nullptr) {}

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>::SyncedVector(size_t size) noexcept
    : state_(kUninitialized),
      size_(size),
      cpu_data_(nullptr),
      gpu_data_(nullptr) {}

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>::~SyncedVector() {
  MyMemFreeCPU(cpu_data_);
  MyMemFreeGPU(gpu_data_);
}

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>::SyncedVector(const SyncedVector<T>& vec) noexcept
    : state_(vec.state_),
      size_(vec.size_),
      cpu_data_(nullptr),
      gpu_data_(nullptr) {
  if (state_ == kHeadAtCPU) {
    MyMemFreeCPU(cpu_data_);
    MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size_ * sizeof(T));
    MyMemcpyCPU2CPU(cpu_data_, vec.cpu_data_, size_ * sizeof(T));
  } else if (state_ == kHeadAtGPU) {
    MyMemFreeGPU(gpu_data_);
    MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size_ * sizeof(T));
    MyMemcpyGPU2GPU(gpu_data_, vec.gpu_data_, size_ * sizeof(T));
  } else if (state_ == kSynced) {
    MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size_ * sizeof(T));
    MyMemcpyCPU2CPU(cpu_data_, vec.cpu_data_, size_ * sizeof(T));
    MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size_ * sizeof(T));
    MyMemcpyGPU2GPU(gpu_data_, vec.gpu_data_, size_ * sizeof(T));
  }
}

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>& SyncedVector<T>::operator=(
    const SyncedVector<T>& vec) noexcept {
  if (this == &vec) {
    return *this;
  }
  state_ = vec.state_;
  size_ = vec.size_;
  if (state_ == kHeadAtCPU) {
    MyMemFreeCPU(cpu_data_);
    MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size_ * sizeof(T));
    MyMemcpyCPU2CPU(cpu_data_, vec.cpu_data_, size_ * sizeof(T));
  } else if (state_ == kHeadAtGPU) {
    MyMemFreeGPU(gpu_data_);
    MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size_ * sizeof(T));
    MyMemcpyGPU2GPU(gpu_data_, vec.gpu_data_, size_ * sizeof(T));
  } else if (state_ == kSynced) {
    MyMemFreeCPU(cpu_data_);
    MyMemFreeGPU(gpu_data_);
    MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size_ * sizeof(T));
    MyMemcpyCPU2CPU(cpu_data_, vec.cpu_data_, size_ * sizeof(T));
    MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size_ * sizeof(T));
    MyMemcpyGPU2GPU(gpu_data_, vec.gpu_data_, size_ * sizeof(T));
  }
  return *this;
}

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>::SyncedVector(SyncedVector<T>&& vec) noexcept
    : state_(vec.state_),
      size_(vec.size_),
      cpu_data_(vec.cpu_data_),
      gpu_data_(vec.gpu_data_) {
  vec.cpu_data_ = nullptr;
  vec.gpu_data_ = nullptr;
}

template <typename T>
  requires std::is_arithmetic<T>::value
SyncedVector<T>& SyncedVector<T>::operator=(SyncedVector<T>&& vec) noexcept {
  if (this == &vec) {
    return *this;
  }
  state_ = vec.state_;
  size_ = vec.size_;
  cpu_data_ = vec.cpu_data_;
  gpu_data_ = vec.gpu_data_;
  vec.cpu_data_ = nullptr;
  vec.gpu_data_ = nullptr;
  return *this;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline void SyncedVector<T>::SetCPUData(const T* const data,
                                        size_t size) noexcept {
  MyMemFreeCPU(cpu_data_);
  MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size * sizeof(T));
  MyMemcpyCPU2CPU(cpu_data_, data, size * sizeof(T));
  size_ = size_;
  state_ = kHeadAtCPU;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline void SyncedVector<T>::SetGPUData(const T* const data,
                                        size_t size) noexcept {
  MyMemFreeGPU(gpu_data_);
  MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size * sizeof(T));
  MyMemcpyCPU2GPU(gpu_data_, data, size * sizeof(T));
  size_ = size;
  state_ = kHeadAtGPU;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline const T* SyncedVector<T>::GetCPUPtr() {
  ToCPU();
  return cpu_data_;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline T* SyncedVector<T>::GetMutableCPUPtr() {
  ToCPU();
  state_ = kHeadAtCPU;
  return cpu_data_;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline const T* SyncedVector<T>::GetGPUPtr() {
  ToGPU();
  return gpu_data_;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline T* SyncedVector<T>::GetMutableGPUPtr() {
  ToGPU();
  state_ = kHeadAtGPU;
  return gpu_data_;
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline T SyncedVector<T>::host(size_t index) {
  if (index >= size_) {
    throw VectorError("Index out of range!");
  }
  ToCPU();
  return Visit_CPU(cpu_data_, index);
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline T SyncedVector<T>::device(size_t index) {
  if (index >= size_) {
    throw VectorError("Index out of range!");
  }
  ToGPU();
  return Visit_GPU(gpu_data_, index);
}

template <typename T>
  requires std::is_arithmetic<T>::value
inline void SyncedVector<T>::ToCPU() {
  switch (state_) {
    case kUninitialized:
      MyMemFreeCPU(cpu_data_);
      MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size_ * sizeof(T));
      state_ = kHeadAtCPU;
      break;
    case kHeadAtGPU:
      MyMemFreeCPU(cpu_data_);
      MyMallocCPU(reinterpret_cast<void**>(&cpu_data_), size_ * sizeof(T));
      MyMemcpyGPU2CPU(cpu_data_, gpu_data_, size_ * sizeof(T));
      state_ = kSynced;
      break;
    case kHeadAtCPU:
    case kSynced:
      break;
    default:
      throw VectorError("Unimplemtation error!");
  }
}

template <typename T>
  requires std::is_arithmetic<T>::value
void SyncedVector<T>::ToGPU() {
  switch (state_) {
    case kUninitialized:
      MyMemFreeGPU(gpu_data_);
      MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size_ * sizeof(T));
      state_ = kHeadAtGPU;
      break;
    case kHeadAtCPU:
      MyMemFreeGPU(gpu_data_);
      MyMallocGPU(reinterpret_cast<void**>(&gpu_data_), size_ * sizeof(T));
      MyMemcpyCPU2GPU(gpu_data_, cpu_data_, size_ * sizeof(T));
      state_ = kSynced;
      break;
    case kHeadAtGPU:
    case kSynced:
      break;
    default:
      throw VectorError("Unimplemention error!");
  }
}

template class SyncedVector<>;
template class SyncedVector<int>;
}  // namespace my_tensor
