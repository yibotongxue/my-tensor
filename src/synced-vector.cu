#include <error.h>
#include <synced-vector.cuh>
#include <utils.cuh>

namespace my_tensor {

template <typename T>
SyncedVector<T>::SyncedVector() : state_(kUninitialized) {}

template <typename T>
SyncedVector<T>::SyncedVector(size_t size)
  : state_(kUninitialized), cpu_data_(size), gpu_data_(size) {}

template <typename T>
SyncedVector<T>::SyncedVector(const SyncedVector<T>& vec)
 : state_(vec.state_), cpu_data_(vec.cpu_data_), gpu_data_(vec.gpu_data_) {}

template <typename T>
SyncedVector<T>& SyncedVector<T>::operator=(const SyncedVector<T>& vec) {
  if (this == &vec) {
    return *this;
  }
  state_ = vec.state_;
  cpu_data_.assign(vec.cpu_data_.begin(), vec.cpu_data_.end());
  gpu_data_.assign(vec.gpu_data_.begin(), vec.gpu_data_.end());
  return *this;
}

template <typename T>
SyncedVector<T>::SyncedVector(SyncedVector<T>&& vec)
  : state_(vec.state_), cpu_data_(std::move(vec.cpu_data_)),
    gpu_data_(std::move(vec.gpu_data_)) {}

template <typename T>
SyncedVector<T>& SyncedVector<T>::operator=(SyncedVector<T>&& vec) {
  if (this == &vec) {
    return *this;
  }
  state_ = vec.state_;
  cpu_data_ = std::move(vec.cpu_data_);
  gpu_data_ = std::move(vec.gpu_data_);
  return *this;
}

template <typename T>
const thrust::host_vector<T>& SyncedVector<T>::GetCPUData(){
  CheckInitialized();
  if (state_ == kHeadAtGPU) {
    SyncedMemory();
  }
  return cpu_data_;
}

template <typename T>
thrust::host_vector<T>& SyncedVector<T>::GetMutableCPUData() {
  CheckInitialized();
  if (state_ == kHeadAtGPU) {
    SyncedMemory();
  }
  state_ = kHeadAtCPU;
  return cpu_data_;
}

template <typename T>
const thrust::device_vector<T>& SyncedVector<T>::GetGPUData(){
  CheckInitialized();
  if (state_ == kHeadAtCPU) {
    SyncedMemory();
  }
  return gpu_data_;
}

template <typename T>
thrust::device_vector<T>& SyncedVector<T>::GetMutableGPUData() {
  CheckInitialized();
  if (state_ == kHeadAtCPU) {
    SyncedMemory();
  }
  state_ = kHeadAtGPU;
  return gpu_data_;
}

template <typename T>
void SyncedVector<T>::SetCPUData(const std::vector<T>& data) {
  cpu_data_.assign(data.begin(), data.end());
  state_ = kHeadAtCPU;
}

template <typename T>
void SyncedVector<T>::SetGPUData(const std::vector<T>& data) {
  gpu_data_.assign(data.begin(), data.end());
  state_ = kHeadAtGPU;
}

template <typename T>
const T* SyncedVector<T>::GetCPUPtr(){
  CheckInitialized();
  if (state_ == kHeadAtGPU) {
    SyncedMemory();
  }
  return thrust::raw_pointer_cast(cpu_data_.data());
}

template <typename T>
T* SyncedVector<T>::GetMutableCPUPtr() {
  CheckInitialized();
  if (state_ == kHeadAtGPU) {
    SyncedMemory();
  }
  return thrust::raw_pointer_cast(cpu_data_.data());
}

template <typename T>
const T* SyncedVector<T>::GetGPUPtr(){
  CheckInitialized();
  if (state_ == kHeadAtCPU) {
    SyncedMemory();
  }
  return thrust::raw_pointer_cast(gpu_data_.data());
}

template <typename T>
T* SyncedVector<T>::GetMutableGPUPtr() {
  CheckInitialized();
  if (state_ == kHeadAtCPU) {
    SyncedMemory();
  }
  return thrust::raw_pointer_cast(gpu_data_.data());
}

template <typename T>
void SyncedVector<T>::Resize(size_t size) {
  if (state_ == kHeadAtCPU) {
    cpu_data_.resize(size);
  } else if (state_ == kHeadAtGPU) {
    gpu_data_.resize(size);
  } else {
    cpu_data_.resize(size);
    state_ = kHeadAtCPU;
  }
}

template class SyncedVector<>;
template class SyncedVector<double>;
}  // namespace my_tensor
