#include <error.h>
#include <synced-vector.cuh>
#include <utils.cuh>

namespace my_tensor {

template <typename T>
SyncedVector<T>::SyncedVector() : state_(kUninitialized) {}

template <typename T>
const thrust::host_vector<T>& SyncedVector<T>::GetCPUData() const {
  CheckInitialized();
  if (state_ == kHeadAtGPU) {
    SyncedMemory();
  }
  return cpu_data_;
}

template <typename T>
thrust::host_vector<T>& SyncedVector<T>::GetCPUData() {
  CheckInitialized();
  if (state_ == kHeadAtGPU) {
    SyncedMemory();
  }
  state_ = kHeadAtCPU;
  return cpu_data_;
}

template <typename T>
const thrust::device_vector<T>& SyncedVector<T>::GetGPUData() const {
  CheckInitialized();
  if (state_ == kHeadAtCPU) {
    SyncedMemory();
  }
  return gpu_data_;
}

template <typename T>
thrust::device_vector<T>& SyncedVector<T>::GetGPUData() {
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
}  // namespace my_tensor
