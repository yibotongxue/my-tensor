#include <error.h>
#include <synced-vector.cuh>
#include <utils.cuh>

namespace my_tensor {

template <typename T>
SyncedVector<T>::SyncedVector() : state_(kUninitialized), size_(0) {}

template <typename T>
SyncedVector<T>::SyncedVector(size_t size) : state_(kUninitialized), size_(size) {}

template <typename T>
SyncedVector<T>::SyncedVector(const SyncedVector<T>& vec)
 : state_(vec.state_), size_(vec.size_), cpu_data_(vec.cpu_data_), gpu_data_(vec.gpu_data_) {}

template <typename T>
SyncedVector<T>& SyncedVector<T>::operator=(const SyncedVector<T>& vec) {
  if (this == &vec) {
    return *this;
  }
  state_ = vec.state_;
  size_ = vec.size_;
  cpu_data_.assign(vec.cpu_data_.begin(), vec.cpu_data_.end());
  gpu_data_.assign(vec.gpu_data_.begin(), vec.gpu_data_.end());
  return *this;
}

template <typename T>
SyncedVector<T>::SyncedVector(SyncedVector<T>&& vec)
  : state_(vec.state_), size_(vec.size_), cpu_data_(std::move(vec.cpu_data_)),
    gpu_data_(std::move(vec.gpu_data_)) {}

template <typename T>
SyncedVector<T>& SyncedVector<T>::operator=(SyncedVector<T>&& vec) {
  if (this == &vec) {
    return *this;
  }
  state_ = vec.state_;
  size_ = vec.size_;
  cpu_data_ = std::move(vec.cpu_data_);
  gpu_data_ = std::move(vec.gpu_data_);
  return *this;
}

template <typename T>
inline const thrust::host_vector<T>& SyncedVector<T>::GetCPUData(){
  ToCPU();
  return cpu_data_;
}

template <typename T>
inline thrust::host_vector<T>& SyncedVector<T>::GetMutableCPUData() {
  ToCPU();
  state_ = kHeadAtCPU;
  return cpu_data_;
}

template <typename T>
inline const thrust::device_vector<T>& SyncedVector<T>::GetGPUData(){
  ToGPU();
  return gpu_data_;
}

template <typename T>
inline thrust::device_vector<T>& SyncedVector<T>::GetMutableGPUData() {
  ToGPU();
  state_ = kHeadAtGPU;
  return gpu_data_;
}

template <typename T>
inline void SyncedVector<T>::SetCPUData(const std::vector<T>& data) {
  cpu_data_.assign(data.begin(), data.end());
  size_ = data.size();
  state_ = kHeadAtCPU;
}

template <typename T>
inline void SyncedVector<T>::SetGPUData(const std::vector<T>& data) {
  gpu_data_.assign(data.begin(), data.end());
  size_ = data.size();
  state_ = kHeadAtGPU;
}

template <typename T>
inline const T* SyncedVector<T>::GetCPUPtr(){
  ToCPU();
  return thrust::raw_pointer_cast(cpu_data_.data());
}

template <typename T>
inline T* SyncedVector<T>::GetMutableCPUPtr() {
  ToCPU();
  state_ = kHeadAtCPU;
  return thrust::raw_pointer_cast(cpu_data_.data());
}

template <typename T>
inline const T* SyncedVector<T>::GetGPUPtr(){
  ToGPU();
  return thrust::raw_pointer_cast(gpu_data_.data());
}

template <typename T>
inline T* SyncedVector<T>::GetMutableGPUPtr() {
  ToGPU();
  state_ = kHeadAtGPU;
  return thrust::raw_pointer_cast(gpu_data_.data());
}

// template <typename T>
// void SyncedVector<T>::Resize(size_t size) {
//   size_ = size;
//   if (state_ == kHeadAtCPU) {
//     cpu_data_.resize(size);
//   } else if (state_ == kHeadAtGPU) {
//     gpu_data_.resize(size);
//   } else {
//     cpu_data_.resize(size);
//     state_ = kHeadAtCPU;
//   }
//   switch (state_) {
//   case kHeadAtCPU:
//     cpu_data_.resize(size);
//     break;
//   case kHeadAtGPU:
//     gpu_data_.resize(size);
//     break;
//   case kUninitialized:
//   case kSynced:

//   }
// }

template <typename T>
inline void SyncedVector<T>::ToCPU() {
  switch (state_) {
  case kUninitialized:
    cpu_data_.resize(size_);
    state_ = kHeadAtCPU;
    break;
  case kHeadAtGPU:
    cpu_data_.assign(gpu_data_.begin(), gpu_data_.end());
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
void SyncedVector<T>::ToGPU() {
  switch (state_) {
  case kUninitialized:
    gpu_data_.resize(size_);
    state_ = kHeadAtGPU;
    break;
  case kHeadAtCPU:
    gpu_data_.assign(cpu_data_.begin(), cpu_data_.end());
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
template class SyncedVector<double>;
}  // namespace my_tensor
