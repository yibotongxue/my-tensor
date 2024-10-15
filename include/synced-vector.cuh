#ifndef INCLUDE_SYNCED_VECTOR_CUH_
#define INCLUDE_SYNCED_VECTOR_CUH_

#include <error.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace my_tensor {

template <typename T = float>
class SyncedVector {
 public:
  SyncedVector();
  ~SyncedVector() = default;

  enum VectorState {kUninitialized, kHeadAtCPU, kHeadAtGPU, kSynced};

  const thrust::host_vector<T>& GetCPUData() const;
  thrust::host_vector<T>& GetCPUData();
  const thrust::device_vector<T>& GetGPUData() const;
  thrust::device_vector<T>& GetGPUData();
  void SetCPUData(const std::vector<T>& data);
  void SetGPUData(const std::vector<T>& data);

 private:
  VectorState state_;
  thrust::host_vector<T> cpu_data_;
  thrust::device_vector<T> gpu_data_;

  inline void SyncedVector<T>::SyncedMemory() {
    CheckInitialized();
    if (state_ == kHeadAtCPU) {
      gpu_data_.assign(cpu_data_.begin(), cpu_data_.end());
    } else if (state_ == kHeadAtGPU) {
      cpu_data_.assign(gpu_data_.begin(), gpu_data_.end());
    }
    state_ = kSynced;
  }

  inline void CheckInitialized() const {
    if (state_ == kUninitialized) {
      throw VectorError("The vector is still unintialized!");
    }
  }
};

extern template class SyncedVector<>;
extern template class SyncedVector<double>;
}  // namespace my_tensor

#endif  // INCLUDE_SYNCED_VECTOR_CUH_
