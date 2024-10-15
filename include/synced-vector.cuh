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
  SyncedVector(size_t size);
  ~SyncedVector() = default;

  SyncedVector(const SyncedVector<T>& vec);
  SyncedVector<T>& operator=(const SyncedVector<T>& vec);
  SyncedVector(SyncedVector<T>&& vec);
  SyncedVector<T>& operator=(SyncedVector<T>&& vec);

  enum VectorState {kUninitialized, kHeadAtCPU, kHeadAtGPU, kSynced};

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

  inline size_t size() const {
    CheckInitialized();
    if (state_ == kHeadAtGPU) {
      return gpu_data_.size();
    }
    return cpu_data_.size();
  }

  void Resize(size_t size);

 private:
  VectorState state_;
  thrust::host_vector<T> cpu_data_;
  thrust::device_vector<T> gpu_data_;

  inline void SyncedMemory() {
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
