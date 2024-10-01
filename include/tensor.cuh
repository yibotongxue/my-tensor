#ifndef INCLUDE_TENSOR_CUH_
#define INCLUDE_TENSOR_CUH_

#include <vector>
#include <string>

namespace my_tensor {
// Device type
enum class DeviceType { CPU, GPU };

// Tensor class
class Tensor {
 public:
  // The explicit constructors.
  explicit Tensor(
    const std::vector<int>& shape, DeviceType device_type = DeviceType::CPU);

  // The Tensor is copable.
  Tensor(const Tensor& tensor);
  Tensor& operator=(const Tensor& tensor);
  Tensor(Tensor&& tensor);
  Tensor& operator=(Tensor&& tensor);

  // The destructors, which will free the dynamic allocated memory.
  ~Tensor();

  // Return same but on specific device.
  Tensor cpu();
  Tensor gpu();

  const float* GetData() const { return data_; }
  float* GetMutableData() { return data_; }
  const float* GetDiff() const { return diff_; }
  float* GetMutableDiff() { return diff_; }

  const std::vector<int> GetShape() const { return shape_; }

  int GetSize() const { return size_; }
  std::size_t GetByteSize() const { return size_ * sizeof(float); }

  bool OnCPU() const { return device_type_ == DeviceType::CPU; }
  bool OnGPU() const { return device_type_ == DeviceType::GPU; }

 private:
  DeviceType device_type_;
  float* data_;
  float* diff_;
  std::vector<int> shape_;
  int size_;

  // Helper methods
  void CopyData(const Tensor& tensor, std::size_t cnt);
  void FreeMemory();
  void AllocateMemory();
  Tensor Clone(DeviceType device_type);
};
}  // namespace my_tensor

#endif  // INCLUDE_TENSOR_CUH_
