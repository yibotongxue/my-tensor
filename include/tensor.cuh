#ifndef MYTENSOR_INCLUDE_TENSOR_H_
#define MYTENSOR_INCLUDE_TENSOR_H_

#include <vector>
#include <string>

namespace my_tensor {
// Device type
enum class DeviceType { CPU, GPU };

// Tensor class
class Tensor {
public:
  // The explicit constructors.
  explicit Tensor(const std::vector<int>& shape, DeviceType device_type = DeviceType::CPU);
  // explicit Tensor(const std::vector<int>& shape, float value, DeviceType deviceType = DeviceType::CPU);
  // explicit Tensor(const std::vector<int>& shape, float* data, DeviceType deviceType = DeviceType::CPU);

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

  const std::vector<int> GetShape() const { return shape_; }

  std::size_t GetBytesCount() const;

  bool OnCPU() const { return device_type_ == DeviceType::CPU; }
  bool OnGPU() const { return device_type_ == DeviceType::GPU; }

private:
  DeviceType device_type_;
  float* data_;
  std::vector<int> shape_;

  void CopyData(const Tensor& tensor, std::size_t cnt);
  void FreeMemory();
  void AllocateMemory();
  Tensor Clone(DeviceType device_type);
}; // class Tensor
} // namespace my_tensor

#endif // MYTENSOR_INCLUDE_TENSOR_H_
