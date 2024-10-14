#ifndef INCLUDE_TENSOR_CUH_
#define INCLUDE_TENSOR_CUH_

#include <vector>
#include <string>
#include <thrust/device_vector.h>

namespace my_tensor {

class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;

// Tensor class
class Tensor {
 public:
  // The explicit constructors.
  explicit Tensor(
    const std::vector<int>& shape);

  // The Tensor is copable.
  Tensor(const Tensor& tensor);
  Tensor& operator=(const Tensor& tensor);
  Tensor(Tensor&& tensor);
  Tensor& operator=(Tensor&& tensor);

  // The destructors, which will free the dynamic allocated memory.
  ~Tensor() = default;

  // Get methods.
  const thrust::device_vector<float>& GetData() const { return data_; }
  thrust::device_vector<float>& GetMutableData() { return data_; }
  const thrust::device_vector<float>& GetDiff() const { return diff_; }
  thrust::device_vector<float>& GetMutableDiff() { return diff_; }

  const std::vector<int> GetShape() const { return shape_; }

  int GetSize() const { return size_; }

 private:
  std::vector<int> shape_;
  int size_;
  thrust::device_vector<float> data_;
  thrust::device_vector<float> diff_;
};  // class Tensor
}  // namespace my_tensor

#endif  // INCLUDE_TENSOR_CUH_
