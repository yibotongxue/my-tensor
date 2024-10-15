#ifndef INCLUDE_TENSOR_CUH_
#define INCLUDE_TENSOR_CUH_

#include <vector>
#include <string>
#include <thrust/device_vector.h>

namespace my_tensor {

// Tensor class
template <typename T=float>
class Tensor {
 public:
  // The explicit constructors.
  explicit Tensor(const std::vector<int>& shape);

  // The Tensor is copable.
  Tensor(const Tensor<T>& tensor);
  Tensor<T>& operator=(const Tensor<T>& tensor);
  Tensor(Tensor<T>&& tensor);
  Tensor<T>& operator=(Tensor<T>&& tensor);

  // The destructors, which will free the dynamic allocated memory.
  ~Tensor() = default;

  // Set methods.
  void SetData(const std::vector<T>& data);
  void SetData(std::vector<T>&& data);
  void SetDiff(const std::vector<T>& diff);
  void SetDiff(std::vector<T>&& diff);

  // Get methods.
  const thrust::device_vector<T>& GetData() const { return data_; }
  thrust::device_vector<T>& GetMutableData() { return data_; }
  const thrust::device_vector<T>& GetDiff() const { return diff_; }
  thrust::device_vector<T>& GetMutableDiff() { return diff_; }

  const std::vector<int>& GetShape() const { return shape_; }

  int GetSize() const { return size_; }

  void Clear();

 private:
  std::vector<int> shape_;
  int size_;
  thrust::device_vector<T> data_;
  thrust::device_vector<T> diff_;

  void CheckShape() const;
};  // class Tensor

template <typename T = float>
using TensorPtr = std::shared_ptr<my_tensor::Tensor<T>>;

extern template class my_tensor::Tensor<>;
}  // namespace my_tensor

#endif  // INCLUDE_TENSOR_CUH_
