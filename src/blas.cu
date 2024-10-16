#include <blas.cuh>
#include <handle.cuh>

namespace my_tensor {
extern HandlePtr handle;

template <>
Tensor<> operator+(const Tensor<>& lhs, const Tensor<>& rhs) {
  int n = lhs.GetSize();
  if (rhs.GetSize() != n) {
    throw BlasError("Tensor operator + size not match.");
  }
  Tensor<> result = lhs;
  float alpha = 1.0f;
  cublasSaxpy(handle->GetHandle(), n, &alpha, rhs.GetGPUDataPtr(), 1, result.GetGPUDataPtr(), 1);
  return result;
}

template <>
Tensor<> matmul(const Tensor<>& lhs, const Tensor<>& rhs) {
  const auto& left_shape = lhs.GetShape();
  const auto& right_shape = rhs.GetShape();
  if (left_shape.size() != 2 || right_shape.size() != 2 || left_shape[1] != right_shape[0]) {
    throw BlasError("Tensor matmul shape not match.");
  }
  int m = left_shape[0], k = left_shape[1], n = right_shape[1];
  Tensor<> result({m, n});
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle->GetHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, rhs.GetGPUDataPtr(), n, lhs.GetGPUDataPtr(), k, &beta, result.GetGPUDataPtr(), n);
  return result;
}
}  // namespace my_tensor
