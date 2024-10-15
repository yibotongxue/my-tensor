#include <blas.cuh>

namespace my_tensor {
template <>
Tensor<> operator+(const Tensor<>& lhs, const Tensor<>& rhs) {
  int n = lhs.GetSize();
  if (rhs.GetSize() != n) {
    throw BlasError("Tensor operator + size not match.");
  }
  Tensor<> result = lhs;
  float alpha = 1.0f;
  cublasSaxpy(handle->GetHandle(), n, &alpha, rhs.GetDataRowPointer(), 1, result.GetDataRowPointer(), 1);
  return result;
}
}  // namespace my_tensor
