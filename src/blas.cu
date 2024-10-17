#include <blas.cuh>
#include <handle.cuh>
#include <utils.cuh>

namespace my_tensor {
extern HandlePtr handle;

template <>
Tensor<> operator+(const Tensor<>& lhs, const Tensor<>& rhs) {
  int n = lhs.GetSize();
  if (rhs.GetSize() != n) {
    throw BlasError("Tensor operator + sizes not match.");
  }
  Tensor<> result = lhs;
  float alpha = 1.0f;
  CUBLAS_ERROR_CHECK(cublasSaxpy(handle->GetHandle(),
    n, &alpha, rhs.GetGPUDataPtr(), 1, result.GetGPUDataPtr(), 1));
  return result;
}

template <>
Tensor<> matmul(const Tensor<>& lhs, const Tensor<>& rhs) {
  const auto& left_shape = lhs.GetShape();
  const auto& right_shape = rhs.GetShape();
  if (left_shape.size() != 2 || right_shape.size() != 2 || left_shape[1] != right_shape[0]) {
    throw BlasError("Tensor matmul shapes not match.");
  }
  int m = left_shape[0], k = left_shape[1], n = right_shape[1];
  Tensor<> result({m, n});
  float alpha = 1.0f;
  float beta = 0.0f;
  // result<sup>T</sup> = rhs<sup>T</sup>lhs<sup>T</sup>
  // also result = (lhs)(rhs)
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_N,  // no transpose of lhs<sup>T</sup>
    CUBLAS_OP_N,  // no transpose of rhs<sup>T</sup>
    n,  // row number of rhs<sup>T</sup> and row number of result<sup>T</sup>
    m,  // col number of lhs<sup>T</sup> and col number of result<sup>T</sup>
    k,  // col number of rhs<sup>T</sup> and row number of lhs<sup>T</sup>
    &alpha,  // alpha
    rhs.GetGPUDataPtr(),  // rhs pointer, in cublas will be rhs<sup>T</sup>
    n,  // leading dimension of rhs<sup>T</sup>
    lhs.GetGPUDataPtr(),  // lhs pointer, in cublas will be lhs<sup>T</sup>
    k,  // leading dimension of lhs<sup>T</sup>
    &beta,  // beta
    result.GetGPUDataPtr(),  // result pointer, in cublas will be result<sup>T</sup>
    n  // leading dimension of result<sup>T</sup>
  ));
  return result;
}
}  // namespace my_tensor
