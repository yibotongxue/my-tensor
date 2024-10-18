#include <blas.cuh>
#include <cublasLt.h>
#include <handle.cuh>
#include <utils.cuh>

namespace my_tensor {
extern HandlePtr handle;

template <>
Tensor<> operator+(const Tensor<>& lhs, const Tensor<>& rhs) {
  if (lhs.GetShape() != rhs.GetShape()) {
    throw BlasError("Tensor operator + shape not match.");
  }
  int n = lhs.GetSize();
  Tensor<> result = lhs;
  float alpha = 1.0f;
  CUBLAS_ERROR_CHECK(cublasSaxpy(handle->GetHandle(),
    n, &alpha, rhs.GetGPUDataPtr(), 1, result.GetGPUDataPtr(), 1));
  return result;
}

namespace {
__global__ void SetAllOnes(float *ones, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    ones[i] = 1.0f;
  }
}
}  // namespace

template <>
Tensor<> add_vector(const Tensor<>& tensor, const Tensor<>& vec, bool at_grad) {
  if (tensor.GetShape().size() != 2 ||
      vec.GetShape().size() != 2 ||
      tensor.GetShape()[0] != vec.GetShape()[0] ||
      vec.GetShape()[1] != 1) {
    throw BlasError("Tensor operator add vector shape not match.");
  }
  int m = tensor.GetShape()[0], n = tensor.GetShape()[1];
  Tensor<> result(tensor);
  float alpha = 1.0f;
  float beta = 1.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, n * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(n), kCudaThreadNum>>>(ones, n);
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    n,
    m,
    1,
    &alpha,
    ones,
    n,
    AT_GRAD_GPU_DATA(vec),
    1,
    &beta,
    AT_GRAD_GPU_DATA(result),
    n));
  cudaFree(ones);
  return result;
}

template <>
Tensor<> matmul(const Tensor<>& lhs, const Tensor<>& rhs, bool at_grad) {
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
    AT_GRAD_GPU_DATA(rhs),  // rhs pointer, in cublas will be rhs<sup>T</sup>
    n,  // leading dimension of rhs<sup>T</sup>
    AT_GRAD_GPU_DATA(lhs),  // lhs pointer, in cublas will be lhs<sup>T</sup>
    k,  // leading dimension of lhs<sup>T</sup>
    &beta,  // beta
    AT_GRAD_GPU_DATA(result),  // result pointer, in cublas will be result<sup>T</sup>
    n  // leading dimension of result<sup>T</sup>
  ));
  return result;
}

template <>
Tensor<> transpose(const Tensor<>& tensor) {
  const auto& shape = tensor.GetShape();
  if (shape.size() != 2) {
    throw BlasError("Tensor transpose shape not two dimension.");
  }
  int m = shape[0], n = shape[1];
  Tensor<> result({n, m});
  float alpha = 1.0f;
  float beta = 0.0f;
  CUBLAS_ERROR_CHECK(cublasSgeam(handle->GetHandle(),
    CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
    tensor.GetGPUDataPtr(), n, &beta, nullptr,
    m, result.GetGPUDataPtr(), m));
  return result;
}

template<>
Tensor<> transpose_matmul(const Tensor<>& lhs, const Tensor<>& rhs, bool at_grad) {
  const auto& left_shape = lhs.GetShape();
  const auto& right_shape = rhs.GetShape();
  if (left_shape.size() != 2 || right_shape.size() != 2 || left_shape[0] != right_shape[0]) {
    throw BlasError("Tensor transpose matmul shapes not match.");
  }
  int k = left_shape[0], m = left_shape[1], n = right_shape[1];
  Tensor<> result({m, n});
  float alpha = 1.0f;
  float beta = 0.0f;
  // result<sup>T</sup> = (rhs<sup>T</sup>)(lhs)
  // also result = (lhs<sup>T</sup>)(rhs)
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_N,  // no transpose of rhs<sup>T</sup>
    CUBLAS_OP_T,  // transpose of lhs<sup>T</sup>
    n,  // row number of rhs<sup>T</sup> and row number of result<sup>T</sup>
    m,  // col number of lhs and col number of result<sup>T</sup>
    k,  // col number of rhs<sup>T</sup> and row number of lhs
    &alpha,  // alpha
    AT_GRAD_GPU_DATA(rhs),  // rhs pointer, in cublas will be rhs<sup>T</sup>
    n,  // leading dimension of rhs<sup>T</sup>
    AT_GRAD_GPU_DATA(lhs),  // lhs pointer, in cublas will be lhs<sup>T</sup>
    m,  // leading dimension of lhs<sup>T</sup>
    &beta,  // beta
    AT_GRAD_GPU_DATA(result),  // result pointer, in cublas will be result<sup>T</sup>
    n  // leading dimension of result<sup>T</sup>
  ));
  return result;
}

template <>
Tensor<> matmul_transpose(const Tensor<>& lhs, const Tensor<>& rhs, bool at_grad) {
  const auto& left_shape = lhs.GetShape();
  const auto& right_shape = rhs.GetShape();
  if (left_shape.size() != 2 || right_shape.size() != 2 || left_shape[1] != right_shape[1]) {
    throw BlasError("Tensor transpose matmul shapes not match.");
  }
  int m = left_shape[0], k = left_shape[1], n = right_shape[0];
  Tensor<> result({m, n});
  float alpha = 1.0f;
  float beta = 0.0f;
  // result<sup>T</sup> = (rhs)(lhs<sup>T</sup>)
  // also result = (lhs)(rhs<sup>T</sup>)
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_T,  // transpose of rhs<sup>T</sup>
    CUBLAS_OP_N,  // no transpose of lhs<sup>T</sup>
    n,  // row number of rhs and row number of result<sup>T</sup>
    m,  // col number of lhs<sup>T</sup> and col number of result<sup>T</sup>
    k,  // col number of rhs and row number of lhs<sup>T</sup>
    &alpha,  // alpha
    AT_GRAD_GPU_DATA(rhs),  // rhs pointer, in cublas will be rhs<sup>T</sup>
    k,  // leading dimension of rhs<sup>T</sup>
    AT_GRAD_GPU_DATA(lhs),  // lhs pointer, in cublas will be lhs<sup>T</sup>
    k,  // leading dimension of lhs<sup>T</sup>
    &beta,  // beta
    AT_GRAD_GPU_DATA(result),  // result pointer, in cublas will be result<sup>T</sup>
    n  // leading dimension of result<sup>T</sup>
  ));
  return result;
}

template <>
Tensor<> transpose_matmul_transpose(const Tensor<>& lhs, const Tensor<>& rhs, bool at_grad) {
  const auto& left_shape = lhs.GetShape();
  const auto& right_shape = rhs.GetShape();
  if (left_shape.size() != 2 || right_shape.size() != 2 || left_shape[0] != right_shape[1]) {
    throw BlasError("Tensor transpose matmul transpose shapes not match.");
  }
  int k = left_shape[0], m = left_shape[1], n = right_shape[0];
  Tensor<> result({m, n});
  float alpha = 1.0f;
  float beta = 0.0f;
  // result<sup>T</sup> = (rhs)(lhs)
  // also result = (lhs<sup>T</sup>)(rhs<sup>T</sup>)
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_T,  // transpose of rhs<sup>T</sup>
    CUBLAS_OP_T,  // transpose of lhs<sup>T</sup>
    n,  // row number of rhs and row number of result<sup>T</sup>
    m,  // col number of lhs and col number of result<sup>T</sup>
    k,  // col number of rhs and row number of lhs<sup>T</sup>
    &alpha,  // alpha
    AT_GRAD_GPU_DATA(rhs),  // rhs pointer, in cublas will be rhs<sup>T</sup>
    k,  // leading dimension of rhs<sup>T</sup>
    AT_GRAD_GPU_DATA(lhs),  // lhs pointer, in cublas will be lhs<sup>T</sup>
    m,  // leading dimension of lhs<sup>T</sup>
    &beta,  // beta
    AT_GRAD_GPU_DATA(result),  // result pointer, in cublas will be result<sup>T</sup>
    n  // leading dimension of result<sup>T</sup>
  ));
  return result;
}
}  // namespace my_tensor
