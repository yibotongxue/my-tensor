// Copyright 2024 yibotongxue

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "blas.hpp"
#include "common.hpp"

namespace my_tensor {

#define DEFINE_ABC_VEC(broadcast)                                      \
  int stride_A = (broadcast == 1) ? 0 : (m * k);                       \
  thrust::device_vector<const float *> A_vec(batch_count);             \
  thrust::transform(thrust::counting_iterator<int>(0),                 \
                    thrust::counting_iterator<int>(batch_count),       \
                    A_vec.begin(),                                     \
                    [A, stride_A] __device__(int i) -> const float * { \
                      return A + i * stride_A;                         \
                    });                                                \
  int stride_B = (broadcast == 2) ? 0 : (k * n);                       \
  thrust::device_vector<const float *> B_vec(batch_count);             \
  thrust::transform(thrust::counting_iterator<int>(0),                 \
                    thrust::counting_iterator<int>(batch_count),       \
                    B_vec.begin(),                                     \
                    [B, stride_B] __device__(int i) -> const float * { \
                      return B + i * stride_B;                         \
                    });                                                \
  thrust::device_vector<float *> C_vec(batch_count);                   \
  thrust::transform(                                                   \
      thrust::counting_iterator<int>(0),                               \
      thrust::counting_iterator<int>(batch_count), C_vec.begin(),      \
      [C, m, n] __device__(int i) -> float * { return C + i * m * n; });

template <>
void matmul_gpu(const float *A, const float *B, float *C, const int m,
                const int k, const int n, const int batch_count,
                const int broadcast) {
  float alpha = 1.0f;
  float beta = 0.0f;
  DEFINE_ABC_VEC(broadcast)
  // C<sup>T</sup> = (B<sup>T</sup>)(A<sup>T</sup>)
  // also C = AB
  CUBLAS_CHECK(cublasSgemmBatched(
      MyTensorContext::cublas_handle(),  // handle
      CUBLAS_OP_N,                       // no transpose of A<sup>T</sup>
      CUBLAS_OP_N,                       // no transpose of B<sup>T</sup>
      n,       // row number of B<sup>T</sup> and row number of C<sup>T</sup>
      m,       // col number of A<sup>T</sup> and col number of C<sup>T</sup>
      k,       // col number of B<sup>T</sup> and row number of A<sup>T</sup>
      &alpha,  // alpha
      RAW_PTR(B_vec),  // B pointer, in cublas will be B<sup>T</sup>
      n,               // leading dimension of B<sup>T</sup>
      RAW_PTR(A_vec),  // A pointer, in cublas will be A<sup>T</sup>
      k,               // leading dimension of A<sup>T</sup>
      &beta,           // beta
      RAW_PTR(C_vec),  // C pointer, in cublas will be C<sup>T</sup>
      n,               // leading dimension of C<sup>T</sup>
      batch_count));
}

template <>
void transpose_matmul_gpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast) {
  DEFINE_ABC_VEC(broadcast)
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B<sup>T</sup>)(A)
  // also C = (A<sup>T</sup>)B
  CUBLAS_CHECK(cublasSgemmBatched(
      MyTensorContext::cublas_handle(),  // handle
      CUBLAS_OP_N,                       // no transpose of A<sup>T</sup>
      CUBLAS_OP_T,                       // no transpose of B<sup>T</sup>
      n,       // row number of B<sup>T</sup> and row number of C<sup>T</sup>
      m,       // col number of A and col number of C<sup>T</sup>
      k,       // col number of B<sup>T</sup> and row number of A
      &alpha,  // alpha
      RAW_PTR(B_vec),  // B pointer, in cublas will be B<sup>T</sup>
      n,               // leading dimension of B<sup>T</sup>
      RAW_PTR(A_vec),  // A pointer, in cublas will be A<sup>T</sup>
      m,               // leading dimension of A<sup>T</sup>
      &beta,           // beta
      RAW_PTR(C_vec),  // C pointer, in cublas will be C<sup>T</sup>
      n,               // leading dimension of C<sup>T</sup>
      batch_count));
}

template <>
void matmul_transpose_gpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast) {
  DEFINE_ABC_VEC(broadcast)
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B)(A<sup>T</sup>)
  // also C = A(B<sup>T</sup>)
  CUBLAS_CHECK(cublasSgemmBatched(
      MyTensorContext::cublas_handle(),  // handle
      CUBLAS_OP_T,                       // no transpose of A<sup>T</sup>
      CUBLAS_OP_N,                       // no transpose of B<sup>T</sup>
      n,       // row number of B and row number of C<sup>T</sup>
      m,       // col number of A<sup>T</sup> and col number of C<sup>T</sup>
      k,       // col number of B and row number of A<sup>T</sup>
      &alpha,  // alpha
      RAW_PTR(B_vec),  // B pointer, in cublas will be B<sup>T</sup>
      k,               // leading dimension of B<sup>T</sup>
      RAW_PTR(A_vec),  // A pointer, in cublas will be A<sup>T</sup>
      k,               // leading dimension of A<sup>T</sup>
      &beta,           // beta
      RAW_PTR(C_vec),  // C pointer, in cublas will be C<sup>T</sup>
      n,               // leading dimension of C<sup>T</sup>
      batch_count));
}

template <>
void transpose_matmul_transpose_gpu(const float *A, const float *B, float *C,
                                    const int m, const int k, const int n,
                                    const int batch_count,
                                    const int broadcast) {
  DEFINE_ABC_VEC(broadcast)
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B)(A)
  // also C = (A<sup>T</sup>)(B<sup>T</sup>)
  CUBLAS_CHECK(cublasSgemmBatched(
      MyTensorContext::cublas_handle(),  // handle
      CUBLAS_OP_T,                       // no transpose of A<sup>T</sup>
      CUBLAS_OP_T,                       // no transpose of B<sup>T</sup>
      n,               // row number of B and row number of C<sup>T</sup>
      m,               // col number of A and col number of C<sup>T</sup>
      k,               // col number of B and row number of A
      &alpha,          // alpha
      RAW_PTR(B_vec),  // B pointer, in cublas will be B<sup>T</sup>
      k,               // leading dimension of B<sup>T</sup>
      RAW_PTR(A_vec),  // A pointer, in cublas will be A<sup>T</sup>
      m,               // leading dimension of A<sup>T</sup>
      &beta,           // beta
      RAW_PTR(C_vec),  // C pointer, in cublas will be C<sup>T</sup>
      n,               // leading dimension of C<sup>T</sup>
      batch_count));
}

namespace {
__global__ void SetAllOnes(float *ones, int n) {
  CUDA_KERNEL_LOOP(i, n) { ones[i] = 1.0f; }
}

__global__ void RepeatVec(const float *vec, float *result, const int m,
                          const int batch_count) {
  CUDA_KERNEL_LOOP(i, m * batch_count) { result[i] = vec[i % m]; }
}
}  // namespace

template <>
void add_row_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count) {
  float alpha = 1.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, n * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(n), kCudaThreadNum>>>(ones, n);
  float *repeat_vec = nullptr;
  cudaMalloc(&repeat_vec, m * batch_count * sizeof(float));
  RepeatVec<<<CudaGetBlocks(m * batch_count), kCudaThreadNum>>>(vec, repeat_vec,
                                                                m, batch_count);
  CUBLAS_CHECK(cublasSger(MyTensorContext::cublas_handle(), n, m * batch_count,
                          &alpha, ones, 1, repeat_vec, 1, mat, n));
  cudaFree(ones);
  cudaFree(repeat_vec);
}

template <>
void add_col_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count) {
  float alpha = 1.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, m * batch_count * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(m * batch_count), kCudaThreadNum>>>(
      ones, m * batch_count);
  CUBLAS_CHECK(cublasSger(MyTensorContext::cublas_handle(), n, m * batch_count,
                          &alpha, vec, 1, ones, 1, mat, n));
  cudaFree(ones);
}

template <>
float tensor_sum_gpu(const float *tensor, const int cnt) {
  return thrust::reduce(thrust::device_pointer_cast(tensor),
                        thrust::device_pointer_cast(tensor + cnt), 0.0f,
                        thrust::plus<float>());
}

template <>
void row_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count) {
  float alpha = 1.0f;
  float beta = 0.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, n * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(n), kCudaThreadNum>>>(ones, n);
  CUBLAS_CHECK(cublasSgemv(MyTensorContext::cublas_handle(), CUBLAS_OP_T, n,
                           m * batch_count, &alpha, mat, n, ones, 1, &beta,
                           result, 1));
  cudaFree(ones);
}

template <>
void col_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count) {
  float alpha = 1.0f;
  float beta = 0.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, m * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(m), kCudaThreadNum>>>(ones, m * batch_count);
  thrust::device_vector<const float *> mat_vec(batch_count);
  int mat_stride = m * n;
  thrust::transform(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(batch_count),
                    mat_vec.begin(),
                    [mat, mat_stride] __device__(int i) -> const float * {
                      return mat + i * mat_stride;
                    });
  thrust::device_vector<const float *> ones_vec(batch_count);
  thrust::fill(ones_vec.begin(), ones_vec.end(), ones);
  thrust::device_vector<float *> result_vec(batch_count);
  thrust::transform(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(batch_count), result_vec.begin(),
      [result, n] __device__(int i) -> float * { return result + i * n; });
  CUBLAS_CHECK(cublasSgemvBatched(MyTensorContext::cublas_handle(), CUBLAS_OP_N,
                                  n, m, &alpha, RAW_PTR(mat_vec), n,
                                  RAW_PTR(ones_vec), 1, &beta,
                                  RAW_PTR(result_vec), 1, batch_count));
  cudaFree(ones);
}

template <>
void add_two_vec_gpu(float *lhs, const float *rhs, const float k, const int n) {
  CUBLAS_CHECK(
      cublasSaxpy(MyTensorContext::cublas_handle(), n, &k, rhs, 1, lhs, 1));
}

#undef DEFINE_ABC_VEC

}  // namespace my_tensor
