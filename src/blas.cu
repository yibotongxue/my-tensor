#include <blas.cuh>
#include <handle.cuh>

#include <cublas_v2.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace my_tensor {
extern HandlePtr handle;

#define DEFINE_ABC_VEC(broadcast)\
  int stride_A = (broadcast == 1) ? 0 : (m * k);\
  thrust::device_vector<const float *> A_vec(batch_count);\
  thrust::transform(thrust::counting_iterator<int>(0),\
                    thrust::counting_iterator<int>(batch_count),\
                    A_vec.begin(),\
                    [A, stride_A] __device__ (int i) -> const float* {\
                        return A + i * stride_A;\
                    });\
  int stride_B = (broadcast == 2) ? 0 : (k * n);\
  thrust::device_vector<const float *> B_vec(batch_count);\
  thrust::transform(thrust::counting_iterator<int>(0),\
                    thrust::counting_iterator<int>(batch_count),\
                    B_vec.begin(),\
                    [B, stride_B] __device__ (int i) -> const float* {\
                        return B + i * stride_B;\
                    });\
  thrust::device_vector<float *> C_vec(batch_count);\
  thrust::transform(thrust::counting_iterator<int>(0),\
                    thrust::counting_iterator<int>(batch_count),\
                    C_vec.begin(),\
                    [C, m, n] __device__ (int i) -> float* {\
                        return C + i * m * n;\
                    });

// template <>
// void matmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
//   float alpha = 1.0f;
//   float beta = 0.0f;
//   // C<sup>T</sup> = (B<sup>T</sup>)(A<sup>T</sup>)
//   // also C = AB
//   CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
//     CUBLAS_OP_N,  // no transpose of A<sup>T</sup>
//     CUBLAS_OP_N,  // no transpose of B<sup>T</sup>
//     n,  // row number of B<sup>T</sup> and row number of C<sup>T</sup>
//     m,  // col number of A<sup>T</sup> and col number of C<sup>T</sup>
//     k,  // col number of B<sup>T</sup> and row number of A<sup>T</sup>
//     &alpha,  // alpha
//     B,  // B pointer, in cublas will be B<sup>T</sup>
//     n,  // leading dimension of B<sup>T</sup>
//     A,  // A pointer, in cublas will be A<sup>T</sup>
//     k,  // leading dimension of A<sup>T</sup>
//     &beta,  // beta
//     C,  // C pointer, in cublas will be C<sup>T</sup>
//     n  // leading dimension of C<sup>T</sup>
//   ));
// }

template <>
void transpose_matmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B<sup>T</sup>)(A)
  // also C = (A<sup>T</sup>)B
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_N,  // no transpose of A<sup>T</sup>
    CUBLAS_OP_T,  // no transpose of B<sup>T</sup>
    n,  // row number of B<sup>T</sup> and row number of C<sup>T</sup>
    m,  // col number of A and col number of C<sup>T</sup>
    k,  // col number of B<sup>T</sup> and row number of A
    &alpha,  // alpha
    B,  // B pointer, in cublas will be B<sup>T</sup>
    n,  // leading dimension of B<sup>T</sup>
    A,  // A pointer, in cublas will be A<sup>T</sup>
    m,  // leading dimension of A<sup>T</sup>
    &beta,  // beta
    C,  // C pointer, in cublas will be C<sup>T</sup>
    n  // leading dimension of C<sup>T</sup>
  ));
}

template <>
void matmul_transpose(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B)(A<sup>T</sup>)
  // also C = A(B<sup>T</sup>)
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_T,  // no transpose of A<sup>T</sup>
    CUBLAS_OP_N,  // no transpose of B<sup>T</sup>
    n,  // row number of B and row number of C<sup>T</sup>
    m,  // col number of A<sup>T</sup> and col number of C<sup>T</sup>
    k,  // col number of B and row number of A<sup>T</sup>
    &alpha,  // alpha
    B,  // B pointer, in cublas will be B<sup>T</sup>
    k,  // leading dimension of B<sup>T</sup>
    A,  // A pointer, in cublas will be A<sup>T</sup>
    k,  // leading dimension of A<sup>T</sup>
    &beta,  // beta
    C,  // C pointer, in cublas will be C<sup>T</sup>
    n  // leading dimension of C<sup>T</sup>
  ));
}

template <>
void transpose_matmul_transpose(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B)(A)
  // also C = (A<sup>T</sup>)(B<sup>T</sup>)
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_T,  // no transpose of A<sup>T</sup>
    CUBLAS_OP_T,  // no transpose of B<sup>T</sup>
    n,  // row number of B and row number of C<sup>T</sup>
    m,  // col number of A and col number of C<sup>T</sup>
    k,  // col number of B and row number of A
    &alpha,  // alpha
    B,  // B pointer, in cublas will be B<sup>T</sup>
    k,  // leading dimension of B<sup>T</sup>
    A,  // A pointer, in cublas will be A<sup>T</sup>
    m,  // leading dimension of A<sup>T</sup>
    &beta,  // beta
    C,  // C pointer, in cublas will be C<sup>T</sup>
    n  // leading dimension of C<sup>T</sup>
  ));
}

template <>
void matmul(const float *A, const float *B, float *C, const int m, const int k, const int n, const int batch_count, const int broadcast) {
  float alpha = 1.0f;
  float beta = 0.0f;
  DEFINE_ABC_VEC(broadcast)
  // C<sup>T</sup> = (B<sup>T</sup>)(A<sup>T</sup>)
  // also C = AB
  CUBLAS_ERROR_CHECK(cublasSgemmBatched(handle->GetHandle(),  // handle
    CUBLAS_OP_N,  // no transpose of A<sup>T</sup>
    CUBLAS_OP_N,  // no transpose of B<sup>T</sup>
    n,  // row number of B<sup>T</sup> and row number of C<sup>T</sup>
    m,  // col number of A<sup>T</sup> and col number of C<sup>T</sup>
    k,  // col number of B<sup>T</sup> and row number of A<sup>T</sup>
    &alpha,  // alpha
    thrust::raw_pointer_cast(B_vec.data()),  // B pointer, in cublas will be B<sup>T</sup>
    n,  // leading dimension of B<sup>T</sup>
    thrust::raw_pointer_cast(A_vec.data()),  // A pointer, in cublas will be A<sup>T</sup>
    k,  // leading dimension of A<sup>T</sup>
    &beta,  // beta
    thrust::raw_pointer_cast(C_vec.data()),  // C pointer, in cublas will be C<sup>T</sup>
    n, // leading dimension of C<sup>T</sup>
    batch_count
  ));
}

namespace {
__global__ void SetAllOnes(float *ones, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    ones[i] = 1.0f;
  }
}
}  // namespace

template <>
void add_row_vector(float *mat, const float *vec, const int m, const int n) {
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
    vec,
    1,
    &beta,
    mat,
    n));
  cudaFree(ones);
}

template <>
void add_col_vector(float *mat, const float *vec, const int m, const int n) {
  float alpha = 1.0f;
  float beta = 1.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, m * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(m), kCudaThreadNum>>>(ones, m);
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    n,
    m,
    1,
    &alpha,
    vec,
    n,
    ones,
    1,
    &beta,
    mat,
    n));
  cudaFree(ones);
}

template <>
float tensor_sum(const float *tensor, const int cnt) {
  return thrust::reduce(thrust::device_pointer_cast(tensor),
                        thrust::device_pointer_cast(tensor + cnt),
                        0.0f, thrust::plus<float>());
}

template <>
void row_sum(const float *mat, float *result, const int m, const int n) {
  float *ones = nullptr;
  cudaMalloc(&ones, n * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(n), kCudaThreadNum>>>(ones, n);
  matmul(mat, ones, result, m, n, 1);
  cudaFree(ones);
}

template <>
void col_sum(const float *mat, float *result, const int m, const int n) {
  float *ones = nullptr;
  cudaMalloc(&ones, m * sizeof(float));
  SetAllOnes<<<CudaGetBlocks(m), kCudaThreadNum>>>(ones, m);
  transpose_matmul(mat, ones, result, n, m, 1);
  cudaFree(ones);
}

}  // namespace my_tensor
