// Copyright 2024 yibotongxue

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "blas.hpp"
#include "cuda-context.hpp"

namespace my_tensor {

template <typename T>
__global__ static void AssignRisingValues(T *data, const int n, T start,
                                          int stride) {
  CUDA_KERNEL_LOOP(i, n) { data[i] = start + i * stride; }
}

#define DEFINE_ABC_VEC(broadcast)                                          \
  const_float_ptr *A_vec_raw = nullptr;                                    \
  CUDA_CHECK(cudaMalloc(&A_vec_raw, batch_count * sizeof(const float *))); \
  int stride_A = (broadcast == 1) ? 0 : (m * k);                           \
  AssignRisingValues<<<CudaGetBlocks(batch_count), kCudaThreadNum>>>(      \
      A_vec_raw, batch_count, A, stride_A);                                \
  const_float_ptr *B_vec_raw = nullptr;                                    \
  CUDA_CHECK(cudaMalloc(&B_vec_raw, batch_count * sizeof(const float *))); \
  int stride_B = (broadcast == 2) ? 0 : (k * n);                           \
  AssignRisingValues<<<CudaGetBlocks(batch_count), kCudaThreadNum>>>(      \
      B_vec_raw, batch_count, B, stride_B);                                \
  float **C_vec_raw = nullptr;                                             \
  CUDA_CHECK(cudaMalloc(&C_vec_raw, batch_count * sizeof(float *)));       \
  AssignRisingValues<<<CudaGetBlocks(batch_count), kCudaThreadNum>>>(      \
      C_vec_raw, batch_count, C, m * n);

#define FREE_ABC_VEC               \
  CUDA_CHECK(cudaFree(A_vec_raw)); \
  CUDA_CHECK(cudaFree(B_vec_raw)); \
  CUDA_CHECK(cudaFree(C_vec_raw));

using const_float_ptr = const float *;

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
      CudaContext::cublas_handle(),  // handle
      CUBLAS_OP_N,                   // no transpose of A<sup>T</sup>
      CUBLAS_OP_N,                   // no transpose of B<sup>T</sup>
      n,          // row number of B<sup>T</sup> and row number of C<sup>T</sup>
      m,          // col number of A<sup>T</sup> and col number of C<sup>T</sup>
      k,          // col number of B<sup>T</sup> and row number of A<sup>T</sup>
      &alpha,     // alpha
      B_vec_raw,  // B pointer, in cublas will be B<sup>T</sup>
      n,          // leading dimension of B<sup>T</sup>
      A_vec_raw,  // A pointer, in cublas will be A<sup>T</sup>
      k,          // leading dimension of A<sup>T</sup>
      &beta,      // beta
      C_vec_raw,  // C pointer, in cublas will be C<sup>T</sup>
      n,          // leading dimension of C<sup>T</sup>
      batch_count));
  FREE_ABC_VEC
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
      CudaContext::cublas_handle(),  // handle
      CUBLAS_OP_N,                   // no transpose of A<sup>T</sup>
      CUBLAS_OP_T,                   // no transpose of B<sup>T</sup>
      n,          // row number of B<sup>T</sup> and row number of C<sup>T</sup>
      m,          // col number of A and col number of C<sup>T</sup>
      k,          // col number of B<sup>T</sup> and row number of A
      &alpha,     // alpha
      B_vec_raw,  // B pointer, in cublas will be B<sup>T</sup>
      n,          // leading dimension of B<sup>T</sup>
      A_vec_raw,  // A pointer, in cublas will be A<sup>T</sup>
      m,          // leading dimension of A<sup>T</sup>
      &beta,      // beta
      C_vec_raw,  // C pointer, in cublas will be C<sup>T</sup>
      n,          // leading dimension of C<sup>T</sup>
      batch_count));
  FREE_ABC_VEC
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
      CudaContext::cublas_handle(),  // handle
      CUBLAS_OP_T,                   // no transpose of A<sup>T</sup>
      CUBLAS_OP_N,                   // no transpose of B<sup>T</sup>
      n,          // row number of B and row number of C<sup>T</sup>
      m,          // col number of A<sup>T</sup> and col number of C<sup>T</sup>
      k,          // col number of B and row number of A<sup>T</sup>
      &alpha,     // alpha
      B_vec_raw,  // B pointer, in cublas will be B<sup>T</sup>
      k,          // leading dimension of B<sup>T</sup>
      A_vec_raw,  // A pointer, in cublas will be A<sup>T</sup>
      k,          // leading dimension of A<sup>T</sup>
      &beta,      // beta
      C_vec_raw,  // C pointer, in cublas will be C<sup>T</sup>
      n,          // leading dimension of C<sup>T</sup>
      batch_count));
  FREE_ABC_VEC
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
      CudaContext::cublas_handle(),  // handle
      CUBLAS_OP_T,                   // no transpose of A<sup>T</sup>
      CUBLAS_OP_T,                   // no transpose of B<sup>T</sup>
      n,          // row number of B and row number of C<sup>T</sup>
      m,          // col number of A and col number of C<sup>T</sup>
      k,          // col number of B and row number of A
      &alpha,     // alpha
      B_vec_raw,  // B pointer, in cublas will be B<sup>T</sup>
      k,          // leading dimension of B<sup>T</sup>
      A_vec_raw,  // A pointer, in cublas will be A<sup>T</sup>
      m,          // leading dimension of A<sup>T</sup>
      &beta,      // beta
      C_vec_raw,  // C pointer, in cublas will be C<sup>T</sup>
      n,          // leading dimension of C<sup>T</sup>
      batch_count));
  FREE_ABC_VEC
}

namespace {
__global__ void SetAllValue(float *ones, int n, float value) {
  CUDA_KERNEL_LOOP(i, n) { ones[i] = value; }
}

__global__ void RepeatVec(const float *vec, float *result, const int m,
                          const int batch_count) {
  CUDA_KERNEL_LOOP(i, m * batch_count) { result[i] = vec[i % m]; }
}
}  // namespace

template <>
void add_row_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale) {
  float alpha = 1.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, n * sizeof(float));
  SetAllValue<<<CudaGetBlocks(n), kCudaThreadNum>>>(ones, n, scale);
  float *repeat_vec = nullptr;
  cudaMalloc(&repeat_vec, m * batch_count * sizeof(float));
  RepeatVec<<<CudaGetBlocks(m * batch_count), kCudaThreadNum>>>(vec, repeat_vec,
                                                                m, batch_count);
  CUBLAS_CHECK(cublasSger(CudaContext::cublas_handle(), n, m * batch_count,
                          &alpha, ones, 1, repeat_vec, 1, mat, n));
  cudaFree(ones);
  cudaFree(repeat_vec);
}

template <>
void add_col_vector_gpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale) {
  float alpha = 1.0f;
  float *ones = nullptr;
  cudaMalloc(&ones, m * batch_count * sizeof(float));
  SetAllValue<<<CudaGetBlocks(m * batch_count), kCudaThreadNum>>>(
      ones, m * batch_count, scale);
  CUBLAS_CHECK(cublasSger(CudaContext::cublas_handle(), n, m * batch_count,
                          &alpha, vec, 1, ones, 1, mat, n));
  cudaFree(ones);
}

namespace {
template <typename T>
__global__ void MutiplyRowVecDevice(T *mat, const T *vec, const int m,
                                    const int n, const int total_cnt) {
  extern __shared__ T shared_vec[];
  CUDA_BLOCK_LOOP(i, m) { shared_vec[i] = vec[i]; }
  __syncthreads();
  CUDA_KERNEL_LOOP(i, total_cnt) {
    int row_idx = (i / n) % m;
    mat[i] *= shared_vec[row_idx];
  }
}
}  // namespace

template <typename T>
void multiply_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count) {
  int total_cnt = m * n * batch_count;
  MutiplyRowVecDevice<<<CudaGetBlocks(total_cnt), kCudaThreadNum,
                        m * sizeof(T)>>>(mat, vec, m, n, total_cnt);
}

template void multiply_row_vector_gpu<float>(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

namespace {
template <typename T>
__global__ void MultiplyColVecDevice(T *mat, const T *vec, const int m,
                                     const int n, const int total_cnt) {
  CUDA_KERNEL_LOOP(i, total_cnt) {
    int col_idx = i % n;
    mat[i] *= vec[col_idx];
  }
}
}  // namespace

template <typename T>
void multiply_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count) {
  int total_cnt = m * n * batch_count;
  MultiplyColVecDevice<<<CudaGetBlocks(total_cnt), kCudaThreadNum>>>(
      mat, vec, m, n, total_cnt);
}

template void multiply_col_vector_gpu<float>(float *mat, const float *vec,
                                             const int m, const int n,
                                             const int batch_count);

namespace {
template <typename T>
__global__ void DivideRowVecDevice(T *mat, const T *vec, const int m,
                                   const int n, const int total_cnt,
                                   const T eps) {
  extern __shared__ T shared_vec[];
  CUDA_BLOCK_LOOP(i, m) { shared_vec[i] = vec[i]; }
  __syncthreads();
  CUDA_KERNEL_LOOP(i, total_cnt) {
    int row_idx = (i / n) % m;
    mat[i] /= (shared_vec[row_idx] + eps);
  }
}
}  // namespace

template <typename T>
void divide_row_vector_gpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count, const T eps) {
  int total_cnt = m * n * batch_count;
  DivideRowVecDevice<<<CudaGetBlocks(total_cnt), kCudaThreadNum,
                       m * sizeof(T)>>>(mat, vec, m, n, total_cnt, eps);
}

template void divide_row_vector_gpu<float>(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

namespace {
template <typename T>
__global__ void DivideColVecDevice(T *mat, const T *vec, const int m,
                                   const int n, const int total_cnt,
                                   const T eps) {
  CUDA_KERNEL_LOOP(i, total_cnt) {
    int col_idx = i % n;
    mat[i] /= (vec[col_idx] + eps);
  }
}
}  // namespace

template <typename T>
void divide_col_vector_gpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count, const T eps) {
  int total_cnt = m * n * batch_count;
  DivideColVecDevice<<<CudaGetBlocks(total_cnt), kCudaThreadNum>>>(
      mat, vec, m, n, total_cnt, eps);
}

template void divide_col_vector_gpu<float>(float *mat, const float *vec,
                                           const int m, const int n,
                                           const int batch_count,
                                           const float eps);

template <>
float tensor_sum_gpu(const float *tensor, const int cnt) {
  return thrust::reduce(thrust::device_pointer_cast(tensor),
                        thrust::device_pointer_cast(tensor + cnt), 0.0f,
                        thrust::plus<float>());
}

template <>
void row_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count, float *help_vec) {
  float alpha = 1.0f;
  float beta = 0.0f;
  bool need_malloc_help_vec = help_vec == nullptr;
  if (need_malloc_help_vec) {
    CUDA_CHECK(cudaMalloc(&help_vec, n * sizeof(float)));
    SetAllValue<<<CudaGetBlocks(n), kCudaThreadNum>>>(help_vec, n, 1.0f);
  }
  CUBLAS_CHECK(cublasSgemv(CudaContext::cublas_handle(), CUBLAS_OP_T, n,
                           m * batch_count, &alpha, mat, n, help_vec, 1, &beta,
                           result, 1));
  if (need_malloc_help_vec) {
    CUDA_CHECK(cudaFree(help_vec));
  }
}

template <>
void col_sum_gpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count, float *help_vec) {
  float alpha = 1.0f;
  float beta = 0.0f;
  bool need_malloc_help_vec = help_vec == nullptr;
  if (need_malloc_help_vec) {
    CUDA_CHECK(cudaMalloc(&help_vec, m * sizeof(float)));
    SetAllValue<<<CudaGetBlocks(m), kCudaThreadNum>>>(help_vec, m, 1.0f);
  }
  thrust::device_vector<const float *> mat_vec(batch_count);
  int mat_stride = m * n;
  thrust::transform(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(batch_count),
                    mat_vec.begin(),
                    [mat, mat_stride] __device__(int i) -> const float * {
                      return mat + i * mat_stride;
                    });
  thrust::device_vector<const float *> ones_vec(batch_count);
  thrust::fill(ones_vec.begin(), ones_vec.end(), help_vec);
  thrust::device_vector<float *> result_vec(batch_count);
  thrust::transform(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(batch_count), result_vec.begin(),
      [result, n] __device__(int i) -> float * { return result + i * n; });
  CUBLAS_CHECK(cublasSgemvBatched(
      CudaContext::cublas_handle(), CUBLAS_OP_N, n, m, &alpha, RAW_PTR(mat_vec),
      n, RAW_PTR(ones_vec), 1, &beta, RAW_PTR(result_vec), 1, batch_count));
  if (need_malloc_help_vec) {
    CUDA_CHECK(cudaFree(help_vec));
  }
}

template <>
void add_two_vec_gpu(float *lhs, const float *rhs, const float k, const int n) {
  CUBLAS_CHECK(
      cublasSaxpy(CudaContext::cublas_handle(), n, &k, rhs, 1, lhs, 1));
}

template <>
void scale_gpu(float *x, const int n, const float k) {
  CUBLAS_CHECK(cublasSscal(CudaContext::cublas_handle(), n, &k, x, 1));
}

template <typename T>
void square_gpu(const T *x, T *y, const int n) {
  thrust::device_ptr<const T> x_ptr(x);
  thrust::device_ptr<T> y_ptr(y);
  thrust::transform(x_ptr, x_ptr + n, y_ptr, thrust::square<T>());
}

template void square_gpu<float>(const float *x, float *y, const int n);

template <typename T>
void sqrt_gpu(const T *x, T *y, const int n) {
  thrust::device_ptr<const T> x_ptr(x);
  thrust::device_ptr<T> y_ptr(y);
  thrust::transform(x_ptr, x_ptr + n, y_ptr, [] __device__(T a) {
    return static_cast<T>(std::sqrt(a));
  });
}

template void sqrt_gpu<float>(const float *x, float *y, const int n);

template <typename T>
void divide_two_vec_gpu(const T *lhs, const T *rhs, T *result, const int n) {
  thrust::device_ptr<const T> lhs_ptr(lhs);
  thrust::device_ptr<const T> rhs_ptr(rhs);
  thrust::device_ptr<T> result_ptr(result);
  thrust::transform(lhs_ptr, lhs_ptr + n, rhs_ptr, result_ptr,
                    thrust::divides<T>());
}

template void divide_two_vec_gpu<float>(const float *lhs, const float *rhs,
                                        float *result, const int n);

template <typename T>
void multiply_two_vec_gpu(const T *lhs, const T *rhs, T *result, const int n) {
  thrust::device_ptr<const T> lhs_ptr(lhs);
  thrust::device_ptr<const T> rhs_ptr(rhs);
  thrust::device_ptr<T> result_ptr(result);
  thrust::transform(lhs_ptr, lhs_ptr + n, rhs_ptr, result_ptr,
                    thrust::multiplies<T>());
}

template void multiply_two_vec_gpu<float>(const float *lhs, const float *rhs,
                                          float *result, const int n);

template <typename T>
void vec_add_num_gpu(const T *vec, T *result, const T num, const int n) {
  thrust::device_ptr<const T> vec_ptr(vec);
  thrust::device_ptr<T> result_ptr(result);
  thrust::transform(vec_ptr, vec_ptr + n, result_ptr,
                    [num] __device__(T a) { return a + num; });
}

template void vec_add_num_gpu<float>(const float *vec, float *result,
                                     const float num, const int n);

template <typename T>
void vec_divide_num_gpu(const T *vec, T *result, const T num, const int n) {
  thrust::device_ptr<const T> vec_ptr(vec);
  thrust::device_ptr<T> result_ptr(result);
  thrust::transform(vec_ptr, vec_ptr + n, result_ptr,
                    [num] __device__(T a) { return a / num; });
}

template void vec_divide_num_gpu<float>(const float *vec, float *result,
                                        const float num, const int n);

#undef DEFINE_ABC_VEC
#undef FREE_ABC_VEC

}  // namespace my_tensor
