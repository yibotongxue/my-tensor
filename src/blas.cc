// Copyright 2024 yibotongxue

#include "blas.hpp"

#include <cblas.h>

namespace my_tensor {

template <>
void matmul_cpu(const float *A, const float *B, float *C, const int m,
                const int k, const int n, const int batch_count,
                const int broadcast) {
  float alpha = 1.0f;
  float beta = 0.0f;

  int stride_A = (broadcast == 1) ? 0 : m * k;
  int stride_B = (broadcast == 2) ? 0 : k * n;

  for (int i = 0; i < batch_count; i++) {
    const float *A_batch = A + i * stride_A;
    const float *B_batch = B + i * stride_B;
    float *C_batch = C + i * m * n;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                A_batch, k, B_batch, n, beta, C_batch, n);
  }
}

template <>
void transpose_matmul_cpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast) {
  float alpha = 1.0f;
  float beta = 0.0f;

  int stride_A = (broadcast == 1) ? 0 : m * k;
  int stride_B = (broadcast == 2) ? 0 : k * n;

  for (int i = 0; i < batch_count; i++) {
    const float *A_batch = A + i * stride_A;
    const float *B_batch = B + i * stride_B;
    float *C_batch = C + i * m * n;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha,
                A_batch, m, B_batch, n, beta, C_batch, n);
  }
}

template <>
void matmul_transpose_cpu(const float *A, const float *B, float *C, const int m,
                          const int k, const int n, const int batch_count,
                          const int broadcast) {
  float alpha = 1.0f;
  float beta = 0.0f;

  int stride_A = (broadcast == 1) ? 0 : m * k;
  int stride_B = (broadcast == 2) ? 0 : k * n;

  for (int i = 0; i < batch_count; i++) {
    const float *A_batch = A + i * stride_A;
    const float *B_batch = B + i * stride_B;
    float *C_batch = C + i * m * n;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                A_batch, k, B_batch, k, beta, C_batch, n);
  }
}

template <>
void transpose_matmul_transpose_cpu(const float *A, const float *B, float *C,
                                    const int m, const int k, const int n,
                                    const int batch_count,
                                    const int broadcast) {
  float alpha = 1.0f;
  float beta = 0.0f;

  int stride_A = (broadcast == 1) ? 0 : m * k;
  int stride_B = (broadcast == 2) ? 0 : k * n;

  for (int i = 0; i < batch_count; i++) {
    const float *A_batch = A + i * stride_A;
    const float *B_batch = B + i * stride_B;
    float *C_batch = C + i * m * n;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, k, alpha, A_batch,
                m, B_batch, k, beta, C_batch, n);
  }
}

template <>
void add_row_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count) {
  // TODO(yibotongxue)
}

template <>
void add_col_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count) {
  // TODO(yibotongxue)
}

template <>
float tensor_sum_cpu(const float *tensor, const int cnt) {
  // TODO(yibotongxue)
  return 0.0f;
}

template <>
void row_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count) {
  // TODO(yibotongxue)
}

template <>
void col_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count) {
  // TODO(yibotongxue)
}

}  // namespace my_tensor
