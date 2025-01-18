// Copyright 2024 yibotongxue

#include "blas.hpp"

#include <cblas.h>
#include <string.h>

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
                        const int batch_count, const float scale) {
  for (int i = 0; i < batch_count; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < n; k++) {
        mat[i * m * n + j * n + k] += vec[j] * scale;
      }
    }
  }
}

template <>
void add_col_vector_cpu(float *mat, const float *vec, const int m, const int n,
                        const int batch_count, const float scale) {
  for (int i = 0; i < batch_count * m; i++) {
    for (int j = 0; j < n; j++) {
      mat[i * n + j] += vec[j] * scale;
    }
  }
}

template <typename T>
void multiply_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count) {
  for (int i = 0; i < batch_count; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < n; k++) {
        mat[i * m * n + j * n + k] *= vec[j];
      }
    }
  }
}

template void multiply_row_vector_cpu(float *mat, const float *vec, const int m,
                                      const int n, const int batch_count);

template <typename T>
void multiply_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                             const int batch_count) {
  for (int i = 0; i < batch_count * m; i++) {
    for (int j = 0; j < n; j++) {
      mat[i * n + j] *= vec[j];
    }
  }
}

template void multiply_col_vector_cpu(float *mat, const float *vec, const int m,
                                      const int n, const int batch_count);

template <typename T>
void divide_row_vector_cpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count, const T eps) {
  for (int i = 0; i < batch_count; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < n; k++) {
        mat[i * m * n + j * n + k] /= (vec[j] + eps);
      }
    }
  }
}

template void divide_row_vector_cpu(float *mat, const float *vec, const int m,
                                    const int n, const int batch_count,
                                    const float eps);

template <typename T>
void divide_col_vector_cpu(T *mat, const T *vec, const int m, const int n,
                           const int batch_count, const T eps) {
  for (int i = 0; i < batch_count * m; i++) {
    for (int j = 0; j < n; j++) {
      mat[i * n + j] /= (vec[j] + eps);
    }
  }
}

template void divide_col_vector_cpu(float *mat, const float *vec, const int m,
                                    const int n, const int batch_count,
                                    const float eps);

template <>
float tensor_sum_cpu(const float *tensor, const int cnt) {
  float sum = 0.0f;
  for (int i = 0; i < cnt; i++) {
    sum += tensor[i];
  }
  return sum;
}

template <>
void row_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count) {
  memset(result, 0, sizeof(float) * batch_count * m);
  for (int i = 0; i < batch_count * m; i++) {
    for (int j = 0; j < n; j++) {
      result[i] += mat[i * n + j];
    }
  }
}

template <>
void col_sum_cpu(const float *mat, float *result, const int m, const int n,
                 const int batch_count) {
  memset(result, 0, sizeof(float) * batch_count * n);
  for (int i = 0; i < batch_count; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < n; k++) {
        result[i * n + k] += mat[i * m * n + j * n + k];
      }
    }
  }
}

template <>
void add_two_vec_cpu(float *lhs, const float *rhs, const float k, const int n) {
  cblas_saxpy(n, k, rhs, 1, lhs, 1);
}

template <>
void scale_cpu(float *x, const int n, const float k) {
  cblas_sscal(n, k, x, 1);
}

template <typename T>
void square_cpu(const T *x, T *y, const int n) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] * x[i];
  }
}

template void square_cpu(const float *x, float *y, const int n);

template <typename T>
void sqrt_cpu(const T *x, T *y, const int n) {
  for (int i = 0; i < n; i++) {
    y[i] = std::sqrt(x[i]);
  }
}

template void sqrt_cpu(const float *x, float *y, const int n);

template <typename T>
void divide_two_vec_cpu(const T *lhs, const T *rhs, T *result, const int n) {
  for (int i = 0; i < n; i++) {
    result[i] = lhs[i] / rhs[i];
  }
}

template void divide_two_vec_cpu(const float *lhs, const float *rhs,
                                 float *result, const int n);

template <typename T>
void multiply_two_vec_cpu(const T *lhs, const T *rhs, T *result, const int n) {
  for (int i = 0; i < n; i++) {
    result[i] = lhs[i] * rhs[i];
  }
}

template void multiply_two_vec_cpu(const float *lhs, const float *rhs,
                                   float *result, const int n);

template <typename T>
void vec_add_num_cpu(const T *vec, T *result, const T num, const int n) {
  for (int i = 0; i < n; i++) {
    result[i] = vec[i] + num;
  }
}

template void vec_add_num_cpu(const float *vec, float *result, const float num,
                              const int n);

template <typename T>
void vec_divide_num_cpu(const T *vec, T *result, const T num, const int n) {
  for (int i = 0; i < n; i++) {
    result[i] = vec[i] / num;
  }
}

template void vec_divide_num_cpu(const float *vec, float *result,
                                 const float num, const int n);

}  // namespace my_tensor
