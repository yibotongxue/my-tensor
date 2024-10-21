#include <blas.cuh>
#include <handle.cuh>

#include <cublas_v2.h>
#include <iostream>

namespace my_tensor {
template <>
void matmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  float alpha = 1.0f;
  float beta = 0.0f;
  // C<sup>T</sup> = (B<sup>T</sup>)(A<sup>T</sup>)
  // also C = AB
  CUBLAS_ERROR_CHECK(cublasSgemm(handle->GetHandle(),  // handle
    CUBLAS_OP_N,  // no transpose of A<sup>T</sup>
    CUBLAS_OP_N,  // no transpose of B<sup>T</sup>
    n,  // row number of B<sup>T</sup> and row number of C<sup>T</sup>
    m,  // col number of A<sup>T</sup> and col number of C<sup>T</sup>
    k,  // col number of B<sup>T</sup> and row number of A<sup>T</sup>
    &alpha,  // alpha
    B,  // B pointer, in cublas will be B<sup>T</sup>
    n,  // leading dimension of B<sup>T</sup>
    A,  // A pointer, in cublas will be A<sup>T</sup>
    k,  // leading dimension of A<sup>T</sup>
    &beta,  // beta
    C,  // C pointer, in cublas will be C<sup>T</sup>
    n  // leading dimension of C<sup>T</sup>
  ));
}
}  // namespace my_tensor
