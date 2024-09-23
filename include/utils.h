#ifndef MYTENSOR_INCLUDE_UTILS_H_
#define MYTENSOR_INCLUDE_UTILS_H_

constexpr int kCudaThreadNum = 512;

inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadNum - 1) / kCudaThreadNum;
}

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#endif // MYTENSOR_INCLUDE_UTILS_H_
