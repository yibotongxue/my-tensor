#include <sum.cuh>
#include <utils.cuh>

namespace my_tensor {
namespace {
__global__ void CudaSum(float* result, float* data, int n) {
  int thread_id = threadIdx.x;
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float shared[];
  shared[thread_id] = global_id < n ? *(data + global_id) : 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (thread_id < s) {
      shared[thread_id] += shared[thread_id + s];
      __syncthreads();
    }
  }
  if (thread_id == 0) {
    *(result + blockIdx.x) = shared[0];
  }
}
}  // namespace

void Sum(float* result, const std::shared_ptr<Tensor> tensor) {
  if (tensor->OnCPU()) {
    int n = tensor->GetSize();
    float sum = 0;
    const float *data = tensor->GetData();
    for (int i = 0; i < n; i++) {
      sum += *(data + i);
    }
    *result = sum;
  } else {
    int n = tensor->GetSize();
    float *data = nullptr;
    cudaMalloc(&data, n * sizeof(float));
    cudaMemcpy(data, tensor->GetData(), n * sizeof(float), cudaMemcpyDeviceToDevice);
    float *temp_sum;
    cudaMalloc(&temp_sum, CudaGetBlocks(n) * sizeof(float));
    int num = n;
    std::size_t shared_bytes = kCudaThreadNum * sizeof(float);
    while (num > 1) {
      int block_num = CudaGetBlocks(num);
      CudaSum<<<block_num, kCudaThreadNum, shared_bytes>>>(temp_sum, data, num);
      num = block_num;
      std::swap(data, temp_sum);
    }
    cudaMemcpy(result, data, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(data);
    data = nullptr;
    cudaFree(temp_sum);
    temp_sum = nullptr;
  }
}
}  // namespace my_tensor
