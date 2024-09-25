#include <tensor.cuh>
#include <utils.cuh>

#include <numeric>
#include <iostream>

namespace my_tensor {
Tensor::Tensor(const std::vector<int>& shape, DeviceType deviceType)
  : shape_(shape), device_type_(deviceType) {
  size_ = std::accumulate(
    shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  AllocateMemory();
}

Tensor::Tensor(const Tensor& tensor)
  : shape_(tensor.shape_), device_type_(tensor.device_type_),
    size_(tensor.size_) {
    AllocateMemory();
    CopyData(tensor, size_ * sizeof(float));
}

void Tensor::CopyData(const Tensor& tensor, std::size_t cnt) {
  if (this->OnCPU() && tensor.OnGPU()) {
    ERROR_CHECK(cudaMemcpy(data_, tensor.data_, cnt, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaMemcpy(diff_, tensor.diff_, cnt, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  } else if (this->OnGPU() && tensor.OnCPU()) {
    ERROR_CHECK(cudaMemcpy(data_, tensor.data_, cnt, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(diff_, tensor.diff_, cnt, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
  } else if (this->OnGPU() && tensor.OnGPU()) {
    ERROR_CHECK(cudaMemcpy(data_, tensor.data_, cnt, cudaMemcpyDeviceToDevice));
    ERROR_CHECK(cudaMemcpy(diff_, tensor.diff_, cnt, cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
  } else {
    memcpy(data_, tensor.data_, cnt);
    memcpy(diff_, tensor.diff_, cnt);
  }
}

void Tensor::FreeMemory() {
  if (OnCPU()) {
    free(data_);
    free(diff_);
  } else {
    ERROR_CHECK(cudaFree(data_));
    ERROR_CHECK(cudaFree(diff_));
    cudaDeviceSynchronize();
  }
  data_ = nullptr;
  diff_ = nullptr;
}

void Tensor::AllocateMemory() {
  std::size_t cnt = size_ * sizeof(float);
  if (OnCPU()) {
    data_ = reinterpret_cast<float*>(malloc(cnt));
    diff_ = reinterpret_cast<float*>(malloc(cnt));
  } else {
    ERROR_CHECK(cudaMalloc(&data_, cnt));
    ERROR_CHECK(cudaMalloc(&diff_, cnt));
    cudaDeviceSynchronize();
  }
  if (data_ == nullptr || diff_ == nullptr) {
    std::cerr << "Malloc failed in the line " << __LINE__
     << " of the file " << __FILE__ << std::endl;
    throw std::bad_alloc();
  }
}

Tensor& Tensor::operator=(const Tensor& tensor) {
  if (this == &tensor) {
    return *this;
  }
  shape_ = tensor.shape_;
  size_ = tensor.size_;
  FreeMemory();
  device_type_ = tensor.device_type_;
  AllocateMemory();
  CopyData(tensor, size_ * sizeof(float));
  return *this;
}

Tensor::Tensor(Tensor&& tensor)
  : shape_(tensor.shape_), device_type_(tensor.device_type_),
    size_(tensor.size_),
    data_(tensor.data_), diff_(tensor.diff_) {
    tensor.data_ = nullptr;
    tensor.diff_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) {
  shape_ = tensor.shape_;
  device_type_ = tensor.device_type_;
  size_ = tensor.size_;
  data_ = tensor.data_;
  diff_ = tensor.diff_;
  tensor.data_ = nullptr;
  tensor.diff_ = nullptr;
  return *this;
}

Tensor::~Tensor() {
  FreeMemory();
}

Tensor Tensor::cpu() {
  return Clone(DeviceType::CPU);
}

Tensor Tensor::gpu() {
  return Clone(DeviceType::GPU);
}

Tensor Tensor::Clone(DeviceType device_type) {
  Tensor tensor { shape_, device_type };
  tensor.CopyData(*this, size_ * sizeof(float));
  return tensor;
}
}  // namespace my_tensor
