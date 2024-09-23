#include <tensor.cuh>
#include <error.h>

#include <numeric>
#include <sstream>

namespace my_tensor {
Tensor::Tensor(const std::vector<int>& shape, DeviceType deviceType)
  : shape_(shape), device_type_(deviceType) {
  AllocateMemory();
}

Tensor::Tensor(const Tensor& tensor)
  : shape_(tensor.shape_), device_type_(tensor.device_type_) {
    AllocateMemory();
}

void Tensor::CopyData(const Tensor& tensor, std::size_t cnt) {
  if (this->OnCPU() && tensor.OnGPU()) {
    cudaMemcpy(data_, tensor.data_, cnt, cudaMemcpyDeviceToHost);
  } else if (this->OnGPU() && tensor.OnCPU()) {
    cudaMemcpy(data_, tensor.data_, cnt, cudaMemcpyHostToDevice);
  } else if (this->OnGPU() && tensor.OnGPU()) {
    cudaMemcpy(data_, tensor.data_, cnt, cudaMemcpyDeviceToDevice);
  } else {
    memcpy(data_, tensor.data_, cnt);
  }
}

std::size_t Tensor::GetBytesCount() const {
  return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>()) * sizeof(float);
}

void Tensor::FreeMemory() {
  if (OnCPU()) {
    free(data_);
  } else {
    cudaFree(data_);
  }
}

void Tensor::AllocateMemory() {
  std::size_t cnt = GetBytesCount();
  if (OnCPU()) {
    data_ = (float*) malloc(cnt);
  } else {
    cudaMalloc(&data_, cnt);
  }
  if (data_ == nullptr) {
    std::stringstream ss;
    ss << "Malloc failed in the line " << __LINE__ << " of the file " << __FILE__;
    throw MemoryError(ss.str());
  }
}

Tensor& Tensor::operator=(const Tensor& tensor) {
  if (this == &tensor) {
    return *this;
  }
  shape_ = tensor.shape_;
  FreeMemory();
  AllocateMemory();
  CopyData(tensor, GetBytesCount());
  device_type_ = tensor.device_type_;
}

Tensor::Tensor(Tensor&& tensor)
  : shape_(tensor.shape_), device_type_(tensor.device_type_), data_(tensor.data_) {
    tensor.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) {
  shape_ = tensor.shape_;
  device_type_ = tensor.device_type_;
  data_ = tensor.data_;
  tensor.data_ = nullptr;
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
  tensor.CopyData(*this, GetBytesCount());
  return tensor;
}
}
