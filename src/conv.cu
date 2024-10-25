#include <blas.cuh>
#include <conv.cuh>
#include <error.h>
#include <im2col.cuh>

namespace my_tensor {

#define DEFINE_SHAPE_DATA\
  const std::vector<int>& bottom_shape = bottom->GetShape();\
  const std::vector<int>& top_shape = top->GetShape();\
  const std::vector<int>& kernel_shape = this->params_[0]->GetShape();\
  int n = bottom_shape[0], c_in = bottom_shape[1];\
  int height = bottom_shape[2], width = bottom_shape[3];\
  int c_out = top_shape[1];\
  int kernel_height = kernel_shape[2];\
  int kernel_width = kernel_shape[3];

template <typename T>
Convolution<T>::Convolution(const std::vector<TensorPtr<T>>& params) : Layer<T>(params) {
  if (this->params_.size() != 1) {
    throw ConvError("Convolution parameters number should be 1.");
  }
  const std::vector<int>& kernel_shape = this->params_[0]->GetShape();
  if (kernel_shape.size() != 4) {
    throw ConvError("Convolution kernel should be 4 dimension tensor.");
  }
}

template <typename T>
void Convolution<T>::ForwardCPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
  const auto& kernel_data = this->params_[0]->GetCPUData();
  const auto& bottom_data = bottom->GetCPUData();
  auto& top_data = top->GetCPUData();
  std::vector<int> temp_shape {n,
      c_in * kernel_height * kernel_width, height * width};
  auto temp_tensor = std::make_shared<Tensor<T>>(temp_shape);
  Im2col_CPU(n, bottom->GetCPUDataPtr(), c_in, height, width,
      kernel_height, kernel_width, temp_tensor->GetCPUDataPtr());
  const auto& temp_data = temp_tensor->GetCPUData();
  int kernel_size = kernel_height * kernel_width;
  int im_size = height * width;
  for (int t = 0; t < n; t++) {
    for (int i = 0; i < c_out; i++) {
      for (int j = 0; j < im_size; j++) {
        for (int k = 0; k < c_in * kernel_size; k++) {
          top_data[t * c_out * im_size + i * im_size + j]
              += kernel_data[i * c_in * kernel_size + k]
              * temp_data[t * c_in * im_size * kernel_size + k * im_size + j];
        }
      }
    }
  }
}

template <typename T>
void Convolution<T>::BackwardCPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
}

template <typename T>
void Convolution<T>::ForwardGPU(const TensorPtr<T>& bottom, TensorPtr<T>& top) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
  std::vector<int> temp_shape {n,
      c_in * kernel_height * kernel_width, height * width};
  auto temp_tensor = std::make_shared<Tensor<T>>(temp_shape);
  Im2col_GPU(n, bottom->GetGPUDataPtr(), c_in, height, width,
      kernel_height, kernel_width, temp_tensor->GetGPUDataPtr());
  int kernel_size = kernel_height * kernel_width;
  int im_size = height * width;
  matmul(this->params_[0]->GetGPUDataPtr(), temp_tensor->GetGPUDataPtr(),
      top->GetGPUDataPtr(), c_out, c_in * kernel_size, im_size, n, 1);
}

template <typename T>
void Convolution<T>::BackwardGPU(const TensorPtr<T>& top, TensorPtr<T>& bottom) {
  CheckShape(bottom, top);
  DEFINE_SHAPE_DATA
}

template <typename T>
void Convolution<T>::CheckShape(const TensorPtr<T>& bottom, const TensorPtr<T>& top) const {
  const std::vector<int>& kernel_shape = this->params_[0]->GetShape();
  const std::vector<int>& bottom_shape = bottom->GetShape();
  const std::vector<int>& top_shape = top->GetShape();
  if (bottom_shape.size() != 4) {
    throw ConvError("The dimension of the input of convolution layer should be 4.");
  }
  if (top_shape.size() != 4) {
    throw ConvError("The dimension of the output of convolution layer should be 4.");
  }
  if (bottom_shape[0] != top_shape[0]) {
    throw ConvError("The batch size of input and output of convolution layer not match.");
  }
  if (bottom_shape[2] != top_shape[2]) {
    throw ConvError("The height of input and output of convolution layer not match.");
  }
  if (bottom_shape[3] != top_shape[3]) {
    throw ConvError("The width size of input and output of convolution layer not match.");
  }
  if (bottom_shape[1] != kernel_shape[1]) {
    throw ConvError("The input channels not match kernel.");
  }
  if (top_shape[1] != kernel_shape[0]) {
    throw ConvError("The output channels not match kernel.");
  }
}

template class Convolution<>;

#undef DEFINE_SHAPE_DATA

}  // namespace my_tensor
