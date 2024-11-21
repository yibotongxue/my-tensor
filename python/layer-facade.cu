#include "layer.cuh"
#include "relu.cuh"
#include "tensor-facade.cuh"
#include "layer-parameter.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class ReluFacade {
 public:
  ReluFacade() : relu_(nullptr) {
    my_tensor::LayerParameterPtr param = std::make_shared<my_tensor::ReluParameter>();
    relu_.reset(new my_tensor::Relu<float>(param));
  }

  // 前向传播，缓存输入并返回输出
  TensorFacade Forward(const TensorFacade& input) {
    input_cache_ = input;  // 缓存输入
    TensorFacade output(input.GetShape());  // 创建输出 Tensor

    // 设置并执行 ReLU 前向操作
    relu_->SetUp({input.tensor_}, {output.tensor_});
    relu_->ForwardGPU({input.tensor_}, {output.tensor_});

    return output;  // 返回输出
  }

  // 反向传播，计算并返回梯度
  TensorFacade Backward(const TensorFacade& output) {
    TensorFacade input_grad(input_cache_.GetShape());  // 创建梯度 Tensor

    // 设置并执行 ReLU 反向操作
    relu_->BackwardGPU({output.tensor_}, {input_cache_.tensor_}, {input_grad.tensor_});

    return input_grad;  // 返回输入的梯度
  }

 private:
  std::shared_ptr<my_tensor::Relu<float>> relu_;
  TensorFacade input_cache_;  // 缓存输入
};
