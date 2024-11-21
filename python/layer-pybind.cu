#include "layer.cuh"
#include "relu.cuh"
#include "tensor-facade.cuh"
#include "layer-parameter.h"

namespace py = pybind11;

class ReluFacade {
 public:
  ReluFacade() : relu_(nullptr) {
    my_tensor::LayerParameterPtr param = std::make_shared<my_tensor::ReluParameter>();
    relu_.reset(new my_tensor::Relu<float>(param));
  }

 private:
  std::shared_ptr<my_tensor::Relu<float>> relu_;
};
