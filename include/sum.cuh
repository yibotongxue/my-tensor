#ifndef INCLUDE_SUM_CUH_
#define INCLUDE_SUM_CUH_

#include <tensor.cuh>

#include <memory>

namespace my_tensor {
void Sum(float *result, const std::shared_ptr<Tensor> tensor);
}  // namespace my_tensor


#endif  // INCLUDE_SUM_CUH_
