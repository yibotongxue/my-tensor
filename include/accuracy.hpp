// Copyright 2024 yibotongxue

#ifndef INCLUDE_ACCURACY_HPP_
#define INCLUDE_ACCURACY_HPP_

#include <vector>

#include "layer-parameter.hpp"
#include "layer.hpp"
#include "utils.hpp"

namespace my_tensor {

/**
 * @brief Accuracy layer.
 *
 * @tparam T Data type.
 *
 * This layer is used to calculate the accuracy of the model.
 * The input of this layer should be the output of the model and the label.
 * The output of this layer is the accuracy.
 */
template <typename T = float>
class Accuracy final : public Layer<T> {
 public:
  explicit Accuracy(LayerParameterPtr param) : Layer<T>(param) {}

  DISABLE_LAYER_COPY(Accuracy)

  void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                        const std::vector<TensorPtr<T>>& top) const override;
  void Reshape(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) const override;
  void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;

  virtual ~Accuracy() = default;

  /**
   * @brief Forward calculation on CPU.
   *
   * @param bottom Input tensor.
   * @param top Output tensor.
   *
   * The forward calculation is to compare the output of the model with the
   * label. The accuracy is calculated by the number of correct predictions
   * divided by the total number of predictions.
   */
  void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

  void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                  const std::vector<TensorPtr<T>>& top) override;
  void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                   const std::vector<TensorPtr<T>>& bottom) override;

 private:
  int batch_size_;
  int features_;

  void CheckShape(const TensorPtr<T> input, const TensorPtr<T> label,
                  const TensorPtr<T> output) const;
};

extern template class my_tensor::Accuracy<>;

}  // namespace my_tensor

#endif  // INCLUDE_ACCURACY_HPP_
