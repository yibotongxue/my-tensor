// Copyright 2024 yibotongxue

#ifndef INCLUDE_NET_HPP_
#define INCLUDE_NET_HPP_

#define CPU_ONLY

#include <memory>
#include <vector>

#include "data-loader.hpp"
#include "dataset.hpp"
#include "layer-parameter.hpp"
#include "layer.hpp"
#include "tensor.hpp"

namespace my_tensor {
/**
 * @brief The class represents net.
 * @note The Net class now only support one input, one output, and one bottom
 * and top except loss layer. So Don't Add Mutiple Bottom Or Mutiple Top Layer
 * Except Loss Layer.
 *
 * @todo Add mutiple bottom and mutiple top hidden layers support.
 */
template <typename T>
class Net {
 public:
  explicit Net(const std::vector<LayerParameterPtr>& layers);

  /**
   * @brief Fetch data and update input and label.
   * @return true if data fetched succefully, false if all the batches are
   * fetched.
   */
  bool RefetchData();

  /**
   * @brief Forward method, calling all the forward methods of the layers in a
   * specific order.
   */
  void Forward() {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->Forward(bottom_vec_[i], top_vec_[i]);
    }
  }

  /**
   * @brief Backward method, calling all the backward methods of the layers in a
   * specific order.
   */
  void Backward() {
    for (int i = layers_.size() - 1; i >= 0; i--) {
      layers_[i]->Backward(top_vec_[i], bottom_vec_[i]);
    }
  }

  /**
   * @brief Return all the learnable parameters in the net.
   * @return all the learnable parameters in the net.
   */
  const std::vector<TensorPtr<T>>& GetLearnableParams() const noexcept {
    return learnable_params_;
  }

  /**
   * @brief Reset the dataloader.
   */
  void Reset() noexcept { dataloader_->Reset(); }

 protected:
  // layer parameters
  std::vector<LayerParameterPtr> layer_parameters_;
  // data loader
  std::shared_ptr<DataLoader> dataloader_;
  // learnable parameters
  std::vector<TensorPtr<T>> learnable_params_;
  // layers in topo order
  std::vector<LayerPtr<T>> layers_;
  // bottom of the layers
  std::vector<std::vector<TensorPtr<T>>> bottom_vec_;
  // top of the layers
  std::vector<std::vector<TensorPtr<T>>> top_vec_;

  void Init(DatasetPtr dataset, int batch_size);

  // 拓扑排序
  static std::vector<LayerParameterPtr> TopoSort(
      const std::vector<LayerParameterPtr>& layers);

  // 检查网络是否合法
  void CheckNetValid() const {
    CheckNoSplitPoint();
    CheckOneInput();
    CheckOneOutput();
    CheckNoCircle();
  }

  // 检查网络没有孤立的点
  void CheckNoSplitPoint() const;
  // 检查网络只有一个输入
  void CheckOneInput() const;
  // 检查网络只有一个输出
  void CheckOneOutput() const;
  // 检查网络无环
  void CheckNoCircle() const;

  /**
   * @brief Connect the layers bottom and top. All the bottom of the layers will
   * be set to a top of other layer or the input. The implemention will make the
   * connected bottom and top point to the same memory.
   * @note The Net doesn't support mutiple bottom or mutiple top now, so the
   * implemention is simply connecting all the layers in order.
   * @todo Add mutiple bottom and mutiple top support.
   */
  void ConnectBottomAndTop();
  /**
   * @brief Update the input and the lable. The implemention will change the
   * pointer of the input and the label, since the input and label won't be the
   * top of other layers, this is safe.
   */
  void InitBottom();
  /**
   * @brief Set up the net structures. This method will call all the SetUp
   * method of the layers in the net.
   */
  // void SetUp();
};  // class Net

template <typename T = float>
using NetPtr = std::shared_ptr<Net<T>>;
}  // namespace my_tensor

#endif  // INCLUDE_NET_HPP_
