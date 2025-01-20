// Copyright 2024 yibotongxue

#ifndef INCLUDE_LAYER_HPP_
#define INCLUDE_LAYER_HPP_

#include <spdlog/spdlog.h>

#include <memory>
#include <vector>

#include "common.hpp"
#include "layer-parameter.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace my_tensor {
/**
 * @brief 所有网络层的基类
 * @tparam T 数据类型
 *
 * Layer类是所有网络层的基类，定义了网络层的基本接口，包括
 * - SetUp：设置网络层及其输入输出，在网络构建的时候调用
 * - Forward: 前向传播
 * - Backward: 反向传播
 * - GetLearnableParameters: 获取可学习参数
 */
template <Arithmetic T>
  requires std::is_arithmetic<T>::value
class Layer {
 public:
  /**
   * @brief 构造函数
   *
   * @param param 网络层参数，包含网络层的名称、类型等信息
   */
  explicit Layer(LayerParameterPtr param) : layer_param_(param) {}

  /**
   * @brief 设置网络层及其输入输出
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * 依次调用CheckTensorCount、LayerSetUp和Reshape方法，完成网络层的设置
   */
  void SetUp(const std::vector<TensorPtr<T>>& bottom,
             const std::vector<TensorPtr<T>>& top) {
    CheckTensorCount(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    spdlog::info("Layer {} setup done.", layer_param_->name_);
  }

  /**
   * @brief 检查输入输出张量的数量
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * 检查输入输出张量的数量是否符合网络层的要求
   *
   * @note 不要直接调用，由SetUp方法调用
   * @note 纯虚函数，需要在子类中实现
   */
  virtual void CheckTensorCount(const std::vector<TensorPtr<T>>& bottom,
                                const std::vector<TensorPtr<T>>& top) const = 0;

  /**
   * @brief 设置网络层
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * 设置网络层的参数，包括网络层的输入输出张量的形状等
   *
   * @note 不要直接调用，由SetUp方法调用
   * @note 虚函数，可以在子类中重写，默认为空实现
   */
  virtual void LayerSetUp(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) {}

  /**
   * @brief 重塑输出的形状
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * 重塑输出张量的形状
   *
   * @note 不要直接调用，由SetUp方法调用
   * @note 纯虚函数，需要在子类中实现
   */
  virtual void Reshape(const std::vector<TensorPtr<T>>& bottom,
                       const std::vector<TensorPtr<T>>& top) const = 0;

  /**
   *  @brief 获取可学习参数
   *
   * @return std::vector<TensorPtr<T>> 可学习参数
   *
   * 获取网络层的可学习参数，包括权重、偏置等
   *
   * @note 虚函数，可以在子类中重写，默认返回空
   */
  virtual std::vector<TensorPtr<T>> GetLearnableParameters() { return {}; }

  // The layer can not be copied or moved.
  DISABLE_LAYER_COPY(Layer)

  virtual ~Layer() = default;

  /**
   * @brief 前向传播
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * 根据当前的计算设备，调用ForwardCPU或ForwardGPU方法
   */
  void Forward(const std::vector<TensorPtr<T>>& bottom,
               const std::vector<TensorPtr<T>>& top) {
    if (MyTensorContext::on_cpu()) {
      ForwardCPU(bottom, top);
    } else {
      ForwardGPU(bottom, top);
    }
  }

  /**
   * @brief 反向传播
   *
   * @param top 网络层的输出
   * @param bottom 网络层的输入
   *
   * 根据当前的计算设备，调用BackwardCPU或BackwardGPU方法
   */
  void Backward(const std::vector<TensorPtr<T>>& top,
                const std::vector<TensorPtr<T>>& bottom) {
    if (MyTensorContext::on_cpu()) {
      BackwardCPU(top, bottom);
    } else {
      BackwardGPU(top, bottom);
    }
  }

  /**
   * @brief CPU前向传播
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * @note 不要直接调用，由Forward方法调用
   * @note 纯虚函数，需要在子类中实现
   */
  virtual void ForwardCPU(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) = 0;

  /**
   * @brief CPU反向传播
   *
   * @param top 网络层的输出
   * @param bottom 网络层的输入
   *
   * @note 不要直接调用，由Backward方法调用
   * @note 纯虚函数，需要在子类中实现
   */
  virtual void BackwardCPU(const std::vector<TensorPtr<T>>& top,
                           const std::vector<TensorPtr<T>>& bottom) = 0;

  /**
   * @brief GPU前向传播
   *
   * @param bottom 网络层的输入
   * @param top 网络层的输出
   *
   * @note 不要直接调用，由Forward方法调用
   * @note 纯虚函数，需要在子类中实现
   */
  virtual void ForwardGPU(const std::vector<TensorPtr<T>>& bottom,
                          const std::vector<TensorPtr<T>>& top) = 0;

  /**
   * @brief GPU反向传播
   *
   * @param top 网络层的输出
   * @param bottom 网络层的输入
   *
   * @note 不要直接调用，由Backward方法调用
   * @note 纯虚函数，需要在子类中实现
   */
  virtual void BackwardGPU(const std::vector<TensorPtr<T>>& top,
                           const std::vector<TensorPtr<T>>& bottom) = 0;

  /**
   * @brief 设置网络层为训练状态
   */
  void SetTrain() { is_train_ = true; }
  /**
   * @brief 设置网络层为测试状态
   */
  void SetTest() { is_train_ = false; }

 protected:
  // 网络层参数
  LayerParameterPtr layer_param_;
  // 是否为训练状态
  bool is_train_ = true;
};

// Layer pointer.
template <Arithmetic T>
using LayerPtr = std::shared_ptr<my_tensor::Layer<T>>;

// 实例化模板类
extern template class my_tensor::Layer<float>;
}  // namespace my_tensor

#endif  // INCLUDE_LAYER_HPP_
