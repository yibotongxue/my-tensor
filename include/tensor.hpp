// Copyright 2024 yibotongxue

#ifndef INCLUDE_TENSOR_HPP_
#define INCLUDE_TENSOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "synced-vector.hpp"

namespace my_tensor {

/**
 * @brief 张量类
 *
 * @tparam T 数据类型，要求为算术类型
 *
 * Tensor类用于表示张量，支持在CPU和GPU上分别获取和设置数据，包含数据和梯度两部分，梯度默认为空
 */
template <typename T = float>
  requires std::is_arithmetic<T>::value
class Tensor {
 public:
  /**
   * @brief 构造函数
   *
   * 默认构造函数，初始化一个空的张量
   */
  Tensor();

  /**
   * @brief 构造函数
   *
   * @param shape 张量的形状
   *
   * 构造一个指定形状的张量
   */
  explicit Tensor(const std::vector<int>& shape);

  /**
   * @brief 拷贝构造函数
   *
   * @param tensor 源张量
   *
   * 拷贝构造函数，将 tensor 的数据拷贝到新的张量中
   */
  Tensor(const Tensor<T>& tensor);
  /**
   * @brief 拷贝赋值运算符
   *
   * @param tensor 源张量
   *
   * 拷贝赋值运算符，将 tensor 的数据拷贝到当前张量中
   */
  Tensor<T>& operator=(const Tensor<T>& tensor);
  /**
   * @brief 移动构造函数
   *
   * @param tensor 源张量
   *
   * 移动构造函数，将 tensor 的数据移动到新的张量中
   */
  Tensor(Tensor<T>&& tensor);
  /**
   * @brief 移动赋值运算符
   *
   * @param tensor 源张量
   *
   * 移动赋值运算符，将 tensor 的数据移动到当前张量中
   */
  Tensor<T>& operator=(Tensor<T>&& tensor);

  ~Tensor() = default;

  /**
   * @brief 设置CPU数据
   *
   * @param data 数据指针
   * @param size 数据大小
   *
   * 将数据拷贝到CPU上，调用SyncedVector的SetCPUData方法
   */
  void SetCPUData(const T* const data, size_t size) {
    data_->SetCPUData(data, size);
  }
  /**
   * @brief 设置CPU梯度
   *
   * @param diff 梯度指针
   * @param size 梯度大小
   *
   * 将梯度拷贝到CPU上，调用SyncedVector的SetCPUData方法，如果梯度为空则分配内存
   */
  void SetCPUDiff(const T* const diff, size_t size) {
    AllocateDiff();
    diff_->SetCPUData(diff, size);
  }
  /**
   * @brief 设置GPU数据
   *
   * @param data 数据指针
   * @param size 数据大小
   *
   * 将数据拷贝到GPU上，调用SyncedVector的SetGPUData方法
   */
  void SetGPUData(const T* const data, size_t size) {
    data_->SetGPUData(data, size);
  }
  /**
   * @brief 设置GPU梯度
   *
   * @param diff 梯度指针
   * @param size 梯度大小
   *
   * 将梯度拷贝到GPU上，调用SyncedVector的SetGPUData方法，如果梯度为空则分配内存
   */
  void SetGPUDiff(const T* const diff, size_t size) {
    AllocateDiff();
    diff_->SetGPUData(diff, size);
  }

  /**
   * @brief 获取CPU数据
   *
   * @param index 索引
   * @return T CPU数据
   *
   * 获取CPU数据，调用SyncedVector的host方法
   *
   * @throw VectorError 索引超出范围
   * @note 仅在调试的时候使用，其他情况下请使用GetCPUDataPtr，然后通过指针访问
   */
  [[nodiscard]] T GetCPUData(size_t index) const { return data_->host(index); }
  T GetCPUDiff(size_t index) const {
    AllocateDiff();
    return diff_->host(index);
  }
  /**
   * @brief 获取GPU数据
   *
   * @param index 索引
   * @return T GPU数据
   *
   * 获取GPU数据，调用SyncedVector的device方法
   * @throw VectorError 索引超出范围
   * @note 仅在调试的时候使用，其他情况下请使用GetGPUDataPtr，然后通过指针访问
   */
  [[nodiscard]] T GetGPUData(size_t index) const {
    return data_->device(index);
  }
  /**
   * @brief 获取GPU梯度
   *
   * @param index 索引
   * @return T GPU梯度
   *
   * 获取GPU梯度，调用SyncedVector的device方法
   * @throw VectorError 索引超出范围
   * @note 仅在调试的时候使用，其他情况下请使用GetGPUDiffPtr，然后通过指针访问
   */
  [[nodiscard]] T GetGPUDiff(size_t index) const {
    AllocateDiff();
    return diff_->device(index);
  }

  /**
   * @brief 获取数据
   *
   * @param index 索引
   * @return T 数据
   *
   * 获取数据，如果在CPU上调用GetCPUData，否则调用GetGPUData
   */
  [[nodiscard]] inline T GetData(size_t index) const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUData(index);
    } else {
      return GetGPUData(index);
    }
  }

  /**
   * @brief 获取梯度
   *
   * @param index 索引
   * @return T 梯度
   *
   * 获取梯度，如果在CPU上调用GetCPUDiff，否则调用GetGPUDiff
   */
  [[nodiscard]] inline T GetDiff(size_t index) const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDiff(index);
    } else {
      return GetGPUDiff(index);
    }
  }

  /**
   * @brief 获取CPU数据指针
   *
   * @return const T* CPU数据指针
   */
  [[nodiscard]] const T* GetCPUDataPtr() const { return data_->GetCPUPtr(); }
  /**
   * @brief 获取可变CPU数据
   *
   * @return T* CPU数据指针
   */
  [[nodiscard]] T* GetCPUDataPtr() { return data_->GetMutableCPUPtr(); }
  /**
   * @brief 获取CPU梯度指针
   *
   * @return const T* CPU梯度指针
   */
  [[nodiscard]] const T* GetCPUDiffPtr() const {
    AllocateDiff();
    return diff_->GetCPUPtr();
  }
  /**
   * @brief 获取可变CPU梯度
   *
   * @return T* CPU梯度指针
   */
  [[nodiscard]] T* GetCPUDiffPtr() {
    AllocateDiff();
    return diff_->GetMutableCPUPtr();
  }

  /**
   * @brief 获取GPU数据指针
   *
   * @return const T* GPU数据指针
   *
   * 获取GPU数据指针，调用SyncedVector的GetGPUPtr方法
   */
  [[nodiscard]] const T* GetGPUDataPtr() const { return data_->GetGPUPtr(); }
  /**
   * @brief 获取可变GPU数据
   *
   * @return T* GPU数据指针
   *
   * 获取可变GPU数据指针，调用SyncedVector的GetMutableGPUPtr方法
   */
  [[nodiscard]] T* GetGPUDataPtr() { return data_->GetMutableGPUPtr(); }
  /**
   * @brief 获取GPU梯度指针
   *
   * @return const T* GPU梯度指针
   *
   * 获取GPU梯度指针，调用SyncedVector的GetGPUPtr方法
   */
  [[nodiscard]] const T* GetGPUDiffPtr() const {
    AllocateDiff();
    return diff_->GetGPUPtr();
  }
  /**
   * @brief 获取可变GPU梯度
   *
   * @return T* GPU梯度指针
   *
   * 获取可变GPU梯度指针，调用SyncedVector的GetMutableGPUPtr方法
   */
  [[nodiscard]] T* GetGPUDiffPtr() {
    AllocateDiff();
    return diff_->GetMutableGPUPtr();
  }

  /**
   * @brief 获取数据指针
   *
   * @return const T* 数据指针
   *
   * 获取数据指针，如果在CPU上调用GetCPUDataPtr，否则调用GetGPUDataPtr
   */
  [[nodiscard]] const T* GetDataPtr() const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDataPtr();
    } else {
      return GetGPUDataPtr();
    }
  }

  /**
   * @brief 获取可变数据
   *
   * @return T* 数据指针
   *
   * 获取可变数据指针，如果在CPU上调用GetCPUDataPtr，否则调用GetGPUDataPtr
   */
  [[nodiscard]] T* GetDataPtr() {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDataPtr();
    } else {
      return GetGPUDataPtr();
    }
  }

  /**
   * @brief 获取梯度指针
   *
   * @return const T* 梯度指针
   *
   * 获取梯度指针，如果在CPU上调用GetCPUDiffPtr，否则调用GetGPUDiffPtr
   */
  [[nodiscard]] const T* GetDiffPtr() const {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDiffPtr();
    } else {
      return GetGPUDiffPtr();
    }
  }

  /**
   * @brief 获取可变梯度
   *
   * @return T* 梯度指针
   *
   * 获取可变梯度指针，如果在CPU上调用GetCPUDiffPtr，否则调用GetGPUDiffPtr
   */
  [[nodiscard]] T* GetDiffPtr() {
    if (MyTensorContext::on_cpu()) {
      return GetCPUDiffPtr();
    } else {
      return GetGPUDiffPtr();
    }
  }

  /**
   * @brief 获取形状
   *
   * @return const std::vector<int>& 形状
   *
   * 获取张量的形状
   */
  [[nodiscard]] const std::vector<int>& GetShape() const { return shape_; }

  /**
   * @brief 获取大小
   *
   * @return int 大小
   *
   * 获取张量的大小
   */
  [[nodiscard]] int GetSize() const { return size_; }

  /**
   * @brief 重置形状
   *
   * @param shape 新的形状
   *
   * 重置张量的形状
   */
  void Reshape(const std::vector<int>& shape);

  /**
   * @brief 重置形状
   *
   * @param shape 新的形状
   *
   * 重置张量的形状，可以改变张量的大小
   */
  void Resize(const std::vector<int>& shape);

 private:
  // 张量形状
  std::vector<int> shape_;
  // 张量大小
  int size_;
  // 数据
  SyncedVectorPtr<T> data_;
  // 梯度
  mutable SyncedVectorPtr<T> diff_;

  /**
   * @brief 分配梯度
   *
   * 分配梯度，如果梯度为空则分配内存
   */
  void AllocateDiff() const;

  /**
   * @brief 检查形状
   *
   * 检查张量的形状是否合法
   * @throw TensorError 形状不合法
   */
  void CheckShape() const;
};  // class Tensor

template <typename T = float>
using TensorPtr = std::shared_ptr<my_tensor::Tensor<T>>;

extern template class my_tensor::Tensor<>;
extern template class my_tensor::Tensor<int>;
}  // namespace my_tensor

#endif  // INCLUDE_TENSOR_HPP_
