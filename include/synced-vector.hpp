// Copyright 2024 yibotongxue

#ifndef INCLUDE_SYNCED_VECTOR_HPP_
#define INCLUDE_SYNCED_VECTOR_HPP_

#include <memory>

#include "error.hpp"
#include "utils.hpp"

namespace my_tensor {

/**
 * @brief 同步向量，用于在CPU和GPU之间同步数据
 *
 * @tparam T 数据类型
 *
 * SyncedVector类用于在CPU和GPU之间同步数据，支持在CPU和GPU上分别获取和设置数据
 */
template <Arithmetic T>
class SyncedVector {
 public:
  /**
   * @brief 构造函数
   *
   * 默认构造函数，初始化一个空的向量
   */
  SyncedVector() noexcept;
  /**
   * @brief 构造函数
   *
   * @param size 向量的大小
   *
   * 构造一个指定大小的向量
   */
  explicit SyncedVector(size_t size) noexcept;
  ~SyncedVector();

  /**
   * @brief 拷贝构造函数
   *
   * @param vec 源向量
   *
   * 拷贝构造函数，将 vec 的数据拷贝到新的向量中
   */
  SyncedVector(const SyncedVector<T>& vec) noexcept;
  /**
   * @brief 拷贝赋值运算符
   *
   * @param vec 源向量
   *
   * 拷贝赋值运算符，将 vec 的数据拷贝到当前向量中
   */
  SyncedVector<T>& operator=(const SyncedVector<T>& vec) noexcept;
  /**
   * @brief 移动构造函数
   *
   * @param vec 源向量
   *
   * 移动构造函数，将 vec 的数据移动到新的向量中
   */
  SyncedVector(SyncedVector<T>&& vec) noexcept;
  /**
   * @brief 移动赋值运算符
   *
   * @param vec 源向量
   *
   * 移动赋值运算符，将 vec 的数据移动到当前向量中
   */
  SyncedVector<T>& operator=(SyncedVector<T>&& vec) noexcept;

  /**
   * @brief 向量状态
   *
   * 向量状态用于标识向量的数据在 CPU 和 GPU 之间的同步状态
   * kUninitialized: 未初始化
   * kHeadAtCPU: 数据在 CPU 上
   * kHeadAtGPU: 数据在 GPU 上
   * kSynced: 数据在 CPU 和 GPU 之间同步
   */
  enum VectorState { kUninitialized, kHeadAtCPU, kHeadAtGPU, kSynced };

  /**
   * @brief 设置 CPU 数据
   *
   * @param data 数据指针
   * @param size 数据大小
   *
   * 将数据拷贝到 CPU 上
   */
  void SetCPUData(const T* const data, size_t size) noexcept;
  /**
   * @brief 获取 CPU 数据
   *
   * @return const T* CPU 数据指针
   *
   * 获取 CPU 数据指针，如果数据未初始化，会先分配内存；如果数据在 GPU
   * 上，会先将数据拷贝到 CPU 上
   */
  [[nodiscard]] const T* GetCPUPtr();
  /**
   * @brief 获取可变 CPU 数据
   *
   * @return T* CPU 数据指针
   *
   * 获取可变 CPU 数据指针，如果数据未初始化，会先分配内存；如果数据在 GPU
   * 上，会先将数据拷贝到 CPU 上
   */
  [[nodiscard]] T* GetMutableCPUPtr();
  /**
   * @brief 设置 GPU 数据
   *
   * @param data 数据指针
   * @param size 数据大小
   *
   * 将数据拷贝到 GPU 上，并设置状态为 kOnGPU
   */
  void SetGPUData(const T* const data, size_t size) noexcept;
  /**
   * @brief 获取 GPU 数据
   *
   * @return const T* GPU 数据指针
   *
   * 获取 GPU 数据指针，如果数据未初始化，会先分配内存；如果数据在 CPU
   * 上，会先将数据拷贝到 GPU 上
   */
  [[nodiscard]] const T* GetGPUPtr();
  /**
   * @brief 获取可变 GPU 数据
   *
   * @return T* GPU 数据指针
   *
   * 获取可变 GPU 数据指针，如果数据未初始化，会先分配内存；如果数据在 CPU
   * 上，会先将数据拷贝到 GPU 上
   */
  [[nodiscard]] T* GetMutableGPUPtr();

  /**
   * @brief 根据索引获取 CPU 数据
   *
   * @param index 索引
   * @return T CPU 数据
   *
   * @note 仅在调试的时候使用，其他情况下请使用 GetCPUDataPtr ，然后通过指针访问
   * @throw VectorError 索引超出范围
   */
  [[nodiscard]] T host(size_t index);
  /**
   * @brief 根据索引获取 GPU 数据
   *
   * @param index 索引
   * @return T GPU 数据
   *
   * @note 仅在调试的时候使用，其他情况下请使用 GetCPUDataPtr ，然后通过指针访问
   * @throw VectorError 索引超出范围
   */
  [[nodiscard]] T device(size_t index);

  /**
   * @brief 获取向量大小
   * @return size_t 向量大小
   * 获取向量的大小
   */
  [[nodiscard]] constexpr size_t size() const noexcept { return size_; }

  /**
   * @brief 重置向量大小
   *
   * @param size 新的向量大小
   *
   * 重置向量的大小
   */
  void Resize(size_t size) noexcept { size_ = size; }

 private:
  // 向量状态
  VectorState state_;
  // 向量大小
  size_t size_;
  // CPU 数据指针
  T* cpu_data_;
  // GPU 数据指针
  T* gpu_data_;

  /**
   * @brief 将数据拷贝到 CPU 上
   *
   * 将数据拷贝到 CPU 上
   *
   * @note 如果数据在 GPU 上，会先将数据拷贝到 CPU
   * 上，如果为初始化，会先分配内存
   * @note 仅在内部使用
   * @throw VectorError 未定义的状态
   */
  void ToCPU();
  /**
   * @brief 将数据拷贝到 GPU 上
   *
   * 将数据拷贝到 GPU 上
   *
   * @note 如果数据在 CPU 上，会先将数据拷贝到 GPU
   * 上，如果为初始化，会先分配内存
   * @note 仅在内部使用
   * @throw VectorError 未定义的状态
   */
  void ToGPU();
};

template <Arithmetic T>
using SyncedVectorPtr = std::shared_ptr<SyncedVector<T>>;

extern template class SyncedVector<float>;
extern template class SyncedVector<int>;
}  // namespace my_tensor

#endif  // INCLUDE_SYNCED_VECTOR_HPP_
