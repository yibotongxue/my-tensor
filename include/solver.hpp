// Copyright 2024 yibotongxue

#ifndef INCLUDE_SOLVER_HPP_
#define INCLUDE_SOLVER_HPP_

#include <memory>
#include <string>

#include "net.hpp"
#include "scheduler.hpp"
#include "solver-parameter.hpp"

namespace my_tensor {

/**
 * @brief Solver 类，用于训练网络
 *
 * @tparam T 数据类型，必须为算术类型
 *
 * 这是一个抽象类，派生类需要实现 UpdateParam 等函数，以实现不同的优化算法
 * 学习率调度算法由 scheduler_ 字段实现
 *
 * 目前支持的优化算法有：
 * - SGD
 * - 带有动量的 SGD
 * - AdamW
 *
 * 支持的学习率调度算法有：
 * - Step
 * - Exponential
 * - Cosine Annealing
 *
 * 这些都可以通过 SolverParameter 类的成员来设置
 */
template <Arithmetic T>
class Solver {
 public:
  explicit Solver(SolverParameterPtr param)
      : param_(param), training_iter_(0) {}

  /**
   * @brief 设置解决器
   *
   * 该函数会调用 CommonSetUp 和 SpecialSetUp 函数
   */
  void SetUp();

  virtual ~Solver() = default;

  /**
   * @brief 训练网络
   */
  void Solve();

  /**
   * @brief 测试网络
   *
   * @return 测试结果
   */
  float Test();

 protected:
  NetPtr<T> net_;
  SolverParameterPtr param_;
  int training_iter_;
  lr_scheduler scheduler_;
  int max_iter_;
  int current_epoch_;
  float base_lr_;
  float l2_;
  int test_step_;
  int save_step_;

  std::string save_model_path_;
  std::string load_model_path_;

  /**
   * @brief 保存模型
   *
   * @param model_path 模型保存路径
   */
  void SaveModel(const std::string& model_path);
  /**
   * @brief 加载模型
   *
   * @param model_path 模型加载路径
   */
  void LoadModel(const std::string& model_path);

  /**
   * @brief 通用设置
   *
   * 该函数会设置网络、学习率调度器等
   */
  void CommonSetUp();
  /**
   * @brief 特殊设置
   *
   * 该函数会设置特定优化算法的参数
   */
  virtual void SpecialSetUp() {}

  /**
   * @brief 训练一次
   */
  void Step();

  /**
   * @brief 更新参数
   *
   * 不同的优化算法需要实现不同的更新参数的方法
   */
  virtual void UpdateParam() = 0;

  /**
   * @brief 获取学习率
   *
   * @return 学习率
   */
  float GetLearningRate() { return scheduler_(base_lr_, current_epoch_); }
};

extern template class Solver<float>;

template <Arithmetic T>
using SolverPtr = std::shared_ptr<Solver<T>>;

}  // namespace my_tensor

#endif  // INCLUDE_SOLVER_HPP_
