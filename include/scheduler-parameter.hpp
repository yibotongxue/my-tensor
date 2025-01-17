// Copyright 2024 yibotongxue

#ifndef INCLUDE_SCHEDULER_PARAMETER_HPP_
#define INCLUDE_SCHEDULER_PARAMETER_HPP_

#include <memory>
#include <string>

#include "json-utils.hpp"
#include "nlohmann/json.hpp"

namespace my_tensor {

enum class SchedulerType {
  kStep,
  kExponential,
  kCosineAnnealing
};  // enum class SchedulerType

class SchedulerParameter {
 public:
  explicit SchedulerParameter(SchedulerType type) : type_(type) {}

  SchedulerType type_;

  virtual ~SchedulerParameter() = default;

  virtual void Deserialize(const nlohmann::json& js) = 0;
};

class StepSchedulerParameter final : public SchedulerParameter {
 public:
  StepSchedulerParameter() : SchedulerParameter(SchedulerType::kStep) {}

  void Deserialize(const nlohmann::json& js) override {
    gama_ = LoadWithKey<float>(js, "gama");
    stepsize_ = LoadWithKey<int>(js, "stepsize");
  }

  float gama_;
  int stepsize_;
};

class ExponentialSchedulerParameter final : public SchedulerParameter {
 public:
  ExponentialSchedulerParameter()
      : SchedulerParameter(SchedulerType::kExponential) {}

  void Deserialize(const nlohmann::json& js) override {
    gama_ = LoadWithKey<float>(js, "gama");
    stepsize_ = LoadWithKey<int>(js, "stepsize");
  }

  float gama_;
  int stepsize_;
};

class CosineAnnealingSchuduelerParameter final : public SchedulerParameter {
 public:
  CosineAnnealingSchuduelerParameter()
      : SchedulerParameter(SchedulerType::kCosineAnnealing) {}

  void Deserialize(const nlohmann::json& js) override {
    max_epoch_ = LoadWithKey<int>(js, "max_epoch");
  }

  int max_epoch_;
};

using SchedulerParameterPtr = std::shared_ptr<SchedulerParameter>;

inline SchedulerParameterPtr CreateSchedulerParameterPtr(
    const std::string& type) {
  if (type == "step") {
    return std::make_shared<StepSchedulerParameter>();
  } else if (type == "exponential") {
    return std::make_shared<ExponentialSchedulerParameter>();
  } else if (type == "cosine_annealing") {
    return std::make_shared<CosineAnnealingSchuduelerParameter>();
  } else {
    throw SchedulerError("Unknown scheduler type" + type);
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_SCHEDULER_PARAMETER_HPP_
