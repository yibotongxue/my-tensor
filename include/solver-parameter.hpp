// Copyright 2024 yibotongxue

#ifndef INCLUDE_SOLVER_PARAMETER_HPP_
#define INCLUDE_SOLVER_PARAMETER_HPP_

#include <memory>
#include <string>

#include "json-utils.hpp"
#include "net-parameter.hpp"
#include "scheduler-parameter.hpp"

namespace my_tensor {

enum class SolverType {
  kSgd,
  kSgdWithMomentum,
  kAdamW
};  // enum class SolverType

class SolverParameter {
 public:
  SolverType type_;
  int max_iter_;
  float base_lr_;
  float l2_;
  int test_step_;
  int save_step_;
  std::string save_model_path_;
  std::string load_model_path_;
  SchedulerParameterPtr scheduler_param_;
  NetParameterPtr net_param_;

  explicit SolverParameter(SolverType type,
                           SchedulerParameterPtr scheduler_param,
                           NetParameterPtr net_param)
      : type_(type), scheduler_param_(scheduler_param), net_param_(net_param) {}

  void Deserialize(const nlohmann::json& js) {
    ParseCommonField(js);
    ParseSettingParameters(js);
  }

 protected:
  void ParseCommonField(const nlohmann::json& js) {
    max_iter_ = LoadWithKey<int>(js, "max_iter");
    base_lr_ = LoadWithKey<float>(js, "base_lr");
    l2_ = LoadWithKey<float>(js, "l2");
    test_step_ = LoadWithKey<int>(js, "test_step");
    save_step_ = LoadWithKey<int>(js, "save_step");
    save_model_path_ = LoadWithKey<std::string>(js, "save_model_path");
    load_model_path_ = LoadWithKey<std::string>(js, "load_model_path");
  }

  virtual void ParseSettingParameters(const nlohmann::json& js) = 0;
};  // class SolverParameter

class SgdSolverParameter final : public SolverParameter {
 public:
  SgdSolverParameter(SchedulerParameterPtr scheduler_param,
                     NetParameterPtr net_param)
      : SolverParameter(SolverType::kSgd, scheduler_param, net_param) {}

 private:
  void ParseSettingParameters(const nlohmann::json& js) override {}
};  // class SgdSolverParameter

class SgdWithMomentumSolverParameter final : public SolverParameter {
 public:
  SgdWithMomentumSolverParameter(SchedulerParameterPtr scheduler_param,
                                 NetParameterPtr net_param)
      : SolverParameter(SolverType::kSgdWithMomentum, scheduler_param,
                        net_param) {}

  float momentum_;

 private:
  void ParseSettingParameters(const nlohmann::json& js) override {
    momentum_ = LoadWithKey<float>(js, "momentum");
  }
};  // class SgdWithMomentumSolverParameter

class AdamWSolverParameter final : public SolverParameter {
 public:
  AdamWSolverParameter(SchedulerParameterPtr scheduler_param,
                       NetParameterPtr net_param)
      : SolverParameter(SolverType::kAdamW, scheduler_param, net_param) {}

  float beta1_;
  float beta2_;
  float epsilon_;

 private:
  void ParseSettingParameters(const nlohmann::json& js) override {
    beta1_ = LoadWithKey<float>(js, "beta1");
    beta2_ = LoadWithKey<float>(js, "beta2");
    epsilon_ = LoadWithKey<float>(js, "epsilon");
  }
};  // class AdamWSolver

using SolverParameterPtr = std::shared_ptr<SolverParameter>;

inline std::function<SolverParameterPtr(SchedulerParameterPtr, NetParameterPtr)>
GetSolverParameterCreater(const std::string& type) {
  if (type == "sgd") {
    return [](SchedulerParameterPtr scheduler_param,
              NetParameterPtr net_param) -> SolverParameterPtr {
      return std::make_shared<SgdSolverParameter>(scheduler_param, net_param);
    };
  } else if (type == "sgd_with_momentum") {
    return [](SchedulerParameterPtr scheduler_param,
              NetParameterPtr net_param) -> SolverParameterPtr {
      return std::make_shared<SgdWithMomentumSolverParameter>(scheduler_param,
                                                              net_param);
    };
  } else if (type == "adamw") {
    return [](SchedulerParameterPtr scheduler_param,
              NetParameterPtr net_param) -> SolverParameterPtr {
      return std::make_shared<AdamWSolverParameter>(scheduler_param, net_param);
    };
  } else {
    throw SolverError("Unknown solver type: " + type);
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_SOLVER_PARAMETER_HPP_
