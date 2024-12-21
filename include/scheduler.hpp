// Copyright 2024 yibotongxue

#ifndef INCLUDE_SCHEDULER_HPP_
#define INCLUDE_SCHEDULER_HPP_

#include <cmath>
#include <functional>
#include <numbers>  // NOLINT

#include "scheduler-parameter.hpp"

namespace my_tensor {

using lr_scheduler = std::function<float(float, int)>;

inline lr_scheduler CreateScheduler(SchedulerParameterPtr param) {
  if (param->type_ == SchedulerType::kStep) {
    auto step_param = std::dynamic_pointer_cast<StepSchedulerParameter>(param);
    float gama = step_param->gama_;
    int stepsize = step_param->stepsize_;
    return [gama, stepsize](float base_lr, int current_epoch) -> float {
      return base_lr *
             std::pow(gama, std::floor(static_cast<float>(current_epoch) /
                                       static_cast<float>(stepsize)));
    };
  } else if (param->type_ == SchedulerType::kExponential) {
    auto exponetial_param =
        std::dynamic_pointer_cast<ExponentialSchedulerParameter>(param);
    float gama = exponetial_param->gama_;
    int stepsize = exponetial_param->stepsize_;
    return [gama, stepsize](float base_lr, int current_epoch) -> float {
      return base_lr * std::pow(gama, static_cast<float>(current_epoch) /
                                          static_cast<float>(stepsize));
    };
  } else if (param->type_ == SchedulerType::kCosineAnnealing) {
    auto cosine_annealing_param =
        std::dynamic_pointer_cast<CosineAnnealingSchuduelerParameter>(param);
    int max_epoch = cosine_annealing_param->max_epoch_;
    return [max_epoch](float base_lr, int current_epoch) -> float {
      return base_lr / 2 *
             (1.0f + std::cos(static_cast<float>(current_epoch) /
                              static_cast<float>(max_epoch) *
                              std::numbers::pi_v<float>));
    };
  } else {
    // TODO(yibotongxue) add specific exception type and description.
    throw std::runtime_error("");
  }
}

}  // namespace my_tensor

#endif  // INCLUDE_SCHEDULER_HPP_
