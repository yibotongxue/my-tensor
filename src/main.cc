// Copyright 2024 yibotongxue

#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"
#include "data-loader.hpp"
#include "dataset.hpp"
#include "json-loader.hpp"
#include "layer-factory.hpp"
#include "layer-parameter.hpp"
#include "net-parameter.hpp"
#include "net.hpp"
#include "sgd-solver.hpp"
#include "solver-factory.hpp"
#include "solver.hpp"
#include "tensor.hpp"

int main(int argc, char** argv) {
  my_tensor::MyTensorContext::set_device_type(
      my_tensor::MyTensorContext::DeviceType::CPU);
  std::string config_file;
  bool has_config = false, has_device = false;
  bool is_train = true, has_phase = false;
  for (int i = 1; i < argc; i++) {
    std::string_view arg = argv[i];
    if (arg.starts_with("--config=")) {
      if (has_config) {
        spdlog::warn("Multiple config file detected, use the last one.");
      }
      config_file = arg.substr(9);
      has_config = true;
    } else if (arg.starts_with("--device=")) {
      if (has_device) {
        spdlog::warn("Multiple device type detected, use the last one.");
      }
      if (arg.substr(9) == "cpu") {
        my_tensor::MyTensorContext::set_device_type(
            my_tensor::MyTensorContext::DeviceType::CPU);
      } else if (arg.substr(9) == "gpu") {
        my_tensor::MyTensorContext::set_device_type(
            my_tensor::MyTensorContext::DeviceType::GPU);
      } else {
        spdlog::error("Unknown device type: {}", arg.substr(9));
        return 1;
      }
      has_device = true;
    } else if (arg.starts_with("--phase")) {
      if (has_phase) {
        spdlog::warn("Multiple phase detected, use the last one.");
      }
      if (arg.substr(8) == "train") {
        is_train = true;
      } else if (arg.substr(8) == "test") {
        is_train = false;
      } else {
        spdlog::error("Unknown phase: {}", arg.substr(8));
        return 1;
      }
      has_phase = true;
    } else if (arg == "--help") {
      spdlog::info(
          "Usage: {} --config=CONFIG_FILE [--device=DEVICE_TYPE] "
          "[--phase=PHASE]",
          argv[0]);
      spdlog::info("  --config=CONFIG_FILE: specify the config file.");
      spdlog::info(
          "  --device=DEVICE_TYPE: specify the device type, CPU or GPU.");
      spdlog::info("  --phase=PHASE: specify the phase, train or test.");
      return 0;
    } else {
      spdlog::error("Unknown argument: {}", arg);
      return 1;
    }
  }
  if (!has_config) {
    spdlog::error("No config file specified.");
    return 1;
  }
  if (!has_device) {
    spdlog::warn("No device type specified, use CPU as default.");
  }
  if (!has_phase) {
    spdlog::warn("No phase specified, use train as default.");
  }
  my_tensor::JsonLoader loader(config_file);
  my_tensor::SolverPtr<float> solver =
      my_tensor::CreateSolver<float>(loader.LoadSolver());
  solver->SetUp();
  if (is_train) {
    solver->Solve();
  }
  spdlog::info("Final test accuracy = {}", solver->Test());

  return 0;
}
