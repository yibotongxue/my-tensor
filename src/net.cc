// Copyright 2024 yibotongxue

#include "net.hpp"

#include <spdlog/spdlog.h>

#include <format>

#include "error.hpp"
#include "layer-factory.hpp"

namespace my_tensor {

template <Arithmetic T>
bool Net<T>::RefetchData() {
  bool result = false;
  if (!GetDataLoader()->HasNext()) {
    GetDataLoader()->Reset();
    result = true;
  }
  if (!GetDataLoader()->HasNext()) {
    throw NetError("The batch size is larger than the dataset size.");
  }
  auto [image, label] = GetDataLoader()->GetNext();
  *curr_image_ = *image;
  *curr_label_ = *label;
  return result;
}

template <Arithmetic T>
void Net<T>::SetUp() {
  net_name_ = net_parameter_->name_;
  train_dataloader_ = CreateDataLoader(net_parameter_->data_parameter_, true);
  test_dataloader_ = CreateDataLoader(net_parameter_->data_parameter_, false);
  CheckNetValid(net_parameter_->layer_params_);
  layers_.clear();
  for (auto&& layer_param : TopoSort(net_parameter_->layer_params_)) {
    layers_.push_back(CreateLayer<T>(layer_param));
  }
  bottom_vec_.resize(layers_.size());
  top_vec_.resize(layers_.size());
  curr_image_ = std::make_shared<Tensor<T>>(train_dataloader_->GetImageShape());
  curr_label_ = std::make_shared<Tensor<T>>(train_dataloader_->GetLabelShape());
  InitTop();
  ConnectBottomAndTop();
  SetUpBottomAndTop();
  learnable_params_.clear();
  for (auto&& layer : layers_) {
    auto&& learnable_param_of_layer = layer->GetLearnableParameters();
    if (learnable_param_of_layer.size() > 0) {
      learnable_params_.insert(learnable_params_.end(),
                               learnable_param_of_layer.begin(),
                               learnable_param_of_layer.end());
    }
  }
  spdlog::info("Net {} setup successfully.", net_name_);
}

template <Arithmetic T>
T Net<T>::GetOutput() const {
  if (phase_ == Phase::kTrain) {
    return top_vec_[top_vec_.size() - 2][0]->GetCPUData(0);
  }
  if (phase_ == Phase::kTest) {
    return top_vec_[top_vec_.size() - 1][0]->GetCPUData(0);
  }
  throw NetError("Unsupport phase.");
}

template <Arithmetic T>
std::vector<std::vector<T>> Net<T>::GetModelData() const {
  std::vector<std::vector<T>> result;
  for (auto&& learnable_param : GetLearnableParams()) {
    auto&& temp_span = SPAN_DATA(learnable_param, T);
    result.emplace_back(std::vector<T>(temp_span.begin(), temp_span.end()));
  }
  return result;
}

template <Arithmetic T>
void Net<T>::SetModelData(std::vector<std::vector<T>>&& data) {
  for (int i = 0; i < data.size(); i++) {
    // TODO(yibotongxue) should turn to SetData, auto detect the device.
    learnable_params_[i]->SetCPUData(data[i].data(), data[i].size());
  }
}

template <Arithmetic T>
void Net<T>::CopyFrom(const std::vector<TensorPtr<T>>& learnable_params) {
  for (int i = 0; i < learnable_params.size(); i++) {
    *(learnable_params_[i]) = *(learnable_params[i]);
  }
}

template <Arithmetic T>
void Net<T>::SetTrain() {
  phase_ = Phase::kTrain;
  for (auto&& layer : layers_) {
    layer->SetTrain();
  }
  spdlog::info("Set phase to train.");
}

template <Arithmetic T>
void Net<T>::SetTest() {
  phase_ = Phase::kTest;
  for (auto&& layer : layers_) {
    layer->SetTest();
  }
  spdlog::info("Set phase to test.");
}

template <Arithmetic T>
std::shared_ptr<DataLoader> Net<T>::GetDataLoader() const {
  if (phase_ == Phase::kTrain) {
    return train_dataloader_;
  }
  if (phase_ == Phase::kTest) {
    return test_dataloader_;
  }
  throw NetError("Unsupport phase.");
}

template <Arithmetic T>
std::vector<LayerParameterPtr> Net<T>::TopoSort(
    const std::vector<LayerParameterPtr>& layers) {
  return layers;
}

template <Arithmetic T>
void Net<T>::CheckNoSplitPoint(
    const std::vector<LayerParameterPtr>& layer_parameters) {
  for (auto&& layer : layer_parameters) {
    if (layer->bottoms_.size() == 0 || layer->tops_.size() == 0) {
      throw NetError("A split point occurs.");
    }
  }
}

template <Arithmetic T>
void Net<T>::CheckOneInput(
    const std::vector<LayerParameterPtr>& layer_parameters) {
  int input_cnt = 0;
  for (auto&& layer : layer_parameters) {
    for (auto&& bottom : layer->bottoms_) {
      if (bottom == "data") {
        input_cnt += 1;
        if (input_cnt > 1) {
          throw NetError("The count of the input is not one.");
        }
      }
    }
  }
}

template <Arithmetic T>
void Net<T>::CheckTwoOutput(
    const std::vector<LayerParameterPtr>& layer_parameters) {
  for (int i = 0; i < layer_parameters.size() - 2; i++) {
    if (layer_parameters[i]->tops_[0] == "loss" ||
        layer_parameters[i]->tops_[1] == "accuracy") {
      throw NetError("The count of the output is not two.");
    }
  }
  if (layer_parameters[layer_parameters.size() - 2]->tops_[0] != "loss") {
    throw NetError("The last but one of the net is not loss layer.");
  }
  if (layer_parameters[layer_parameters.size() - 1]->tops_[0] != "accuracy") {
    throw NetError("The last of the net is not accuracy layer.");
  }
}

template <Arithmetic T>
void Net<T>::CheckNoCircle(
    const std::vector<LayerParameterPtr>& layer_parameters) {
  // TODO(yibotongxue)
}

template <Arithmetic T>
void Net<T>::InitTop() {
  for (int i = 0; i < top_vec_.size(); i++) {
    top_vec_[i] = {std::make_shared<Tensor<T>>()};
  }
}

template <Arithmetic T>
void Net<T>::ConnectBottomAndTop() {
  for (int i = 1; i < layers_.size(); i++) {
    for (auto&& bottom : net_parameter_->layer_params_[i]->bottoms_) {
      bool has_top = false;
      for (int j = 0; j < i; j++) {
        if (bottom == net_parameter_->layer_params_[j]->name_) {
          bottom_vec_[i] = top_vec_[j];
          has_top = true;
        }
      }
      if (!has_top) {
        throw NetError(std::format("The bottom {} has no top.", bottom));
      }
    }
  }
  bottom_vec_[0] = {curr_image_};
  // the top two layer should be loss and accuracy layer, thus have label as
  // bottom
  bottom_vec_[bottom_vec_.size() - 1].push_back(curr_label_);
  bottom_vec_[bottom_vec_.size() - 2].push_back(curr_label_);
}

template <Arithmetic T>
void Net<T>::SetUpBottomAndTop() {
  for (int i = 0; i < layers_.size(); i++) {
    layers_[i]->SetUp(bottom_vec_[i], top_vec_[i]);
  }
}

template class Net<float>;
}  // namespace my_tensor
