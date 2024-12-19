// Copyright 2024 yibotongxue

#include "net.hpp"

#include <format>

#include "error.hpp"

namespace my_tensor {
template <typename T>
Net<T>::Net(const std::vector<LayerParameterPtr>& layers)
    : layer_parameters_(layers),
      dataloader_(nullptr),
      learnable_params_(),
      layers_(layers.size(), nullptr),
      bottom_vec_(layers.size(), nullptr),
      top_vec_(layers.size(), nullptr) {}

template <typename T>
bool Net<T>::RefetchData() {
  if (dataloader_->HasNext()) {
    auto [image, label] = dataloader_->GetNext();
    bottom_vec_[0][0] = image;
    top_vec_[top_vec_.size() - 1][0] = image;
    bottom_vec_[bottom_vec_.size() - 1][1] = label;
    return true;
  } else {
    return false;
  }
}

template <typename T>
void Net<T>::Init(DatasetPtr dataset, int batch_size) {
  CheckNetValid();
  dataloader_.reset(new DataLoader(dataset, batch_size));
  for (auto&& layer_param : TopoSort(layer_parameters_)) {
    layers_.push_back(CreateLayer(layer_param));
  }
  for (auto&& layer : layers_) {
    if (layer->GetLearnableParams().size() > 0) {
      learnable_params_.insert(learnable_params_.end(),
                               layer->GetLearnableParams().begin(),
                               layer->GetLearnableParams().end());
    }
  }
  ConnectBottomAndTop();
  InitBottom();
}

template <typename T>
std::vector<LayerParameterPtr> Net<T>::TopoSort(
    const std::vector<LayerParameterPtr>& layers) {
  return layers;
}

template <typename T>
void Net<T>::CheckNoSplitPoint() const {
  for (auto&& layer : layer_parameters_) {
    if (layer->bottoms_.size() == 0 || layer->tops_.size() == 0) {
      throw NetError("A split point occurs.");
    }
  }
}

template <typename T>
void Net<T>::CheckOneInput() const {
  int input_cnt = 0;
  for (auto&& layer : layer_parameters_) {
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

template <typename T>
void Net<T>::CheckOneOutput() const {
  int output_cnt = 0;
  for (auto&& layer : layer_parameters_) {
    for (auto&& top : layer->tops_) {
      if (top == "output") {
        output_cnt += 1;
        if (output_cnt > 1) {
          throw NetError("The count of the output is not one.");
        }
      }
    }
  }
}

template <typename T>
void Net<T>::CheckNoCircle() const {
  // TODO(yibotongxue)
}

template <typename T>
void Net<T>::ConnectBottomAndTop() {
  for (int i = 0; i < layers_.size(); i++) {
    for (auto&& bottom : layer_parameters_[i]->bottoms_) {
      bool has_top = false;
      for (int j = 0; j < layers_.size(); j++) {
        if (bottom == layer_parameters_[j]->name_) {
          bottom_vec_[i] = top_vec_[j];
          has_top = true;
        }
      }
      if (!has_top) {
        throw NetError(std::format("The bottom {} has no top.", bottom));
      }
    }
  }
  bottom_vec_[0] = {std::make_shared<Tensor<T>>(dataloader_->GetDataShape())};
}

template <typename T>
void Net<T>::InitBottom() {
  bottom_vec_[0] = {std::make_shared<Tensor<T>>(dataloader_->GetDataShape())};
  for (int i = 1; i < layers_.size() - 1; i++) {
    layers_[i]->SetUp(bottom_vec_[i], top_vec_[i]);
  }
}
}  // namespace my_tensor
