#include "layer-facade.cuh"

TensorFacade ReluFacade::Forward(TensorFacade input) {
  input_cache_ = input;
  TensorFacade output;
  relu_->SetUp({input.GetTensor()}, {output.GetTensor()});
  if (input.OnCPU()) {
    relu_->ForwardCPU({input.GetTensor()}, {output.GetTensor()});
  } else {
    relu_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
  }
  return output;
}

TensorFacade ReluFacade::Backward(TensorFacade output) {
  if (output.OnCPU()) {
    relu_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  } else {
    relu_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  }
  return input_cache_;
}

TensorFacade SigmoidFacade::Forward(TensorFacade input) {
  input_cache_ = input;
  TensorFacade output;
  sigmoid_->SetUp({input.GetTensor()}, {output.GetTensor()});
  if (input.OnCPU()) {
    sigmoid_->ForwardCPU({input.GetTensor()}, {output.GetTensor()});
  } else {
    sigmoid_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
  }
  return output;
}

TensorFacade SigmoidFacade::Backward(TensorFacade output) {
  if (output.OnCPU()) {
    sigmoid_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  } else {
    sigmoid_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  }
  return input_cache_;
}

TensorFacade LinearFacade::Forward(TensorFacade input) {
  input_cache_ = input;
  TensorFacade output;
  linear_->SetUp({input.GetTensor()}, {output.GetTensor()});
  if (param_set_) {
    if (input.OnCPU()) {
      linear_->GetWeight()->SetCPUData(weight_cache_.GetTensor()->GetCPUData().begin(), weight_cache_.GetTensor()->GetCPUData().end());
      linear_->GetBias()->SetCPUData(bias_cache_.GetTensor()->GetCPUData().begin(), bias_cache_.GetTensor()->GetCPUData().end());
    } else {
      linear_->GetWeight()->SetGPUData(weight_cache_.GetTensor()->GetCPUData().begin(), weight_cache_.GetTensor()->GetCPUData().end());
      linear_->GetBias()->SetGPUData(bias_cache_.GetTensor()->GetCPUData().begin(), bias_cache_.GetTensor()->GetCPUData().end());
    }
  }
  weight_cache_.SetTensor(linear_->GetWeight());
  bias_cache_.SetTensor(linear_->GetBias());
  if (input.OnCPU()) {
    linear_->ForwardCPU({input.GetTensor()}, {output.GetTensor()});
  } else {
    linear_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
  }
  return output;
}

TensorFacade LinearFacade::Backward(TensorFacade output) {
  if (output.OnCPU()) {
    linear_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  } else {
    linear_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  }
  return input_cache_;
}

TensorFacade ConvolutionFacade::Forward(TensorFacade input) {
  input_cache_ = input;
  TensorFacade output;
  conv_->SetUp({input.GetTensor()}, {output.GetTensor()});
  if (param_set_) {
    if (input.OnCPU()) {
      conv_->GetKernel()->SetCPUData(kernel_cache_.GetTensor()->GetCPUData().begin(), kernel_cache_.GetTensor()->GetCPUData().end());
      conv_->GetBias()->SetCPUData(bias_cache_.GetTensor()->GetCPUData().begin(), bias_cache_.GetTensor()->GetCPUData().end());
    } else {
      conv_->GetKernel()->SetGPUData(kernel_cache_.GetTensor()->GetCPUData().begin(), kernel_cache_.GetTensor()->GetCPUData().end());
      conv_->GetBias()->SetGPUData(bias_cache_.GetTensor()->GetCPUData().begin(), bias_cache_.GetTensor()->GetCPUData().end());
    }
  }
  kernel_cache_.SetTensor(conv_->GetKernel());
  bias_cache_.SetTensor(conv_->GetBias());
  if (input.OnCPU()) {
    conv_->ForwardCPU({input.GetTensor()}, {output.GetTensor()});
  } else {
    conv_->ForwardGPU({input.GetTensor()}, {output.GetTensor()});
  }
  return output;
}

TensorFacade ConvolutionFacade::Backward(TensorFacade output) {
  if (output.OnCPU()) {
    conv_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  } else {
    conv_->BackwardGPU({output.GetTensor()}, {input_cache_.GetTensor()});
  }
  return input_cache_;
}
