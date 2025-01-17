// Copyright 2025 yibotongxue

#include "adamw-solver.hpp"

#include "blas.hpp"
#include "memory-util.hpp"

namespace my_tensor {

template <typename T>
void AdamWSolver<T>::UpdateParam() {
  time_step_++;
  const auto& learnable_params = this->net_->GetLearnableParams();
  for (int i = 0; i < learnable_params.size(); i++) {
    TensorPtr<T> param = learnable_params[i];
    T* grad_ptr = param->GetDiffPtr();
    scale<T>(m_data_[i]->GetDataPtr(), param->GetSize(), beta1_);
    add_two_vec<T>(m_data_[i]->GetDataPtr(), grad_ptr,
                   static_cast<T>(1.0 - beta1_), param->GetSize());
    scale<T>(v_data_[i]->GetDataPtr(), param->GetSize(), beta2_);
    T* square_grad_ptr = nullptr;
    if (MyTensorContext::on_cpu()) {
      MyMallocCPU(reinterpret_cast<void**>(&square_grad_ptr),
                  param->GetSize() * sizeof(T));
    } else {
      MyMallocGPU(reinterpret_cast<void**>(&square_grad_ptr),
                  param->GetSize() * sizeof(T));
    }
    square<T>(grad_ptr, square_grad_ptr, param->GetSize());
    add_two_vec<T>(v_data_[i]->GetDataPtr(), square_grad_ptr,
                   static_cast<T>(1.0 - beta2_), param->GetSize());
    T* moment2_unbiase = square_grad_ptr;
    vec_divide_num<T>(v_data_[i]->GetDataPtr(), moment2_unbiase,
                      static_cast<T>(1.0 - std::pow(beta2_, time_step_)),
                      param->GetSize());
    sqrt<T>(moment2_unbiase, moment2_unbiase, param->GetSize());
    vec_add_num<T>(moment2_unbiase, moment2_unbiase, epsilon_,
                   param->GetSize());
    T* temp = moment2_unbiase;
    divide_two_vec<T>(m_data_[i]->GetDataPtr(), temp, temp, param->GetSize());
    auto lr = this->GetLearningRate();
    scale<T>(param->GetDataPtr(), param->GetSize(),
             static_cast<T>(1.0 - 2 * lr * this->l2_));
    T* temp_cpu = reinterpret_cast<T*>(malloc(param->GetSize() * sizeof(T)));
    MyMemcpyGPU2CPU(temp_cpu, temp, param->GetSize() * sizeof(T));
    // for (int j = 0; j < param->GetSize(); j++) {
    //   std::cout << temp_cpu[j] << " ";
    // }
    // std::cout << std::endl;
    free(temp_cpu);
    // std::cout << static_cast<T>(-lr / (1.0 - std::pow(beta1_, time_step_)))
    // << std::endl;
    add_two_vec<T>(param->GetDataPtr(), temp,
                   static_cast<T>(-lr / (1.0 - std::pow(beta1_, time_step_))),
                   param->GetSize());
    if (MyTensorContext::on_cpu()) {
      MyMemFreeCPU(square_grad_ptr);
    } else {
      MyMemFreeGPU(square_grad_ptr);
    }
  }
}

template <typename T>
void AdamWSolver<T>::SpecialSetUp() {
  const auto& learnable_params = this->net_->GetLearnableParams();
  m_data_.resize(learnable_params.size());
  v_data_.resize(learnable_params.size());
  for (int i = 0; i < learnable_params.size(); i++) {
    m_data_[i] = std::make_shared<Tensor<T>>(learnable_params[i]->GetShape());
    v_data_[i] = std::make_shared<Tensor<T>>(learnable_params[i]->GetShape());
    if (MyTensorContext::on_cpu()) {
      Fill_CPU<T>(m_data_[i]->GetDataPtr(), learnable_params[i]->GetSize(),
                  static_cast<T>(0));
      Fill_CPU<T>(v_data_[i]->GetDataPtr(), learnable_params[i]->GetSize(),
                  static_cast<T>(0));
    } else {
      Fill_GPU<T>(m_data_[i]->GetDataPtr(), learnable_params[i]->GetSize(),
                  static_cast<T>(0));
      Fill_GPU<T>(v_data_[i]->GetDataPtr(), learnable_params[i]->GetSize(),
                  static_cast<T>(0));
    }
  }
  auto adamw_param =
      std::dynamic_pointer_cast<AdamWSolverParameter>(this->param_);
  beta1_ = adamw_param->beta1_;
  beta2_ = adamw_param->beta2_;
  epsilon_ = adamw_param->epsilon_;
  time_step_ = 0;
}

template class AdamWSolver<float>;

}  // namespace my_tensor
