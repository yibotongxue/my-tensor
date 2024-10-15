#include <handle.cuh>

namespace my_tensor {
Handle* Handle::handle_ = nullptr;

my_tensor::Handle* handle = my_tensor::Handle::GetInstance();
}  // namespace my_tensor
