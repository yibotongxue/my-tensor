# 作业说明

这是人工智能中的编程第一次作业第二部分的作业的代码。

## 项目结构

项目有四个目录，一个 `main.cpp` 文件，一个 `CMakeLists.txt` ，和一个 `README.md` 。 `include` 目录下包含项目需要的头文件， `src` 目录下包含源文件， `test` 目录下包含测试文件， `third_patrs` 包含第三方库。项目使用 `GoogleTest` 框架编写单元测试， `GoogleTest` 静态连接库和头文件在 `third_parts` 目录下，可以直接使用。项目使用 `cmake` 进行编译构建， `CMakeListst.txt` 在项目根目录下。

## 环境

项目在以下环境已经得到验证，可以正常运行：

| 操作系统     | Ubuntu24.04.1 LTS |
| ------------ | ----------------- |
| CUDA Toolkit | 12.4              |
| gcc/g++      | 13.2              |
| CMake        | 3.28.3            |
| Make         | 4.3               |

一般地，使用其他 Linux 发行版或者使用 WSL2 ，CMake 版本高于3.20，gcc/g++ 版本支持 C++11及以上，是可以完成项目的编译构建的。如果使用 Windows 操作系统，一般使用 MinGW 也可以，MSVC 不确定能否编译构建本项目。

## 编译运行

进入项目根目录，依次执行下面的命令：

```bash
mkdir build && cd build
cmake .. # 默认 Debug 模式，如果需要使用 Release 模式，可以执行 cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 # 如果CPU核心数小于4,应该修改-j选项
# make 命令将会生成三个可执行文件，my_tensor, tensor_test, layer_test，后面两个是测试
./my_tensor # 将会打印 Hello, from my_tensor!
./tensor_test # 将会测试 Tensor 类的方法
./layer_test # 将会测试两个 Layer 类的方法
```

## 项目代码解析

下面简单介绍一下项目的代码。

### `Tensor` 类

`Tensor` 类定义在 `include/tensor.cuh` 文件中，在命名空间 `my_tensor` 下，包含属性 `device_type_` , `data_` , `diff_` , `shape_` 和 `size_` 。 `Tensor` 类的对象可以从 `const std::vector<int>& shape, my_tensor::DeviceType::device` 构造，也可以进行复制构造和移动构造。可以进行拷贝和移动，也可以返回 `CPU` 和 `GPU` 上的数据。

### `Layer` 类

`Layer` 类是一个抽象类，不能拷贝和移动，有 `Forward` 和 `Backward` 两个纯虚函数。

#### `Relu` 类

`Relu` 类是 `Layer` 类的子类，实现了 `Forward` 和 `Backwrd` 方法，分别实现 `Relu` 激活函数的前向和反向传播函数。

#### `Sigmoid` 类

`Sigmoid` 类是 `Layer` 类的子类，实现了 `Forward` 和 `Backwrd` 方法，分别实现 `Sigmoid` 激活函数的前向和反向传播函数。

### 测试样例

测试代码在 `test` 目录下的两个源文件中。

1. `tensor-test.cu` 测试 `Tensor` 类的方法，有18个test suite，70个test case，分别测试类型转换构造函数、赋值构造函数、移动构造函数、复制赋值、移动赋值、`cpu` 方法、 `gpu` 方法，对每一个方法的测试一般是形状是否正确、设备类型是否设置正确、数据是否在应该在的设备上（这个通过 `cudaMemcpy` 返回的错误代码判断）、数据是否拷贝或者移动成功（类型转换函数没有这个测试）等。
2. `laer-test.cu` 测试两个 `Layer` 子类的方法，有4个test suite，8个test case，分别测试 `Relu` 和 `Sigmoid` 类的 `Forward` 和 `Backward` 方法，其中 `Backward` 方法的测试是通过 10000 个随即生成的数据，通过每个数据加 0.001，通过微分求出梯度，再与 `Backward` 方法求出的梯度比较。

