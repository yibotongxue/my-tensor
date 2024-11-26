[toc]

# 作业说明

这是人工智能中的编程第三次作业的代码。

## 项目结构

项目有五个目录，一个 `main.cpp` 文件，一个 `CMakeLists.txt` ，和一个 `README.md` 。 `include` 目录下包含项目需要的头文件， `src` 目录下包含源文件， `test` 目录下包含测试文件， `third_patrs` 包含第三方库，`python` 目录下包括将代码绑定到 `python` 的相关代码。项目使用 `cmake` 进行编译构建， `CMakeListst.txt` 在项目根目录下。

## 环境

项目在以下环境已经得到验证，可以正常运行：

| 操作系统     | Ubuntu24.04.1 LTS |
| ------------ | ----------------- |
| CUDA Toolkit | 12.4              |
| gcc/g++      | 13.2              |
| CMake        | 3.28.3            |
| Make         | 4.3               |

一般地，使用其他 Linux 发行版或者使用 WSL2 ，CMake 版本高于3.20，gcc/g++ 和 nvcc 版本支持 C++20或以上，是可以完成项目的编译构建的。如果使用 Windows 操作系统，一般使用 MinGW 也可以（这没有经过实验，不推荐这样做），MSVC 不确定能否编译构建本项目。

对于 `python` 的使用，本项目还需要 `numpy` 和 `pytorch` 包。

## 编译运行

### 编译项目

进入项目根目录，如果你希望在后面的 `python` 部分使用虚拟环境，请先激活虚拟环境，然后再依次执行下面的命令：

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 # 如果CPU核心数小于4,应该修改-j选项
```

### 安装包

可以直接在 `build` 目录执行这条命令：

```bash
make install
```

这会将上面生成的 `mytensor.YOUR_PLATFORM_INFO.so` 安装到你的 `python` 环境的 `site-packages` 中。建议创建一个虚拟环境，否则会安装到系统的 `python` 环境。如果你确定要安装到系统的环境中，上面的命令需要改为

```bash
sudo make install
```

如果你使用虚拟环境，你应该在执行 `cmake` 命令的时候就已经进入虚拟环境，否则还是会安装到系统的 `python` 环境。如果在执行 `cmake` 的时候没有进入虚拟环境，又希望使用虚拟环境，可以进入 `build` 目录，执行

```bash
rm CMakeCache.txt
```

然后重新执行 `cmake` 命令。

### 运行测试

如果你已经安装了 `mytensor` 包和 `pytorch` 和 `numpy` ，并下载了 `mnist` 数据集，那么可以进入 `build` 目录，执行

```bash
ctest
```

即可运行测试。如果没有安装，则需要在 `build` 目录手动运行 `python` 命令进行测试，直接运行上面的命令不能成功。

注意你可能需要修改 `dataset-test.py` 中 `Mnist` 数据集的相对路径。

### 可能的环境问题

#### libstdc++.so.6: version `GLIBCXX_3.4.29' not found

如果你使用的是 `conda` 虚拟环境，可能会在 `make` 的时候遇到这样的报错：

```bash
libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

类似的问题，根据[这个issue](https://github.com/pybind/pybind11/discussions/3453)，可以按照[这个评论](https://github.com/pybind/pybind11/discussions/3453#discussioncomment-8010023)的方法删除 `conda` 环境中 `libstdc++.so.6` 的符号链接以解决。

#### 安装的时候权限不够

很可能是因为安装路径设置为系统的 `python` 环境了，如果你确信要安装到安装到系统环境，请使用 `sudo` 权限，如果这不是你的本意，很可能是在 `cmake` 的时候忘记先激活虚拟环境了，删除缓存重新构建即可。

#### 提示找不到 `Python.h` 文件

如果是在 `Linux` 平台批阅，可能是需要安装 `python3-dev` 包，以 `Ubuntu` 系统为例，执行下面的命令：

```bash
sudo apt install python3-dev
```

#### 编译出错

请检查编译器是否支持 `C++20` ，也就是应该使用 `g++11` 以上和 `Cuda Toolkit 12` 以上。

#### 其他环境问题

如果有需要，可以使用我打包的 `Docker` 镜像，里面包含了项目运行需要的环境。考虑 `DockerHub` 不好访问，我把镜像放在了学校的服务器上，执行下面的命令拉取镜像：

```bash
docker pull 10.129.81.243:5000/mytensor:latest # 可能需要 sudo 权限
```

然后执行下面的命令可以创建并进入容器，检查代码运行结果

```bash
docker run -it -v /dev/shm:/dev/shm --name mytensor --gpus all 10.129.81.243:5000/mytensor:latest /bin/bash
```

镜像里的工作目录就是项目的目录，文件跟上传教学网的差不多，仅仅是少了 `README.pdf` ，并为了批阅的方便，将 `Mnist` 数据集下载到了 `data` 目录，可以直接使用。如果你需要验证容器没有提前安装 `mytensor` ，可以执行下面的命令以验证：

```bash
python -c "import mytensor"
```

可以发现并没有事先安装 `mytensor` 。如果你希望用教学网下载的文件进行测试，可以进入工作目录，删除所有文件，然后将本地从教学网下载的作业文件拷贝到容器，也就是在教学网下载的文件所在的根目录执行下面的命令：

```bash
docker cp ./* mytensor:/workspace/
```

## 项目代码解析

### C++ 和 CUDA 部分

#### `Tensor` 类

`Tensor` 类定义在 `include/tensor.cuh` 文件中，在命名空间 `my_tensor` 下，包含属性 `data_` , `diff_` , `shape_` 和 `size_` 。 `Tensor` 类的对象可以从 `const std::vector<int>& shape` 构造，也可以进行复制构造和移动构造。可以进行拷贝和移动，其中 `data_` 和 `diff_` 是 `SyncedVector` 类型，用来同步 CPU 和 GPU 上的数据。

#### `Layer` 类

`Layer` 类是一个抽象类，不能拷贝和移动，有 `Forward` 和 `Backward` 两个纯虚函数。 `Layer` 从 `LayerParameter` 对象构造，在 `SetUp` 方法中设置一些相关的参数。其他的网络层都是继承这个抽象类。在本项目中，我们定义了 `Relu` 和 `Sigmoid` 作为激活层， `Linear` 作为全连接层， `Convolution` 作为卷积层， `Pooling` 作为池化层， `Softmax` 作为分类层， `LossWithSoftmax` 作为损失层。

### python 和 pybind11 部分

主要包括 `python` 目录下的文件和 `test/python` 目录下的文件， `python/tensor-facade.cuh` 定义了 `TensorFacade` 类，对 `Tensor` 进行了简单的封装，主要是为了给 `python` 提供更友好的接口。注意尽管这个类命名为 `TensorFacade` ，但这不是外观设计模式。同样的，我们在 `python/layer-facade.cuh` 定义了第二次作业实现的若干个类的封装。在 `python/tensor-pybind.cu` 中我们将我们的封装再进行封装，以向 `python` 提供接口。我们实现的包为 `mytensor` 。

## 作业相关要求的实现

### `Tensor` 类的封装

封装的代码如下

```c++
PYBIND11_MODULE(mytensor, m) {
    py::class_<TensorFacade>(m, "Tensor", py::buffer_protocol())
        .def(py::init<const std::vector<int>&>(), py::arg("shape"))
        .def("reshape", &TensorFacade::Reshape, py::arg("shape"))
        .def("set_data", &TensorFacade::SetData, py::arg("data"))
        .def("set_grad", &TensorFacade::SetGrad, py::arg("grad"))
        .def("data", &TensorFacade::GetData)
        .def("grad", &TensorFacade::GetGrad)
        .def("to_cpu", &TensorFacade::ToCPU)
        .def("to_gpu", &TensorFacade::ToGPU)
        .def_static("from_numpy", &TensorFacade::FromNumpy, py::arg("data"))
        .def("shape", &TensorFacade::GetShape)
        .def_buffer([](TensorFacade &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                m.GetShape().size(),
                m.GetShape(),
                m.GetByteStride()
            );
        });
}
```

对 `TensorFacade` 类进行封装，其中 `TensorFacade` 包括了一个 `Tensor` 类的指针作为成员变量，对 `python` 端提供了从形状数组进行构造的构造方法，和修改形状、数据和梯度的方法，获取形状、数据和梯度的方法，改变设备位置的方法，从 `numpy` 构造的方法，以及转换为 `numpy` 数组的方法。

在 `python` 段可以这样使用

```python
# 导入包
import mytensor as ts
import numpy as np

# 从形状构造
tensor = ts.Tensor((1, 2))
# 从 numpy 数组构造
tensor = ts.Tensor.from_numpy(np.array([1, 2, 3]))
#获取形状
print(tensor.shape())
# 修改形状
tensor.reshape((1, 3))
# 设置数据
tensor.set_data([2, 3])
# 转换为 numpy 数组
numpy_tensor = np.array(tensor)
# 获取数据
print(tensor.data())
```

### 神经网络层的封装

我们首先对若干个神经网络层的类进行了封装，具体的代码在 `python/layer-facade.cuh` 和 `python/layer-facade.cu` 中。在 `python/tensor-pybind` 我们对这些封装好的类再次封装，以提供 `python` 接口。普遍的，我们提供从若干个网络层的配置数据构造的构造方法，获取和设置内部参数的方法，前向传播和反向传播的方法的接口，以全连接层为例，其封装代码如下

```c++
PYBIND11_MODULE(mytensor, m) {
    py::class_<LinearFacade>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &LinearFacade::Forward, py::arg("input"), "Perform forward propagation with Linear")
        .def("backward", &LinearFacade::Backward, py::arg("output"), "Perform backward propagation with Linear")
        .def("weight", &LinearFacade::GetWeight)
        .def("bias", &LinearFacade::GetBias)
        .def("set_weight", &LinearFacade::SetWeight, py::arg("weight"))
        .def("set_bias", &LinearFacade::SetBias, py::arg("bias"));
}
```

以 `Relu` 为例，在 `python` 端可以这样使用

```python
import mytensor as ts
relu = ts.Relu()
input = ts.Tensor.from_numpy(np.array([[1, 2, -1], [-1, -2, 1]]))
output = relu.forward(input)
print(output.data())
output.set_grad([1, 2, 3, -2, -3, -4])
input = relu.backward(output)
print(input.grad())
```

### `Mnist` 数据的读取

由于第二次作业我们已经实现了 `Dataset` 类，这里直接对其进行封装， `Dataset` 类的定义在 `include/dataset.h` 中，封装代码如下

```c++
PYBIND11_MODULE(mytensor, m) {
    py::class_<my_tensor::MnistDataset, std::shared_ptr<my_tensor::MnistDataset>>(m, "MnistDataset")
        .def(py::init<const std::string&, const std::string&>())
        .def("load_data", &my_tensor::MnistDataset::LoadData)
        .def("get_height", &my_tensor::MnistDataset::GetHeight)
        .def("get_width", &my_tensor::MnistDataset::GetWidth)
        .def("get_image", &my_tensor::MnistDataset::GetImage)
        .def("get_label", &my_tensor::MnistDataset::GetLabel)
        .def("get_size", &my_tensor::MnistDataset::GetSize);
}
```

在 `python` 端可以这样使用

```python
import mytensor as ts
import numpy as np
# 定义数据集
dataset = ts.MnistDataset("path/to/images", "path/to/labels")
# 加载数据
dataset.load_data()
# 获取图片高和宽，以及数据集大小
print("Height:", dataset.get_height())
print("Width:", dataset.get_width())
print("Size:", dataset.get_size())
# 获取图片数据和标签数据
images = np.array(dataset.get_image()).reshape((-1, 28, 28))
labels = np.array(dataset.get_label())
# 转换为 ts.Tensor 对象
images, labels = ts.Tensor.from_numpy(images), ts.Tensor.from_numpy(labels)
```

### 单元测试

我们在 `test/python` 定义了若干个测试文件。运行这些代码，需要安装我们构建的 `mytensor` 包和 `numpy` 和 `pytorch` 包，由于 `dataset-test.py` 涉及到文件路径，所以需要进入 `test/python` 目录以执行。

#### `Tensor` 实例化

在 `test/python/tensor-test.py` 中我们尝试对 `mytensor.Tensor` 实例化，如果包的安装成功的话，应该可以正常的得到一些输出。

#### 算子的测试

在 `relu-test.py` ， `sigmoid-test.py` ， `linear-test.py` ， `conv-test.py` ， `pooling-test.py` ， `softmax-tes.py` 和 `cross-entropy-loss-test.py` 中我们定义了对相关算子的测试，主要的测试思路是与 `pytorch` 进行对照，输入为随机生成的数据，将结果都转换为 `numpy` 数组进行比较。使用的框架是 `unittest` ，所以这一部分需要安装 `pytorch` ， `numpy` 和 `unittest` 的相关的包。除了 `softmax-test.py` 只包含了前向传播的测试，其他所有测试都包括了 `CPU` 和 `GPU` 上的前向传播和反向传播测试。进入 `test/python` 目录，然后执行下面的命令，可以进行测试：

```bash
python3 relu-test.py
python3 sigmoid-test.py
python3 linear-test.py
python3 conv-test.py
python3 pooling-test.py
python3 softmax-test.py
python3 cross-entropy-loss-test.py
```

测试结果为全部通过。

#### 读取 `Mnist` 数据

相关的示例在 `dataset-test.py` 中，为了运行这个测试，你需要首先在项目根目录下创建一个新目录 `data` ，并下载 `Mnist` 数据到这个新目录。官网的数据似乎无法下载了，这里提供从 `pytorch` 提供的渠道下载的方法，进入项目根目录，然后依次执行下面的命令：

```bash
mkdir data && cd data
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *
```

然后进入 `test/python` 目录，运行 `dataset-test.py` 即可检查是否正确读取了 `Mnist` 数据集，我的运行结果是

```bash
Height: 28
Width: 28
Size: 60000
[60000, 28, 28] [60000]
```

注意 `dataset-test.py` 使用了相对路径，所以你需要进入 `test/python` 目录再执行，否则可能会得到类似这样的输出，这时可以选择进入 `test/python` 目录再执行命令，或者修改 `dataset-test` 的相对路径。

```bash
Height: -1337850181
Width: 930611200
Size: 0
[0, 28, 28] [0]
```

## 卸载包

如果直接安装了 `mytensor` 包，可以直接找到安装的位置删除其，或者使用下面的命令删除。

```bash
rm $(python -c "import site; print(site.getsitepackages()[0])")/mytensor*.so
```
