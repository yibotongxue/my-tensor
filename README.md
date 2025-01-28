[toc]

# 作业说明

这是人工智能中的编程大作业第三部分的代码。

## 项目结构

项目有个目录，分别为

- `data` 目录，用以存放训练、测试的数据
- `examples` 目录，存放示例的配置文件
- `include` 目录，包含项目的头文件
- `model` 目录，存放训练的模型参数
- `scripts` 目录，存放一些脚本
- `src` 目录包含源文件
- `test` 目录，包括一些测试文件

和文件[xmake.lua](./xmake.lua)，`README.md` 说明文件，`Doxyfile` 文档生成文件等。项目使用C++和CUDA实现。

## 环境

项目在以下环境已经得到验证，可以正常运行：

| 操作系统     | Ubuntu24.04.1 LTS |
| ------------ | ----------------- |
| CUDA Toolkit | 12.4              |
| gcc/g++      | 13.2              |
| xmake | 2.9.7 |

一般地，使用其他 Linux 发行版或者使用 WSL2 ，gcc/g++ 和 nvcc 版本支持 C++20 或以上，是可以完成项目的编译构建的。如果使用 Windows 操作系统，一般使用 MinGW 也可以（这没有经过实验，不推荐这样做），MSVC 不确定能否编译构建本项目。

## 编译运行

项目使用[xmake](https://xmake.io)构建，你需要安装有 `xmake` 。

### 编译项目

进入项目根目录，执行下面的命令：

```bash
xmake
```

### 运行测试

可以通过如下命令运行测试用例

```bash
xmake run test
```

### 运行示例配置

#### 运行 `MNIST` 数据集示例

首先需要下载 `MNIST` 数据，在**项目根目录**运行脚本，命令如下

```bash
./scripts/download_mnist.sh
```

然后运行脚本

```bash
./scripts/run_mnist.sh
```

以运行示例。

#### 运行 `Cifar-10` 数据示例

首先需要下载 `Cifar-10` 数据，在**项目根目录**运行脚本，命令如下

```bash
./scripts/download_cifar10.sh
```

然后运行脚本

```bash
./scripts/run_cifar10.sh
```

以运行示例。

## 项目代码解析

### 生成文档

为了批阅的方便，我对一些类（主要是基类）进行了必要的注释，其他的类由于类似并没有过多注释。可以通过如下命令生成文档

```bash
doxygen Doxyfile
```

### 张量与内存管理

项目使用 `SyncedVector` 同步 `CPU` 和 `GPU` 的内存，张量定义为含有两个 `SyncedVector` 指针的类型，分别表示数据和梯度，相关的定义在[synced-vector.hpp](./include/synced-vector.hpp)和[tensor.hpp](./include/tensor.hpp)，这两个类都有详细的文档，具体的可以参考上面生成的文档，或者具体查看源码。

### 数据处理

项目支持 `MNIST` 数据集、 `Cifar-10` 数据集和 `ImageNet` 数据集，通过统一的抽象类 `Dataset` 对外提供接口，而 `Dataloader` 类负责数据加载的工作，更具体的可以参考上面生成的文档，或者查看源码，文件为[dataset.hpp](./include/dataset.hpp)和[data-loader.hpp](./include/data-loader.hpp)。

### 神经网络层

项目实现了若干种常见的神经网络层，以支持一些神经网络的构建，这些层都为抽象类 `Layer` 的派生类，基本的包括

- `ReLU` 类，实现了 `ReLU` 激活层
- `Sigmoid` 类，实现了 `Sigmoid` 激活层
- `Linear` 类，实现了全连接网络层
- `Convolution` 类，实现了卷积层
- `Pooling` 类，实现了最大池化层
- `BatchNorm` 类，实现了批量归一化层
- `Softmax` 类，实现了 `Softmax` 层
- `LossWithSoftmax` 类，实现了损失层
- `Accuracy` 类，实现了准确率计算层
- `Flatten` 类，实现了展平操作

`Layer` 抽象类提供一些层对象需要的公共接口，主要包括前向传播、反向传播、获取可学习参数等，这部分有详细的注释，可以查看上面生成的文档，或者查看源码。

### 网络

基于上面实现的神经网络层，我们实现了一个类 `Net` ，用以管理若干个层组成的网络，包括

- 网络的构建，输入网络参数，解析数据加载、网络层等，将网络层按顺序写入数组，顺序的确定依照网络定义的时候指定的输入和输出，使用拓扑排序
- 前向传播，按照构建的计算图进行前向传播，依次调用排序后的层的前向传播函数
- 反向传播，按照构建的计算图进行反向传播，依次调用排序后的层的反向传播函数
- 诸如网络训练或测试状态的设置、模型参数的保存和加载等其他方法

具体的可以参考文档或源码。

### `Solver`

这个命名参考了[Caffe](http://caffe.berkeleyvision.org/)，主要负责网络的训练、应用等。主要包括两个成员，网络和学习率调度器。不同于[Caffe](http://caffe.berkeleyvision.org/)的 `Solver` 类，我们把学习率调度器作为 `Solver` 的一个变量，实现更好的可拓展性。不同的优化器算法通过继承 `Solver` 基类实现，当前实现的包括 `SGD` ，带动量的 `SGD` 和 `AdamW` 。具体的可以查看上面生成的文档或者源码。

### 主函数

项目的使用，包括网络的设计、数据的选取、优化算法的选择、学习率调度算法的选择等主要通过编写配置文件完成，配置文件使用 `Json` 文件，通过 `JsonLoader` 类读取解析。主函数在[main.cc](./src/main.cc)中，主要解析命令行参数、创建 `Solver` 和进行训练、测试。你需要指定的参数包括

- 配置文件的路径，必须指定
- 使用的设备，分为 CPU 和 GPU，默认为 CPU
- 模式，分为 train 和 test ，默认为 train

样例配置文件可以参考 `examples` 目录下的几个配置文件。

## 实验结果

### MNIST 数据集实验结果

在 `MNIST` 数据上，经过 $500$ 次迭代，训练的模型准确率为 $0.9824219$ 网络即为 `examples` 目录下的配置文件[mnist.json](./examples/mnist.json)所定义的。复现方式可以在项目根目录执行

```bash
./scripts/run_mnist.sh
```

或者执行

```bash
xmake run main --config=../../../../examples/mnist.json --device=gpu --phase=train
```

你可以在 `model` 目录下看到模型训练参数的文件 `lenet.model` ，如果你中途中断了训练，可以继续在上一次保存的模型之后继续训练，而不必从头开始。

### ImageNet 数据集实验结果

首先需要下载数据，但我没有找到可以通过命令行下载的方法，所以请手动下载到 `data` 目录下的imagenet目录或者其他任意你喜欢的目录（这个目录下应该有train和test子目录），然后修改[imagenet.json](./examples/imagenet.json)中的对应路径，类似上面的方法即可运行。

在 `ImageNet` 数据集上，我们定义了一个 `VGG-16` 网络，配置文件为[imagenet.json](./examples/imagenet.json)，但训练没有取的好的效果。

如果你只是想验证代码能运行，可以尝试运行一下，确实是可以运行的，但这个网络并没有好的效果，所以请不要浪费时间在这上面，不必等训练结束，可以直接当这份作业的ImageNet部分的训练没有效果，然后继续其他的批阅工作。

关于为什么训练得不到好的效果，我的猜测是迭代次数不够，或者参数需要调整。我添加了批量归一化层（文件[batchnorm.hpp](./include/batchnorm.hpp)）和对模型参数进行 `Xavier` 初始化和 `Kaiming` 初始化（文件[filler.hpp](./include/filler.hpp)），也确对数据进行了处理（文件[dataset.cc](./src/dataset.cc)），所有的网络层都经过充分的测试，并在MNIST数据集上取得较好的效果，而模型设计使用的也是[ImageNet挑战赛](https://image-net.org/challenges/LSVRC/)上取得很好效果的`VGG-16`，所以我相信这个模型如果进行调试是可能较好地完成ImageNet的分类任务的，后续需要针对的进行改进。
