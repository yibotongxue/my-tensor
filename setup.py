from setuptools import setup, Extension
import os

# 获取 .so 文件所在目录
ext_modules = [
    Extension(
        name="my_tensor.mytensor",  # Python 模块名称
        sources=[],  # 没有源文件，因为我们已经有了 .so 文件
        library_dirs=[os.path.abspath('my_tensor')],  # .so 文件所在目录
        libraries=["mytensor"],  # 只需要写库的名字，不包括 .so 后缀
    ),
]

setup(
    name="my_tensor",
    version="0.1.0",
    packages=["my_tensor"],  # 确保这里指定了正确的包目录
    ext_modules=ext_modules,
)
