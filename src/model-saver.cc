// Copyright 2024 yibotongxue

// from ChatGPT

#include "model-saver.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>  // 用于抛出异常
#include <string>
#include <vector>

namespace my_tensor {

template <typename T>
void ModelSaver::Save(const std::vector<std::vector<T>>& data,
                      const std::string& file_path) {
  std::ofstream out(
      file_path, std::ios::binary | std::ios::trunc);  // 打开文件并清空原内容
  if (!out) {
    throw std::runtime_error("Error opening file for writing: " + file_path);
  }

  size_t rows = data.size();
  out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));  // 写入行数

  for (const auto& row : data) {
    size_t cols = row.size();
    out.write(reinterpret_cast<const char*>(&cols),
              sizeof(size_t));  // 写入每行的列数
    out.write(reinterpret_cast<const char*>(row.data()),
              cols * sizeof(T));  // 写入每行的数据
  }

  out.close();
}

// 从文件读取数据
template <typename T>
std::vector<std::vector<T>> ModelSaver::Load(const std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Error opening file for reading: " + file_path);
  }

  size_t rows;
  in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));  // 读取行数

  // std::vector<std::vector<T>> data(rows);

  // 检查文件中的行数是否与传入的data的大小匹配
  // if (rows != data.size()) {
  //   throw std::runtime_error(
  //       "Data size mismatch: file rows do not match data size.");
  // }

  std::vector<std::vector<T>> data;

  for (size_t i = 0; i < rows; ++i) {
    size_t cols;
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));  // 读取列数
    std::vector<T> row(cols);
    in.read(reinterpret_cast<char*>(row.data()),
            cols * sizeof(T));  // 读取一行数据
    data.push_back(std::move(row));
  }

  in.close();
  return data;
}

template void ModelSaver::Save<float>(
    const std::vector<std::vector<float>>& data, const std::string& file_path);

template std::vector<std::vector<float>> ModelSaver::Load<float>(
    const std::string& file_path);

}  // namespace my_tensor
