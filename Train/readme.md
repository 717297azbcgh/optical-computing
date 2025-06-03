# 深度学习与光计算协同处理说明

本项目提供了对医学图像（BraTS 数据集）和手写数字图像（MNIST 数据集）在深度学习与光计算协同下的训练与推理框架。  
涵盖数据划分、模型定义与训练、光计算输入输出转换、以及模型与光计算混合推理分析。

---

## 一、BraTS 数据集相关脚本

### 数据处理与模型训练

- `get_benign_data.py`  
  将原始 BraTS 数据集划分为训练集、验证集和测试集，并生成索引文件。

- `dataset.py`  
  加载划分后的 BraTS 图像数据，并构建适用于 PyTorch 的数据加载器。

- `model.py`  
  定义基于 U-Net 架构的医学图像分割模型结构。

- `train.py`  
  设置训练参数并执行模型训练。

- `train_benign.py`  
  支持断点续训的训练脚本，可加载已有模型继续训练。

- `utils.py`  
  辅助函数集合，包含：  
  - 模型保存与加载  
  - 均值方差统计  
  - 数据加载器构建  
  - 准确率评估  
  - 验证图像结果保存  

- `my_checkpoint.pth.tar`  
  已训练完成的 U-Net 模型权重文件，可用于加载进行推理或继续训练。

- `training_metrics.csv`  
  BraTS 数据集训练过程中的性能参数记录文件。

### 光计算协同分析

- `read.ipynb`  
  Jupyter Notebook 脚本，功能包括：  
  - 加载训练好的模型  
  - 导出第一层卷积层输入（作为光计算输入）  
  - 加载光计算输出并继续后续神经网络推理  
  - 分析光计算替代部分计算后对整体模型性能的影响  

---

## 二、MNIST 数据集相关脚本

- `train_and_read_mnist.ipynb`  
  Jupyter Notebook 文件，功能包括：  
  - 构建并训练用于 MNIST 手写数字识别的小型卷积神经网络  
  - 输出第一层卷积层输入供光计算处理  
  - 加载光计算卷积输出，并导入后续网络得到最终预测结果  

- `mnist_cnn.pth`  
  MNIST 数据集训练完成后的卷积神经网络模型权重文件，用于加载进行推理或继续训练。

---

## 使用建议流程

1. **BraTS 数据集训练与推理**：  
   - 使用 `get_benign_data.py` 进行数据划分  
   - 使用 `train.py` 或 `train_benign.py` 进行模型训练  
   - 使用 `read.ipynb` 提取卷积输入并进行光计算协同推理  

2. **MNIST 数据集训练与推理**：  
   - 直接运行 `train_and_read_mnist.ipynb` 完成训练与混合推理分析  

---

## 环境要求

- BraTS 数据集相关处理请使用 `BraTs.yaml` 配置的 conda 环境。  
- MNIST 数据集相关处理请使用 `MNIST.yaml` 配置的 conda 环境。

