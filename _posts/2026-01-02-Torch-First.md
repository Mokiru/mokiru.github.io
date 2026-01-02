---
title: PyTorch使用
date: 2025-06-22 22:00:00 +0800
categories: [AI, Base]
tags: [implement]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: 简单介绍PyTorch基本使用
comments: false # 评论
pin: false # top 
math: true
---

## 训练模型

训练模型通常包括以下步骤：

1. 数据准备
2. 定义模型
3. 选择损失函数
4. 选择优化器
5. 前向传播
6. 计算损失
7. 反向传播
8. 参数更新
9. 迭代优化：重复步骤5-8
10. 评估和测试
11. 模型调优
12. 部署模型

### 损失函数

PyTorch提供了常用的损失函数，我们可以将其按照模型任务进行划分：

- 分类任务
    - 交叉熵损失
        - `nn.CrossEntropyLoss`：适用于多类别分类，内部自动对输入应用`LogSoftmax` + `NLLLoss`
        - `nn.NLLLoss`：适用于已应用`LogSoftmax`的输出
        - `nn.BCELoss`：用于二分类或多标签分类(每个样本可以属于多个类别)，输入前需要经`Sigmoid`激活
        - `nn.BCEWithLogitsLoss`：将`Sigmoid` + `BCELoss`合并，数值更稳定
    - 其他分类损失
        - `nn.KLDivLoss`：分布匹配
        - `nn.HingeEmbeddingLoss`：用于二分类的`hinge loss`变体
        - `nn.MultiMarginLoss`：支持多分类的`hinge loss`
        - `nn.MultiLabelMarginLoss`：多标签版本的`margin loss`
- 回归任务
    - 均方误差(MSE)
        - `nn.MSELoss`：常用于预测连续值，对异常值敏感
    - 平均绝对误差(MAE)
        - `nn.L1Loss`：对异常值更健壮
    - 平滑L1损失
        - `nn.SmoothL1Loss`：结合 MSE 和 MAE 在误差小时用平方，大时用绝对值，常用于目标检测
    - 其他回归损失
        - `nn.HuberLoss`：与平滑L1损失类似，但参数化更清晰
        - `nn.PoissonNLLLoss`：用于泊松分布的负对数似然(如计数预测)
- 嵌入式与度量学习
    - `nn.CosineEmbeddingLoss`：基于余弦相似度，用于学习向量的语义相似性（如句子/图像嵌入）
    - `nn.MarginRankingLoss`：用于排序任务（如 x1 应比 x2 分数高）
    - `nn.TripletMarginLoss`：三元组损失，用于人脸识别、ReID 等（拉近锚点与正样本，推远负样本）
    - `nn.TripletMarginWithDistanceLoss`：支持自定义距离函数的三元组损失
- 生成与概率建模
    - `nn.GaussianNLLLoss`：高斯负对数似然，用于预测均值和方差的回归模型
    - `nn.CTCLoss`：用于序列建模（如语音识别、OCR），处理输入输出长度不一致
- 通用与组合损失
    - 通过继承`nn.Module`实现自定义损失函数
        
    
### 优化器

PyTorch也提供了常用的优化器在`torch.optim`中：

- 基础优化器
    - SGD(随机梯度下降优化器)
- 自适应学习率优化器
    - Adam(自适应动量估计)
    - RMSSprop()
    - Adagrad
    - Adadelta
- 改进型Adam系列
    - AdamW
    - NAdam
    - RAdam
- 其他优化器
    - LBFGS
    - SparseAdam
    - ASGD


## 定义一个简单的前馈神经网络

```python
import torch
import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义一个输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(2, 2)  # 输入 2 个特征，输出 2 个特征
        # 定义一个隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(2, 1)  # 输入 2 个特征，输出 1 个预测值
    
    def forward(self, x):
        # 前向传播过程
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.fc2(x)  # 输出层
        return x

# 创建模型实例
model = SimpleNN()

# 打印模型
print(model)
```

PyTorch提供了常见的神经网络层：
- `nn.Linear`：全连接层
- `nn.Conv2d`：2D卷积层，用于图像处理
- `nn.MaxPool2d`：2D最大池化层，用于降维
- `nn.ReLU`：ReLU激活函数，用于隐藏层，增加非线性
- `nn.Softmax`：Softmax激活函数，通常用于输出层，适用于多分类问题
- `nn.Sigmoid`：Sigmoid激活函数，将结果映射到0-1之间

接下来我们以一个二分类任务为例，实现一个简单的前馈神经网络，我们可以使用`nn.Sequential`直接创建一个顺序模型：

```python
model = nn.Sequential(
   nn.Linear(n_in, n_h),  # 输入层到隐藏层的线性变换
   nn.ReLU(),            # 隐藏层的ReLU激活函数
   nn.Linear(n_h, n_out),  # 隐藏层到输出层的线性变换
   nn.Sigmoid()           # 输出层的Sigmoid激活函数
)
```

如此，我们就创建好了一个前馈计算顺序，现在我们想要完成模型训练还需要创造一个输入，一个优化器和损失函数计算：

```python
# 定义输入层大小、隐藏层大小、输出层大小和批量大小
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# 创建虚拟输入数据和目标数据
x = torch.randn(batch_size, n_in)  # 随机生成输入数据
y = torch.tensor([[1.0], [0.0], [0.0],
                 [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])  # 目标输出数据

# 定义均方误差损失函数和随机梯度下降优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率为0.01
```

最后，我们就可以开始训练这个模型：

```python
# 执行梯度下降算法进行模型训练
for epoch in range(50):  # 迭代50次
   y_pred = model(x)  # 前向传播，计算预测值
   loss = criterion(y_pred, y)  # 计算损失
   print('epoch: ', epoch, 'loss: ', loss.item())  # 打印损失值

   optimizer.zero_grad()  # 清零梯度
   loss.backward()  # 反向传播，计算梯度
   optimizer.step()  # 更新模型参数
```

我们还可以对Loss变化进行可视化，即随着训练轮次变化，Loss值的变化，以及：

```python
losses = [] # 记录每轮损失值

# 可视化损失变化曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()
```