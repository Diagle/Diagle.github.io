---
layout: post
title: "积分梯度算法：Integrated Gradients"
date: 2026-10-30 15:00:00 +0800
categories: [MachineLearnings, Research]
tags: [Integrated Gradients, Interpretability, Deep Learning]
---

# 积分梯度算法Interpretability or Explainability

> 论文(Axiomatic Attribution for Deep Networks)

## 可解释性

机器学习领域，指对一个算法的原理或者得出结果的依据做出解释

Interpretability：主要指的传统的统计学习方法的判决机理给出解释，白盒模型

Explainability：主要指的是对于复杂的神经网络进行判决归因，黑盒模型

## 满足公理

基线值：一件事情的发生归因于某个原因时，把没有该原因导致的情况作为基线值

> 图像识别的系统，基线值可以为全黑图
>
> NLP系统，基线值可以是全部值为0的词向量

### 敏感度(Sensitivity)

针对某一特征，当输入值为非基线值时获得有状态的预测(基线值为无状态)；当取两个相对于基线值不同的输入值时，模型应该输出不同的值，即该特征为非零归因，其梯度处处不为0

### 实现不变性(Implementation Invariance)：

内部实现不一致，对于所有的输入，输出结果一致的两个模型称为功能等效。功能等效的模型进行归因分析而获得的相同的成因归因算法满足实现不变性，得到输出的责任仅归结于输入，与算法内部无关

## 可解释性算法

### 梯度归因Gradients

输出对输入的直接梯度就是对模型参数的自然模拟，然而网络可视为多个非线性函数的堆栈体。非线性函数不可避免的存在梯度饱和的阶段，该阶段内梯度趋于0而违背了敏感性准则

> ReLU函数在x<1处梯度为0，违背了敏感性准则
>
> 对于不变性来说，基于输入输出的直接梯度可以忽略模型内部实现细节，故满足实现不变性

### 反向传播归因Back-propagation based approaches

DeepLift和LRP通过设置跨度较大的离散值保障梯度处处存在，满足敏感性准则。离散函数不满足链式法则，内部实现细节不可忽略，不满足实现不变性

### 积分梯度算法Integrated Gradients

IG算法结合了直接梯度和基于反向传播的归因技术分而治之的设计思想，满足敏感性和实现不变性的公理

x为输入,x'为基线值，函数映射为F，α为积分路径，对输入的第i个维度求积分梯度为：
$$
IG_i=(x_i-x_i')\times \int_{\alpha=0}^1\frac{\partial F(x_i'+\alpha\times(x_i-x_i'))}{\partial x_i'}d\alpha
$$
文章提出了一个新的公理为完整度Completeness

> 归因技术以积分的方式加和了从基线值到输入间的输出值称为完整度。直接梯度当输入的特征值刚好处于饱和阶段时，归因结果在该特征上的比重微乎其微但并非不重要

应用变式

m为积分的黎曼和近似的步值，选取20到300步值足以在5%的偏差内近似积分
$$
IG_i=\frac{x_i-x_i'}{m}\times\sum_{k=1}^m\frac{F(x'+\frac km\times(x-x'))}{x_i}
$$
**应用**

- 安装

  ```sh
  conda install captum
  ```

  or

  ```sh
  pip install captum
  ```

- 示例

  ```python
  from captum.attr import IntegratedGradients
  
  net = best_bkmd
  X_train = torch.tensor(data_s.iloc[:100].values,dtype=torch.float32 )
  X_test = torch.tensor(data_s.iloc[100:150].values,dtype=torch.float32)
  
  # 选择对类别1计算特征贡献
  target_class_index = 1
  net.eval()
  
  # applying integrated gradients on the SoftmaxModel and input data point
  ig = IntegratedGradients(net)
  attributions, approximation_error = ig.attribute(X_test, 			target=target_class_index,return_convergence_delta=True)
  
  # The input and returned corresponding attribution have the
  # same shape and dimensionality.
  ```

- 参数

  - `input`：输入样本x
  - `baseline`：基线输入（baseline），通常是全零（默认）、全黑或平均值。用于代表“没有信号”
  - `target`：目标分类数量
  - `return_convergence_delta`：返回置信度
  - `n_steps`：黎曼和近似的步值

- 返回：与输入相同维度的各特征的贡献度

**示例**：

- SCAD：计算并记录五倍交叉验证的每一折的目标域测试集基因的贡献得分，最后平均化减少简单折分的偏重。前后5%贡献得分的基因选为域药物敏感异质性有关的耐药基因。差异基因分析过滤得到显著相关的基因

  ![](C:\Users\WuMing\Desktop\code.png)

- scDEAL：应用IG得分识别模型的关键输入特征。IG得分代表输入的基因表达值验证baseline（0表达值）到表达水平梯度的累计。

  ![](E:\code.png)
