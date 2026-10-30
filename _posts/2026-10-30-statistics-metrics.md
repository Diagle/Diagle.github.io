---
layout: post
title: "统计学指标：Statistics Metrics"
date: 2026-10-30 15:00:00 +0800
categories: [MachineLearnings, Research]
tags: [Statistics Metrics, Assessments]
---

# 用于模型评估的统计学指标


## 一、聚类效果

### 1、CH得分

CH得分指的是卡林斯基-哈拉巴兹指数（**CH**，Calinski–Harabasz Index），也称为方差比准则，通过比较类间离散度和类内离散度来评估聚类效果。
- 簇之间的距离越大，簇内数据点之间的距离越小，则得分越高，聚类效果越好。

对于一个包含，

**定义**

CH得分是类间散度与类内散度之比，并按各自的自由度归一化。

例如，n个数据点的集合{x1,...,xn}被化为k个簇{c1,...,ck}
$$
\text {CH} =\frac{\text {BCSS}/(k-1) } {\text {WCSS}/(n-k)}
$$
其中，类间平方和（**BCSS**，*Between-cluster Sum of Squares*）是通过计算每个簇的质心到整体质心的欧几里得距离平方，并按簇中样本数量加权求和得到
$$
\text{BCSS}=\sum_{i=1}^kn_i||c_i-c||^2
$$
- ni为簇Ci的数据点的数量，ci为簇Ci的质心，c为总体质心。

类内平方和（**WCSS**，*Within-cluster Sum of Squares*）衡量的是聚类的紧密程度，指每个数据点到其所在簇质心的欧几里得距离的平方总和，其值越小表示聚类越紧凑。
$$
\text{WCSS}=\sum_{i=1}^k\sum_{x\in C_i}||x-c_i||^2
$$

**代码实现**

```python
import numpy as np
import pandas as pd
'''
data：pandas对象，行为样本，列为特征
label：标签，行为样本，列为类别（整数值表示）
'''
def calculate_CHscore(data,label):
    df = data.values
    lb = label.values.reshape(1,-1)[0]
    counts = pd.value_counts(lb).sort_index()
    n = len(df)
    k = len(counts)
    c = df.mean(axis=0)
    WCSS = 0
    BCSS = 0
    for i in range(k):
        Ci = df[lb==i,]
        ci = Ci.mean(axis=0)
        WCSS += counts[0] * (Ci.var(axis=0).sum())
        BCSS += counts[0] * np.dot(ci - c,ci - c)
    CH  = (BCSS/(k-1)) / ((WCSS)/(n-k))
    return CH
```

**应用**

OmiCLIP的图像嵌入捕获了组织的形态学，而其转录嵌入表示其基因特征。因为OmiCLIP同时包含转录与图像编码器，作者采用CH得分评估了对比学习是否增强了其任一编码器较之原始编码器表示其组织类型的能力。

*示例1*

- 数据：ST-bank具有已知专家注释细胞类型的组织数据（空转数据与切片染色图像数据）输入到OmiCLIP模型产生的嵌入
- 目的：评估对比学习的加入能否提高模型嵌入的细胞类型划分能力
- 计算：标签为细胞类型，特征为模型产生的嵌入，计算CH得分反应嵌入对细胞类型的聚类能力
- 结果：采用对比模型的OmiCLIP，其嵌入的CH得分显著提高

![image-20250625142732376](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250625142732376.png)

*示例2*

- 数据：ST-bank中不具有专家注释的组织数据，其类别由Leiden算法得到
- 目的：比较对比学习是否显著提升嵌入表示其组织类型的能力
- 计算：标签为各自的聚类标签，特征分别为空转和染色切片的嵌入，计算CH评估聚类效果
- 结果：经过对比学习后的聚类效果更好，如肝癌和脊髓细胞的聚类效果显著提升

![image-20250625142720478](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250625142720478.png)

## 二、相关性

### 1、PCC

皮尔森相关系数（**PCC**，Pearson correlation coefficient），是一种衡量两个变量之间 **线性相关性（linear correlation）** 的统计指标。

**定义**

对给定数据$\{(x_1,y_1),...,(x_n,y_n)\}$，共包含n对样本，PCC表示为$r_{xy}$，定义为
$$
r_{xy}=\frac{n\sum x_iy_i-\sum x_i\sum y_i}{\sqrt{n\sum x_i^2-(\sum x_i)^2}\sqrt{n\sum y_i^2-(\sum y_i)^2}}
$$
n代表样本数量，xi和yi分别代表位置i的独立样本点。

**代码实现**

```python
from scipy.stats import pearsonr

'''参数说明（参数在下面也没展示，df.iloc是啥，换成xy这种不就行了）
- x: 类数组
- y：类数组
- alternative：{ 'two-sided', 'less', 'greater' }，可指定备择假设的形式。默认'two-sided'

- return r 相关系数[-1, 1]; p 显著性
'''
r,p = pearsonr(df.iloc[0,],df.iloc[1,])
```

**应用**

作者在八个相邻小肠组织切片数据集上评估了Loki 对齐的性能，以源数据与目标数据对齐后嵌入的相关性作为评估指标。

*示例*

- 数据：经过CPD降维后的OmiCLIP、PASTE、GPSA对齐的源数据与目标数据的嵌入。
- 目的：通过比较源数据与目标数据嵌入的相关性，比较不同对齐工具的性能。
- 计算：Loki对齐分别计算7个源数据的空转嵌入和染色切片嵌入，对齐到目标数据；参考工具计算7个源数据的空转嵌入，对齐到目标数据。经过CPD算法对嵌入进行降维，对降维后的嵌入计算其与目标数据降维后的嵌入之间的PCC相关性。
- 结果：Loki image-to-ST和ST-to-ST对齐的PCC中位数高于参考工具，证明其性能优于参考工具。

![image-20250625155254059](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250625155254059.png)

### 2、Kendall's tau coefficiant

是一种用来衡量两个变量之间排序一致性（秩相关性）的非参数统计量，用于评估它们是否单调相关。核心思想是所有样本点在两个变量中的相对排序是否一致。特别适合于：

- 等级数据、非正态数据
- 小样本
- 数据中含有极端值的情况

**定义**

对给定数据$\{(x_1,y_1),...,(x_n,y_n)\}$，共包含n对样本，
$$
\tau=\frac{P-Q}{\sqrt{(P+Q+T)(P+Q+U)}}
$$
P代表一致的对数，Q代表不一致的对数，而T和U分别表示仅在x和y出现的平对（什么意思）的数量。

**代码**

```python
from scipy.stats import kendalltau
'''
- x：类数组
- y：类数组
- alternative：{ 'two-sided', 'less', 'greater' }，可指定备择假设的形式。默认'two-sided'

- return r 相关系数[-1, 1]; p 显著性
'''
r,p = kendalltau(x,y)
```

**应用**

作者在八个相邻小肠组织切片数据集上评估了Loki 对齐的性能，以源数据与目标数据对齐后嵌入的相关性作为评估指标。

*示例*

- 数据：经过CPD降维后的OmiCLIP、PASTE、GPSA对齐的源数据与目标数据的嵌入。
- 目的：通过比较源数据与目标数据嵌入的相关性，比较不同对齐工具的性能。
- 计算：Loki对齐分别计算7个源数据的空转嵌入和染色切片嵌入，对齐到目标数据；参考工具计算7个源数据的空转嵌入，对齐到目标数据。经过CPD算法对嵌入进行降维，对降维后的嵌入计算其与目标数据降维后的嵌入之间的Kendall‘s tau系数。

- 结果：Loki的ST-to-ST和image-to-ST得到最高的kendall's tau系数，性能优于其他两个平台。

![image-20250707162353080](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250707162353080.png)

## 三、相似度

### 1、余弦相似度

余弦相似度（**Cosine Similarity**）是一种非常常见的**向量相似度度量**方法，常用于文本分析、推荐系统、图像比较以及其他高维空间中的样本对比。它衡量的是两个向量之间的**夹角**，而不是它们的欧几里得距离。

**定义**
$$
\text{CosineSimilarity}(x,y)=\frac{\sum_i^nx_iy_i}{\sqrt{\sum_i^nx_i}{\sqrt{\sum_i^ny_i}}}
$$
代码

```python
def calculate_CosineSimilarity(x,y):
    return np.dot(x,y)/(np.sqrt( np.dot(x,x) ) * np.sqrt(np.dot(y,y)))
calculate_CosineSimilarity(x,y)
```

**应用**

评估不同处理条件下的数据，比较OmiCLIP嵌入的余弦相似度

*示例1*

评估不同测序深度对模型的影响，分析了ST-bank样本为高、中和低测序深度组的测序深度下的模型性能

- 数据：OmiCLIP，PLIP和OpenAI CLIP对不同测序深度组的数据编码的嵌入
- 目的：评估不同测序计数对OmiCLIP的影响，采用余弦相似度衡量不同测序计数下的数据与原始数据的嵌入相似性
- 计算：计算不同测序条件下的数据的OmiCLIP嵌入，比较其与原始数据嵌入的余弦相似度
- 结果：证明OmiCLIP对测序深度的鲁棒性，突出其对不同技术产生数据集的适用能力

![image-20250625103320987](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250625103320987.png)

*示例2*

为了评估技术的限制导致的样本图像质量差异对OmiCLIP的影响，作者对图像添加高斯噪声模拟低质量的 H&E 图像，并比较了由OmiCLIP图像和转录编码器编码的配对原始图像和转录组嵌入、和模拟低质量图像和转录组嵌入的相似性得分

- 数据：OmiCLIP，PLIP和OpenAI CLIP对不同测序深度组的数据编码的嵌入
- 目的：评估不同图像质量对OmiCLIP的影响，采用余弦相似度衡量不同图像质量的数据与原始数据的嵌入相似性
- 计算：计算不同图像质量数据的OmiCLIP嵌入，比较其与原始数据嵌入的余弦相似度
- 结果：证明OmiCLIP对测序深度的鲁棒性，证实其对图不同像质量数据集的适用能力

![image-20250625103219594](C:\Users\WuMing\Documents\tencent files\1990866968\filerecv\assets\image-20250625103219594.png)

### 2、JS散度

**Jensen-Shannon 散度**是衡量两个概率分布之间相似度的一种对称、平滑的度量方式。它是由 KL 散度（Kullback-Leibler divergence）扩展而来，并具有以下特点：
- **取值范围**：通常标准化为 [0,1]，越接近 0，表示两个分布越相似。
- **对称性**：JS(P‖Q) = JS(Q‖P)
- **有界性**：不像 KL 散度可能为无穷大，JS 散度总是有限。
- **可开平方**：其平方根称为 **JS 距离**，满足三角不等式，因此是一个**真正的距离度量**。

**定义**

为了计算两个概率分布P和Q之间的JS散度，首先计算它们的逐点平均分布，
$$
M=\frac12(P+Q)
$$
接着，计算每个分布相对于平均分布M的**Kullback–Leibler(KL)散度**，即$D_{\mathrm{KL}}(P \parallel M)$ 和 $D_{\mathrm{KL}}(Q \parallel M)$。KL散度是一种衡量一个概率分布相对于另一个分布偏离程度的度量：
$$
\text {KL}(P||M)=\sum P(x)\log\big(\frac{P(x)}{M(x)}\big)\\
\text {KL}(Q||M)=\sum Q(x)\log\big(\frac{Q(x)}{M(x)}\big)
$$
JS散度为这两个KL散度的均值
$$
D_\text{JS}(P||Q)=\frac12D_\text{KL}(P||M)+\frac12D_{KL}(Q||M)
$$

如果将n个样本的数据转换成连续概率密度，需要用到核密度估计（不太理解，前面不是已经推导出来）

核密度估计（KDE, Kernel Density Estimation）是一种**非参数概率密度估计方法**，用来从样本数据中**估计一个连续的概率密度函数（PDF）**。原理是将每个样本点视为一个核密度函数，将n个函数加起来归一化得到平滑的密度函数。
$$
\hat f(x)=\frac{1}{nh}\sum_{i=1}^nK\big(\frac{x-x_i}{h}\big)
$$


h：带宽（平滑系数）

K：核函数，满足积分为1，非负，常用高斯核来模拟

**代码**（没有参数解释，而且这个指标没有现成函数？）==涉及到数据转化为概率分布问题，处理较复杂，故简化==

```python
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import numpy as np

# KDE 拟合
kde_A = gaussian_kde(x)
kde_B = gaussian_kde(y)

# 创建统一评估网格
a = np.linspace(-1, 1, 1000)
p = kde_A(a)
q = kde_B(a)

# 归一化（变成离散概率分布）
p /= p.sum()
q /= q.sum()

# 计算 JS 散度
jsd = jensenshannon(p, q, base=2)**2
print(f"JS散度 ≈ {jsd:.4f}")
```

> JS散度 ≈ 0.0030

**应用**

为评估Loki Decompose将切片数据映射到单细胞类型的预测能力，采用JS散度评估预测结果与真实结果分布的距离相似性

*示例*

- 数据：不同的分解方法对自建的三阴性乳腺癌数据集单细胞预测结果与真实结果，Xenium数据将组织分为三种主要的细胞类型：肿瘤上皮细胞，免疫细胞和基质细胞

- 目的：评估Loki Decompose将切片数据映射到单细胞类型的预测能力

- 计算：根据切片嵌入与单细胞转录嵌入的相似性，为切片嵌入分配一个相似性最高的单细胞，通过JS散度测度预测的细胞标签分布与真实的单细胞标签分布之间的距离

- 结果：Loki Decompose在ST模型和图像模型中的影响得分分别为1.32和1.11，在所有方法中排前2。优于其他SOTA方法，包括Tangram, Spatial Seurat, CARD, CytoSPACE, Cell2location, SpatialDWLS 和 RCTD，这些方法的影响得分为0.87-1.82。

![image-20250627212632855](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250627212632855.png)

### 3、SSIM

**结构相似性指数**（SSIM, Structural Similarity Index））是一种**图像质量评价指标**，用来衡量两幅图像在**感知结构上的相似程度**。它广泛应用于图像压缩、图像重建、图像预测等任务，尤其适合于衡量**图像内容的保真度**，比传统的MSE更贴近人类视觉感知。

**定义**（xy是什么，只能对图像评估？）

给定两幅图像（或数组） x 和 y，其 SSIM 计算公式为：
$$
\begin{aligned}
\text{SSIM}(x,y)&=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
\end{aligned}
$$
其中，$\mu_x$ 和 $\mu_y$ 是图像 x 和 y 的平均强度（亮度），$\sigma_x^2$ 和 $\sigma_y^2$ 是它们的方差（对比度），$\sigma_{xy}$ 是 x 与 y 的协方差，$C_1$ 和 $C_2$ 是用于防止分母接近于 0 时不稳定的小常数。

**代码**（缺输入描述）

```python
import numpy as np

def calculate_SSIM(x, y):
    x = np.array(x)
    y = np.array(y)

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.std(x)  # 标准差
    sigma_y = np.std(y)
    sigma_xy = np.cov(x, y)[0, 1]  # 协方差

    # 常数项，假设数据范围在 [0,1]
    L = 1
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)

    ssim = numerator / denominator
    return ssim
calculate_SSIM(x,y)
```

**应用**

为评估Loki Decompose将切片数据映射到单细胞类型的预测能力，采用SSIM评估预测结果与真实结果分布的距离相似性

*示例*

- 数据：不同的分解方法对自建的三阴性乳腺癌数据集单细胞预测结果与真实结果，Xenium数据将组织分为三种主要的细胞类型：肿瘤上皮细胞，免疫细胞和基质细胞

- 目的：评估Loki Decompose将切片数据映射到单细胞类型的预测能力

- 计算：根据切片嵌入与单细胞转录嵌入的相似性，为切片嵌入分配一个相似性最高的单细胞，通过SSIM测度预测的细胞标签分布与真实的单细胞标签分布之间的距离

- 结果：Loki Decompose在ST模型和图像模型中的影响得分分别为1.32和1.11，在所有方法中排前2。优于其他SOTA方法，包括Tangram, Spatial Seurat, CARD, CytoSPACE, Cell2location, SpatialDWLS 和 RCTD，这些方法的影响得分为0.87-1.82。

![image-20250627212632855](C:\Users\WuMing\OneDrive - hrbmu.edu.cn\科研\生信分析\单细胞分析\OmiCLIP模型\assets\image-20250627212632855.png)

## 四、分类任务评估指标

### 1、混淆矩阵

![image-20250908151744590](C:\Users\WuMing\Documents\Tencent Files\1990866968\FileRecv\总结\assets\image-20250908151744590.png)

### 2、二级指标

- **精确率（Accuracy）**：分类正确的样本数占样本总数的比例

$$
\text{ACC}=\frac{TP+TN}{TP+FN+TN+FP}
$$



- **准确率（precision）**：是真正例占所有预测为正例的比例

$$
\text{PPV}=\frac{TP}{TP+FP}
$$



- **灵敏度（sensitivity）/召回率（recall）**：真正例占所有实际为正例的比例

$$
TPR=\frac{TP}{TP+FN}
$$



- **特异度（specificity）**：真负例占所有实际为负例的比例

$$
TNR=\frac{TN}{TN+FP}
$$



- **宏（macro-）指标**：在多个数据集上分别计算混淆矩阵的二级指标，取平均
- **微（micro-）指标**：在多个数据集上对混淆矩阵的TN,TP,FN,FP取平均，然后计算二级指标

### 3、三级指标

- **F1-measure**：是准确率和召回率的调和均值
  $$
  F=\frac{(\alpha^2+1)\cdot P\cdot R}{\alpha^2(P+R)}
  $$
  当α=1时即为F1 score，该指标的取值范围是[0, 1]，F1 Score越接近于0，模型的输出结果越差；F1 Score越接近于1，模型的输出结果越好

- **adjusted Rand index（ARI）**：ARI 取值范围为 [-1,1],值越大表明和真实结果越吻合，即聚类效果更好
  $$
  a =TP，b=FN，c=FP，d=TN\\
  \text{ARI}=\frac{2(ad-bc)}{(a+b)(b+d)+(a+c)(c+d)}
  $$
  

- **Fowlkes-Mallows Score**：为准确度和召回率的几何平均值

$$
\text{FMI}=\frac{TP}{\sqrt{(TP+FP)(TP+FN)}}
$$

- **kappa**：用于衡量分类结果（特别是两个评估者、分类器或方法之间）的一致性程度，并考虑了一致性中“偶然一致”的影响。高于 0.8 的分数通常被认为是良好的一致性

$$
\mathcal K=(p_o-p_e)/(1-p_e)\\
p_o=\text{ACC}\\
p_e=\frac{[(TP+FP)×(TP+FN)+(FN+TN)×(FP+TN)]}{(总计×总计)}
$$

$\kappa = 1$：完全一致

$\kappa = 0$：与随机一致程度相同

$\kappa < 0$：比随机一致还差，说明存在系统性不一致

- **AUC曲线**：ROC全称“受试者工作特征”曲线。根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次计算出真正例率TPR和假正例率FPR。

  - **假阳率（False positive rate ，FPR）**：在所有真实的负样本中，被模型错误的判断为正例的比例
    $$
    FPR = \frac{FP}{FP+TN}
    $$
    

  - **真阳率（True Positive Rate，简称TPR）**：在所有真实的正样本中，被模型正确的判断为正例的比例，其实就是召回率

$$
TPR = \frac{TP}{TP+FN}
$$

AUC( Area Under Curve )是曲线下面积，AUC给出的是分类器的平均性能值。 每一条ROC曲线对应一个AUC值。通常，AUC的取值在0.5与1之间。 模型的AUC面积值越大，模型的准确度越高，模型越理想

- **KS曲线**

- **PR曲线**：根据学习器的预测结果对样例进行排序，排在前面的是学习器认为“最可能”是正例的样本，排在最后的则是学习器认为“最不可能”是正例的样本。按此顺序逐个把样本作为正例进行预测，则每次可以计算出当前的查全率、查准率。以查准率为纵轴、查全率为横轴作图，就得到了查准率-查全率曲线，简 称 “P-R曲线〃，显示该曲线的图称为“P-R图”。

##  5、显著性检验

- **Permutation test score（置换检验）**：一种非参数统计方法，通过随机打乱数据标签来检验模型预测效果是否显著优于随机。

  - 原假设 $H_0$：模型和随机预测没有区别

  - 备择假设 $H_1$：模型的预测显著优于随机

  1. 训练模型，在真实数据上计算性能指标 $score_{true}$

  - 多次打乱标签（y 值），重新训练并计算性能得分，得到一个分布 $\{score_i\}$。
  - 计算 p 值：

  $$
  p = \frac{\#\{score_i \geq score_{true}\} + 1}{n\_permutations + 1}
  $$

  如果 p 很小（如 < 0.05），说明模型表现显著优于随机。

- **Bootstrap**

1. 从原始数据中随机抽样，并生成一个样本。
2. 在该样本上计算想要的统计量。
3. 重复前两步 K 次，得到 K 个统计量的估计值。
4. 根据 K 个估计值获得统计量的分布，并计算置信区间
