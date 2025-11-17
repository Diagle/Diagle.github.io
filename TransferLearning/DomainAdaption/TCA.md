[toc]

# Domain Adaptation via Transfer Component Analysis

>  Integrating struc tured biological data by Kernel Maximum Mean Discrepancy

Domain-adversarial training of neural networks.

Marginalized Denoising Autoencoders for Domain Adaptation.

Unsupervised Visual Domain Adaptation Using Subspace Alignment.子空间对齐

Correlation Alignment for Unsupervised Domain Adaptation.

Correlation Alignment for Deep Domain Adaptation.相关对齐

Aggregating Randomized Clustering-Promoting Invariant Projections for Domain Adaptation.

Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors. 

Computational Optimal Transport: With Applications to Data Science.

## 摘要

域适应使得来自源域的知识能够迁移到一个不同但有联系的目标域。直观来说，在不同领域之间发现一种良好的特征表示是至关重要的。在本文中，我们首先提出了通过一种新的学习方法，迁移成分分析（TCA）来对域适应寻找一种表示。TCA在再生核希尔伯特空间中使用最大均值差异来学习不同领域中的迁移成分。在由这些迁移成分所张成的子空间中，数据的固有属性得以保留，同时不同领域中的数据分布彼此更加接近。由此，在该子空间中的新表示下，我们可以在源领域中应用标准的机器学习方法来训练分类器或回归模型，并将其用于目标领域。

此外，为了发掘源领域与目标领域数据标签之间关系中所蕴含的知识，我们在半监督学习的设置下扩展了 TCA，将标签信息编码到迁移成分的学习过程中。我们称这种扩展方法为**半监督 TCA（Semi-supervised TCA）**。

我们的主要贡献在于提出了一种新颖的降维框架，用于在潜在空间中减少不同领域之间的距离，从而实现领域自适应（domain adaptation）。我们提出了无监督和半监督两种特征提取方法，这两种方法通过将数据投影到学习到的迁移成分上，可以显著减小领域分布之间的差异。

最后，我们的方法能够处理大规模数据集，并且自然地具备样本外（out-of-sample）泛化能力。我们通过在五个人工数据集以及两个真实应用场景（跨领域室内 WiFi 定位和跨领域文本分类）上的实验验证了该方法的有效性与高效性。

**索引术语**：降维、领域自适应、分布的希尔伯特空间嵌入、迁移学习

## 介绍

域适应旨在解决目标域的学习问题，通过使用源域的训练数据，即使这些领域可能具有不同的分布。这是一个重要的学习问题，因为标签数据经常很难获取，使得最大程度利用相关的可用数据是非常理想的。例如，室内WiFi定位需要回归学习，很难获取有标签的训练数据，并且代价昂贵。此外，即使获得，这些数据也容易过时，因为WiFi信号强度可能是由包含时间、设备和空间多种动态因子的函数结果。为减少重新校准的工作量，我们可能希望将一个在某个时间段（源领域）中训练的定位模型，适配到新的时间段（目标领域）；或者将一个在某个移动设备（源领域）上训练的定位模型，适配到新的移动设备（目标领域）。

域适应能被视为迁移学习的一种专门设置，旨在在不同但相关任务或领域见迁移共享知识。域适应一个主要的计算问题是如何减少源域和目标域数据间的分布差异。直观地，发现一组新的特征表示不同领域至关重要的。一组良好的特征表示应该能够尽可能地减少领域间的分布差异，与此同时保留原始数据重要的性质（例如几何特性、统计特性或辅助信息），特别是目标域数据。最近，一些方法被提出来学习域适应的一组常见特征表示。Daumé III ^[8]^设计了一个核
Recently, several approaches have been proposed to learn a common feature representation for domain adaptation [8]–[10]. Daumé III [8] designed a heuristic kernel to augment features for solving some specific domain adaptation problems in natural language processing. Blitzer et al. [9] proposed the structural correspondence learning (SCL) algorithm, motivated from [11], to induce correspondences among features from the different domains. This method depends on the heuristic selec-tions of pivot features appearing frequently in both domains.Although it is experimentally shown that SCL can reduce the difference between domains based on the A-distance measure[6], the heuristic criterion of pivot feature selection may be sensitive to different applications.

Most previous feature-based domain adaptation methods do not minimize the distance in distributions between domains explicitly. Recently, von Bünau et al. [12] proposed stationary subspace analysis (SSA) to match distributions in a latent space. However, SSA is focused on the identification of a stationary subspace, without considering the preservation of properties such as data variance in the subspace. Pan et al.[10] proposed a new dimensionality reduction method called maximum mean discrepancy embedding (MMDE) for domain adaptation. MMDE aims at learning a shared latent space underlying the domains where distance between distributions can be reduced while the data variance can be preserved. How-
ever, MMDE suffers from two major limitations: 1) MMDE is transductive, and does not generalize to out-of-sample patterns [13], and 2) MMDE learns the latent space by solving a semi- definite program (SDP), which is computationally expensive.

In this paper, we propose a new feature extraction approach, called transfer component analysis (TCA), for domain adaptation. It tries to learn a set of common transfer components underlying both domains such that the difference in data distributions of the different domains, when projected onto this
subspace, can be dramatically reduced and data properties can be preserved. Standard machine learning methods can then be used in this subspace to train classification or regression models across domains. More specifically, if two domains are related to each other, there may exist several common components (or latent variables) underlying them. Some of these may cause the data distributions between domains to
be different, while others may not. Meanwhile, some of these components may capture the intrinsic structure or discrimi- native information underlying the original data, while others may not. Hence, our goal is to discover those components that do not cause distribution change very much across the domains and can well preserve the structure or task-relevant information of the original data.
Our main contribution is on proposing a novel dimen-
sionality reduction method to reduce the distance between
domains via projecting data onto a learned transfer subspace.
Once the subspace is found, one can use any method for
subsequent classification, regression, and clustering. Further-
more, TCA and its semisupervised extension SSTCA are
much more efficient than MMDE and can handle the out-
of-sample extension problem [13]. The rest of this paper is
organized as follows. In Section II, we first introduce the
domain adaptation problem and traditional dimensionality re-
duction methods and describe the Hilbert space embedding for
distances and dependence measure between distributions. Our
proposed feature extraction methods for domain adaptation are
presented in Sections III and IV. In Section V, we conduct a
series of experiments on some toy datasets and two real-world
application problems to verify the effectiveness and efficiency
of the proposed methods. Finally, we conclude our work in
Section VI.