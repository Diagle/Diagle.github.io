---
layout: post
title: "类别感知样本重加权：Class-Aware Sample Reweighting(CASR)"
date: 2026-10-30 15:00:00 +0800
categories: [TransferLearnings, Research]
tags: [Multi-source Unsupervised Domain Adaptation, TransferLearnings]
---

2023 Neurocomputing IF 5.5

**摘要**

多源域域适应（MSDA）技术因为其将多源域的知识迁移到无标签的目标域的能力而引起广泛关注。最优运输（OT）由于其鲁棒性最近被用以测度分布间的距离。本文提出了一个新颖的基于OT的类别感知样本重赋权（CASR）方法来实现多源域核目标域样本级细粒度水平的对齐。技术上，类别感知采样策略通过从每个域中明确选择样本来确保在运输期间类别水平条件对齐。此外，样本感知模块设计来对每个运输的样本分配特定的质量，这一步考虑了类别的可靠性和空间信息的相关性以得到目标域和多源域的对齐比例并进一步优化运输计划。在一些基准测试上主导的拓展实验表明CASR较之于其他MSDA方法展现出显著的优势，并且可视化分析进一步证实了每个模块的效果。

# 1. 引言

基于深度学习的监督学习方法试图发现特征与标签信息之间的关系，并利用特征融合（feature fusion）或注意力机制（attention mechanism）设计新的模块，以提升网络捕获全局上下文信息的能力，从而完成各种复杂的计算机视觉任务[9–12]。

然而，由于从不同来源收集的数据会受到环境变化或资源差异的影响，往往不遵循相同的数据分布[14]，因此即使是在相同或相似任务上，训练良好的模型在新收集的无标注数据集上也难以达到预期性能[13]。此外，对每一条数据进行人工标注既繁琐又不现实。基于这些原因，大量研究致力于开发能够将训练好的模型应用于新的无标注数据集的方法，这类技术被称为**无监督领域自适应（Unsupervised Domain Adaptation，UDA）**[15]。

在UDA任务中，假设源域（source domain）S和目标域（target domain）T分别服从两个不同的联合分布 $P_S(x,y)$ 和 $P_T(x,y)$。域间差异（domain shift）通常建立在以下几种假设之上：

- **协变量偏移（Covariate Shift）**：$P_S(x) \neq P_T(x)$
- **标签偏移（Label Shift）**：$P_S(y) \neq P_T(y)$
- **条件偏移（Conditional Shift）**：$P_S(x|y) \neq P_T(x|y)$
其中，x 和 y 分别表示特征和标签。为了量化领域差异，常见方法通常通过显式利用距离度量来衡量分布差异[16]，或采用最小—最大（min-max）对抗框架生成不可区分的领域特征[17,18]。
在更贴近现实的场景中，通常存在多个来自不同来源且带有标签的数据集。多源领域自适应（Multi-Source Domain Adaptation，MSDA）技术旨在将多个标签丰富的源域知识迁移到一个无标签目标域[19]。与UDA相比，由于所有领域之间普遍存在更大的领域差距（domain gap），MSDA面临更大的挑战[20–22]。

当前主流的MSDA方法大多建立在分布加权假设（distribution-weighted hypothesis）之上，即认为目标域可以表示为多个源域的凸组合（convex combination）[23]。因此，研究者通常显式学习每个源域与目标域之间相关性的迁移权重，或者通过计算多个源域分类器预测结果的加权和来获得目标域预测结果。然而，这类基于权重的方法忽略了多源域之间存在的图像分布差异。学习得到的粗粒度权重缺乏足够的可解释性，也难以捕捉领域之间复杂的关系，因此往往只能获得次优的性能。

为了解决上述挑战，本文提出了一种基于**最优传输（Optimal Transport，OT）**[24]的简单而有效的方法。OT最初用于解决资源分配问题，非常适合刻画复杂概率分布的几何结构。与其他常用距离度量相比，即使两个分布的支持集（support）没有重叠，OT仍能提供有意义的距离度量。

同时，OT通过构建样本之间的传输计划（transport plan），并最小化样本质量（mass）迁移的总成本来衡量分布差异。这种方式本质上是在样本层面实现领域对齐（domain alignment），并且便于直接从不同领域中选择样本建立传输关系，从而进一步实现领域间的对齐。
在本文中，我们并非显式学习每个源域的组合权重，而是提出一种更加细粒度的、基于OT的类别感知样本重加权方法（Class-aware Sample Reweighting，CASR）。该方法能够自适应地选择与目标样本更相关的源域样本，从而实现目标域与多源域子集之间的潜在对齐（latent alignment）。

具体而言：

- 首先，设计了一种类别**感知采样策略（class-aware sampling strategy）**，控制采样过程，确保传输两侧的类别尽可能重叠。
- 然后，为了抑制无关源样本参与传输，我们综合考虑：
	- 目标样本伪分类概率（pseudo-classification probability）的可靠性；
	- 与多源样本之间的空间信息相关性（spatial information correlation）；

为每个样本分配权重，从而更细致地挖掘跨域结构信息，并持续优化传输计划。

本文的主要创新和贡献总结如下：
- 提出一种基于OT的MSDA方法，用于探索多源领域自适应任务中的复杂分布结构。利用类别感知采样策略，显式满足**类别级条件对齐（class-level conditional alignment）** 的要求。
- 设计了一种基于样本可靠性与相关性的样本重加权模块，实现细粒度最优传输，减少由多源样本均匀分布所带来的无差别对齐问题。
- 不同于分别将目标域与各个源域单独对齐的方法，本文逐步地、有选择地将目标域适配到组合源域。整个框架支持端到端（end-to-end）训练。实验结果验证了各模块的有效性，并表明CASR相较于现有先进方法具有竞争力的性能。

本文其余部分安排如下：

- 第2节介绍MSDA相关研究工作；
- 第3节给出任务定义以及MSDA中常用的最优传输方法；
- 第4节详细阐述本文提出的CASR方法；
- 第5节通过大量实验验证方法有效性，并与其他方法进行性能比较；
- 第6节总结全文。
# 2. 相关工作

本节总结与本文方法最相关的研究工作。我们主要关注多源领域学习场景下的领域自适应技术以及最优传输（Optimal Transport）的应用。

## 2.1 单源域适应

单源领域自适应（Single-Source Domain Adaptation，SSDA）方法主要从两个角度解决领域偏移（domain shift）问题。

- 第一类方法是基于**对抗学习（adversarial-based）** 的方法，通过学习领域不变特征（domain-invariant features）来实现两个领域之间的对齐[25,26]。

- 第二类方法是基于**分布差异（discrepancy-based）** 的方法，其致力于探索不同的度量学习（metric learning）策略，以最小化两个概率分布之间的统计差异（statistical divergence）[27–29]。

本文进一步关注更加复杂的数据分布场景下的领域自适应问题。
## 2.2 多源域适应
多源领域自适应（Multi-Source Domain Adaptation，MSDA）假设训练数据来自多个具有不同分布的场景。

一些早期研究基于理论保证，即目标分布可以被看作多个源分布的凸组合（convex combination）[20]，因此建议重点关注目标域与多个源域之间的分布差异。

- **DCTN（Deep Cocktail Network）** [30] 通过对抗学习最小化多个源域与目标域之间的分布差异。
- **MDDA** [18] 为每个源域分别学习特征提取器和分类器，并估计权重以获得最终预测结果。
- **DAN-MA** [31] 利用两个对抗网络学习领域不变特征，并为每个源域使用两个分类器，使目标样本接近各源域的决策边界。
- **DARN** [32] 动态调整各源域权重，通过增加有效样本数量来提升自适应性能。

此外，近年来还有一些方法关注源域之间的分布偏移问题。这些方法认为，多源域之间的领域冲突会导致领域无关模型（domain-agnostic model）性能下降。

- **M3SDA** [16] 采用矩匹配（moment matching）方法实现源域之间的对齐。
- **MDMN** [33] 基于相似思想，利用Wasserstein距离减小领域间差异。
- **DRT** [34] 提出动态迁移（dynamic transfer）机制，为每个样本生成不同模型，并促进目标域与多源域中相关部分的对齐。
## 2.3 最优运输
最优传输（Optimal Transport，OT）通过计算分布之间在给定底层代价（ground cost）下所需的最小运输成本来衡量分布间距离。

受内存瓶颈限制，原始OT的大规模计算通常通过求解来自各分布的小批量（mini-batch）子问题来近似实现[35]。

在这种情况下，为了纠正多批次映射相对于完整OT传输产生的不准确性，研究人员提出了：

- 部分最优传输（Partial Optimal Transport）[36]；
- 非平衡最优传输（Unbalanced Optimal Transport）[37,38]。

在更具体的MSDA应用中：

- **WJDOT** [39] 通过优化多个源域加权组合与目标域之间的Wasserstein距离实现迁移。
- **JCPOT** [40] 通过求解Wasserstein重心（barycentric）问题来学习权重，并在目标偏移（target shift）情况下估计类别比例。
- **WBT** [41] 通过重心映射（barycentric mapping）生成一个代表所有源域的中间域，然后计算该中间域与目标域之间的OT传输计划。

本文提出一种**类别感知样本策略（class-aware sample strategy）**，用于控制参与传输的样本类别，并在小批量最优传输（mini-batch OT，m-OT）过程中提供显式指导，从而消除由于领域间类别不平衡导致的不准确条件分布对齐问题。

# 3. 预备知识

本节首先介绍MSDA任务的定义及其通用框架，然后说明最优传输在MSDA中的应用，并指出现有方法的不足之处。

## 3.1 任务定义

MSDA中，假设我们具有M个源域$D_s=\{S^i\}_{i=1}^M$，每个域被表征为$N_i$个独立同分布的带标签样本$X_{S^i}=\{x_i^j,y_i^j\}_{j}^{N_i}$，其边缘分布为$\mu_{S^i}=P(X_{S^i})$。类似的，$D_T$为单个目标域表征为$N_T$个未带有标签的样本 $X_T=\{X_T^i\}_{i=1}^{N_T}$，其分布为$\mu_T=P(X_T)$，并且$\mu_{S^i}\ne\mu_T$。本文中，我们假设所有域的共享相同的标签空间$\mathcal Y$，即$y_{S^i},y_T\in\mathcal Y$。根据计算理论，我们的目标是寻找以给假设h，使得最小化目标域上的经验风险$\epsilon_T(h)$。假设$h$通常由两个部分构建：
- 特征提取器$G_\theta$：将样本映射到共享空间并提取所有源域和目标域信息间的联系
- 分类器$F_\varphi$：根据所提取的特征构建条件概率分布。
一般来说，特征提取器$G_\theta$和标签函数$F_\varphi$同时优化。因为MSDA数据集中大量的数据，实践中常常在每个训练iteration随机选择一部分子集。在此，我们使用$\hat D=\{\hat S^1,\hat S^2\,...,\hat S^M, \hat T\}$表示源域和目标域，$N_b$为每个域子集的样本数量。技术上，每个域的标准分类损失可以描述如下，其中l表示交叉熵损失：
$$
L_{cls_i}(X_{\hat S^i})=\frac{1}{N_b}l(F(G(x_i^j;\theta);\varphi),y_i^j)\tag{1}
$$
## 3.2 用于MSDA的最优运输

最优传输（Optimal Transport，OT）作为一种经典工具，因其诸多优势而被广泛用于衡量概率分布之间的距离。

在MSDA中，每个领域都可以看作由给定数量样本构成的离散测度（discrete measure）。由 $N_i$​ 和 $N_T$​ 个Dirac测度均匀加权求和得到的经验测度（empirical measures）定义如下：
$$
\hat u_{S^i}:=\frac{1}{N_i}\sum_{j=1}^{N_i}\delta_{x_i^j},\quad \hat\mu_T:=\frac{1}{N_T}\sum_{j=1}^{N_T}\delta_{x_T^j}​\tag{2}
$$
其中$\delta _{x^j}$表示位于$x^j$的Dirac测度。通常，各领域中的样本相互独立，其权重一般设置为相同，即采用均匀加权（uniform weighting）方式。基于OT的Wasserstein距离被用作经验分布之间的距离度量，其对应于一个基础代价矩阵（ground cost matrix）C。其正则化形式定义如下：
$$
W_C(\hat\mu_{s^i},\hat\mu_{T})=\min_{\gamma\in\Pi(\hat\mu_{S^j},\hat\mu_T)}\big< C,\gamma\big>_F-\epsilon H(\gamma)\tag{3}
$$
其中$\big<.,.\big>_F$表示Frobenius点积，H(γ)表示γ的正则项。在MSDA中，域之间的Wasserstein距离计算为批次尺度以避免在大规模应用中的经验计算问题。因此，每个iteration中域间的批次尺度OT损失描述如下：
$$
\begin{aligned}
L_d(\hat\mu_S,\hat\mu_T)
&=\sum_{i=1}^MW_c(\hat\mu_{S^i},\hat\mu_T)
\\
&=\min_{\gamma_{\hat S^iT}\in\Delta}\sum_{i=1}^MC(X_{\hat S^i},Y_{\hat S^i};X_{\hat T},F(G(X_{\hat T})))\gamma_{\hat S^iT}\\
&-\epsilon H(\gamma_{\hat S^iT})
\end{aligned}\tag{4}
$$
其中，$\delta$ 为属于集合 $\Delta$ 的耦合矩阵。通常，分布间的批次尺度运输代价分解为特征和标签水平。对于特征水平测度函数，d设为欧式距离。总体运输矩阵计算为如下：
$$
C=d(G(X_{\hat S^i});G(X_{\hat T}))+\alpha||Y_{\hat S^i},F(G(X_{\hat T}))||_2\tag{5}
$$
然而，与在完整原始分布上计算的OT相比，基于批次的传输（batch-wise transport）存在近似误差。

在训练过程中，随机采样无法保证每个领域中的类别重叠（category overlap），从而导致条件分布（conditional distribution）的错误对齐（misalignment）。此外，均匀加权策略也可能产生负面影响，因为不同样本对于领域对齐的贡献程度并不相同。

下面，我们将进一步优化分布间批量OT距离的计算过程。本文提出采用**类别感知采样策略（class-aware sampling strategy）和样本重加权模块（sample reweighting module）** 来细化每次迭代中的对齐过程，从而避免仅进行边缘分布对齐（marginal distribution alignment）以及无差别样本传输（undifferentiated transport）所导致的负迁移（negative transfer）问题。

# 4. 类别认知样本重加权

为了在OT度量下构建类条件（class-conditional）的细粒度对齐，我们期望能够发现不同类别样本的多模态分布，并刻画多源域中每个样本的重要性。

因此，我们提出**类别感知样本重加权（Class-Aware Sample Reweighting，CASR）** 方法，这是一种新的基于深度最优传输（deep OT）的多源领域自适应模型。整体框架如图1所示。

具体而言，对于类条件对齐，在最小化跨域差异的同时，我们提出一种类别感知采样策略，用于控制每个批次中的样本类别，从而促使OT在同类别样本之间建立传输关系。

与此同时，我们根据分类结果的可靠性，为多源域和目标域的每个样本分配权重，以实现更细粒度的跨域对齐。

具体过程将在后续章节中详细阐述。

![[Pasted image 20260611150350.png]]
> 类别感知模块自动从每个域采样特定类别的集合，共享特征提取器$G_\theta$将样本映射到潜空间。然后，对多源域的高维嵌入$G_\theta(X^S)、G_\theta(X^T)$，样本重加权模块产生重加权向量$w_S,w_T$来完成细粒度对齐。最终，两个分类器对抗训练来获得更好的收敛

## 4.1 类别感知采样策略

在这种情况下，许多已有算法致力于重新组合多源域中的类别比例，以使其适应目标域的类别分布[39,40]。还有一些算法利用基于类别的正则化项（class-based regularization terms），对属于同一类别且彼此接近的源域样本施加约束，避免它们被传输到目标域中的远距离位置[41,42]。

尽管这些方法有助于寻找更优的样本耦合关系（couplings），但在基于批次（batch-wise）的训练过程中，随机采样仍然无法避免类别不确定性（category uncertainty）问题，即从源域和目标域中选取的样本难以具有相似的标签分布。因此，这两个子集之间的对齐过程可能导致域内结构（intra-domain structure）上的类别级错位（class-level misalignment），最终只能获得域级别的次优对齐结果（suboptimal alignment）。

为了解决这一问题，我们采用一种先验的**类别感知采样策略（class-aware sampling strategy）**。该策略根据预先指定的标签集合直接选择样本，从而保证每一对源域–目标域之间的类条件对齐（class-conditional alignment）。具体而言，在每次训练iteration中，为了保证所有领域中的样本类别完全重叠，首先随机选择一个特定的标
签子集：$\tilde{Y}\subset\mathcal Y$，其中$\tilde Y$包括K个元素，其中任一表示一个类别。任一$y_k\in\tilde Y$，我们在每个域中对类别$y_k$采样m个样本作为当前批次的训练样本，其中m是预先指定的，每个域的样本总数为$N_b=mK$。

对于任一源域$S^i$，$\cal Y$中任一类别中样本能够根据金标准标签$Y_S$直接选择，当前训练批次的总训练样本可以表示为$X_{\hat S}=[X_{\hat S^1};X_{\hat S^2}; ...; X_{\hat S^M}]$，其中包含$N=M\times N_b$个样本。然而，对于目标域，类别特异选择依赖于预测的伪标签$\hat Y_{\hat T}=F(G(X_{\hat T}))$。除了基于分类器的单一预测之外，大量方法用于提高伪标签的可靠性。目标域的样本训练批次包含$N_b$个样本并表示为$X_{\hat T}$。类别感知模块采样策略的效果如图2所示。

![[Pasted image 20260611153835.png]]
> 根据不同样本策略域之间的运输映射不同，其中$M=1,m=1,K=5$
> （左）随机采样，其中，传输计划中出现了类别对齐不明确的问题。
> （右）类别感知采样，相同类别的运输确定于高概率。


在训练过程中，与式（4）不同，多源域与目标域之间的分布差异并不是分别计算的。我们将当前迭代中的所有样本进行组合，并构建一个统一的集成传输映射（ensemble transport map）。因此，领域间差异可计算如下：
$$L_d(\hat\mu_S,\hat\mu_T)=W_C(\hat\mu_S,\hat\mu_T)=\min_{\gamma_{\hat S\hat T}\in\Delta}\big<C,\gamma_{\hat S\hat T}-\epsilon H(\gamma_{\hat S\hat T}) \big>\tag{6}
$$

其中$C\in\mathbb R^{N\times N_b}$为任一训练批次的代价矩阵，$\gamma_{\hat S\hat T}$为多源域和目标域的运输映射。

## 4.2 样本重加权模块

根据最优传输（OT）理论[35,44]，传输计划（transport plan）的构建不仅依赖于样本分布，还依赖于传输代价矩阵（transport cost matrix）。根据第4.1节的设定，我们假设当前训练批次中的样本类别已经满足避免负迁移（negative transfer）所需的必要条件。理想情况下，每个目标域样本都应与源域中属于相同类别的样本进行对齐。在传统设置下，源域和目标域的概率质量（probability mass）通常被设置为均匀分布，即：$m_{\hat S}^i=\frac1N\sum_{i=1}^N\delta _{X_{\hat S}^i}, m_{\hat T}^i=\frac{1}{N_b}\sum_{i=1}^{N_b}\delta _{X_{\hat T}^i}$。然而，然而，这种设置会带来以下两个问题：

1. 受样本概率质量约束的影响，即使存在同类别样本，仅依赖代价矩阵进行匹配，在复杂分布情况下仍可能产生不期望的跨类别传输（cross-class transfer）。
2. 即便构建的传输关系满足类别级条件对齐（class-level conditional alignment），均匀加权分布仍可能导致目标样本无差别地与多个源域样本进行对齐，从而产生次优的适应效果（suboptimal adaptation）。
为了缓解上述负传输（negative transport）带来的影响，已有研究尝试通过修正样本间的传输代价矩阵来解决这一问题。
- RWOT [45] 考虑收缩子空间可靠性（shrinking subspace reliability），结合原型空间信息（prototypical spatial information）和域内信息（intra-domain information）来提升传输效率。
- ETD [46] 则利用注意力机制（attention mechanism）来修正样本之间的相似性度量。
与这些直接修改代价矩阵的方法不同，本文提出了一种**样本重加权模块（sample-reweighting module）**。该模块分别学习源域和目标域训练样本的重加权向量$w_S,w_T$ ,通过调整样本的传输比例（transport proportion），进一步隐式增强传输过程的鲁棒性（transport robustness）。
对于目标域，我们采用基于熵（entropy-based）的分类可靠性度量[26]来为不同样本分配权重。其基本思想是：预测结果越可靠（即预测正确概率越高）的样本，应被赋予越高的优先级权重。对于类别 $p$ 中的目标样本 $X_{\hat T}^i$​，其对应的重加权向量$w_{\hat T}$计算如下：
$$
w_{\hat T}(i,p)=e^{\frac1m\sum_{i=1}^m\sum_{j=1}^K\hat y_{T}^{ij}\log\hat y_T^{ij}}\tag{7}
$$
其中$\hat y_{\hat T}^{ij}$表示第i个目标样本属于第j个类别的概率。

对于源域，我们利用目标域的原型信息（target prototypical information）来调整样本的概率质量（probability mass）。其基本思想在于：单纯的平方欧氏距离（squared Euclidean distance）并不能真实反映不同领域之间的实际关系。例如，位于源域分布边缘的样本，可能与位于目标域分布另一侧的目标样本距离较远。如果强制要求多源域与目标域之间进行均匀传输，则可能导致不必要的对齐（unnecessary alignment）以及较低的收敛效率（convergence efficiency）。因此，优先将目标样本与位于相邻条件分布（adjacent conditional distributions）中的源域样本进行对齐是有益的。首先，必须保证传输计划两侧的概率质量保持平衡。换句话说，对于类别 p，源域样本的总质量需要与目标域中同类别样本的总质量相等，即：$h_{\hat S}^p=\sum_{i=1}^N\delta(y^i=p)m_{\hat S}^i=h^p_{\hat T}$。其中两边分别表示源域和目标域中类别 $p$ 的总概率质量。随后，对于类别 $p$ 中的每个源域样本$X_{\hat S}^i$，其对应的重加权向量$w_{\hat S}$可计算如下：
$$
w_{\hat S}(i,p)=\frac{e^{-d(G(x^i_{\hat S}),c_T^p)}}{\sum_{j=1}^{m\cdot M}e^{-d(G(x^j_{\hat S}),c_T^p)}}\cdot h^p_{\hat S}\tag{8}
$$
其中，$d(G(x_{\hat S}^i), c_T^p)$为嵌入源域样本与第p个目标类别中心$c_T^p$的距离，$c_T^p$为表示目标域类别 *p* 的全局原型（global target prototype）,该原型基于目标域中具有伪标签（pseudo-label）的样本构建，通过对类别 $p$ 内所有样本的特征嵌入（feature embeddings）求均值得到。
基于重加权向量$w_S,w_T$，边缘分布写为$\hat\mu_S'=(w_S)^S\delta_{X_{\hat S}},\hat\mu_T'=(w_T)^T\delta_{X_{\hat T}}$。运输映射$\gamma'_{\hat S\hat T}=\underset{\gamma\in\Pi(\hat\mu_S',\hat\mu_T')}{\arg\min}\big<C,\gamma\big>_F-\epsilon H(\gamma)$，类似于等式6，域间的差异损失描述为如下：
$$
L_d(\hat\mu_S,\hat\mu_T)=\gamma'_{\hat S\hat T}\cdot C\tag{9}
$$
## 4.3 优化
优化方法（Optimization methods）旨在利用问题目标函数表面的梯度信息，在合理时间内搜索并返回最优解或次优解[47–49]。本节介绍CASR的优化过程，其伪代码如算法1所示。在训练阶段，我们将所提出的CASR方法部署到基于MDD（Margin Disparity Discrepancy）的领域差异度量框架中，该框架引入了边际损失（margin loss）。分类器$F_\varphi$和$F_\phi$通过联合优化来保证分类结果的一致性（consistency of classification results）。多源域的分类损失（classification loss）定义为各个源域分类损失之和，即如式（1）所示：
$$
L_{cls}(X_{\hat S})=\frac1M\sum_{i=1}^ML_{cls}(X^i_{\hat S_i})=\frac1M\cdot\frac{1}{N_b}\sum_{j=1}^{N_b} l(F(G(x_i^j;\theta);\varphi),y_j^i)\tag{10}
$$
分类器的对抗损失为：
$$
L_{adv}(X_{\hat S},X_{\hat T})=\sup|\sum_{X_{\hat T}}[F_\varphi\ne F_\phi]-\sum_{X_{\hat S}}[F_{\varphi}-F_\phi]|\tag{11}
$$
考虑到基于OT类别感知域差异，CASR算法通过求解以下最优化问题来训练：
$$
\min \mathcal L=\mathcal L_{cls}+\alpha\mathcal L_{adv}+\beta\mathcal L_d\tag{12}
$$
其中，β 和 γ 是超参数，用于在不同数据集上分别调节各部分损失（或目标项）之间的权衡关系。
在评估阶段，分类器$F_\varphi$用以预测目标标签
$$
\hat y_i=F(G(x^i_T;\theta;\varphi))\tag{13}
$$
# 5. 实验

## 实施细节

- 特征提取器采用预训练-微调策略。
- 分类器为单层全连接神经网络，随机初始化训练。
- 优化器：带动量的SGD优化器，lr初始化为1e-3，权重衰减为5e-4。
- 对于权衡参数（trade-off parameters）β和γ，采用网格搜索$\{0.1,0.5,1,5,10\}$
- 正则化系数ε设置为1
- 最大迭代次数设置为10000，所有迁移任务重复5次，取平均结果。
# 6 结果

## 消融实验

对比了分类损失、分类损失+对抗损失、在前两者基础上分别添加原始OT/类别感知采样CAS/CAS+样本重加权SR

## 类别感知模块的效果评估

对比不同类别比例下类别感知采样与随机采样对ACC的影响

## 联合运输的效果

评估联合运输与分别运输对ACC的影响
## 收敛性

评估SR/CAS/CASR ACC的收敛迭代次数

## 可视化

**运输计划与OT损失曲线的可视化**

**T-SNE嵌入**

比较域适应前后各个类别中源域和目标域混合情况，并且目标域不是强制对齐所有源域还是某个特定的域。