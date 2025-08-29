基于适配器的方法在冻结的预训练模型的注意力层和全连接层之后添加额外的可训练参数，以减少内存占用并加快训练速度。

可以只是一个额外的添加层，也可以将权重更新 $\Delta W$ 表示为权重矩阵的低秩分解。无论哪种方式，适配器通常都很小，但表现出与完全微调模型相当的性能，并且能够以更少的资源训练更大的模型。

# Low-Rank Adaptation (LoRA)

LoRA 通过低秩分解将权重更新 $\Delta W$ 表示为两个较小的矩阵（称为更新矩阵 ）。这些新矩阵可以进行训练以适应新数据，同时保持较低的参数总数。原始权重矩阵保持不变，不会进行任何进一步的更新。LoRA 与其他参数高效方法正交，通常仅应用于 Transformer 模型中的注意力模块，可训练参数的数量取决于更新矩阵的大小，而更新矩阵的大小主要由秩 r 和原始权重矩阵的形状决定。

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)<br>

# Mixture of LoRA Experts (X-LoRA)

X-LoRA 是一种 LoRA 混合专家模型，其工作原理是使用密集或稀疏门控来动态激活 LoRA 专家。LoRA 专家模型和基础模型在训练期间均处于冻结状态，因此只需训练门控层即可降低参数数量。

门控层输出的缩放比例（取决于配置）在层和 token 级别上是精细的。此外，在推理过程中，X-LoRA 会动态激活 LoRA 适配器来调用知识并有效地进行混合

对于每个 step ，X-LoRA 都需要基础模型运行两次：首先，在不使用任何 LoRA 适配器的情况下获取隐藏状态；其次，使用隐藏状态计算应用于 LoRA 适配器的缩放比例，并再次运行模型。第二次运行的输出即为模型步骤的结果。通过双重前向传递方案反思其知识，并动态地重新配置架构。

[X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design](https://arxiv.org/abs/2402.07148)<br>

# Low-Rank Hadamard Product (LoHa)

低秩分解会影响性能，因为权重更新仅限于低秩空间，这会限制模型的表达能力。LoHa 使用 Hadamard 积 （逐元素积）而非矩阵积。 $\Delta W$ 由四个较小的矩阵表示，而不是像 LoRA 中那样由两个矩阵表示，并且每对低秩矩阵都与 Hadamard 积组合。因此， $\Delta W$ 可以具有相同数量的可训练参数，但具有更高的秩和表达能力。

[FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning](https://arxiv.org/abs/2108.06098)<br>

# Low-Rank Kronecker Product (LoKr)

LoKr 与 LoRA 和 LoHa 非常相似，也主要应用于扩散模型，也可以将其用于其他类型的模型。LoKr 将矩阵积替换为克罗内克积 。克罗内克积分解会创建一个分块矩阵，该分块矩阵保留了原始权重矩阵的秩。克罗内克积的另一个优点是可以通过堆叠矩阵列来实现向量化。

[Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation](https://arxiv.org/abs/2309.14859)<br>

# Orthogonal Finetuning (OFT)

OFT 尝试在同一层中所有成对神经元之间保持相同的余弦相似度（超球面能量），因为这可以更好地捕捉神经元之间的语义信息。这意味着 OFT 更能保留主体，并且更适合可控生成（类似于 ControlNet ）

OFT 通过学习神经元的正交变换来保持它们之间的余弦相似度不变，从而保留超球面能量，即将正交矩阵与预训练权重矩阵进行矩阵乘积。

为了提高参数效率，正交矩阵被表示为具有秩 $r$ 个块的块对角矩阵

[Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280)<br>

# Adaptive Low-Rank Adaptation (AdaLoRA)

AdaLoRA 通过为更适合特定任务的重要权重矩阵分配更多参数（即更高的秩 $r$ ）并修剪不太重要的权重矩阵来管理从 LoRA 引入的参数

$\Delta W$ 的秩根据重要性得分进行调整。 $\Delta W$ 被划分为三元组，并根据每个三元组对模型性能的贡献进行评分。重要性得分低的三元组将被修剪，重要性得分高的三元组将被保留以进行微调。

AdaLoRA 的训练分为三个阶段： init 阶段、 budgeting 阶段和 final 阶段。 init 阶段不采用 budgeting ，因此 randk 不会受到影响。在 budgeting 阶段，将应用上述流程，并根据 budgeting 重新分配 rank ，旨在给予更重要的适配器更高的 rank ，而给予不太重要的层更低的 rank 。当到达 final 阶段时， budgeting 已经结束， rank 将重新分配，使用重新分配的 rank 继续训练一段时间，以进一步提升性能。

[Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)<br>

# Llama-Adapter

Llama-Adapter 将 Llama 适配到指令跟随模型的方法。为了帮助模型适应指令跟随，该适配器使用 52K 指令输出数据集进行训练。

一组可学习的自适应提示被添加到输入指令标记的前缀中。这些提示被插入到模型的上层，因为使用预训练模型的高级语义进行学习效果更佳。添加到输入前缀的指令输出标记会引导自适应提示生成上下文响应。

为了避免给 token 增加噪声，适配器使用了零初始化的注意力机制。在此基础上，适配器还添加了一个可学习的门控因子（以零初始化），以便在训练过程中逐步向模型添加信息。这可以防止新学习到的指令淹没模型的预训练知识。

[LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/abs/2303.16199)<br>

# Householder Reflection Adaptation (HRA)

HRA 为连接 LoRA 和 OFT 提供了一个新的视角，这意味着它可以利用两种策略的优势，减少参数和计算成本，同时惩罚预训练知识的丢失。

HRA 构建了 $r$ 可训练的 Householder 反射（HR）链。由于 Householder 反射矩阵是正交矩阵，且正交矩阵的乘积也是正交矩阵，因此 HRA 满足正交微调（OFT）的理论保证。同时，通过改写公式，HRA 也可以看作是一个低秩微调适配器。

$r$ 越高，可训练参数越多，模型容量越大，性能也越好。此外，由于链式结构，HR 平面的正交性会影响 HRA 的容量和正则性。为了在模型容量和正则性之间取得平衡，在损失函数中添加了 HR 平面的正交正则化器。权重 $\lambda$ 可以控制正则化器的强度。

[Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation](https://arxiv.org/abs/2405.17484)<br>

