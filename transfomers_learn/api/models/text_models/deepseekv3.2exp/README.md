# Contribution

论文中提出了 DeepSeek 稀疏注意力（DeepSeek Sparse Attention, DSA）机制。
工作并非从零开始构建一个全新的模型，而是在 Dense Model DeepSeek-V3.1-Terminus 的基础上，通过持续训练的方式，将模型平滑地迁移到稀疏注意力架构上。
这种方法实现了继承现有强模型能力，同时赋予其长上下文效率优势。

# Background

LLM 推理分为两个阶段： prefill 阶段和 decode 阶段
1. prefill 阶段：模型通过 casual mask ，对全部的 Prompt tokens 一次性并行计算，最终会生成第一个输出token
2. decode 阶段：每次生成一个 token ，直到生成 EOS（end-of-sequence）token ，产出最终的 response

LLM 生成是一个基于前向 token 序列预测下一个 token 的过程，序列中的 token 只与它前面的 token 交互来计算 casual attention ，前面的 token 作为 key 和 value 在 prefill 中要被重复计算

KV-cache 机制：为了加速训练和推理的效率，在推理过程中，避免重复计算前序的 $k,v$ ，把前序计算好的缓存起来
- prefill 阶段：计算每一层所有的 prompt tokens 的 $k,v$ 并缓存到 `past_key_values` 中
- decode 阶段：每次生成一个 token ，只计算当前 token 的 $k,v$ 并追加到 `past_key_values` 中

⚠️ KV-cache 是通过空间换时间的方法，通过显存来保存 KV-cache 会带来访存的瓶颈

以  Qwen-72B-Chat 为例，
$l=80, n_h=64, d_h=128$ ，
对于每个 token ，显存占用为
$$
num_{kv}=2\times l \times n_h = 2 \times 80 \times 64 = 2 \times 80 \times 64 = 10k
$$
假设使用半精度 fp16 ，每个数值占用 2 bytes ，一个 token 存储占用
$$
token_mem = 2 \times num_{kv} \times d_h = 2 \times 10k \times 128 = 2.62(MB)
$$
Batch和序列设置： $B = 1 ， S = 2048$ 
$$
mem_{kv} = B \times S \times token_mem = 5.366(GB)
$$

那么就要减少 kv 的显存占用，有多个优化的思路
- 共享 KV：多个 Head 共享使用 1 组 KV ，来压缩KV的存储，比如 GQA，MQA 等
- 窗口 KV：只缓存最近的 $W$ 个 token 的 KV ，超过窗口的丢掉
- 量化压缩：对 KV 进行 INT8 量化压缩
- 计算优化：减少显存交换，比如 flashAttention

[MQA 和 GQA 的图]

DeepSeek V2 提出了 MLA ，通过缓存低秩压缩的 KV ，来减少显存占用
- $d_h$ 是单个 head 的维度， $n_h$ 是 head 的数量
- $d=d_h\times n_h$ 是模型的隐藏维度
- $d_c$ 是 MLA 低秩压缩的维度，其中 $d_c=4\times d_h$

[mla 架构图]

MLA 的核心原理：矩阵吸收计算
$$
q_i^T\times k_j= (W^q c_i^q)^T \times (W^k c_j^{kv}) = (c_i^q)^T \times (W^q)^T \times W^k \times c_j^{kv}
$$

可以提前计算好 $ W^{kq} = (W^q)^T \times W^k$ ，也就是把 $W^{k}$ 吸收到 $W^{q}$ 中 ，在对 q 做变换的时候，同时计算了 $W^{k}$ 的矩阵计算

这样只需要缓存低维的 $c_j^{kv}$ ，而不是缓存完整的 $k_j = W^k c_j^{kv}$ 

原来每个 key 需要缓存 $d_ = 64\times d_h$ 维度（deepseek V2 的 $n_h=64$），现在只需要缓存 $d_c = 4\times d_h$ 维度

现在如果直接加上 response
$$
q_i^T\times k_j = (\mathcal{R}_i W^q c_i^q)^T \times (\mathcal{R}_j W^k c_j^{kv}) 
= (c_i^q)^T \times (W^q)^T \mathcal{R}_i^T \mathcal{R}_j W^k \times c_j^{kv}
$$

中间的 $\mathcal{R}_i^T \mathcal{R}_j$ 不能提前计算，因为 response 是每次 decode 生成一个 token 才会更新，所以和论文中认为 RoPE 和低秩变换不兼容

通过增加一个很小的 $q,k$ 分量，引入 RoPE ，将 $d_h$ 维度的 $q,k$ 分量拆分成两部分
$$
q_i^T\times k_j= [q_i^{mla} ; q_i^{rope}]^T \times [k_j^{mla} ; k_j^{rope}]
= {q_i^{mla}}^T \times k_j^{mla} + {q_i^{rope}}^T \times k_j^{rope}
$$
- $d_h - r$ 维度的分量使用 MLA 低秩，通过矩阵吸收的方式只缓存低秩的 $c_j^{kv}$
- $r$ 维度的分量按照正常的 MQA 方式计算，整个 Head 只缓存一个共享的 $k_j^{rope}$

类似的方式，可以将 $v_j$ 的变换矩阵 $W^v$ 吸收到结果变换矩阵 $W^o$ 中

[性能压缩效果对比]

[MLA 的两种不同的模式]

# Architecture

DSA 的原型设计主要由两个相互协作的核心组件构成： Lightning Indexer 和 Fine-grained Token Selection Mechanism

1. Lightning Indexer：为每一个查询 Token高效地计算之前的上下文 Token 之间的相关性得分
- 计算查询 token 与前序 token 的 **索引分数** $I_{t,s}$ ，用于确定哪些 token 应被关注。
- 复杂度较低，可快速选择 top-k 的关键 token。

设输入序列长度为 $L$ ，隐藏维度为 $d$ ，对于每个查询 token $h_t \in \mathbb{R}^d$，计算与前序 token 的 **索引分数** $I_{t,s}$
$$
I_{t,s} = \sum_{j=1}^{H_I} w^I_{t,j} \cdot \text{ReLU}( q^I_{t,j} \cdot k^I_s )
$$
其中：
- $H_I$ ：索引器的头数，远小于主注意力头数；
- $q^I_{t,j}, w^I_{t,j}$ ：由查询 token $h_t$ 通过线性变换得到；
- $k^I_s$ ：由前序 token $h_s$ 通过线性变换得到；
- $w_{t,j}^I$ 是一个权重标量，由当前 query token 决定的 head 的重要性分数，不同 head 的匹配结果分配不同的权重。
- $\text{ReLU}$ 作为激活函数，用于提高计算上的高吞吐量（throughput）。与 Softmax 等需要全局归一化的函数相比，ReLU 仅需进行一次简单的阈值操作，计算成本低廉。
- 由于只是一个排序模块，可以使用 **FP8 精度** 量化而不损失精度

2. Fine-Grained Token Selection：从所有历史 Token 中，挑选出索引分最高的 Top-k 个 Token。
- 得到 $I_{t,s}$ 后，DSA 选取每个 query token 对应的前 k 个最相关的 token 索引
$$
S_t = \{ s ,|, I_{t,s} \in \text{Top-}k(I_{t,:}) \}
$$
- 在这些 token 上执行标准注意力操作，其中 $c_s$ 为 key-value 组合的 latent vector
$$
u_t = \text{Attn}(h_t, {c_s \mid s \in S_t})
$$
- 将原本 $O(L^2)$ 的注意力复杂度降为 $O(L\cdot k)$

3. DSA 在 MLA 架构下的实现
- 为了能够从 DeepSeek-V3.1-Terminus 进行持续训练，DSA 的具体实现是基于 MLA（Multi-Head Attention）架构的 MQA（Multi-Query Attention）模式下实例化 DSA
- 所有的查询头共享同一组键（Key）和值（Value）向量，这极大地减少了 KV 缓存（KV Cache）的大小

[图 mqa 模式的 mla 架构下的 dsa 实现]

```json
{
  "architectures": [
    "DeepseekV32ForCausalLM"
  ],

  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 2048,

  "kv_lora_rank": 512,
  "num_attention_heads": 128,
  "num_key_value_heads": 128,

  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "use_cache": true
}
```

# Training

## Continued Pre-Training

1. Dense Warm-up Stage
- 初始化 Lightning Indexer，冻结主模型参数，快速让 Indexer 具备初步的、可靠的 Token 筛选能力
- 用 KL 散度损失使索引分数分布对齐主注意力分布
$$
L_I = \sum_t D_{\mathrm{KL}}\big( p_{t,:} ,|, \mathrm{Softmax}(I_{t,:}) \big)
$$

训练细节：
- 学习率： $1e-3$
- 训练步数： 1000 步
- 每步包含的序列数： 16 条
- 序列长度： 128K Tokens
- 总计训练 Token 数： $1000\times 16 \times 128K = 2.048 \text{B}$

2. Sparse Training Stage
- 对于主模型：继续进行标准的语言建模任务，优化其语言建模损失
- 对于 Lightning Indexer ：继续将其输出与主注意力的分布对齐，但与热身阶段不同的是，此时的对齐只考虑被 Top-k 机制选中的 Token 集合 $S_t$
$$
L_I = \sum_t D_{\mathrm{KL}}\big( p_{t,S_t} ,|, \mathrm{Softmax}(I_{t,S_t}) \big)
$$
- 在计算图中将索引器的输入与主模型的计算图分离开（detach），主模型的梯度不会通过 Indexer 传播， Indexer 的优化完全由其自身的损失函数驱动，而主模型的优化则完全由语言建模损失驱动。
有助于稳定训练过程，避免两个不同目标的梯度相互干扰。

训练细节：
- 学习率： $7.3\times 1e-6$
- 训练步数： 15000 步
- 每步包含的序列数： 480 条
- 序列长度： 128K Tokens
- 总计训练 Token 数： $15000\times 480 \times 128K = 943.7 \text{B}$

## Post-Training

1. Specialist Distillation: 
    1. 训练专家模型：从预训练好的 DeepSeek-V3.2 基础 checkpoint 出发，为每个特定领域（如数学、编程竞赛、逻辑推理、代码智能体、搜索智能体等）分别微调出一个专家模型。
    2. 生成高质量数据：专家模型被用来为各自的领域生成大量高质量的训练数据，这些数据包含了长链式思维和直接回答两种模式。
    3. 蒸馏到主模型：将所有专家模型生成的领域数据汇集起来，用于训练最终的模型。

2.  Mixed RL Training: 使用 GRPO（Group Relative Policy Optimization）
    - 算法：继续采用 GRPO 算法
    - 混合强化学习：将推理、智能体和人类对齐这三个方面的训练合并到了一个 RL 阶段中，这种方法能够有效地平衡模型在不同领域的能力，同时避免了多阶段训练中常见的灾难性遗忘问题。


## Evaluations

### Model Capabilities

- 总体上看，**无明显性能损失，部分任务略有提升**。
- V3.2-Exp 在生成推理时倾向于产生更少的推理 Token，在一些需要详尽推理步骤的评测中导致失分，与模型的生成策略有关，而非核心能力的损失。

[图]

### Inference Costs

DSA 将核心注意力的复杂度从 $O(L^2)$ 降至 $O(Lk)$ 。尽管 Indexer 本身仍有 $O(L^2)$ 的复杂度，但其极低的计算常数使得总体计算量显著减少。

[图]