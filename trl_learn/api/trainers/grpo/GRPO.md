# GRPO (Group Relative Policy Optimization) 算法详解

## 概述

GRPO（Group Relative Policy Optimization）是一种强化学习算法，源自 DeepSeekMath 论文《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》。与 PPO 类似，GRPO 也是一种异策略算法，但其核心创新在于使用组内相对优势函数，无需训练独立的 Critic 模型。

## 算法核心思想

### 与 PPO 的主要区别

1. **优势函数设计不同**：
   - PPO：需要训练 Critic 模型估计状态价值函数
   - GRPO：使用组内奖励标准化作为优势函数，无需 Critic

2. **计算效率**：
   - GRPO 通过组内相对奖励计算基线，大幅降低计算成本
   - 避免了训练和推理 Critic 模型的开销

## 数学原理

### 目标函数

GRPO 的优化目标：

$$
\mathcal{J}(\theta)=\mathbb{E}_{q\sim P(Q),\{o_i\}\sim \pi_{\theta_\text{old}}(O\vert q)}\\
\frac{1}{G}\sum_{i=1}^G(
\min(\frac{\pi_\theta(o_i\vert q)}{\pi_{\theta_\text{old}}(o_i\vert q)}A_i,
\text{clip}(\frac{\pi_\theta(o_i\vert q)}{\pi_{\theta_\text{old}}(o_i\vert q)},1-\varepsilon,1+\varepsilon)
)-
\beta\mathbb{D}_\text{KL}(\pi_\theta\parallel\pi_{\theta_\text{old}})
)
$$

### 优势函数计算

GRPO 的关键创新在于组内相对优势：

$$
A_i=\frac{r_i-\text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}
$$

其中：
- $r_i$ 是第 $i$ 个生成的奖励
- $G$ 是组大小（每个问题生成的答案数量）
- 使用组内均值和标准差进行标准化

## 核心组件

### 1. 生成策略

每个问题生成 $G$ 个答案：

```python
# 伪代码
for question in batch:
    completions = []
    for i in range(G):  # G = num_generations
        completion = model.generate(question, generation_config)
        completions.append(completion)
```

### 2. 奖励计算

支持多种奖励函数类型：

```python
# 奖励函数可以是：
# 1. 预训练模型
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model_id")

# 2. 自定义函数
def custom_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        # 基于规则或启发式的奖励计算
        reward = calculate_reward(completion)
        rewards.append(reward)
    return rewards

# 3. 多奖励函数组合
reward_funcs = [reward_model1, custom_reward, reward_model2]
```

### 3. 优势标准化

```python
def compute_advantages(rewards, num_generations):
    # rewards shape: (batch_size * num_generations,) - 扁平化的奖励值
    # 例如: batch_size=4, num_generations=3 -> rewards.shape=(12,)
    
    # 重塑为 (batch_size, num_generations)
    grouped_rewards = rewards.view(-1, num_generations)  # shape: (4, 3)
    
    # 计算组内统计量
    mean_rewards = grouped_rewards.mean(dim=1)  # shape: (4,) - 每组的平均奖励
    std_rewards = grouped_rewards.std(dim=1)    # shape: (4,) - 每组的奖励标准差
    
    # 标准化优势 - 使用 repeat_interleave 复制均值和标准差
    advantages = (rewards - mean_rewards.repeat_interleave(num_generations)) / (std_rewards + 1e-4)
    # advantages shape: (12,) - 标准化后的优势值
    
    return advantages
```

## API 详解

### GRPOTrainer 类

```python
class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],           # 训练模型
        reward_funcs: Union[RewardFunc, list[RewardFunc]],  # 奖励函数
        args: Optional[GRPOConfig] = None,            # 训练配置
        train_dataset: Optional[Dataset] = None,      # 训练数据集
        eval_dataset: Optional[Dataset] = None,       # 评估数据集
        processing_class: Optional[PreTrainedTokenizerBase] = None,  # 分词器
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,  # 奖励模型分词器
        callbacks: Optional[list[TrainerCallback]] = None,  # 回调函数
        optimizers: tuple = (None, None),              # 优化器和学习率调度器
        peft_config: Optional["PeftConfig"] = None,    # PEFT 配置
    )
```

### 关键参数说明

#### 模型相关
- `model`: 可以是模型路径字符串或 PreTrainedModel 对象
- `processing_class`: 用于处理文本的分词器
- `peft_config`: PEFT（参数高效微调）配置

#### 奖励系统
- `reward_funcs`: 奖励函数，支持单个或多个
- `reward_processing_classes`: 奖励模型的分词器
- `reward_weights`: 多奖励函数的权重（可选）

#### 训练策略
- `num_generations`: 每个问题生成的答案数量（G）
- `num_iterations`: 迭代次数（μ）
- `steps_per_generation`: 生成步骤间隔
- `temperature`: 生成温度参数
- `top_p`, `top_k`: 生成采样参数

#### 损失函数
- `beta`: KL 散度权重
- `epsilon_low`, `epsilon_high`: 裁剪范围
- `loss_type`: 损失类型（"grpo", "bnpo", "dr_grpo"）

### GRPOConfig 配置

```python
GRPOConfig(
    output_dir="./results",
    # 生成参数
    max_prompt_length=512,
    max_completion_length=128,
    num_generations=4,  # G
    temperature=0.9,
    top_p=0.9,
    
    # 训练参数
    num_iterations=2,  # μ
    steps_per_generation=4,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    
    # 损失参数
    beta=0.1,
    epsilon=0.2,
    epsilon_high=0.2,
    loss_type="grpo",
    
    # vLLM 加速
    use_vllm=False,
    vllm_mode="colocate",
    
    # 其他
    log_completions=True,
    disable_dropout=True,
)
```

## 训练流程与 Tensor Shape 变化详解

### 1. 数据准备与输入 Shape

```python
# 数据集格式
dataset = load_dataset("trl-lib/tldr", split="train")

# 数据样本必须包含 "prompt" 列
# 支持两种格式：
# 1. 标准格式：{"prompt": "问题文本"}
# 2. 对话格式：{"prompt": [{"role": "user", "content": "问题"}]}

# 假设配置参数：
# per_device_train_batch_size = 2
# num_generations = 3
# max_prompt_length = 512
# max_completion_length = 128
```

### 2. 批次处理与 Shape 变化

#### 2.1 输入批次 Shape

```python
# 1. 原始输入批次 (RepeatSampler 处理后)
inputs = [
    {"prompt": "问题1"}, {"prompt": "问题1"}, {"prompt": "问题1"},  # 问题1生成3个答案
    {"prompt": "问题2"}, {"prompt": "问题2"}, {"prompt": "问题2"}   # 问题2生成3个答案
]
# 批次大小: 6 = per_device_train_batch_size * num_generations

# 2. Tokenization 后的 shape
prompt_ids = tokenizer(prompts, padding=True, truncation=True, max_length=max_prompt_length)
# prompt_ids.shape: (6, 512) - 6个样本，每个512个token
# prompt_mask.shape: (6, 512) - 对应的attention mask
```

#### 2.2 生成阶段的 Shape 变化

```python
# 3. 生成完成阶段
generation_outputs = model.generate(
    input_ids=prompt_ids,
    attention_mask=prompt_mask,
    max_new_tokens=max_completion_length,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.9
)

# 生成输出的 shape 变化：
# input_ids.shape: (6, 512) - 输入提示
# completion_ids.shape: (6, variable_length) - 生成的完成文本（长度可变）
# 实际长度在 1 到 max_completion_length 之间
```

#### 2.3 Padding 和 Masking

```python
# 4. 统一长度和创建 mask
completion_ids = pad(completion_ids_list, pad_value=tokenizer.pad_token_id)
# completion_ids.shape: (6, 128) - 统一到最大长度

# 创建完成 mask（用于计算损失时忽略 padding）
completion_mask = (completion_ids != tokenizer.pad_token_id).long()
# completion_mask.shape: (6, 128)

# 组合完整的输入序列
input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
# input_ids.shape: (6, 640) - 512 + 128
# attention_mask.shape: (6, 640)
```

### 3. 奖励计算与优势标准化

#### 3.1 奖励计算 Shape

```python
# 5. 奖励函数计算
rewards_per_func = []  # 多个奖励函数的结果
for reward_func in reward_funcs:
    rewards = reward_func(completions_text)  # 文本形式的奖励计算
    rewards_tensor = torch.tensor(rewards, device=device)
    # rewards_tensor.shape: (6,) - 每个完成文本的奖励值
    rewards_per_func.append(rewards_tensor)

# 组合多个奖励函数的结果
rewards_per_func = torch.stack(rewards_per_func, dim=1)  # shape: (6, num_reward_funcs)
rewards = (rewards_per_func * reward_weights).sum(dim=1)  # shape: (6,)
```

#### 3.2 优势标准化详细过程

```python
# 6. 优势标准化 - 关键步骤
# 原始奖励: shape (6,) = [r1, r2, r3, r4, r5, r6]

# 重塑为组形式
grouped_rewards = rewards.view(-1, num_generations)  # shape: (2, 3)
# 例如: [[r1, r2, r3],   # 问题1的3个完成
#        [r4, r5, r6]]   # 问题2的3个完成

# 计算组内统计量
mean_grouped_rewards = grouped_rewards.mean(dim=1)  # shape: (2,)
# 例如: [mean1, mean2] 其中 mean1 = (r1+r2+r3)/3

std_grouped_rewards = grouped_rewards.std(dim=1)    # shape: (2,)
# 例如: [std1, std2] 其中 std1 = std([r1, r2, r3])

# 复制统计量以匹配原始形状
mean_expanded = mean_grouped_rewards.repeat_interleave(num_generations)  # shape: (6,)
# 例如: [mean1, mean1, mean1, mean2, mean2, mean2]

std_expanded = std_grouped_rewards.repeat_interleave(num_generations)    # shape: (6,)
# 例如: [std1, std1, std1, std2, std2, std2]

# 计算优势
advantages = (rewards - mean_expanded) / (std_expanded + 1e-4)  # shape: (6,)
# 例如: [(r1-mean1)/std1, (r2-mean1)/std1, (r3-mean1)/std1,
#        (r4-mean2)/std2, (r5-mean2)/std2, (r6-mean2)/std2]
```

### 4. 损失计算的 Shape 变化

#### 4.1 Log Probability 计算

```python
# 7. 计算 log probabilities
logits = model(input_ids, attention_mask=attention_mask).logits
# logits.shape: (6, 640, vocab_size) - 完整序列的logits

# 只需要完成部分的 logits（排除最后一个位置）
logits = logits[:, :-1, :]  # shape: (6, 639, vocab_size)

# 计算完成部分的 log probabilities
completion_logits = logits[:, -logits_to_keep:, :]  # shape: (6, 128, vocab_size)
completion_input_ids = input_ids[:, -logits_to_keep:]  # shape: (6, 128)

per_token_logps = selective_log_softmax(completion_logits, completion_input_ids)
# per_token_logps.shape: (6, 128) - 每个token的log probability
```

#### 4.2 损失计算的详细 Shape

```python
# 8. PPO 损失计算
old_per_token_logps = per_token_logps.detach()  # shape: (6, 128)
per_token_logps = per_token_logps               # shape: (6, 128)

# 计算重要性权重
ratio = torch.exp(per_token_logps - old_per_token_logps)  # shape: (6, 128)

# PPO 裁剪
clip_low = 1 - epsilon_low   # 例如: 0.8
clip_high = 1 + epsilon_high # 例如: 1.2
clipped_ratio = torch.clamp(ratio, clip_low, clip_high)  # shape: (6, 128)

# 计算策略损失
advantages_expanded = advantages.unsqueeze(1)  # shape: (6, 1) -> (6, 1)
advantages_expanded = advantages_expanded.expand(-1, 128)  # shape: (6, 128)

per_token_loss1 = ratio * advantages_expanded          # shape: (6, 128)
per_token_loss2 = clipped_ratio * advantages_expanded # shape: (6, 128)

per_token_loss = -torch.min(per_token_loss1, per_token_loss2)  # shape: (6, 128)

# 应用 mask 和平均
masked_loss = per_token_loss * completion_mask  # shape: (6, 128)
loss = masked_loss.sum() / completion_mask.sum()  # scalar
```

### 5. KL 散度惩罚

```python
# 9. KL 散度计算（如果 beta > 0）
if beta > 0:
    ref_per_token_logps = ref_model(input_ids, attention_mask=attention_mask)  # shape: (6, 128)
    
    # 计算 KL 散度
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps) - 
        (ref_per_token_logps - per_token_logps) - 1
    )  # shape: (6, 128)
    
    kl_loss = beta * (per_token_kl * completion_mask).sum() / completion_mask.sum()  # scalar
    
    total_loss = loss + kl_loss
```

### 6. Shape 总结

| 阶段 | Tensor | Shape | 说明 |
|------|--------|-------|------|
| **输入** | `prompt_ids` | `(batch_size * G, P)` | 批次大小×生成数，提示长度 |
| **生成** | `completion_ids` | `(batch_size * G, C)` | 完成长度（可变） |
| **组合** | `input_ids` | `(batch_size * G, P+C)` | 完整输入序列 |
| **Logits** | `logits` | `(batch_size * G, P+C-1, V)` | 模型输出logits |
| **LogPs** | `per_token_logps` | `(batch_size * G, C)` | 完成部分的log概率 |
| **奖励** | `rewards` | `(batch_size * G,)` | 每个完成的奖励值 |
| **优势** | `advantages` | `(batch_size * G,)` | 标准化后的优势 |
| **损失** | `per_token_loss` | `(batch_size * G, C)` | 每个token的损失 |

其中：
- `batch_size`: 每个设备的批次大小
- `G`: `num_generations` - 每个问题生成的答案数量
- `P`: `max_prompt_length` - 最大提示长度
- `C`: `max_completion_length` - 最大完成长度
- `V`: 词汇表大小

## GRPO 采样策略详解

### 1. RepeatSampler 设计原理

GRPO 的核心采样策略是通过 `RepeatSampler` 实现的，其主要目的是：

1. **确保每个 prompt 生成多个答案**：用于组内优势计算
2. **复用生成结果**：在多个训练步骤中重复使用相同的生成结果
3. **多进程一致性**：确保不同 GPU 上的相同 prompt 能够正确分组

### 2. RepeatSampler 工作机制

```python
class RepeatSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,      # = num_generations (G)
        batch_size: int = 1,        # 每批唯一 prompt 数量
        repeat_count: int = 1,       # = num_iterations * steps_per_generation
        shuffle: bool = True,
        seed: Optional[int] = None,
    )
```

#### 采样过程示例

假设参数：
- `data_source`: 7 个样本 [0, 1, 2, 3, 4, 5, 6]
- `mini_repeat_count`: 3 (每个 prompt 生成 3 个答案)
- `batch_size`: 3 (每批 3 个唯一 prompt)
- `repeat_count`: 2 (重复 2 次)

```python
# 步骤 1: 随机打乱数据
indexes = torch.randperm(7).tolist()  # 例如: [2, 4, 3, 1, 0, 6, 5]

# 步骤 2: 分组成批次
chunks = [[2, 4, 3], [1, 0, 6]]  # 最后一个不完整的批次 [5] 被丢弃

# 步骤 3: 对每个批次进行重复采样
for chunk in chunks:
    for _ in range(repeat_count):          # 重复 2 次
        for index in chunk:                 # 遍历批次中的每个索引
            for _ in range(mini_repeat_count):  # 每个 prompt 重复 3 次
                yield index

# 最终采样结果：
# [2, 2, 2, 4, 4, 4, 3, 3, 3,   # 第一轮，第一批次
#  2, 2, 2, 4, 4, 4, 3, 3, 3,   # 第二轮，第一批次
#  1, 1, 1, 0, 0, 0, 6, 6, 6,   # 第一轮，第二批次
#  1, 1, 1, 0, 0, 0, 6, 6, 6]   # 第二轮，第二批次
```

### 3. 多 GPU 环境下的采样策略

#### 3.1 数据分布策略

```python
# 假设有 2 个 GPU，每个 GPU 的 batch_size=2，num_generations=3

# GPU 0 收到的索引：
# [0, 0, 0, 1, 1, 1,   # 第一个 generation batch
#  0, 0, 0, 1, 1, 1,   # 第二个 generation batch
#  2, 2, 2, 3, 3, 3,   # 第三个 generation batch
#  2, 2, 2, 3, 3, 3]   # 第四个 generation batch

# GPU 1 收到的索引：
# [4, 4, 4, 5, 5, 5,   # 第一个 generation batch
#  4, 4, 4, 5, 5, 5,   # 第二个 generation batch
#  6, 6, 6, 7, 7, 7,   # 第三个 generation batch
#  6, 6, 6, 7, 7, 7]   # 第四个 generation batch
```

#### 3.2 组内奖励标准化的重要性

```python
# 在多 GPU 环境下，每个 prompt 的所有生成必须在同一 GPU 上
# 这样才能正确计算组内奖励的均值和标准差

# 正确的分组：
# GPU 0: prompt0 的 3 个生成 -> 可以计算优势
# GPU 1: prompt0 的 3 个生成 -> 可以计算优势

# 错误的分组（如果随机分布）：
# GPU 0: prompt0 的 2 个生成 + prompt1 的 1 个生成 -> 无法正确计算优势
# GPU 1: prompt0 的 1 个生成 + prompt1 的 2 个生成 -> 无法正确计算优势
```

### 4. Generation Batch vs Training Batch

GRPO 使用两种不同的批次概念：

#### 4.1 Generation Batch

```python
# Generation Batch 是实际进行生成的批次大小
generation_batch_size = per_device_train_batch_size * num_generations
# 例如：2 * 3 = 6

# 每个 generation batch 包含：
# - 2 个唯一 prompt
# - 每个 prompt 生成 3 个答案
# - 总共 6 个生成结果
```

#### 4.2 Training Batch

```python
# Training Batch 是用于计算损失的批次大小
training_batch_size = per_device_train_batch_size
# 例如：2

# 每个 training batch 包含：
# - 从之前生成的 6 个结果中选择 2 个
# - 用于计算策略梯度
```

### 5. 生成结果的复用机制

#### 5.1 Buffer 机制

```python
# GRPO 维护一个生成结果缓冲区
self._buffered_inputs = None

# 当 steps_per_generation > 1 时，同一个 generation batch
# 的结果会被用于多个 training steps
```

#### 5.2 复用流程

```python
# 假设 steps_per_generation = 4, num_iterations = 2

# Step 0: 生成新的 completions
# - 生成 6 个 completions
# - 存储到 buffer
# - 使用前 2 个计算损失

# Step 1: 复用之前的 completions
# - 从 buffer 读取 6 个 completions
# - 使用接下来的 2 个计算损失

# Step 2: 继续复用
# - 从 buffer 读取 6 个 completions
# - 使用接下来的 2 个计算损失

# Step 3: 继续复用
# - 从 buffer 读取 6 个 completions
# - 使用最后的 2 个计算损失

# Step 4: 开始新的迭代
# - 生成新的 6 个 completions
# - 重复上述过程
```

### 6. 采样策略的数学解释

#### 6.1 采样概率

```python
# 在 RepeatSampler 中，每个样本的采样概率是均匀的
# 但是由于重复机制，实际的训练分布会有所不同

# 理论采样概率：
P_sample(i) = 1 / len(data_source)  # 对于每个唯一样本

# 实际训练频率：
P_train(i) = mini_repeat_count * repeat_count / total_samples
           = G * μ * S / (N * G * μ * S)
           = 1 / N
```

#### 6.2 梯度估计的无偏性

```python
# GRPO 的采样策略保持了梯度估计的无偏性
# 因为每个样本都有相同的概率被选中

# 重要性权重：
importance_weight = P_π(s) / P_μ(s)
                    = (π(s) / μ(s)) / (π(s) / μ(s))
                    = 1

# 因此不需要额外的重要性采样校正
```

### 7. 采样策略的优化效果

#### 7.1 计算效率

```python
# 传统 PPO：
# - 每个 training step 都需要生成新的 completions
# - 生成开销 = steps_per_epoch * generation_cost

# GRPO：
# - 每个 generation batch 可以复用 steps_per_generation 次
# - 生成开销 = (steps_per_epoch / steps_per_generation) * generation_cost
# - 节省 = steps_per_generation 倍的生成开销
```

#### 7.2 内存效率

```python
# 内存使用优化：
# - 生成结果存储在 GPU 内存中
# - 避免了重复生成的内存分配开销
# - 通过 gradient accumulation 进一步减少内存使用
```

### 8. 实际配置示例

```python
# 典型的 GRPO 配置
training_args = GRPOConfig(
    per_device_train_batch_size=8,      # B
    num_generations=4,                  # G
    steps_per_generation=4,             # S
    num_iterations=2,                    # μ
    gradient_accumulation_steps=2,      # G_acc
    
    # 实际的 generation batch size:
    # generation_batch_size = B * G = 8 * 4 = 32
    
    # 实际的训练频率：
    # 每 S * G_acc = 4 * 2 = 8 个 training steps 生成一次
)

# 对应的 RepeatSampler 配置
sampler = RepeatSampler(
    data_source=dataset,
    mini_repeat_count=4,                    # G
    batch_size=32 // 4,                     # generation_batch_size // G
    repeat_count=2 * 4,                     # μ * S
    shuffle=True,
    seed=42,
)
```

### 9. 采样策略的限制和注意事项

#### 9.1 内存限制

```python
# 大的 steps_per_generation 会增加内存使用
# 因为需要存储更多的生成结果

# 内存使用估算：
memory_usage = generation_batch_size * max_sequence_length * dtype_size
```

#### 9.2 收敛性考虑

```python
# 过大的 steps_per_generation 可能导致：
# - 策略更新滞后
# - 生成结果与当前策略不匹配
# - 训练不稳定

# 建议的平衡：
steps_per_generation <= gradient_accumulation_steps
```

### 2. 采样策略

GRPO 使用特殊的采样器 `RepeatSampler`：

```python
def get_train_sampler(self):
    return RepeatSampler(
        data_source=dataset,
        mini_repeat_count=self.num_generations,    # 每个样本重复次数
        batch_size=self.args.generation_batch_size // self.num_generations,
        repeat_count=self.num_iterations * self.args.steps_per_generation,
        shuffle=True,
        seed=self.args.seed,
    )
```

### 3. 训练步骤

```python
def training_step(self, model, inputs):
    # 1. 生成多个完成
    generation_batch = self._generate_and_score_completions(inputs)
    
    # 2. 计算奖励和优势
    rewards = self._calculate_rewards(generation_batch)
    advantages = self._compute_advantages(rewards)
    
    # 3. 计算策略损失
    loss = self.compute_loss(model, generation_batch)
    
    # 4. 反向传播
    loss.backward()
    
    return loss.detach()
```

### 4. 损失计算

```python
def compute_loss(self, model, inputs):
    # 获取新旧策略的 log probabilities
    old_per_token_logps = inputs["old_per_token_logps"]
    per_token_logps = self._get_per_token_logps(model, inputs)
    
    # 计算重要性权重
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    
    # PPO 裁剪
    clip_low = 1 - self.epsilon_low
    clip_high = 1 + self.epsilon_high
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
    
    # 策略损失
    advantages = inputs["advantages"]
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # KL 散度惩罚
    if self.beta != 0:
        ref_logps = inputs["ref_per_token_logps"]
        kl_loss = self.beta * self._compute_kl_divergence(per_token_logps, ref_logps)
        policy_loss += kl_loss
    
    return policy_loss
```

## 实际应用示例

### 基础使用

```python
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和数据
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
dataset = load_dataset("trl-lib/tldr", split="train")

# 定义奖励函数
def reward_func(completions, **kwargs):
    # 示例：奖励长度接近 20 的完成
    return [-abs(20 - len(completion)) for completion in completions]

# 训练配置
training_args = GRPOConfig(
    output_dir="./grpo_results",
    num_generations=4,
    learning_rate=5e-5,
    num_train_epochs=3,
)

# 创建训练器
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()
```

### 多奖励函数

```python
# 多个奖励函数
def reward_length(completions, **kwargs):
    return [len(completion) for completion in completions]

def reward_diversity(completions, **kwargs):
    # 计算词汇多样性
    return [len(set(completion.split())) for completion in completions]

# 使用多个奖励函数
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_length, reward_diversity],
    reward_weights=[0.7, 0.3],  # 权重分配
    args=training_args,
    train_dataset=dataset,
)
```

### 使用预训练奖励模型

```python
from transformers import AutoModelForSequenceClassification

# 加载预训练奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "OpenAssistant/reward-model-deberta-v3-large-v2", 
    num_labels=1
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_model,
    args=training_args,
    train_dataset=dataset,
)
```

## 性能优化

### 1. vLLM 加速

```python
training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",  # 或 "server"
    vllm_gpu_memory_utilization=0.9,
    vllm_tensor_parallel_size=1,
)
```

### 2. Liger Kernel 优化

```python
training_args = GRPOConfig(
    use_liger_loss=True,
    loss_type="grpo",
)
```

### 3. 内存优化

```python
training_args = GRPOConfig(
    gradient_checkpointing=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)
```

## 监控和日志

### 1. 完成样本日志

```python
training_args = GRPOConfig(
    log_completions=True,
    num_completions_to_print=5,
    wandb_log_unique_prompts=True,
)
```

### 2. 关键指标

- `rewards/{func_name}/mean`: 各奖励函数的平均值
- `rewards/{func_name}/std`: 各奖励函数的标准差
- `kl`: KL 散度
- `clip_ratio`: 裁剪比例
- `completions/mean_length`: 完成长度统计

## 总结

GRPO 的优势：

1. **无需 Critic 模型**：通过组内相对奖励计算优势函数
2. **计算效率高**：避免了 Critic 模型的训练和推理开销
3. **实现简单**：基于 PPO 框架，易于理解和实现
4. **灵活性强**：支持多种奖励函数和优化配置

GRPO 特别适用于：
- 数学推理任务
- 代码生成
- 需要多个候选答案比较的场景
- 计算资源受限的环境