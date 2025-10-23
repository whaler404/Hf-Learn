# PrefixEncoder

PrefixEncoder 是用于编码前缀（prefix）的 PyTorch 模块，是 Prefix Tuning 方法的核心组件。它将虚拟令牌转换为用于注意力机制的键值对（key-value pairs），从而在不修改原始模型参数的情况下调整模型的行为。

PrefixEncoder 通过将虚拟前缀令牌编码为与 transformer 架构兼容的 past_key_values 格式，实现了参数高效的前缀调优。该模块支持两种编码策略：直接嵌入和投影变换，为不同的使用场景提供了灵活性。

## 参数

- **config** (`PrefixTuningConfig`): 前缀编码器的配置对象，包含以下关键参数：
  - `num_virtual_tokens` (`int`): 虚拟令牌的数量
  - `token_dim` (`int`): 每个令牌的维度（通常是模型的隐藏维度）
  - `num_layers` (`int`): transformer 层的数量
  - `encoder_hidden_size` (`int`): 编码器隐藏层的大小（仅在 prefix_projection=True 时使用）
  - `prefix_projection` (`bool`): 是否对前缀嵌入进行投影变换
  - `inference_mode` (`bool`): 是否处于推理模式

## 输入形状

- `prefix`: `(batch_size, num_virtual_tokens)` - 虚拟前缀令牌的张量

## 输出形状

- `past_key_values`: `(batch_size, num_virtual_tokens, 2*num_layers*token_dim)` - 编码后的键值对，用于 transformer 的注意力机制

## 使用案例

```python
from peft import PrefixEncoder, PrefixTuningConfig

# 创建前缀调优配置
config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_hidden_size=768,
)

# 创建前缀编码器
prefix_encoder = PrefixEncoder(config)

# 使用编码器
virtual_tokens = torch.arange(20).unsqueeze(0).expand(4, -1)  # batch_size=4, num_virtual_tokens=20
past_key_values = prefix_encoder(virtual_tokens)
print(past_key_values.shape)  # torch.Size([4, 20, 18432]) -> 4, 20, 2*12*768
```

# 方法

## `__init__`

```python
def __init__(self, config):
    super().__init__()
    # 保存是否使用前缀投影的配置
    self.prefix_projection = config.prefix_projection

    # 从配置中获取关键参数
    token_dim = config.token_dim  # 令牌维度，通常是模型的隐藏维度
    num_layers = config.num_layers  # transformer 层数
    encoder_hidden_size = config.encoder_hidden_size  # 编码器隐藏层大小
    num_virtual_tokens = config.num_virtual_tokens  # 虚拟令牌数量

    if self.prefix_projection and not config.inference_mode:
        # 使用 MLP 进行前缀投影的情况
        # 创建嵌入层：将虚拟令牌索引映射到 token_dim 维度的向量
        # shape: [num_virtual_tokens, token_dim]
        self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)

        # 创建两层的 MLP 变换器
        # 输入: token_dim -> 隐藏层: encoder_hidden_size -> 输出: 2*num_layers*token_dim
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(token_dim, encoder_hidden_size),  # 线性层: token_dim -> encoder_hidden_size
            torch.nn.Tanh(),  # 激活函数: 添加非线性变换
            torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),  # 线性层: encoder_hidden_size -> 2*num_layers*token_dim
        )
    else:
        # 直接嵌入的情况，不进行额外的投影变换
        # 创建嵌入层：直接将虚拟令牌索引映射到 2*num_layers*token_dim 维度
        # shape: [num_virtual_tokens, 2*num_layers*token_dim]
        self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
```

## `forward`

```python
def forward(self, prefix: torch.Tensor):
    # 根据配置选择不同的编码策略
    if self.prefix_projection:
        # 前缀投影模式：使用 MLP 进行复杂变换
        # 1. 将虚拟令牌索引转换为嵌入向量
        # 输入: [batch_size, num_virtual_tokens]
        # 输出: [batch_size, num_virtual_tokens, token_dim]
        prefix_tokens = self.embedding(prefix)

        # 2. 通过 MLP 变换为最终的 past_key_values 格式
        # 输入: [batch_size, num_virtual_tokens, token_dim]
        # 输出: [batch_size, num_virtual_tokens, 2*num_layers*token_dim]
        past_key_values = self.transform(prefix_tokens)
    else:
        # 直接嵌入模式：直接使用嵌入层的输出
        # 输入: [batch_size, num_virtual_tokens]
        # 输出: [batch_size, num_virtual_tokens, 2*num_layers*token_dim]
        past_key_values = self.embedding(prefix)

    return past_key_values
```

## 架构说明

PrefixEncoder 的核心思想是将少量可训练的虚拟令牌插入到 transformer 模型的每一层中，作为注意力机制的键值对。这种方法的优势：

1. **参数效率高**：只需训练少量虚拟令牌，而不需要修改整个模型
2. **保持模型完整性**：原始模型参数保持不变，便于模型复用
3. **灵活的编码策略**：
   - **直接嵌入**：简单高效，适合大多数场景
   - **MLP投影**：提供更强的表达能力，适合复杂任务

输出张量的最后一个维度 `2*num_layers*token_dim` 表示：
- `2`：每个注意力头需要 key 和 value 两个矩阵
- `num_layers`：模型的层数
- `token_dim`：每个 key/value 向量的维度

这种设计使得输出可以直接用作 transformer 模型的 past_key_values 参数，实现无缝集成。

# PrefixTuningConfig

PrefixTuningConfig 是 Prefix Tuning 方法的配置类，继承自 PromptLearningConfig，专门用于配置前缀编码器的行为和参数。

## 参数

- **encoder_hidden_size** (`int`): 前缀编码器的隐藏层大小。当 `prefix_projection=True` 时，这个参数定义了 MLP 中间层的维度，影响前缀变换的表达能力（默认值：None）

- **prefix_projection** (`bool`): 是否对前缀令牌进行投影变换。设置为 `True` 时使用两层 MLP 进行变换，设置为 `False` 时直接使用嵌入层（默认值：False）

## 继承的参数

由于继承自 `PromptLearningConfig`，还包含以下通用参数：

- **peft_type** (`str`): PEFT 类型，自动设置为 "PREFIX_TUNING"
- **task_type** (`TaskType`): 任务类型（如 SEQ_2_SEQ_LM、CAUSAL_LM 等）
- **num_virtual_tokens** (`int`): 虚拟令牌的数量
- **token_dim** (`int`): 每个令牌的维度（通常是模型的隐藏维度）
- **num_transformer_submodules** (`int`): transformer 子模块的数量
- **num_attention_heads** (`int`): 注意力头的数量
- **num_layers** (`int`): transformer 层的数量
- **inference_mode** (`bool`): 是否处于推理模式

## 配置示例

```python
from peft import PrefixTuningConfig

# 简单配置：直接嵌入模式
config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
)

# 复杂配置：使用 MLP 投影
config_with_projection = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_hidden_size=1024,  # MLP 隐藏层大小
    prefix_projection=True,    # 启用 MLP 投影
)
```