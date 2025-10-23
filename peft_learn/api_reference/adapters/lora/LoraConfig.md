# LoraConfig

## 概述

`LoraConfig` 是用于存储 [`LoraModel`] 配置的配置类。LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，通过在预训练模型的权重矩阵旁添加低秩矩阵来实现微调，从而大幅减少可训练参数的数量。

## 核心参数

### r (`int`): LoRA 注意力维度（秩）
LoRA 矩阵的秩，控制适配器的容量和参数数量。较低的值（如 4, 8）产生较少参数但表达能力有限，较高的值（如 32, 64）提供更强表达能力但参数更多。

**默认值**: `8`

```python
# 常见配置示例
r = 8    # 标准配置，平衡性能和效率
r = 4    # 轻量级配置，参数更少
r = 16   # 高容量配置，适用于复杂任务
```

### lora_alpha (`int`): LoRA 缩放因子
控制 LoRA 适配器的缩放比例，通常设置为 r 的 1-2 倍。最终的缩放因子为 `lora_alpha/r`（或 `lora_alpha/sqrt(r)` 当使用 RSLoRA 时）。

**默认值**: `8`

### lora_dropout (`float`): LoRA 层的丢弃概率
应用于 LoRA 适配器的 dropout 概率，用于防止过拟合。

**默认值**: `0.0`（不使用 dropout）

## 目标模块配置

### target_modules (`Optional[Union[List[str], str]]`): 应用适配器的模块名称
指定要应用 LoRA 适配器的模块名称列表或正则表达式。

**默认值**: `None`（根据模型架构自动选择）

**支持格式**:
- **字符串列表**: `['q_proj', 'v_proj']` - 精确匹配模块名或检查模块名是否以指定字符串结尾
- **正则表达式字符串**: `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'` - 正则匹配
- **通配符**: `'all-linear'` - 匹配所有线性/Conv1D 层（PreTrainedModel 的输出层除外）
- **空列表**: `[]` - 不 targeting 任何模块（配合 `target_parameters` 使用）

```python
# 常见配置示例
target_modules = ['q_proj', 'v_proj']                    # 精确指定
target_modules = '.*attention.*(q|v)$'                   # 正则表达式
target_modules = 'all-linear'                           # 所有线性层
target_modules = []                                     # 配合 target_parameters 使用
```

### exclude_modules (`Optional[Union[List[str], str]]`): 排除的模块名称
指定不应用 LoRA 适配器的模块名称列表或正则表达式。当传入字符串时执行正则匹配，当传入字符串列表时执行精确匹配或检查模块名是否以指定字符串结尾。

**默认值**: `None`

### target_parameters (`Optional[List[str]]`): 目标参数名称列表
指定要替换为 LoRA 的参数名称列表或正则表达式。此参数的行为类似于 `target_modules`，但是传递参数名称而非模块名称。

**默认值**: `None`

```python
# 适用于 MoE 层的示例
target_parameters = ['feed_forward.experts.gate_up_proj', 'feed_forward.experts.down_proj']
```

## 初始化配置

### init_lora_weights (`bool | Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number]", "corda", "loftq", "orthogonal"]`): LoRA 权重初始化方式
指定如何初始化 LoRA 适配器层的权重。

**默认值**: `True`

**可选值**:
- **`True`**: 默认初始化，LoRA B 权重设为 0，使适配器在训练前为无操作
- **`False`**: 随机初始化 LoRA A 和 B，用于调试目的
- **`"gaussian"`**: 按秩缩放的高斯初始化
- **`"eva"`**: 基于数据的 Explained Variance Adaptation 初始化
- **`"olora"`**: OLoRA 初始化
- **`"pissa"`**: Principal Singular values and Singular vectors Adaptation 初始化
- **`"pissa_niter_[number]"`**: Fast-SVD-based PiSSA 初始化，[number] 为迭代次数
- **`"corda"`**: Context-Oriented Decomposition Adaptation 初始化
- **`"loftq"`**: LoftQ 初始化（需配合 `loftq_config`）
- **`"orthogonal"`**: 正交初始化 LoRA A 和 B

## 高级适配器配置

### use_rslora (`bool`): 使用 Rank-Stabilized LoRA
启用 [Rank-Stabilized LoRA](https://huggingface.co/papers/2312.03732)，将适配器缩放因子设置为 `lora_alpha/sqrt(r)`，被证明效果更好。

**默认值**: `False`

### use_dora (`bool`): 启用 DoRA
启用 'Weight-Decomposed Low-Rank Adaptation' (DoRA)。该技术将权重更新分解为幅度和方向两部分，方向由普通 LoRA 处理，幅度由单独的可学习参数处理。

**默认值**: `False`

### use_qalora (`bool`): 启用 QALoRA
启用 Quantization-Aware Low-Rank Adaptation (QALoRA)，将量化感知训练与 LoRA 结合，提高量化模型的性能。

**默认值**: `False`

### qalora_group_size (`int`): QALoRA 组大小
QALoRA 池化的组大小参数，控制维度缩减因子。输入维度被分组为指定大小的组，减少计算成本。

**默认值**: `16`

## 层选择和模式配置

### layers_to_transform (`Union[List[int], int]`): 要转换的层索引
指定要转换的层索引列表。如果传递整数列表，将对指定索引的层应用适配器；如果传递单个整数，将只对该索引的层应用转换。

**默认值**: `None`

### layers_pattern (`Optional[Union[List[str], str]]`): 层模式名称
仅在 `layers_to_transform` 不为 None 时使用的层模式名称。应针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

**默认值**: `None`

### rank_pattern (`dict`): 层特定秩配置
从层名称或正则表达式到不同于默认秩的映射。

**默认值**: `{}`

```python
# 示例：为特定层设置不同的秩
rank_pattern = {
    '^model.decoder.layers.0.encoder_attn.k_proj': 16,
    '.*attention.*': 8
}
```

### alpha_pattern (`dict`): 层特定 alpha 配置
从层名称或正则表达式到不同于默认 alpha 的映射。

**默认值**: `{}`

## 偏置和权重配置

### bias (`Literal["none", "all", "lora_only"]`): LoRA 偏置类型
指定 LoRA 的偏置类型。

**默认值**: `"none"`

**可选值**:
- **`"none"`**: 不使用偏置
- **`"all"`**: 所有偏置在训练期间更新
- **`"lora_only"`**: 仅 LoRA 偏置在训练期间更新

### fan_in_fan_out (`bool`): fan_in_fan_out 模式
如果要替换的层以 (fan_in, fan_out) 格式存储权重，则设置为 True。例如，gpt-2 使用以该格式存储权重的 `Conv1D`。

**默认值**: `False`

### lora_bias (`bool`): LoRA B 参数偏置项
是否为 LoRA B 参数启用偏置项。通常应该禁用。主要用例是当 LoRA 权重从完全微调的参数中提取时，可以考虑这些参数的偏置。

**默认值**: `False`

## 模块保存和层复制

### modules_to_save (`List[str]`): 要保存的可训练模块
除了 LoRA 层之外，要设置为可训练并在最终检查点中保存的模块列表。例如，在序列分类或标记分类任务中，最终的 `classifier/score` 层是随机初始化的，需要可训练和保存。

**默认值**: `None`

### layer_replication (`List[Tuple[int, int]]`): 层复制配置
通过根据指定范围重复原始模型的某些层来构建新的层堆栈，从而将模型扩展（或缩小）到更大的模型。新层都将附加单独的 LoRA 适配器。

**默认值**: `None`

```python
# 示例：层复制配置
layer_replication = [[0, 4], [2, 5]]
# 原始模型有5层：[0, 1, 2, 3, 4]
# 最终模型将有：[0, 1, 2, 3, 2, 3, 4]
```

## 特殊配置

### trainable_token_indices (`Optional[Union[List[int], dict[str, List[int]]]]`): 可训练令牌索引
允许指定哪些令牌索引进行选择性微调，而无需重新训练整个嵌入矩阵。

**默认值**: `None`

**格式**:
- **列表**: `[0, 1, 2]` - 目标模型的输入嵌入层
- **字典**: `{'embed_tokens': [0, 1, ...]}` - 指定嵌入模块名称和令牌索引

### loftq_config (`Optional[LoftQConfig]`): LoftQ 配置
如果指定，将使用 LoftQ 量化骨干权重并初始化 LoRA 层。需要同时设置 `init_lora_weights='loftq'`。

**默认值**: `None`

### eva_config (`Optional[EvaConfig]`): EVA 配置
EVA 的配置。如果指定，将使用 EVA 初始化 LoRA 层。需要同时设置 `init_lora_weights='eva'`。

**默认值**: `None`

### corda_config (`Optional[CordaConfig]`): CorDA 配置
CorDA 的配置。如果指定，将使用 CorDA 构建适配器层。需要同时设置 `init_lora_weights='corda'`。

**默认值**: `None`

## Megatron 集成配置

### megatron_config (`Optional[dict]`): Megatron TransformerConfig 参数
用于创建 LoRA 并行线性层的 Megatron TransformerConfig 参数。

**默认值**: `None`

### megatron_core (`Optional[str]`): Megatron 核心模块
要使用的 Megatron 核心模块，用于创建 LoRA 并行线性层。

**默认值**: `"megatron.core"`

## 运行时配置

### runtime_config (`LoraRuntimeConfig`): 运行时配置
运行时配置（不保存或恢复）。

**默认值**: `LoraRuntimeConfig()`