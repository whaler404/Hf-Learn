# 数据加载和处理方法

## `get_train_dataloader()`

返回训练数据加载器 (`torch.utils.data.DataLoader`)。

### 说明
- 如果 `train_dataset` 没有实现 `__len__`，将不使用采样器
- 否则使用随机采样器（根据需要适配分布式训练）
- 自动处理数据集的预处理和批次化

### 返回值
- `DataLoader`: 训练数据加载器

### 异常
- `ValueError`: 如果 `train_dataset` 为 None

### 示例
```python
# 获取训练数据加载器
train_dataloader = trainer.get_train_dataloader()

# 遍历训练数据
for batch in train_dataloader:
    # 训练逻辑
    pass
```

## `get_eval_dataloader()`

返回评估数据加载器 (`torch.utils.data.DataLoader`)。

### 参数
- **eval_dataset** (`str` 或 `torch.utils.data.Dataset`, 可选):
  - 如果是字符串，将使用 `self.eval_dataset[eval_dataset]` 作为评估数据集
  - 如果是 `Dataset`，将覆盖 `self.eval_dataset` 且必须实现 `__len__`
  - 如果是 `datasets.Dataset`，模型 `forward()` 方法不接受的列将自动移除

### 返回值
- `DataLoader`: 评估数据加载器

### 异常
- `ValueError`: 如果 `eval_dataset` 和 `self.eval_dataset` 都为 None

### 示例
```python
# 使用默认评估数据集
eval_dataloader = trainer.get_eval_dataloader()

# 使用指定的评估数据集
from datasets import Dataset
eval_data = Dataset.from_dict({"text": ["test1", "test2"], "label": [0, 1]})
eval_dataloader = trainer.get_eval_dataloader(eval_data)

# 使用命名评估数据集（如果设置了多个评估数据集）
eval_dataloader = trainer.get_eval_dataloader("validation")
```

## `get_test_dataloader()`

返回测试数据加载器 (`torch.utils.data.DataLoader`)。

### 参数
- **test_dataset** (`torch.utils.data.Dataset`): 要使用的测试数据集。如果是 `datasets.Dataset`，模型 `forward()` 方法不接受的列将自动移除。必须实现 `__len__`。

### 返回值
- `DataLoader`: 测试数据加载器

### 示例
```python
from datasets import Dataset
test_data = Dataset.from_dict({"text": ["test1", "test2"], "label": [0, 1]})
test_dataloader = trainer.get_test_dataloader(test_data)

# 进行测试预测
for batch in test_dataloader:
    outputs = trainer.model(**batch)
    # 处理输出
```

## `_get_dataloader()`

从给定数据集创建 `torch.utils.data.DataLoader` 的通用方法。

### 参数
- **dataset** (`Dataset`: 输入数据集
- **description** (`str`): 数据加载器描述（用于日志）
- **batch_size** (`int`): 批次大小
- **sampler_fn** (`Callable`, 可选): 采样器函数
- **is_training** (`bool`, 可选): 是否为训练模式
- **dataloader_key** (`str`, 可选): 数据加载器键（用于持久化工作器）

### 返回值
- `DataLoader`: 配置好的数据加载器

### 说明
- 自动移除数据集中模型不接受的列
- 配置数据加载器参数（工作器数量、内存固定等）
- 支持持久化工作器以避免重复初始化

## `_get_train_sampler()`

获取训练数据采样器。

### 参数
- **train_dataset** (`Dataset`, 可选): 训练数据集

### 返回值
- `Sampler` 或 `None`: 训练采样器

### 说明
- 如果启用 `group_by_length`，使用 `LengthGroupedSampler`
- 否则使用 `RandomSampler`
- 对于 `IterableDataset` 或没有长度的数据集返回 None

## `_get_eval_sampler()`

获取评估数据采样器。

### 参数
- **eval_dataset** (`Dataset`): 评估数据集

### 返回值
- `Sampler` 或 `None`: 评估采样器

### 说明
- 支持分布式训练的顺序采样
- 根据配置选择合适的采样器策略
- 处理 XLA、SageMaker 等特殊环境

## `_remove_unused_columns()`

移除数据集中未使用的列。

### 参数
- **dataset** (`datasets.Dataset`): 输入数据集
- **description** (`str`, 可选): 数据集描述（用于日志）

### 返回值
- `Dataset`: 处理后的数据集

### 说明
- 根据 `args.remove_unused_columns` 参数决定是否移除列
- 基于模型 forward 方法的签名确定需要保留的列
- 自动移除模型不需要的列以提高内存效率

### 示例
```python
# 自动移除未使用的列
filtered_dataset = trainer._remove_unused_columns(
    dataset=raw_dataset,
    description="Training"
)
```

## `_get_collator_with_removed_columns()`

创建带有移除未使用列功能的数据整理器包装器。

### 参数
- **data_collator** (`Callable`): 原始数据整理器
- **description** (`str`, 可选): 描述（用于日志）

### 返回值
- `Callable`: 包装后的数据整理器

### 说明
- 如果 `remove_unused_columns` 为 False，直接返回原始整理器
- 否则返回一个在整理数据前自动移除未使用列的包装器

## `_set_signature_columns_if_needed()`

如果需要，设置签名列。

### 说明
- 检查模型的 forward 方法签名
- 确定哪些参数是模型接受的
- 添加标签相关的列名（如 "label", "label_ids"）
- 缓存结果以避免重复检查

## `_align_special_tokens()`

对齐分词器和模型配置的特殊标记。

### 说明
- 对齐 EOS、BOS、PAD 等特殊标记的 ID
- 更新模型配置和生成配置
- 在训练前调用以确保标记一致性
- 如果发现不一致会发出警告