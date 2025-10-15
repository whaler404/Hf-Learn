# 优化器和调度器方法

## `create_optimizer()`

设置优化器。

### 说明
- 提供一个运行良好的合理默认配置
- 如果想使用其他优化器，可以在初始化时通过 `optimizers` 参数传递元组，或者子类化并重写此方法
- 自动处理权重衰减参数分组
- 支持多种优化器类型（AdamW、Adafactor、8bit优化器等）

### 参数分组
方法会自动将参数分为两组：
1. **权重衰减组**: 包含需要权重衰减的参数，权重衰减系数为 `args.weight_decay`
2. **无权重衰减组**: 包含不需要权重衰减的参数（如偏置、层归一化参数），权重衰减系数为 0.0

### 返回值
- `torch.optim.Optimizer`: 创建的优化器实例

### 示例
```python
# 使用默认优化器
trainer.create_optimizer()

# 在初始化时传递自定义优化器
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
trainer = Trainer(model=model, args=args, optimizers=(optimizer, None))
```

## `create_scheduler()`

设置学习率调度器。

### 参数
- **num_training_steps** (`int`): 要执行的训练步数
- **optimizer** (`torch.optim.Optimizer`, 可选): 优化器。如果未提供，使用训练器的优化器

### 返回值
- `torch.optim.lr_scheduler.LambdaLR`: 创建的学习率调度器

### 说明
- 优化器必须在此方法调用之前设置，或者作为参数传递
- 使用 `args.lr_scheduler_type` 指定调度器类型
- 自动计算热身步数
- 支持调度器特定的额外参数

### 示例
```python
# 创建调度器
trainer.create_scheduler(num_training_steps=1000)

# 自定义调度器参数
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer=trainer.optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)
trainer.lr_scheduler = scheduler
```

## `create_optimizer_and_scheduler()`

设置优化器和学习率调度器。

### 参数
- **num_training_steps** (`int`: 要执行的训练步数

### 说明
- 便捷方法，同时创建优化器和调度器
- 按顺序调用 `create_optimizer()` 和 `create_scheduler()`
- 确保正确的初始化顺序

### 示例
```python
# 一同创建优化器和调度器
trainer.create_optimizer_and_scheduler(num_training_steps=1000)
```

## `get_optimizer_cls_and_kwargs()` (静态方法)

根据训练参数返回优化器类和优化器参数。

### 参数
- **args** (`TrainingArguments`): 训练会话的训练参数
- **model** (`PreTrainedModel`, 可选): 模型实例

### 返回值
- `tuple[Any, Any]`: 优化器类和优化器参数的元组

### 说明
- 解析 `args.optim` 和 `args.optim_args`
- 支持多种优化器类型：
  - AdamW（torch、fused、XLA、NPU fused）
  - Adafactor
  - 8-bit优化器（bitsandbytes）
  - GaLore优化器
  - Apollo优化器
  - LOMO优化器
  - GrokAdamW
  - 等等

### 示例
```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./results",
    optim="adamw_torch",
    learning_rate=5e-5,
    weight_decay=0.01,
    optim_args="betas=(0.9,0.999),eps=1e-8"
)

optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
```

## `get_decay_parameter_names()`

获取需要应用权重衰减的参数名称。

### 参数
- **model**: 要检查的模型

### 返回值
- `list[str]`: 需要权重衰减的参数名称列表

### 说明
- 此函数通过两种方式过滤参数：
  1. 按层类型（`ALL_LAYERNORM_LAYERS` 中指定的层实例）
  2. 按参数名称模式（包含 'bias' 或各种 'norm' 变体）
- 自动排除偏置参数和归一化层参数

### 过滤规则
排除包含以下模式的参数：
- "bias"
- "layernorm"
- "rmsnorm"
- "(?:^|\\.)norm(?:$|\\.)"
- "_norm(?:$|\\.)"

### 示例
```python
# 获取需要权重衰减的参数
decay_params = trainer.get_decay_parameter_names(model)

# 手动创建参数组
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_params],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_params],
        "weight_decay": 0.0,
    },
]
```

## `get_num_trainable_parameters()`

获取可训练参数的数量。

### 返回值
- `int`: 可训练参数的总数

### 示例
```python
# 打印可训练参数数量
print(f"可训练参数数量: {trainer.get_num_trainable_parameters():,}")
```

## `get_learning_rates()`

返回优化器中每个参数的学习率。

### 返回值
- `list[float]`: 每个参数组的学习率列表

### 异常
- `ValueError`: 如果优化器为 None

### 示例
```python
# 获取当前学习率
learning_rates = trainer.get_learning_rates()
print(f"当前学习率: {learning_rates}")

# 监控学习率变化
for step in range(num_steps):
    trainer.training_step()
    if step % 100 == 0:
        print(f"Step {step}: LR = {trainer.get_learning_rates()[0]:.2e}")
```

## `get_optimizer_group()`

返回参数对应的优化器组。

### 参数
- **param** (`str` 或 `torch.nn.parameter.Parameter`, 可选): 要获取优化器组的参数

### 返回值
- `dict` 或 `list[dict]`: 如果提供了参数，返回对应的优化器组；否则返回所有参数组的参数

### 异常
- `ValueError`: 如果优化器为 None

### 示例
```python
# 获取特定参数的优化器组
import torch
param = model.embeddings.word_embeddings.weight
group = trainer.get_optimizer_group(param)
print(f"参数组的权重衰减: {group['weight_decay']}")

# 获取所有优化器组
all_groups = trainer.get_optimizer_group()
for i, group in enumerate(all_groups):
    print(f"组 {i}: {len(group['params'])} 个参数")
```

## `num_examples()`

获取数据加载器中样本数量的辅助方法。

### 参数
- **dataloader** (`DataLoader`: 输入数据加载器

### 返回值
- `int`: 样本数量

### 说明
- 通过访问数据加载器的数据集来获取样本数量
- 如果数据加载器.dataset 不存在或没有长度，尽力估算
- 对于 `IterableDatasetShard` 有特殊处理

### 示例
```python
# 获取训练样本数量
train_dataloader = trainer.get_train_dataloader()
num_train_examples = trainer.num_examples(train_dataloader)
print(f"训练样本数: {num_train_examples}")

# 计算训练轮数
num_epochs = trainer.args.num_train_epochs
steps_per_epoch = num_train_examples // trainer.args.per_device_train_batch_size
total_steps = int(steps_per_epoch * num_epochs)
```