# 训练循环方法

## `train()`

主要的训练入口点。

### 参数
- **resume_from_checkpoint** (`str` 或 `bool`, 可选):
  - 如果是字符串，表示先前 Trainer 实例保存的检查点的本地路径
  - 如果是布尔值且为 True，加载 *args.output_dir* 中的最后一个检查点
  - 如果存在，训练将从这里加载的模型/优化器/调度器状态恢复
- **trial** (`optuna.Trial` 或 `dict[str, Any]`, 可选): 用于超参数搜索的试验运行或超参数字典
- **ignore_keys_for_eval** (`list[str]`, 可选): 在训练期间收集预测以进行评估时，应忽略的模型输出中的键列表
- **kwargs** (`dict[str, Any]`, 可选): 用于隐藏已弃用参数的额外关键字参数

### 返回值
- `TrainOutput`: 训练输出，包含全局步数、训练损失等指标

### 说明
- 设置训练环境和状态
- 从检查点恢复训练（如果指定）
- 执行完整的训练循环
- 处理评估、保存和日志记录
- 支持各种训练策略（混合精度、分布式训练等）

### 示例
```python
# 基础训练
trainer.train()

# 从检查点恢复训练
trainer.train(resume_from_checkpoint="./checkpoint-1000")

# 使用超参数搜索
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }
trainer.train(trial=optuna_trial)
```

## `training_step()`

对批次输入执行单个训练步骤。

### 参数
- **model** (`nn.Module`): 要训练的模型
- **inputs** (`dict[str, Union[torch.Tensor, Any]]`): 模型的输入和目标。字典将在提供给模型之前解包。大多数模型期望参数 `labels` 下的目标。
- **num_items_in_batch** (`torch.Tensor`, 可选): 批次中的项目数量

### 返回值
- `torch.Tensor`: 此批次上的训练损失张量

### 说明
- 执行前向传播
- 计算损失
- 执行反向传播
- 更新模型参数
- 处理梯度累积
- 支持上下文并行训练

### 示例
```python
# 自定义训练循环
for batch in train_dataloader:
    loss = trainer.training_step(trainer.model, batch)
    print(f"训练损失: {loss.item()}")
```

## `compute_loss()`

计算模型的损失。

### 参数
- **model** (`nn.Module`): 计算损失的模型
- **inputs** (`dict[str, Union[torch.Tensor, Any]]`): 模型的输入数据
- **return_outputs** (`bool`, 可选, 默认为 `False`): 是否返回模型输出和损失
- **num_items_in_batch** (`torch.Tensor`, 可选): 批次中的项目数量

### 返回值
- 如果 `return_outputs=False`: 损失张量
- 如果 `return_outputs=True`: (损失张量, 模型输出) 元组

### 说明
- 默认情况下，所有模型在第一个元素中返回损失
- 支持标签平滑（使用 `label_smoother`）
- 支持自定义损失函数（使用 `compute_loss_func`）
- 处理过去的键值状态（用于生成任务）

### 示例
```python
# 自定义损失计算
def custom_compute_loss(model, inputs):
    outputs = model(**inputs)
    # 自定义损失逻辑
    loss = outputs.loss * 2.0  # 示例：放大损失
    return loss

# 在初始化时设置
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    compute_loss_func=custom_compute_loss
)
```

## `_inner_training_loop()`

内部训练循环实现。

### 参数
- **args** (`TrainingArguments`): 训练参数
- **resume_from_checkpoint** (`str` 或 `bool`, 可选): 检查点恢复路径
- **trial** (`optuna.Trial` 或 `dict`, 可选): 超参数搜索试验
- **ignore_keys_for_eval** (`list[str]`, 可选): 评估时忽略的键

### 返回值
- `TrainOutput`: 训练输出

### 说明
- 这是 `train()` 方法的核心实现
- 处理 epoch 和 step 循环
- 管理检查点保存和加载
- 处理评估和指标计算
- 支持分布式训练和混合精度

### 主要流程
1. 初始化训练状态
2. 设置数据加载器
3. 创建优化器和调度器
4. 遍历 epochs
5. 在每个 epoch 内遍历数据加载器
6. 执行训练步骤
7. 定期评估和保存
8. 返回训练结果

## `get_total_train_batch_size()`

获取总训练批次大小。

### 参数
- **args** (`TrainingArguments`): 训练参数

### 返回值
- `int`: 总训练批次大小

### 说明
- 计算考虑了以下因素的总批次大小：
  - 每设备批次大小
  - 梯度累积步数
  - 设备数量（分布式训练）
  - 数据并行大小

### 示例
```python
# 计算总批次大小
total_batch_size = trainer.get_total_train_batch_size(trainer.args)
print(f"总批次大小: {total_batch_size}")

# 计算训练步数
num_examples = len(trainer.train_dataset)
steps_per_epoch = num_examples // total_batch_size
total_steps = steps_per_epoch * trainer.args.num_train_epochs
```

## `get_tp_size()`

获取张量并行大小。

### 返回值
- `int`: 张量并行大小

### 说明
- 返回当前训练配置的张量并行大小
- 用于分布式训练中的张量并行
- 通常与 DeepSpeed 或其他分布式训练框架一起使用

## `set_initial_training_values()`

设置初始训练值。

### 说明
- 初始化训练状态变量
- 设置初始指标值
- 配置内存跟踪
- 准备训练环境

## `get_batch_samples()`

获取批次样本信息。

### 参数
- **batch_samples** (`list`): 批次样本列表
- **device** (`torch.device`): 设备

### 返回值
- `int` 或 `None`: 批次中的项目数量

### 说明
- 计算批次中的实际项目数量
- 考虑动态批次大小
- 用于准确的损失计算和指标统计