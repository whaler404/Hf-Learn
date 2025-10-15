# 实用工具方法

## `get_num_trainable_parameters()`

获取可训练参数的数量。

### 返回值
- `int`: 可训练参数的总数

### 示例
```python
# 打印可训练参数数量
num_params = trainer.get_num_trainable_parameters()
print(f"可训练参数数量: {num_params:,}")

# 计算模型大小
param_size = num_params * 4 / (1024**3)  # 假设 float32
print(f"模型大小约: {param_size:.2f} GB")
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
        current_lr = trainer.get_learning_rates()[0]
        print(f"Step {step}: LR = {current_lr:.2e}")

# 获取不同参数组的学习率
for i, lr in enumerate(trainer.get_learning_rates()):
    print(f"参数组 {i}: LR = {lr:.2e}")
```

## `num_examples()`

获取数据加载器中样本数量的辅助方法。

### 参数
- **dataloader** (`DataLoader`): 输入数据加载器

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

# 计算训练轮数和步数
num_epochs = trainer.args.num_train_epochs
steps_per_epoch = num_train_examples // trainer.args.per_device_train_batch_size
total_steps = int(steps_per_epoch * num_epochs)
print(f"预计总步数: {total_steps}")

# 处理不同的数据加载器类型
eval_dataloader = trainer.get_eval_dataloader()
num_eval_examples = trainer.num_examples(eval_dataloader)
print(f"评估样本数: {num_eval_examples}")
```

## `num_tokens()`

计算数据加载器中的 token 数量。

### 参数
- **train_dl** (`DataLoader`): 训练数据加载器
- **max_steps** (`int`, 可选): 最大步数，如果提供则乘以此步数

### 返回值
- `int`: token 总数

### 说明
- 通过遍历数据加载器来计算 token 数量
- 假设输入中包含 "input_ids" 键
- 可以用于估算训练时间和资源需求

### 示例
```python
# 计算训练数据总 token 数
train_dataloader = trainer.get_train_dataloader()
total_tokens = trainer.num_tokens(train_dataloader)
print(f"训练数据总 token 数: {total_tokens:,}")

# 估算训练时间
tokens_per_step = total_tokens / (total_tokens / trainer.args.per_device_train_batch_size)
steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * trainer.args.num_train_epochs

print(f"每步 token 数: {int(tokens_per_step):,}")
print(f"预计总训练步数: {total_steps}")

# 使用最大步数限制
tokens_100_steps = trainer.num_tokens(train_dataloader, max_steps=100)
print(f"前100步的 token 数: {tokens_100_steps:,}")
```

## `floating_point_ops()`

计算前向传播的浮点运算数量。

### 参数
- **inputs** (`dict[str, Union[torch.Tensor, Any]]`): 模型输入

### 返回值
- `int`: 浮点运算数量

### 说明
- 用于估算模型的计算复杂度
- 可以帮助估算训练时间和硬件需求
- 基于 Transformer 架构的标准计算方法

### 示例
```python
# 获取一个训练批次
batch = next(iter(trainer.get_train_dataloader()))

# 计算浮点运算数量
flops = trainer.floating_point_ops(batch)
print(f"前向传播 FLOPs: {flops:,}")

# 估算训练时间
steps_per_epoch = len(trainer.get_train_dataloader())
total_flops = flops * steps_per_epoch * trainer.args.num_train_epochs
print(f"总训练 FLOPs: {total_flops:,}")

# 估算 GPU 时长（假设 GPU 可以执行 10 TFLOPs/s）
gpu_time_hours = total_flops / (10e12 * 3600)
print(f"预计 GPU 时长: {gpu_time_hours:.2f} 小时")
```

## `is_local_process_zero()`

检查当前进程是否是本地进程 0。

### 返回值
- `bool`: 如果是本地进程 0 返回 True

### 说明
- 用于分布式训练环境
- 只有本地进程 0 应该执行某些操作（如日志记录、文件保存等）
- 避免多进程重复操作

### 示例
```python
# 只在主进程打印日志
if trainer.is_local_process_zero():
    print("这是主进程，将执行日志记录和文件保存")

# 条件性保存文件
if trainer.is_local_process_zero():
    with open("training_log.txt", "w") as f:
        f.write("训练开始\n")

# 条件性进度显示
if trainer.is_local_process_zero():
    print("训练进度: 50%")
```

## `is_world_process_zero()`

检查当前进程是否是全局进程 0。

### 返回值
- `bool`: 如果是全局进程 0 返回 True

### 说明
- 用于分布式训练环境
- 全局进程 0 是所有节点中的第一个进程
- 比 `is_local_process_zero()` 更严格

### 示例
```python
# 只在全局主进程执行某些操作
if trainer.is_world_process_zero():
    print("这是全局主进程")
    # 保存模型
    trainer.save_model()

# 在所有进程中执行计算
results = trainer.evaluate()

# 只在全局主进程打印结果
if trainer.is_world_process_zero():
    print(f"评估结果: {results}")
```

## `log()`

记录训练指标。

### 参数
- **logs** (`dict[str, float]`): 要记录的指标字典
- **start_time** (`float`, 可选): 开始时间

### 说明
- 通过所有回调处理日志
- 支持多种日志后端（TensorBoard、WandB等）
- 自动添加时间戳和其他元数据

### 示例
```python
# 记录自定义指标
custom_metrics = {
    "custom_loss": 0.5,
    "custom_accuracy": 0.85,
    "learning_rate": 1e-4
}
trainer.log(custom_metrics)

# 记录带有时间戳的指标
import time
start_time = time.time()
# ... 训练步骤 ...
trainer.log({
    "step": trainer.state.global_step,
    "loss": current_loss
}, start_time=start_time)
```

## `_prepare_input()`

准备单个输入数据。

### 参数
- **data** (`torch.Tensor` 或 `Any`): 输入数据

### 返回值
- `torch.Tensor` 或 `Any`: 准备好的输入数据

### 说明
- 将数据移动到正确的设备
- 处理不同的数据类型
- 为分布式训练做准备

## `_prepare_inputs()`

准备输入数据字典。

### 参数
- **inputs** (`dict[str, Union[torch.Tensor, Any]]`): 输入数据字典

### 返回值
- `dict[str, Union[torch.Tensor, Any]]`: 准备好的输入数据字典

### 说明
- 批量处理输入数据
- 将所有张量移动到正确设备
- 保持字典结构

### 示例
```python
# 手动准备输入
batch = {
    "input_ids": torch.tensor([[1, 2, 3]]),
    "attention_mask": torch.tensor([[1, 1, 1]]),
    "labels": torch.tensor([0])
}

prepared_batch = trainer._prepare_inputs(batch)
# 现在 batch 中的所有张量都在正确的设备上
```

## `store_flos()`

存储浮点运算数量。

### 说明
- 累积训练过程中的浮点运算数量
- 用于计算训练效率指标
- 在 `trainer_state.json` 中保存

## 实用工具使用示例

```python
class TrainingAnalyzer:
    def __init__(self, trainer):
        self.trainer = trainer

    def analyze_model_complexity(self):
        """分析模型复杂度"""
        # 获取一个批次进行分析
        batch = next(iter(self.trainer.get_train_dataloader()))

        # 计算各种指标
        num_params = self.trainer.get_num_trainable_parameters()
        flops = self.trainer.floating_point_ops(batch)
        num_tokens = self.trainer.num_tokens(self.trainer.get_train_dataloader())

        print("=== 模型复杂度分析 ===")
        print(f"可训练参数: {num_params:,}")
        print(f"单步 FLOPs: {flops:,}")
        print(f"训练数据总 token 数: {num_tokens:,}")

        # 估算资源需求
        steps_per_epoch = len(self.trainer.get_train_dataloader())
        total_steps = steps_per_epoch * self.trainer.args.num_train_epochs
        total_flops = flops * total_steps

        print(f"预计总训练步数: {total_steps:,}")
        print(f"预计总 FLOPs: {total_flops:,}")

        # 假设 GPU 性能
        gpu_tflops = 10  # 假设 10 TFLOPs/s
        gpu_hours = total_flops / (gpu_tflops * 1e12 * 3600)
        print(f"预计 GPU 时长: {gpu_hours:.2f} 小时")

    def monitor_training_progress(self):
        """监控训练进度"""
        print(f"当前全局步数: {self.trainer.state.global_step}")
        print(f"当前 epoch: {self.trainer.state.epoch}")

        if self.trainer.optimizer:
            current_lr = self.trainer.get_learning_rates()[0]
            print(f"当前学习率: {current_lr:.2e}")

        if self.trainer.is_local_process_zero():
            print("这是主进程，负责日志记录")

        if self.trainer.is_world_process_zero():
            print("这是全局主进程，负责模型保存")

# 使用示例
analyzer = TrainingAnalyzer(trainer)
analyzer.analyze_model_complexity()
analyzer.monitor_training_progress()
```