# 模型保存和加载方法

## `save_model()`

保存模型，以便可以使用 `from_pretrained()` 重新加载。

### 参数
- **output_dir** (`str`, 可选): 保存模型的输出目录。如果为 None，使用 `self.args.output_dir`
- **_internal_call** (`bool`, 可选): 是否为内部调用（通常用户不需要设置）

### 说明
- 只从主进程保存
- 支持多种分布式训练环境（TPU、SageMaker、FSDP、DeepSpeed等）
- 自动处理不同并行训练策略的模型保存
- 保存模型权重和配置文件

### 支持的保存策略
- **TPU**: 使用 `_save_tpu()` 方法
- **SageMaker Model Parallelism**: 保存包装后的模型状态
- **N-D 并行**: 根据并行配置决定保存策略
- **张量并行**: 保存完整的模型状态
- **FSDP**: 根据 FSDP 插件配置保存状态字典
- **DeepSpeed**: 处理 DeepSpeed 的特殊保存需求

### 示例
```python
# 保存到默认位置
trainer.save_model()

# 保存到指定位置
trainer.save_model("./my_saved_model")

# 从保存的模型加载
from transformers import AutoModel
model = AutoModel.from_pretrained("./my_saved_model")
```

## `_save_checkpoint()`

保存训练检查点。

### 参数
- **model** (`nn.Module`): 要保存的模型
- **trial** (`optuna.Trial` 或 `dict`, 可选): 超参数搜索试验

### 说明
- 保存完整的训练状态，包括：
  - 模型权重
  - 优化器状态
  - 学习率调度器状态
  - 随机数生成器状态
  - 训练器状态
- 支持最佳模型检查点管理
- 自动创建检查点目录结构

### 检查点结构
```
checkpoint-1000/
├── pytorch_model.bin              # 模型权重
├── optimizer.pt                  # 优化器状态
├── scheduler.pt                  # 调度器状态
├── scaler.pt                     # 混合精度缩放器状态
├── trainer_state.json            # 训练器状态
├── training_args.bin             # 训练参数
├── rng_state.pth                 # 随机数状态
└── config.json                   # 模型配置
```

### 示例
```python
# 手动保存检查点（通常在训练循环中自动调用）
trainer._save_checkpoint(trainer.model, trial=None)
```

## `_load_from_checkpoint()`

从检查点恢复训练。

### 参数
- **resume_from_checkpoint** (`str` 或 `bool`): 检查点路径或 True（使用最后一个检查点）
- **model** (`nn.Module`, 可选): 要加载的模型

### 返回值
- `tuple`: (model, resumed) 其中 resumed 是是否成功恢复的布尔值

### 说明
- 从指定或最后的检查点恢复训练状态
- 加载模型权重、优化器状态、调度器状态等
- 恢复训练器状态（全局步数、epoch等）
- 处理不同分布式训练环境的加载

### 示例
```python
# 从特定检查点恢复
model, resumed = trainer._load_from_checkpoint("./checkpoint-1000")

# 从最后一个检查点恢复
model, resumed = trainer._load_from_checkpoint(True)
```

## `_save_optimizer_and_scheduler()`

保存优化器和学习率调度器状态。

### 参数
- **output_dir** (`str`): 输出目录

### 说明
- 保存优化器的状态字典
- 保存学习率调度器的状态字典
- 支持分布式训练环境的特殊处理

### 示例
```python
# 手动保存优化器和调度器
trainer._save_optimizer_and_scheduler("./my_checkpoint")
```

## `_load_optimizer_and_scheduler()`

加载优化器和学习率调度器状态。

### 参数
- **checkpoint** (`str`): 检查点目录路径

### 说明
- 从检查点加载优化器状态
- 从检查点加载调度器状态
- 处理版本兼容性问题
- 支持 DeepSpeed 和 FSDP 的特殊加载逻辑

### 示例
```python
# 手动加载优化器和调度器
trainer._load_optimizer_and_scheduler("./checkpoint-1000")
```

## `_save_rng_state()`

保存随机数生成器状态。

### 参数
- **output_dir** (`str`): 输出目录

### 说明
- 保存 PyTorch、NumPy 和 Python random 的状态
- 确保训练的可重现性
- 在分布式训练中特别重要

### 示例
```python
# 保存随机数状态
trainer._save_rng_state("./checkpoint")
```

## `_load_rng_state()`

加载随机数生成器状态。

### 参数
- **checkpoint** (`str`): 检查点路径

### 说明
- 恢复所有相关库的随机数状态
- 确保从检查点恢复后的训练可重现性
- 处理不同设备和库的兼容性

## `_load_best_model()`

加载最佳模型。

### 说明
- 根据 `metric_for_best_model` 加载最佳检查点的模型
- 在训练结束时或评估期间调用
- 自动处理最佳模型检查点的查找和加载

### 示例
```python
# 手动加载最佳模型
trainer._load_best_model()
```

## `_sorted_checkpoints()`

获取排序后的检查点列表。

### 参数
- **output_dir** (`str`, 可选): 检查点目录
- **checkpoint_prefix** (`str`, 可选): 检查点文件前缀

### 返回值
- `list[str]`: 按步数排序的检查点路径列表

### 说明
- 扫描指定目录中的检查点
- 按步数排序（最新的在最后）
- 用于检查点管理和清理

### 示例
```python
# 获取所有检查点
checkpoints = trainer._sorted_checkpoints("./checkpoints")
print(f"找到 {len(checkpoints)} 个检查点")

# 获取最新的检查点
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"最新检查点: {latest_checkpoint}")
```

## `_rotate_checkpoints()`

轮换检查点，删除最旧的检查点以节省空间。

### 参数
- **use_mtime** (`bool`, 可选): 是否使用修改时间而非步数排序
- **output_dir** (`str`, 可选): 输出目录

### 说明
- 根据 `save_total_limit` 参数删除最旧的检查点
- 保持最新的 N 个检查点
- 可以基于修改时间或步数排序

### 示例
```python
# 手动轮换检查点
trainer._rotate_checkpoints()

# 基于修改时间轮换
trainer._rotate_checkpoints(use_mtime=True)
```

## `_save_scaler()`

保存混合精度缩放器状态。

### 参数
- **output_dir** (`str`): 输出目录

### 说明
- 保存 GradScaler 状态用于混合精度训练
- 确保从检查点恢复后精度缩放的一致性

## `_load_scaler()`

加载混合精度缩放器状态。

### 参数
- **checkpoint** (`str`): 检查点路径

### 说明
- 从检查点加载 GradScaler 状态
- 处理缩放器不存在的情况

## 完整的检查点管理示例

```python
from transformers import Trainer, TrainingArguments
import os

# 自定义检查点管理
class CheckpointManager:
    def __init__(self, trainer):
        self.trainer = trainer

    def save_checkpoint(self, step, custom_name=None):
        """保存自定义检查点"""
        if custom_name:
            checkpoint_dir = os.path.join(
                self.trainer.args.output_dir,
                custom_name
            )
        else:
            checkpoint_dir = os.path.join(
                self.trainer.args.output_dir,
                f"checkpoint-{step}"
            )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.trainer.save_model(checkpoint_dir)
        self.trainer._save_optimizer_and_scheduler(checkpoint_dir)
        self.trainer._save_rng_state(checkpoint_dir)

        print(f"检查点已保存到: {checkpoint_dir}")
        return checkpoint_dir

    def list_checkpoints(self):
        """列出所有检查点"""
        checkpoints = self.trainer._sorted_checkpoints()
        return checkpoints

    def clean_old_checkpoints(self, keep_last=3):
        """清理旧检查点，保留最新的几个"""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                import shutil
                shutil.rmtree(checkpoint)
                print(f"删除旧检查点: {checkpoint}")

# 使用示例
manager = CheckpointManager(trainer)

# 保存自定义检查点
manager.save_checkpoint(step=1000, custom_name="milestone_checkpoint")

# 清理检查点
manager.clean_old_checkpoints(keep_last=3)
```