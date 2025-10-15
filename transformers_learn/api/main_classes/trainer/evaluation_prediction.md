# 评估和预测方法

## `evaluate()`

运行评估并返回指标。

### 参数
- **eval_dataset** (`Dataset` 或 `dict[str, Dataset]`, 可选):
  - 如果要覆盖 `self.eval_dataset`，传递数据集
  - 如果是 `datasets.Dataset`，模型 `forward()` 方法不接受的列将自动移除
  - 如果是字典，将在每个数据集上评估，在指标名称前添加字典键
  - 数据集必须实现 `__len__` 方法
- **ignore_keys** (`list[str]`, 可选): 在收集预测时应忽略的模型输出中的键列表
- **metric_key_prefix** (`str`, 可选, 默认为 `"eval"`): 用作指标键的可选前缀

### 返回值
- `dict[str, float]`: 包含评估损失和潜在指标的字典。字典还包含来自训练状态的 epoch 数量。

### 说明
- 调用脚本负责提供计算指标的方法，因为它们是任务相关的（传递给 init 的 `compute_metrics` 参数）
- 支持多数据集评估
- 自动处理分布式环境中的指标聚合

### 多数据集评估示例
```python
# 评估单个数据集
eval_results = trainer.evaluate()

# 评估多个数据集
multi_eval_dataset = {
    "validation": val_dataset,
    "test": test_dataset,
    "ood": ood_dataset
}
results = trainer.evaluate(eval_dataset=multi_eval_dataset)
# 结果将包含 "eval_validation_loss", "eval_test_loss", "eval_ood_loss" 等

# 与 load_best_model_at_end 一起使用
training_args = TrainingArguments(
    metric_for_best_model="eval_validation_loss",
    load_best_model_at_end=True
)
```

### 示例
```python
# 基础评估
results = trainer.evaluate()
print(f"评估损失: {results['eval_loss']:.4f}")
print(f"困惑度: {math.exp(results['eval_loss']):.2f}")

# 使用自定义前缀
results = trainer.evaluate(metric_key_prefix="val")
print(f"验证损失: {results['val_loss']:.4f}")

# 评估时忽略某些输出
results = trainer.evaluate(ignore_keys=["past_key_values", "hidden_states"])
```

## `predict()`

运行预测并返回预测和潜在指标。

### 参数
- **test_dataset** (`Dataset`): 运行预测的数据集。如果是 `datasets.Dataset`，模型 `forward()` 方法不接受的列将自动移除。必须实现 `__len__` 方法
- **ignore_keys** (`list[str]`, 可选): 在收集预测时应忽略的模型输出中的键列表
- **metric_key_prefix** (`str`, 可选, 默认为 `"test"`): 用作指标键的可选前缀

### 返回值
- `PredictionOutput`: 包含以下键的命名元组：
  - `predictions` (`np.ndarray`): 在 `test_dataset` 上的预测
  - `label_ids` (`np.ndarray`, 可选): 标签（如果数据集包含）
  - `metrics` (`dict[str, float]`, 可选): 潜在的指标字典（如果数据集包含标签）

### 说明
- 根据数据集和用例，测试数据集可能包含标签，此时也会返回指标
- 如果预测或标签有不同的序列长度，预测将被填充（右侧）以允许连接成一个数组，填充索引为 -100
- 支持动态填充场景

### 示例
```python
# 基础预测
predictions = trainer.predict(test_dataset)
print(f"预测形状: {predictions.predictions.shape}")
print(f"标签形状: {predictions.label_ids.shape}")

# 获取预测类别
predicted_classes = predictions.predictions.argmax(axis=-1)

# 使用指标
if predictions.metrics:
    print(f"测试准确率: {predictions.metrics['test_accuracy']:.4f}")

# 处理序列标注任务的动态填充
predictions = trainer.predict(token_classification_dataset)
# 自动处理不同长度的序列
```

## `evaluation_loop()`

评估循环的主要实现。

### 参数
- **dataloader** (`DataLoader`): 评估数据加载器
- **description** (`str`): 循环描述
- **prediction_loss_only** (`bool`, 可选): 是否只计算损失
- **ignore_keys** (`list[str]`, 可选): 忽略的键
- **metric_key_prefix** (`str`): 指标前缀

### 返回值
- `EvalLoopOutput`: 评估循环输出

### 说明
- 这是评估的核心实现
- 处理模型评估、预测收集和指标计算
- 支持分布式训练环境
- 自动处理内存优化和设备管理

## `prediction_loop()`

预测循环的传统实现。

### 参数
- **dataloader** (`DataLoader`): 预测数据加载器
- **description** (`str`): 循环描述
- **ignore_keys** (`list[str]`, 可选): 忽略的键
- **metric_key_prefix** (`str`): 指标前缀

### 返回值
- `PredictionOutput`: 预测输出

### 说明
- 这是 `evaluation_loop` 的简化版本
- 专门用于预测场景
- 在某些情况下可能提供更好的性能

## `prediction_step()`

执行单个预测步骤。

### 参数
- **model** (`nn.Module`): 要评估的模型
- **inputs** (`dict[str, Union[torch.Tensor, Any]]`): 模型的输入和目标
- **prediction_loss_only** (`bool`): 是否只计算损失
- **ignore_keys** (`list[str]`, 可选): 忽略的键

### 返回值
- `tuple`: 包含损失、预测和标签的元组

### 说明
- 可以子类化并重写以注入自定义行为
- 处理单个批次的预测
- 支持生成任务的特殊处理

### 示例
```python
# 自定义预测步骤
class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 自定义预测逻辑
        with torch.no_grad():
            outputs = model(**inputs)

        # 自定义处理
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]

        return (outputs.loss if hasattr(outputs, "loss") else outputs[0],
                logits,
                inputs.get("labels"))
```

## `_gather_and_numpify()`

收集并转换为 NumPy 数组。

### 参数
- **tensors** (`torch.Tensor`): 要收集的张量
- **name** (`str`): 张量名称

### 返回值
- `np.ndarray`: 收集的 NumPy 数组

### 说明
- 在分布式环境中收集张量
- 转换为 NumPy 格式
- 处理不同设备上的张量

## `_nested_gather()`

嵌套收集张量。

### 参数
- **tensors**: 要收集的张量（可以是嵌套结构）
- **name** (`str`, 可选): 张量名称

### 返回值
- 收集的张量结构

### 说明
- 处理嵌套的张量结构
- 在分布式训练中特别有用
- 保持原始结构形状