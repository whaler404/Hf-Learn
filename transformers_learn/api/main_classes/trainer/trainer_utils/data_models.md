# Trainer 数据模型

本文档详细介绍了 Transformers Trainer 中使用的核心数据模型，包括评估预测、输出容器和检查点管理相关的类和函数。

## 概述

这些数据模型为训练、评估和预测过程提供了标准化的数据结构，确保了不同组件之间的一致性和可互操作性。

## 核心数据模型

### 1. EvalPrediction

评估预测结果容器，用于在计算指标时传递模型预测结果和真实标签。

```python
class EvalPrediction:
    """
    评估输出（总是包含标签），用于计算指标。

    Parameters:
        predictions (`np.ndarray`): 模型的预测结果
        label_ids (`np.ndarray`): 要匹配的目标标签
        inputs (`np.ndarray`, *optional*): 传递给模型的输入数据
        losses (`np.ndarray`, *optional*): 评估过程中计算的损失值
    """
```

动态构建的元组，包含所有非 None 的数据组件：
- 基础：`(predictions, label_ids)`
- 如果有 inputs：`(predictions, label_ids, inputs)`
- 如果有 losses：`(predictions, label_ids, inputs, losses)`

### 2. EvalLoopOutput

评估循环的输出容器，包含完整的评估结果。

```python
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, tuple[np.ndarray]]]
    metrics: Optional[dict[str, float]]
    num_samples: Optional[int]
```

### 3. PredictionOutput

预测输出容器，用于 `Trainer.predict()` 方法的结果。

```python
class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, tuple[np.ndarray]] # 模型的预测结果
    label_ids: Optional[Union[np.ndarray, tuple[np.ndarray]]] # 真实标签（如果可用）
    metrics: Optional[dict[str, float]] # 计算的指标（如果可用）
```

### 4. TrainOutput

训练步骤的输出容器，包含训练过程的关键信息。

```python
class TrainOutput(NamedTuple):
    global_step: int # 全局训练步数
    training_loss: float # 当前步骤的训练损失
    metrics: dict[str, float] # 训练相关的其他指标
```

### 5. BestRun

超参数搜索最佳运行结果容器。

```python
class BestRun(NamedTuple):
    """
    超参数搜索找到的最佳运行结果。

    Parameters:
        run_id (`str`): 最佳运行的 ID（如果保存了模型，检查点在以 run-{run_id} 结尾的文件夹中）
        objective (`float`): 该运行获得的目标值
        hyperparameters (`dict[str, Any]`): 获得此运行的超参数
        run_summary (`Optional[Any]`): 调优实验的摘要。Ray 后端为 ray.tune.ExperimentAnalysis 对象
    """
```

## 策略枚举

### IntervalStrategy

定义操作执行间隔的策略。

```python
class IntervalStrategy(ExplicitEnum):
    NO = "no"      # 不执行
    STEPS = "steps"  # 按步数执行
    EPOCH = "epoch"  # 按轮次执行
```

### SaveStrategy

定义模型保存策略。

```python
class SaveStrategy(ExplicitEnum):
    NO = "no"        # 不保存
    STEPS = "steps"  # 按步数保存
    EPOCH = "epoch"  # 按轮次保存
    BEST = "best"    # 当达到最佳指标时保存
```

### EvaluationStrategy

定义评估策略（功能与 IntervalStrategy 相同）。

```python
class EvaluationStrategy(ExplicitEnum):
    NO = "no"      # 不评估
    STEPS = "steps"  # 按步数评估
    EPOCH = "epoch"  # 按轮次评估
```

### HubStrategy

定义推送到 HuggingFace Hub 的策略。

```python
class HubStrategy(ExplicitEnum):
    END = "end"                          # 训练结束时推送
    EVERY_SAVE = "every_save"            # 每次保存时推送
    CHECKPOINT = "checkpoint"            # 推送检查点
    ALL_CHECKPOINTS = "all_checkpoints"  # 推送所有检查点
```