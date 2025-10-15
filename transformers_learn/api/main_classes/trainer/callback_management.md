# 回调管理方法

## `add_callback()`

向当前的 `TrainerCallback` 列表中添加回调。

### 参数
- **callback** (`type` 或 `TrainerCallback`):
  - `TrainerCallback` 类或 `TrainerCallback` 实例
  - 如果是类，将实例化该类的一个成员

### 示例
```python
from transformers import TrainerCallback, Trainer

# 添加回调类
class MyCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"Epoch {state.epoch} 完成")

trainer.add_callback(MyCallback)

# 添加回调实例
my_callback = MyCallback()
trainer.add_callback(my_callback)

# 添加多个回调
trainer.add_callback([MyCallback, AnotherCallback])
```

## `pop_callback()`

从当前的 `TrainerCallback` 列表中移除回调并返回它。

### 参数
- **callback** (`type` 或 `TrainerCallback`):
  - `TrainerCallback` 类或 `TrainerCallback` 实例
  - 如果是类，将弹出在回调列表中找到的该类的第一个成员

### 返回值
- `TrainerCallback` 或 `None`: 如果找到则返回被移除的回调，否则返回 None

### 说明
- 如果找不到回调，返回 None（不会引发错误）
- 可以用于临时移除回调

### 示例
```python
# 移除回调类
removed_callback = trainer.pop_callback(MyCallback)
if removed_callback:
    print("成功移除回调")

# 移除回调实例
removed_callback = trainer.pop_callback(my_callback)

# 临时移除并在之后重新添加
removed_callback = trainer.pop_callback(MyCallback)
# ... 执行一些不需要该回调的操作 ...
if removed_callback:
    trainer.add_callback(removed_callback)
```

## `remove_callback()`

从当前的 `TrainerCallback` 列表中移除回调。

### 参数
- **callback** (`type` 或 `TrainerCallback`):
  - `TrainerCallback` 类或 `TrainerCallback` 实例
  - 如果是类，将移除在回调列表中找到的该类的第一个成员

### 说明
- 与 `pop_callback()` 不同，此方法不返回被移除的回调
- 用于永久移除回调

### 示例
```python
# 移除特定回调类
trainer.remove_callback(MyCallback)

# 移除回调实例
trainer.remove_callback(my_callback)

# 移除内置回调
from transformers.integrations import TensorBoardCallback
trainer.remove_callback(TensorBoardCallback)
```

## 内置回调类型

Trainer 包含多个内置回调：

### 默认回调
- `DefaultFlowCallback`: 管理训练流程
- `ProgressCallback`: 显示进度条
- `PrinterCallback`: 打印日志信息

### 集成回调
- `TensorBoardCallback`: TensorBoard 日志记录
- `WandbCallback`: Weights & Biases 集成
- `MLflowCallback`: MLflow 集成
- `CometCallback`: Comet.ml 集成
- `NeptuneCallback`: Neptune.ai 集成
- `AzureMLCallback`: Azure ML 集成
- `CodeCarbonCallback`: 碳排放跟踪

### 示例：自定义回调
```python
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "train_loss" in logs:
                print(f"Step {state.global_step}: Train Loss = {logs['train_loss']:.4f}")

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Eval Loss = {logs.get('eval_loss', 'N/A'):.4f}")

    def on_save(self, args, state, control, **kwargs):
        print(f"Checkpoint saved at step {state.global_step}")

# 使用自定义回调
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=[CustomLoggingCallback()]
)
```

### 示例：回调管理
```python
# 检查当前回调
print(f"当前回调数量: {len(trainer.callback_handler.callbacks)}")

# 获取特定类型的回调
tensorboard_callback = None
for callback in trainer.callback_handler.callbacks:
    if hasattr(callback, 'writer'):  # TensorBoardCallback 的特征
        tensorboard_callback = callback
        break

# 动态添加和移除回调
class EvaluationCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 每个 epoch 结束时进行额外评估
        eval_results = trainer.evaluate()
        print(f"额外评估结果: {eval_results}")

# 在训练过程中动态添加
trainer.add_callback(EvaluationCallback())

# 训练完成后移除
trainer.remove_callback(EvaluationCallback)
```

## 回调生命周期

回调在训练过程中的不同阶段被调用：

1. **训练开始**: `on_train_begin`
2. **Epoch 开始**: `on_epoch_begin`
3. **Step 开始**: `on_step_begin`
4. **训练步骤**: `on_step_end`
5. **评估**: `on_evaluate`
6. **日志**: `on_log`
7. **保存**: `on_save`
8. **Epoch 结束**: `on_epoch_end`
9. **训练结束**: `on_train_end`

### 示例：完整的回调生命周期
```python
class LifecycleCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("训练开始")

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)} 开始")

    def on_step_begin(self, args, state, control, **kwargs):
        print(f"Step {state.global_step} 开始")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step} 结束")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)} 结束")

    def on_train_end(self, args, state, control, **kwargs):
        print("训练结束")
```