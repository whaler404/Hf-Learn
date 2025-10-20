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

## `_inner_training_loop()`

内部训练循环实现。

### 参数
- **args** (`TrainingArguments`): 训练参数
- **resume_from_checkpoint** (`str` 或 `bool`, 可选): 检查点恢复路径
- **trial** (`optuna.Trial` 或 `dict`, 可选): 超参数搜索试验
- **ignore_keys_for_eval** (`list[str]`, 可选): 评估时忽略的键

### 返回值
- `TrainOutput`: 训练输出

### 主要流程
1. 初始化训练状态
2. 设置数据加载器
3. 创建优化器和调度器
4. 遍历 epochs
5. 在每个 epoch 内遍历数据加载器
6. 执行训练步骤
7. 定期评估和保存
8. 返回训练结果

### 1. 主要功能逻辑

#### 数据准备和参数初始化 (lines 2374-2443)
```python
# 数据加载器和训练步数设置
train_dataloader = self.get_train_dataloader()
if self.is_fsdp_xla_v2_enabled:
    train_dataloader = tpu_spmd_dataloader(train_dataloader)

# 设置训练控制变量：
# number of training epochs: 训练轮数
# number of training steps per epoch: 每轮训练步数
# total number of training steps to execute: 总训练步数
total_train_batch_size = self.get_total_train_batch_size(args)

(
    num_train_epochs,
    num_update_steps_per_epoch,
    num_examples,
    num_train_samples,
    epoch_based,
    len_dataloader,
    max_steps,
) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

# 计算训练令牌数量（可选）
num_train_tokens = None
if self.args.include_tokens_per_second:
    num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
    # 如果按轮数训练，则线性乘以轮数
    if len_dataloader is not None and epoch_based:
        num_train_tokens *= args.num_train_epochs
    # 否则如果是按步数训练，则乘以梯度累积步数
    else:
        num_train_tokens *= args.gradient_accumulation_steps
```

#### 模型包装和准备 (lines 2449-2517)

正确配置模型以支持各种并行/加速策略（如 DDP、FSDP、DeepSpeed、TP、XLA）

| 模式           | 包装后的结构（`model`）                                    |
| ------------ | -------------------------------------------------- |
| 单卡           | 原始模型（无包装）                                          |
| 多卡 DDP       | `torch.nn.parallel.DistributedDataParallel(model)` |
| DeepSpeed    | `deepspeed.DeepSpeedEngine` 包裹后的模型                 |
| FSDP         | `FullyShardedDataParallel(model)`                  |
| SageMaker MP | AWS 上混合精度模型并行包装                                    |

* DDP（Distributed Data Parallel）：每张 GPU 拿一份完整模型副本，各自处理不同数据，最后再 **同步梯度**
* DeepSpeed：使用 ZeRO 优化，**将参数、梯度、优化器状态等全部分片**
* FSDP（FullySharded）：更细粒度的参数分片，每个参数都只保存在一个 GPU 上


```python
# `self.model_wrapped`：是加载好的 Transformers 模型（例如用 `from_pretrained()` 加载的 `AutoModelForCausalLM`）
# `_wrap_model()`：把模型包上一层，如 DDP、FSDP、DeepSpeed、DataParallel、AMP、IPEX 等
model = self._wrap_model(self.model_wrapped)

# Hugging Face 使用 [`Accelerate`](https://huggingface.co/docs/accelerate/index) 框架来统一管理分布式训练的设备、梯度同步、混合精度等
# 由于模型已经被包装，不要使用 `accelerator.prepare`
# 这用于处理特殊情况，例如：
# FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
use_accelerator_prepare = model is self.model

if use_accelerator_prepare and self.is_fsdp_enabled:
    # 如果启用了自动查找批次大小
    # 从子模型中移除 FSDP 包装，否则 batch size 搜索器会失败
    self.model = unwrap_model(self.model, recursive=True)

# 接下来分成了几个分支逻辑，决定如何正确地 `prepare()` 模型/优化器：
# 模型和优化器的分布式准备
if use_accelerator_prepare:
    self.model.train()
    # 情况一：只有模型和优化器，不包含调度器
    if hasattr(self.lr_scheduler, "step"):
        # 仅 prepare model（Apex）
        if self.use_apex:
            model = self.accelerator.prepare(self.model)
        # 仅 prepare optimizer（TP）
        else:
            # 在TP情况下我们应该避免使用accelerate准备模型，因为transformers from_pretrained已经处理了，而且它会进入基于DDP的准备
            if self.is_tp_enabled:
                self.optimizer = self.accelerator.prepare(self.optimizer)
            else:
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    # 情况二：包含学习率调度器（标准场景），prepare model + optimizer + scheduler
    else:
        # 处理传递"DummyScheduler"的情况，例如在DeepSpeed配置中指定时
        model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

# 重要：此时：
# self.model         是Transformers模型
# self.model_wrapped 是DDP(Transformers模型), Deepspeed(Transformers模型),
# FSDP(Transformers模型), Dynamo优化模块(Transformers模型)等
```

#### 核心训练循环 (lines 2578-2678)

**Batch（批次）** ：一次前向传播 / 反向传播操作所处理的数据数量
**Step（训练步）** ：每处理一个 batch，就完成一个训练 step
**Gradient Accumulation（梯度累积）** ：显存不足以支持大的 batch_size 时，用这个方法来模拟更大的 batch

有效 batch 是 128，但显存只能放下 32，可以设置 `gradient_accumulation_steps=4`，每训练 4 个 batch 才做一次优化（`optimizer.step()`）

```
epoch（1次完整遍历）  
 └── step（处理一个 batch）
     └── batch（如 32 条样本）  
     └── accumulate n 个 batch  
         └── optimizer.step()（模型权重更新）

N = 10_000
batch_size = 32
gradient_accumulation_steps = 4
```
那么 dataloader 每个 epoch 会 yield `10_000 / 32 = 313` batch（步）
每个 epoch 有 `313 // 4 = 78` 次参数更新（optimizer.step 调用）

```python
# 从之前训练的 epoch 开始循环（支持 resume）
for epoch in range(epochs_trained, num_train_epochs):
    epoch_dataloader = train_dataloader
    if hasattr(epoch_dataloader, "set_epoch"):
        epoch_dataloader.set_epoch(epoch)

    # 如果需要，在每个epoch开始时重置过去的内存状态
    if args.past_index >= 0:
        self._past = None

    steps_in_epoch = (
        len(epoch_dataloader)
        if len_dataloader is not None
        else args.max_steps * args.gradient_accumulation_steps
    )
    # 每次 epoch 开始时，会触发 callback，做一些日志或模型保存的准备工作
    self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    # 处理梯度累积
    epoch_iterator = iter(epoch_dataloader)
    # 我们将epoch迭代器分块为梯度累积步数 `n` 个批次
    # 处理最后不足一个完整梯度累积 step 的情况（尾 batch）
    remainder = steps_in_epoch % args.gradient_accumulation_steps
    if remainder == 0:
        remainder = args.gradient_accumulation_steps
    update_step = -1
    total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
        remainder < args.gradient_accumulation_steps
    )
    for _ in range(total_updates):
        update_step += 1
        num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
        # 每次取 `gradient_accumulation_steps` 个 batch 组成一个“更新单元”，然后将这些 batch 分别执行训练
        batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
        # 存储当前梯度累积的批次数
        # 这用于在最后一个累积步骤批次数较少时正确缩放损失
        self.current_gradient_accumulation_steps = len(batch_samples)

        for i, inputs in enumerate(batch_samples):
            step += 1
            do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
            # 由于我们执行预取，需要手动设置sync_gradients
            # `do_sync_step=True` 时，训练才真正做权重更新。
            self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

            # 我们明确希望避免依赖 `accelerator.accumulate` 进行生成训练
            context = (
                functools.partial(self.accelerator.no_sync, model=model)
                if i != len(batch_samples) - 1
                and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                else contextlib.nullcontext
            )
            # `context()` 是为了控制是否同步梯度（sync gradients），即是否真的做 optimizer.step。
            with context():
                tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
```

#### 梯度更新和训练完成 (lines 2692-2856)

每当完成一次**梯度累积**之后（即 `do_sync_step=True` 时），就会进入这段代码完成：

✅ 梯度同步
✅ 梯度裁剪
✅ 权重更新（optimizer.step）
✅ 学习率调度（lr_scheduler.step）
✅ 梯度清零
✅ 日志 / 保存 / 评估 / 回调

```python
# 模型参数更新逻辑执行区间
if do_sync_step:
    # 在分布式训练环境中（比如 DDP 或 DeepSpeed），必须告诉 `Accelerator` 现在需要同步各卡的梯度
    self.accelerator.gradient_state._set_sync_gradients(True)

    # 梯度裁剪，防止梯度爆炸
    if args.max_grad_norm is not None and args.max_grad_norm > 0:
        # 梯度裁剪逻辑（支持多种框架）
        with context():
            # 支持分布式、混合精度的版本
            _grad_norm = self.accelerator.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

    # 1. 执行注册的 callback（如日志、EarlyStopping 等）
    # 2. 在参数更新前提供扩展点（可以监控或修改梯度）
    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

    # 兼容 Deepspeed 等框架（比如 optimizer.step 被 hook 掉的场景）
    with context():
        # 执行梯度下降，更新模型权重
        self.optimizer.step()

    # 再次提供钩子机制，允许跟踪、修改优化器行为
    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

    # 更新前获取学习率
    learning_rate = self._get_learning_rate()

    if not self.accelerator.optimizer_step_was_skipped:
        # 延迟优化器调度直到生成指标
        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step()

    # 梯度清零
    # 注意这里不是 `optimizer.zero_grad()`，这是 intentional，兼容一些特殊模型逻辑
    model.zero_grad()
    self.state.global_step += 1
    self.state.epoch = epoch + (step + 1) / steps_in_epoch
    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    self._maybe_log_save_evaluate(
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=learning_rate,
    )
```

### 2. 分布式训练支持

#### 多GPU和分布式训练检查 (lines 2405-2432)
```python
# `--debug underflow_overflow` 是一个数值稳定性诊断工具，用于检查训练中是否发生了：
# 梯度为 NaN
# 梯度下溢（过小变成 0）
# 梯度上溢（过大导致 NaN）
if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    if self.args.n_gpu > 1:
        # nn.DataParallel(model) 复制模型，创建新的变量和模块
        # 这里注册的引用在其他GPU上不再工作，破坏模块
        raise ValueError(
            "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
            " (torchrun or torch.distributed.launch (deprecated))."
        )
    else:
        DebugUnderflowOverflow(self.model)

delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

# 使用FSDP2时无法延迟优化器创建
is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
if is_fsdp2:
    delay_optimizer_creation = False

# 如果用了 DeepSpeed，就用 HuggingFace 的 `deepspeed_init()` 来配置优化器和调度器
# 会读取的 `ds_config.json`，决定用什么 optimizer（如 AdamW, Adam8bit 等）
if self.is_deepspeed_enabled:
    self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

if not delay_optimizer_creation:
    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
```

#### 分布式训练中的梯度同步 (lines 2667-2694)
```python
# 我们明确希望避免依赖 `accelerator.accumulate` 进行生成训练
# 为每个 batch 训练步骤设置了一个是否同步梯度的上下文环境，配合梯度累积（gradient accumulation）
context = (
    # 不是最后一个累积 step时：使用 `accelerator.no_sync()` 禁止同步
    functools.partial(self.accelerator.no_sync, model=model)
    if i != len(batch_samples) - 1
    # DeepSpeed 已经自己管理了梯度同步，因此不需要 `no_sync()` 包裹
    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
    # 在最后一个 step：使用 `contextlib.nullcontext` 正常同步
    else contextlib.nullcontext
)
with context():
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

if do_sync_step:
    # 由于我们执行预取，需要手动将sync_gradients设置为True
    self.accelerator.gradient_state._set_sync_gradients(True)
```

### 3. 内存和性能优化

#### 动态批处理大小优化 (lines 2358-2373)

Trainer 会在训练开始时：
1. 尝试使用提供的原始 batch size（如 64）
2. 如果发生显存溢出（OOM），自动减半 batch size（64 → 32 → 16 → 8 → ...）
3. 直到模型可以成功运行并进行前向+反向传播

```python
# 自动调整批量大小（batch size），用于避免 OOM（内存溢出）错误
if self.args.auto_find_batch_size:
    if self.state.train_batch_size != self._train_batch_size:
        from accelerate.utils import release_memory

        # 释放之前模型占用的内存：`release_memory(self.model_wrapped)`，防止旧模型的内存残留导致 OOM
        (self.model_wrapped,) = release_memory(self.model_wrapped)
        # 替换为未封装的原始模型：`self.model_wrapped = self.model`
        self.model_wrapped = self.model

        # 在初始传递后检查DeepSpeed并修改配置
        if self.is_deepspeed_enabled:
            # 临时取消设置 `self.args.train_batch_size`
            original_bs = self.args.per_device_train_batch_size
            self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
            self.propagate_args_to_deepspeed(True)
            self.args.per_device_train_batch_size = original_bs
    # 更新 `self.state.train_batch_size`，记录当前生效的批量大小
    self.state.train_batch_size = self._train_batch_size
```

#### 梯度检查点启用 (lines 2445-2448)

训练大模型时，内存占用的最大部分来自 **中间层的激活值（activation）**，它们通常要保留到反向传播阶段

Gradient checkpointing 采用了一种 trade-off 策略：用时间换空间
* 正常训练中，每一层的输出都要 **缓存**，以供反向传播使用
* 开启 checkpointing 后：
  * **中间层的输出不缓存**
  * **反向传播时再重新前向计算**

```python
# 如果需要，激活梯度检查点
if args.gradient_checkpointing:
    self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
```

#### 内存跟踪和性能监控 (lines 2830-2835)
```python
self.is_in_train = False

self._memory_tracker.stop_and_update_metrics(metrics)

# 将所有汇总的性能指标日志化（可能包括 tensorboard、wandb、console 输出等），便于监控和调试
self.log(metrics)
```

### 4. 训练监控和日志

#### 损失处理和NaN/Inf过滤 (lines 2676-2690)

避免训练过程因不稳定的 `loss`（如梯度爆炸、数值溢出）导致崩溃。这样做虽不精确，但能提升训练鲁棒性

```python
if (
    args.logging_nan_inf_filter
    and not is_torch_xla_available()
    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
):
    # 用已有的 `tr_loss` 的近似值进行填充
    # 如果损失是NaN或Inf，则简单添加之前记录损失的平均值
    tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
else:
    if tr_loss.device != tr_loss_step.device:
        raise ValueError(
            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
        )
    tr_loss = tr_loss + tr_loss_step

self.current_flos += float(self.floating_point_ops(inputs))
```

#### 梯度范数监控 (lines 2696-2729)

防止梯度爆炸，保证训练稳定。使用梯度裁剪将参数梯度的范数（一般是 L2 范数）控制在指定范围内

```python
# Gradient clipping
# 只在设置了 `max_grad_norm` 时启用梯度裁剪
if args.max_grad_norm is not None and args.max_grad_norm > 0:
    # SageMaker 分布式 + fp16（混合精度）
    if is_sagemaker_mp_enabled() and args.fp16:
        _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
    # 使用 Apex（NVIDIA AMP）
    elif self.use_apex:
        from apex import amp
        _grad_norm = nn.utils.clip_grad_norm_(
            amp.master_params(self.optimizer),
            args.max_grad_norm,
        )
    # 默认方式（Accelerate）
    else:
        with grad_norm_context():
            _grad_norm = self.accelerator.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

    # 当使用 DeepSpeed 分布式训练时，Hugging Face 使用 `model.get_global_grad_norm()` 来获取全局梯度范数
    if (
        is_accelerate_available()
        and self.accelerator.distributed_type == DistributedType.DEEPSPEED
    ):
        grad_norm = model.get_global_grad_norm()
        # 在某些情况下，梯度范数可能不返回float类型
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
    else:
        grad_norm = _grad_norm
```

### 5. 回调系统

#### 训练生命周期回调 (lines 2573-2592)
```python
self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

if args.eval_on_start:
    self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

for epoch in range(epochs_trained, num_train_epochs):
    # ...
    self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
```

#### 步骤生命周期回调 (lines 2663-2767)
```python
if step % args.gradient_accumulation_steps == 0:
    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

# 训练步骤执行后
self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
# 优化器更新
self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
# 步骤结束
self.control = self.callback_handler.on_step_end(args, self.state, self.control)
else:
    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
```

#### 训练结束回调 (lines 2846-2856)
```python
self.control = self.callback_handler.on_train_end(args, self.state, self.control)

# 等待检查点上传完成
self._finish_current_push()

# 训练结束后，我们确保通过移除前向传播后钩子来恢复嵌入层的原始前向传播方法
if self.neftune_noise_alpha is not None:
    self._deactivate_neftune(self.model)

return TrainOutput(self.state.global_step, train_loss, metrics)
```

### 6. 评估和验证

#### 开始时评估 (lines 2575-2576)
```python
if args.eval_on_start:
    self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)
```

#### 定期评估和保存 (lines 2756-2792)
```python
self._maybe_log_save_evaluate(
    tr_loss,
    grad_norm,
    model,
    trial,
    epoch,
    ignore_keys_for_eval,
    start_time,
    learning_rate=learning_rate,
)

# Epoch结束时再次评估
self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
self._maybe_log_save_evaluate(
    tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
)
```

#### 最佳模型加载 (lines 2811-2812)
```python
if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    self._load_best_model()
```

### 7. 高级训练策略

#### 梯度累积处理 (lines 2607-2621)
```python
# 我们将epoch迭代器分块为梯度累积步数 `n` 个批次
remainder = steps_in_epoch % args.gradient_accumulation_steps
if remainder == 0:
    remainder = args.gradient_accumulation_steps
update_step = -1
total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
    remainder < args.gradient_accumulation_steps
)
for _ in range(total_updates):
    update_step += 1
    num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
    batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
    # 存储当前梯度累积的批次数
    # 这用于在最后一个累积步骤批次数较少时正确缩放损失
    self.current_gradient_accumulation_steps = len(batch_samples)
```

#### 令牌计数 (lines 2628-2657)

用于跟踪 **模型在训练过程中见过的输入 token 数量**。这通常用于监控训练进度、估算训练成本、或者用于 FLOPs、token-level logging 等。

`TrainingArguments.include_num_input_tokens_seen` 参数控制是否启用 **token 计数功能**。它的取值可以是：
* `"no"` / `False`：不统计（默认）
* `"all"`：统计所有 token
* `"non_padding"`：只统计非 padding 的 token（更有意义）

```python
if self.args.include_num_input_tokens_seen not in ["no", False]:
    main_input_name = getattr(self.model, "main_input_name", "input_ids")
    # 从模型对象中获取主要输入的名称。默认为 `"input_ids"`，但有些模型（比如语音、图像）可能是别的名称（如 `"pixel_values"` 或 `"input_features"`）
    if main_input_name not in inputs:
        logger.warning(
            "Tried to track the number of tokens seen, however the current model is "
            "not configured properly to know what item is the input. To fix this, add "
            "a `main_input_name` attribute to the model class you are using."
        )
    else:
        # 判断用户是否希望只统计非 padding token：
        if self.args.include_num_input_tokens_seen == "non_padding":
            # `attention_mask=1` 的位置是非 padding，直接 `sum()` 就是非 padding token 数
            if "attention_mask" in inputs:
                input_tokens = inputs["attention_mask"].sum()
            # 如果没有 `attention_mask`，那就用 `pad_token_id` 判断非 padding：不等于 pad 的就算非 padding
            elif (
                self.processing_class is not None
                and hasattr(self.processing_class, "pad_token_id")
                and self.processing_class.pad_token_id is not None
            ):
                input_tokens = (
                    inputs[main_input_name] != self.processing_class.pad_token_id
                ).sum()
            else:
                logger.warning(
                    "Could not determine method to count non-padding tokens, falling back to counting all tokens."
                )
                input_tokens = inputs[main_input_name].numel()
        else:
            input_tokens = inputs[main_input_name].numel()

        # 将 input token 数转为 tensor，以支持分布式训练
        input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
        # 使用 `accelerator.gather()` 把所有进程的 token 数合并（适配分布式），然后累加到 `num_input_tokens_seen` 上
        self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
```

#### 学习率调度 (lines 2744-2750)
```python
# 更新前获取学习率
learning_rate = self._get_learning_rate()

if not self.accelerator.optimizer_step_was_skipped:
    # 延迟优化器调度直到生成指标
    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.lr_scheduler.step()
```

### 8. 模型增强功能

#### NEFTune噪声处理 (lines 2853-2854)
```python
# After training we make sure to retrieve back the original forward pass method
# for the embedding layer by removing the forward post hook.
if self.neftune_noise_alpha is not None:
    self._deactivate_neftune(self.model)
```

#### FSDP QLoRA插件更新 (lines 2463-2466)
```python
if use_accelerator_prepare:
    # 如果有qlora，配置fsdp插件
    self._fsdp_qlora_plugin_updates()
    if self.accelerator.mixed_precision != "fp8":
        self.model = self.accelerator.prepare(self.model)
```

### 9. 实验管理

#### 超参数搜索状态设置 (lines 2439-2440)

**初始化训练状态 `TrainerState` 对象**，其中保存了当前训练/微调任务的元信息，如 epoch、step、最佳模型位置等

```python
self.state = TrainerState(
    stateful_callbacks=[
        # `self.callback_handler.callbacks` 是在训练中用到的回调列表（可能包括 `EarlyStoppingCallback`、`TensorBoardCallback` 等）
        # `self.control` 是 `TrainerControl` 实例
        # `ExportableState` 是一个接口（协议），代表该对象拥有可以导出的状态（用于恢复训练）
        cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
    ]
)
# `trial` 代表当前是否正在进行超参数搜索（如通过 Optuna）
self.state.is_hyper_param_search = trial is not None
```

#### 输出目录管理 (lines 2836-2844)
```python
# 获取运行目录（run_dir）
run_dir = self._get_output_dir(trial)
# 列出所有检查点目录
checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

# 当save_total_limit=1时，如果最后一个检查点与最佳检查点不同且进程允许保存，则删除最后一个检查点
if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    for checkpoint in checkpoints_sorted:
        if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)
```

### 10. 生产部署功能

#### 检查点管理 (lines 2501-2511)
```python
# 检查点加载
if resume_from_checkpoint is not None:
    if self.is_deepspeed_enabled:
        deepspeed_load_checkpoint(
            self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
        )
    elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
        self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

# 检查是否存在保存的优化器或调度器状态
self._load_optimizer_and_scheduler(resume_from_checkpoint)
self._load_scaler(resume_from_checkpoint)
```

#### 模型上传完成 (lines 2848-2849)
```python
# 等待检查点上传完成
self._finish_current_push()
```

#### 最终训练指标 (lines 2819-2835)
```python
metrics = speed_metrics(
    "train",
    start_time,
    num_samples=num_train_samples,
    num_steps=self.state.max_steps,
    num_tokens=num_train_tokens,
)
self.store_flos()
metrics["total_flos"] = self.state.total_flos
metrics["train_loss"] = train_loss

self.is_in_train = False

self._memory_tracker.stop_and_update_metrics(metrics)

self.log(metrics)
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

### 1. 主要功能逻辑

#### 核心训练流程 (lines 4010-4020, 4050-4073)
```python
# 设置模型为训练模式
model.train()
if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    self.optimizer.train()  # 支持一些自定义优化器的特殊行为

# 准备输入数据（移动到设备，数据类型转换等）
inputs = self._prepare_inputs(inputs)

# 计算损失
with self.compute_loss_context_manager():
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

# 多GPU训练时对损失进行平均
if self.args.n_gpu > 1:
    loss = loss.mean()  # mean() to average on multi-gpu parallel training

# 反向传播
if self.use_apex:
    from apex import amp
    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
else:
    # 标准反向传播（默认执行分支），处理梯度累积，
    if (
        not self.model_accepts_loss_kwargs or num_items_in_batch is None
    ) and self.compute_loss_func is None:
        # 如果模型不接受损失参数，需要按梯度累积步数归一化损失
        loss = loss / self.current_gradient_accumulation_steps

    # DeepSpeed 特殊处理（防止重复 scale）
    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
        kwargs["scale_wrt_gas"] = False

    # `Accelerate` 统一管理多种设备（CPU/GPU/TPU）的反向传播，自动处理多种 backend 的差异
    self.accelerator.backward(loss, **kwargs)

# 取出 `loss` 并从计算图中分离（避免影响之后的梯度）
return loss.detach()
```

### 2. 并行训练支持

#### 上下文并行处理 (lines 4004-4009)
```python
# 为上下文并行准备缓冲区
# Context Parallelism（CP）用于异构设备并行训练，比如 CPU + GPU 协同
cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

# 如果未启用CP，上下文管理器为空操作
with cp_context():
    # 训练逻辑在上下文并行环境中执行
    model.train()
    # ... 训练步骤
```

#### 多GPU损失平均 (lines 4050-4051)
```python
# 防止多个 GPU 每个计算的 loss 不一致，做一次平均
if self.args.n_gpu > 1:
    loss = loss.mean()  # mean() to average on multi-gpu parallel training
```

#### 定期缓存清理 (lines 4023-4042)
```python
del inputs  # 显式删除输入数据释放内存

# 定期清理GPU缓存以防止内存泄漏
if (
    self.args.torch_empty_cache_steps is not None
    and self.state.global_step % self.args.torch_empty_cache_steps == 0
):
    # 支持多种硬件平台的缓存清理
    ...
    else:
        torch.cuda.empty_cache()  # 默认CUDA缓存清理
```

### 3. 高级优化器支持

#### LOMO优化器特殊处理 (lines 4046-4048)
```python
# 对于LOMO优化器，需要显式使用学习率
if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    kwargs["learning_rate"] = self._get_learning_rate()
```

### 4. 混合精度训练

#### Apex混合精度 (lines 4053-4057)
```python
if self.use_apex:
    from apex import amp
    # 使用Apex进行混合精度训练的损失缩放
    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
```

#### 标准Accelerate混合精度 (lines 4066-4071)
```python
# DeepSpeed启用时禁用梯度累积相关的损失缩放
# https://github.com/huggingface/transformers/pull/35808
if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    kwargs["scale_wrt_gas"] = False

# 使用Accelerate进行混合精度的反向传播
self.accelerator.backward(loss, **kwargs)
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
# 训练一个对比学习模型（比如 SimCSE）
def my_contrastive_loss(outputs, labels=None, num_items_in_batch=None):
    embeddings = outputs["last_hidden_state"][:, 0, :]  # CLS
    similarity = torch.matmul(embeddings, embeddings.T)
    loss = custom_ntxent(similarity)
    return loss
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_loss_func=my_contrastive_loss,
    model_accepts_loss_kwargs=False
)
```

### 主要功能逻辑

#### 损失计算核心流程 (lines 4101-4147)
```python
# 如果启用了标签平滑或自定义损失函数，从输入中提取labels
if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
    labels = inputs.pop("labels")  # 从输入中移除labels，避免重复处理
else:
    labels = None

# 检查模型是否接受额外的损失参数（如num_items_in_batch）
if self.model_accepts_loss_kwargs:
    kwargs = {}
    if num_items_in_batch is not None:
        kwargs["num_items_in_batch"] = num_items_in_batch
    inputs = {**inputs, **kwargs}  # 将额外参数合并到输入中

# 执行模型前向传播
outputs = model(**inputs)

# 保存过去状态（用于生成任务的缓存）
if self.args.past_index >= 0:
    self._past = outputs[self.args.past_index]
```

#### 自定义损失函数处理 (lines 4117-4127)
```python
# 用户定义的自定义损失函数
if self.compute_loss_func is not None:
    if labels is None:
        logger.warning(
            "Trainer: `compute_loss_func` is defined but `labels=None`. "
            "Your custom loss function will still be called with labels=None. "
        )
    # 调用用户自定义的损失计算函数
    loss = self.compute_loss_func(
        outputs,
        labels,
        num_items_in_batch=num_items_in_batch,
    )
```

#### 标准HF损失处理 (lines 4129-4147)
```python
# 如果没有自定义损失函数，默认HF损失处理（标签平滑）
elif labels is not None:
    # 解包模型以获取真实模型名称
    unwrapped_model = self.accelerator.unwrap_model(model)
    model_name = (
        unwrapped_model.base_model.model._get_name()
        if _is_peft_model(unwrapped_model)  # PEFT模型特殊处理
        else unwrapped_model._get_name()
    )

    # 使用 Label Smoother（仅用于分类/语言建模）
    # 根据模型类型选择不同的标签平滑策略
    if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        loss = self.label_smoother(outputs, labels, shift_labels=True)  # 因果语言模型需要shift标签
    else:
        loss = self.label_smoother(outputs, labels)  # 其他模型直接使用标签平滑
# 如果上面两种都没有，从模型的返回中拿 `loss`（`ModelOutput.loss` 或 tuple 的第一个元素）
else:
    if isinstance(outputs, dict) and "loss" not in outputs:
        raise ValueError(
            "The model did not return a loss from the inputs, only the following keys: "
            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        )
    # 从模型输出中提取损失（支持字典和元组格式）
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
```




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