# TrainingArguments

`TrainingArguments` 是一个包含所有训练配置参数的数据类，用于控制训练过程中的各种行为和设置。

## 概述

`TrainingArguments` 是训练循环相关的参数集合，可以通过 `HfArgumentParser` 转换为命令行参数。它涵盖了从输出目录配置到优化器设置、分布式训练、日志记录等各个方面。

## 参数分类

### 📁 输出和目录配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `output_dir` | `str` | `"trainer_output"` | 模型预测和检查点的输出目录 |
| `overwrite_output_dir` | `bool` | `False` | 是否覆盖输出目录内容。可用于从检查点继续训练 |
| `run_name` | `str` | `output_dir` | 运行描述符，通常用于 wandb、mlflow 等日志记录 |

### 🎯 训练控制

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `do_train` | `bool` | `False` | 是否执行训练。不由 `Trainer` 直接使用，由训练脚本使用 |
| `do_eval` | `bool` | `None` | 是否在验证集上运行评估。如果 `eval_strategy` 不是 `"no"` 则自动设为 `True` |
| `do_predict` | `bool` | `False` | 是否在测试集上运行预测 |
| `num_train_epochs` | `float` | `3.0` | 总训练轮数。如果不是整数，会在最后一轮执行百分比部分后停止 |
| `max_steps` | `int` | `-1` | 如果为正数，覆盖 `num_train_epochs`，执行指定的训练步数 |
| `resume_from_checkpoint` | `str` | `None` | 从指定路径的检查点恢复训练 |


`max_steps` ：训练过程中总共要进行的优化器参数更新的最大次数（也叫训练步数、global steps）
- `epoch-based training`（基于轮数）
```
max_steps = ceil(num_train_epochs * num_update_steps_per_epoch)
```
- `step-based training`（基于步数）
```python
num_train_epochs = 3
train_dataset_size = 10000
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
num_devices = 1

total_train_batch_size = 8 * 1 * 2 = 16
num_update_steps_per_epoch = floor(10000 / 16) = 625
max_steps = 3 * 625 = 1875
```

> Trainer 内部会判断用了哪种： `epoch_based = args.max_steps <= 0`


### 📊 批次和梯度设置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `per_device_train_batch_size` | `int` | `8` | 每设备的训练批次大小。全局批次大小 = `per_device_train_batch_size * 设备数量` |
| `gradient_accumulation_steps` | `int` | `1` | 执行反向/更新前累积梯度的步数 |
| `per_device_eval_batch_size` | `int` | `8` | 每设备的评估批次大小 |
| `eval_accumulation_steps` | `int` | `None` | 评估时累积输出张量的步数，然后将结果移动到 CPU |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪的最大范数 |


### 📈 评估策略

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `eval_strategy` | `str` 或 `IntervalStrategy` | `"no"` | 评估策略：<br>• `"no"`: 训练期间不评估<br>• `"steps"`: 每 `eval_steps` 步评估<br>• `"epoch"`: 每轮结束时评估 |
| `eval_steps` | `int` 或 `float` | `None` | 当 `eval_strategy="steps"` 时，两次评估之间的步数 |
| `eval_delay` | `float` | `None` | 首次评估前等待的轮数或步数（取决于 `eval_strategy`） |
| `eval_on_start` | `bool` | `False` | 训练前是否执行评估步骤（健全性检查） |
| `eval_do_concat_batches` | `bool` | `True` | 是否递归连接批次间的输入/损失/标签/预测 |
| `eval_use_gather_object` | `bool` | `False` | 是否在嵌套列表/元组/字典中递归收集所有设备的对象 |
| `prediction_loss_only` | `bool` | `False` | 执行评估和预测时仅返回损失 |
| `batch_eval_metrics` | `bool` | `False` | 是否在每个批次结束时调用 compute_metrics 累积统计信息 |

### 🔧 优化器配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `learning_rate` | `float` | `5e-5` | AdamW 优化器的初始学习率 |
| `weight_decay` | `float` | `0.0` | 应用于除偏置和 LayerNorm 外所有层的权重衰减 |
| `adam_beta1` | `float` | `0.9` | AdamW 优化器的 beta1 超参数 |
| `adam_beta2` | `float` | `0.999` | AdamW 优化器的 beta2 超参数 |
| `adam_epsilon` | `float` | `1e-8` | AdamW 优化器的 epsilon 超参数 |
| `optim` | `str` 或 `OptimizerNames` | `"adamw_torch"` | 优化器类型，如 "adamw_torch"、"adafactor" 等 |
| `optim_args` | `str` | `None` | 优化器的可选参数 |
| `optim_target_modules` | `str` 或 `list[str]` | `None` | 要优化的目标模块，目前用于 GaLore 和 APOLLO 算法 |

### 📅 学习率调度

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `lr_scheduler_type` | `str` 或 `SchedulerType` | `"linear"` | 学习率调度器类型 |
| `lr_scheduler_kwargs` | `dict` | `{}` | 学习率调度器的额外参数 |
| `warmup_ratio` | `float` | `0.0` | 用于从 0 到 `learning_rate` 线性预热的总训练步数比例 |
| `warmup_steps` | `int` | `0` | 用于从 0 到 `learning_rate` 线性预热的步数，覆盖 `warmup_ratio` |

### 📝 日志记录

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `logging_dir` | `str` | `None` | TensorBoard 日志目录。默认为 `output_dir/runs/**CURRENT_DATETIME_HOSTNAME***` |
| `logging_strategy` | `str` 或 `IntervalStrategy` | `"steps"` | 日志策略：`"no"`、`"epoch"` 或 `"steps"` |
| `logging_steps` | `int` 或 `float` | `500` | 当 `logging_strategy="steps"` 时，两次日志之间的更新步数 |
| `logging_first_step` | `bool` | `False` | 是否记录第一个 `global_step` |
| `logging_nan_inf_filter` | `bool` | `True` | 是否过滤 `nan` 和 `inf` 损失的日志记录 |
| `log_level` | `str` | `"passive"` | 主进程的日志级别：'debug'、'info'、'warning'、'error'、'critical'、'passive' |
| `log_level_replica` | `str` | `"warning"` | 副本进程的日志级别 |
| `log_on_each_node` | `bool` | `True` | 多节点分布式训练时是否每个节点都记录日志 |
| `report_to` | `str` 或 `list[str]` | `"all"` | 报告结果的集成平台：'azure_ml'、'clearml'、'wandb'、'tensorboard' 等 |
| `project` | `str` | `"huggingface"` | 用于日志记录的项目名称 |

### 💾 检查点保存

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `save_strategy` | `str` 或 `SaveStrategy` | `"steps"` | 保存策略：<br>• `"no"`: 不保存<br>• `"epoch"`: 每轮结束时保存<br>• `"steps"`: 每 `save_steps` 步保存<br>• `"best"`: 当达到新的最佳指标时保存 |
| `save_steps` | `int` 或 `float` | `500` | 当 `save_strategy="steps"` 时，两次保存之间的更新步数 |
| `save_total_limit` | `int` | `None` | 限制检查点总数，删除较旧的检查点 |
| `save_safetensors` | `bool` | `True` | 是否使用 safetensors 保存状态字典 |
| `save_on_each_node` | `bool` | `False` | 多节点分布式训练时是否在每个节点保存模型 |
| `save_only_model` | `bool` | `False` | 是否只保存模型，不包括优化器、调度器和 RNG 状态 |

### 🎛️ 模型和数据处理

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `remove_unused_columns` | `bool` | `True` | 是否自动移除模型前向方法未使用的列 |
| `label_names` | `list[str]` | `None` | 输入字典中对应标签的键名列表 |
| `data_seed` | `int` | `None` | 数据采样器的随机种子 |
| `dataloader_drop_last` | `bool` | `False` | 是否丢弃最后一个不完整的批次 |
| `dataloader_num_workers` | `int` | `0` | 数据加载的子进程数量 |
| `dataloader_pin_memory` | `bool` | `True` | 是否在数据加载器中固定内存 |
| `dataloader_persistent_workers` | `bool` | `False` | 数据加载器是否保持工作进程活跃 |
| `dataloader_prefetch_factor` | `int` | `None` | 每个工作进程预加载的批次数 |
| `group_by_length` | `bool` | `False` | 是否将大致相同长度的样本分组（减少填充） |
| `length_column_name` | `str` | `"length"` | 预计算长度的列名 |

### 🚀 性能优化

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `torch_compile` | `bool` | `False` | 是否使用 PyTorch 2.0 的 `torch.compile` 编译模型 |
| `torch_compile_backend` | `str` | `None` | `torch.compile` 使用的后端 |
| `torch_compile_mode` | `str` | `None` | `torch.compile` 使用的模式 |
| `torch_empty_cache_steps` | `int` | `None` | 调用 `torch.<device>.empty_cache()` 前等待的步数 |
| `jit_mode_eval` | `bool` | `False` | 是否对推理使用 PyTorch jit trace |
| `gradient_checkpointing` | `bool` | `False` | 是否使用梯度检查点节省内存 |
| `gradient_checkpointing_kwargs` | `dict` | `None` | 传递给 `gradient_checkpointing_enable` 方法的关键字参数 |
| `auto_find_batch_size` | `bool` | `False` | 是否通过指数衰减自动找到适合内存的批次大小 |
| `use_liger_kernel` | `bool` | `False` | 是否启用 Liger Kernel 进行 LLM 模型训练 |
| `liger_kernel_config` | `dict` | `None` | Liger Kernel 的配置字典 |

### 🔢 精度和数据类型

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `bf16` | `bool` | `False` | 是否使用 bf16 16位（混合）精度训练 |
| `fp16` | `bool` | `False` | 是否使用 fp16 16位（混合）精度训练 |
| `fp16_opt_level` | `str` | `"O1"` | fp16 训练的 Apex AMP 优化级别：'O0'、'O1'、'O2'、'O3' |
| `half_precision_backend` | `str` | `"auto"` | 混合精度训练的后端：`"auto"`、`"apex"`、`"cpu_amp"` |
| `bf16_full_eval` | `bool` | `False` | 评估时是否使用完整的 bfloat16 |
| `fp16_full_eval` | `bool` | `False` | 评估时是否使用完整的 float16 |
| `tf32` | `bool` | `None` | 是否启用 TF32 模式（适用于 Ampere 及更新的 GPU 架构） |

**浮点数精度类型简介**

| 精度类型               | 位数                   | 名称                   | 数值范围       | 精度                   | 常用用途               |
| ------------------ | -------------------- | -------------------- | ---------- | -------------------- | ------------------ |
| `float32`          | 32位                  | 单精度浮点数               | 大（±1e38）   | 高                    | 默认全精度训练            |
| `float16`（`fp16`）  | 16位                  | 半精度浮点数               | ±6.55e4    | 中                    | 混合精度训练（NVIDIA AMP） |
| `bfloat16`（`bf16`） | 16位                  | Brain Floating Point | ±3.39e38   | 中高（指数精度和 float32 一样） | TPU、Ampere GPU推荐   |
| `tf32`             | 19位（10-bit mantissa） | TensorFloat-32       | 类似 float32 | 中高                   | 用于矩阵乘法，训练加速        |

**精度对比**

| 精度      | 存储效率 | 数值范围            | 精度（小数位） | 训练稳定性       | 是否适合大模型     |
| ------- | ---- | --------------- | ------- | ----------- | ----------- |
| float32 | 🟥 差 | 🟩 非常大          | 🟩 高    | 🟩 稳定       | ✅ 是         |
| fp16    | 🟩 高 | 🟧 较小           | 🟥 低    | 🟥 不稳定（需技巧） | ✅ 配合 AMP 使用 |
| **bf16**    | 🟩 高 | 🟩 和 float32 一样 | 🟧 较好   | 🟩 较稳定      | ✅ 推荐大模型训练   |
| tf32    | 🟩 高 | 🟩 和 float32 一样 | 🟧 中    | 🟩 稳定       | ✅ 加速大矩阵乘法   |


### 🌐 分布式训练

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `local_rank` | `int` | `-1` | 分布式训练过程中的进程排名 |
| `ddp_backend` | `str` | `None` | 分布式训练的后端：`"nccl"`、`"mpi"`、`"ccl"`、`"gloo"`、`"hccl"` |
| `ddp_find_unused_parameters` | `bool` | `None` | 传递给 `DistributedDataParallel` 的 `find_unused_parameters` 标志 |
| `ddp_bucket_cap_mb` | `int` | `None` | 传递给 `DistributedDataParallel` 的 `bucket_cap_mb` 标志 |
| `ddp_broadcast_buffers` | `bool` | `None` | 传递给 `DistributedDataParallel` 的 `broadcast_buffers` 标志 |
| `ddp_timeout` | `int` | `1800` | `torch.distributed.init_process_group` 调用的超时时间 |
| `fsdp` | `bool`、`str` 或 `list` | `[]` | PyTorch 全分片数据并行训练配置 |
| `fsdp_config` | `str` 或 `dict` | `None` | FSDP 的配置文件或字典 |
| `deepspeed` | `str` 或 `dict` | `None` | DeepSpeed 配置文件或字典 |
| `accelerator_config` | `str`、`dict` 或 `AcceleratorConfig` | `None` | 内部 `Accelerator` 实现的配置 |
| `parallelism_config` | `ParallelismConfig` | `None` | 训练运行的并行配置 |

### 🎯 设备和加速器

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_cpu` | `bool` | `False` | 是否使用 CPU。如果为 `False`，将使用可用的 cuda 或 mps 设备 |
| `tpu_num_cores` | `int` | `None` | TPU 训练时的 TPU 核心数 |
| `use_mps_device` | `bool` | `False` | **已弃用**：mps 设备将像 cuda 设备一样在可用时自动使用 |

### 🎲 随机性和可重现性

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `seed` | `int` | `42` | 训练开始时设置的随机种子 |
| `full_determinism` | `bool` | `False` | 是否调用 `enable_full_determinism` 确保分布式训练的可重现结果 |

### 🏆 最佳模型选择

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `load_best_model_at_end` | `bool` | `False` | 是否在训练结束时加载找到的最佳模型 |
| `metric_for_best_model` | `str` | `None` | 用于比较不同模型的指标名称 |
| `greater_is_better` | `bool` | `None` | 指标是否越大越好 |

### 🌍 Hub 集成

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `push_to_hub` | `bool` | `False` | 是否在每次保存模型时推送到 Hub |
| `hub_model_id` | `str` | `None` | 要与本地 `output_dir` 保持同步的仓库名称 |
| `hub_strategy` | `str` 或 `HubStrategy` | `"every_save"` | 定义推送到 Hub 的范围和时机 |
| `hub_token` | `str` | `None` | 推送模型到 Hub 的令牌 |
| `hub_private_repo` | `bool` | `None` | 是否使仓库私有 |
| `hub_always_push` | `bool` | `False` | 是否总是推送检查点 |
| `hub_revision` | `str` | `None` | 推送到 Hub 时使用的版本 |

### 🐛 调试和诊断

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `debug` | `str` 或 `list[DebugOption]` | `""` | 启用一个或多个调试功能 |
| `skip_memory_metrics` | `bool` | `True` | 是否跳过内存分析器报告 |
| `disable_tqdm` | `bool` | `None` | 是否禁用 tqdm 进度条 |
| `past_index` | `int` | `-1` | 用于过去隐藏状态的输出索引（适用于 TransformerXL、XLNet 等模型） |
| `torchdynamo` | `str` | `None` | TorchDynamo 的后端编译器 |
| `ray_scope` | `str` | `"last"` | 使用 Ray 进行超参数搜索时的作用域 |

### 📊 指标计算

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `include_for_metrics` | `list[str]` | `[]` | 在 `compute_metrics` 函数中包含的额外数据 |
| `label_smoothing_factor` | `float` | `0.0` | 标签平滑因子，0 表示不进行标签平滑 |
| `ignore_data_skip` | `bool` | `False` | 恢复训练时是否跳过数据加载的阶段 |

### 🚀 高级功能

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `neftune_noise_alpha` | `float` | `None` | NEFTune 噪声嵌入的 alpha 值，可显著提高指令微调性能 |
| `include_tokens_per_second` | `bool` | `False` | 是否计算每设备每秒的令牌数 |
| `include_num_input_tokens_seen` | `bool` | `None` | 是否跟踪训练过程中看到的输入令牌数量 |
| `average_tokens_across_devices` | `bool` | `True` | 是否跨设备平均令牌数 |
| `restore_callback_states_from_checkpoint` | `bool` | `False` | 是否从检查点恢复回调状态 |

### 🌐 日志集成平台

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `trackio_space_id` | `str` | `"trackio"` | 使用 Trackio 时部署的 Hugging Face Space ID |

## 使用示例

```python
from transformers import TrainingArguments

# 基本配置
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 高级配置（混合精度、分布式训练）
training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,  # 混合精度训练
    dataloader_num_workers=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    max_grad_norm=1.0,
    max_steps=-1,
    num_train_epochs=5,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    logging_steps=100,
    report_to=["tensorboard", "wandb"],
    seed=42,
    data_seed=42,
)
```

## 注意事项

1. **梯度累积**: 使用梯度累积时，一步计为一次反向传递，因此日志记录、评估、保存将每 `gradient_accumulation_steps * xxx_step` 个训练样本执行一次。

2. **内存管理**: `torch_empty_cache_steps` 可以通过降低峰值 VRAM 使用来避免 CUDA 内存不足错误，但性能会降低约 10%。

3. **分布式训练**: 使用 `fsdp` 或 `deepspeed` 时，确保模型在初始化 `TrainingArguments` 之后才初始化。

4. **检查点恢复**: 当 `load_best_model_at_end=True` 时，`save_strategy` 需要与 `eval_strategy` 相同。

5. **Hub 推送**: 如果 `output_dir` 已存在，它需要是目标仓库的本地克隆。

## 并行计算案例

🧮 一、训练参数设定

| 参数                            | 值                      |
| ----------------------------- | ---------------------- |
| 总样本数 (`examples`)             | 30000                  |
| GPU 数 (`n_gpus`)              | 3                      |
| `per_device_train_batch_size` | 8                      |
| `gradient_accumulation_steps` | 2                      |
| 并行策略                          | Deepspeed ZeRO Stage 3 |


✅ 二、计算 global effective batch size

```python
effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × n_gpus

effective_batch_size = 8 × 2 × 3 = 48
steps = total_examples / effective_batch_size
      = 30000 / 48 ≈ 625（取整，Trainer 会向上取整）
```

🧠 三、GPU 每张卡上的梯度和反向传播行为（ZeRO Stage 3）

ZeRO Stage 3 的核心逻辑

| 特征       | 描述                                                                     |
| -------- | ---------------------------------------------------------------------- |
| 参数分片     | 模型的所有参数都被切片分布在各个 GPU 上，每个 GPU 只存自己负责的那部分参数                             |
| 优化器状态也分片 | 如 Adam 的 `m`, `v` 也被分片存储                                               |
| 梯度分片     | 每个 GPU **只负责自己那部分参数的梯度**，不需要同步整个模型的梯度                                  |
| 通信模式     | 在反向传播时，使用 **Reduce-Scatter**，然后在参数更新后使用 **All-Gather** 来重建模型参数（用于前向计算） |

假设在训练第 `i` 次 step，下面是每张卡的工作流程：

| 阶段                | 每张 GPU 上发生的事                                             |
| ----------------- | -------------------------------------------------------- |
| 🔄 **前向传播**       | 每张 GPU 用 all-gather 重构模型参数（只在需要时），然后跑 forward pass       |
| 🧮 **计算 Loss**    | 各 GPU 独立计算自己 batch 的 loss                                |
| 🔁 **反向传播**       | 每张 GPU **只计算并持有自己负责参数部分的梯度**。这时用 Reduce-Scatter 通信技术聚合梯度 |
| ⏳ **梯度累积（如果有）**   | 累积梯度直到 `gradient_accumulation_steps` 次                   |
| 🔧 **参数更新（step）** | 各 GPU 分别更新自己负责的那部分参数（局部 optimizer step）                  |
| 📦 **释放内存**       | 清除不再需要的中间状态，节省显存                                         |
