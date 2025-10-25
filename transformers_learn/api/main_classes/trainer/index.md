# Trainer 类方法索引

## 概览

本文档提供了 🤗 Transformers 库中 `Trainer` 类的完整方法参考。`Trainer` 类是一个功能强大且易于使用的训练框架，专为 Transformer 模型优化。

核心方法
- [init](./initialization.md#参数)
- [_inner_training_loop](./training_loop.md#核心训练循环-lines-2578-2678)
    - [核心训练循环](training_loop.md#核心训练循环-lines-2578-2678)
        - [training_step](./training_loop.md#training_step)
            - [核心训练流程](./training_loop.md#核心训练流程-lines-4010-4020-4050-4073)
                - [compute_loss](./training_loop.md#compute_loss)
        - [梯度更新和训练完成](./training_loop.md#梯度更新和训练完成-lines-2692-2856)

### 快速开始

```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset

# 加载数据集和模型
dataset = load_dataset("imdb", split="train")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 配置训练参数
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_dir='./logs',
)

# 初始化训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()
```

## 方法分类

### 📚 [初始化和配置](./initialization.md)
- [`__init__()`](./initialization.md#__init__) - 初始化训练器
- [`create_accelerator_and_postprocess()`](./initialization.md#create_accelerator_and_postprocess) - 创建加速器
- [`tokenizer`](./initialization.md#tokenizer-属性-已弃用) - 分词器属性（已弃用）

### 📊 [数据加载和处理](./data_loading.md)
- [`get_train_dataloader()`](./data_loading.md#get_train_dataloader) - 获取训练数据加载器
- [`get_eval_dataloader()`](./data_loading.md#get_eval_dataloader) - 获取评估数据加载器
- [`get_test_dataloader()`](./data_loading.md#get_test_dataloader) - 获取测试数据加载器
- [`_get_dataloader()`](./data_loading.md#_get_dataloader) - 通用数据加载器创建
- [`_remove_unused_columns()`](./data_loading.md#_remove_unused_columns) - 移除未使用的列
- [`_get_collator_with_removed_columns()`](./data_loading.md#_get_collator_with_removed_columns) - 数据整理器包装器
- [`_set_signature_columns_if_needed()`](./data_loading.md#_set_signature_columns_if_needed) - 设置签名列
- [`_align_special_tokens()`](./data_loading.md#_align_special_tokens) - 对齐特殊标记

### ⚙️ [优化器和调度器](./optimizer_scheduler.md)
- [`create_optimizer()`](./optimizer_scheduler.md#create_optimizer) - 创建优化器
- [`create_scheduler()`](./optimizer_scheduler.md#create_scheduler) - 创建学习率调度器
- [`create_optimizer_and_scheduler()`](./optimizer_scheduler.md#create_optimizer_and_scheduler) - 创建优化器和调度器
- [`get_optimizer_cls_and_kwargs()`](./optimizer_scheduler.md#get_optimizer_cls_and_kwargs-静态方法) - 获取优化器类和参数
- [`get_decay_parameter_names()`](./optimizer_scheduler.md#get_decay_parameter_names) - 获取权重衰减参数名
- [`get_num_trainable_parameters()`](./optimizer_scheduler.md#get_num_trainable_parameters) - 获取可训练参数数量
- [`get_learning_rates()`](./optimizer_scheduler.md#get_learning_rates) - 获取学习率
- [`get_optimizer_group()`](./optimizer_scheduler.md#get_optimizer_group) - 获取优化器组
- [`num_examples()`](./optimizer_scheduler.md#num_examples) - 获取样本数量
- [`num_tokens()`](./optimizer_scheduler.md#num_tokens) - 获取 token 数量

### 🏃‍♂️ [训练循环](./training_loop.md)
- [`train()`](./training_loop.md#train) - 主训练入口点
- [`training_step()`](./training_loop.md#training_step) - 单步训练
- [`compute_loss()`](./training_loop.md#compute_loss) - 计算损失
- [`_inner_training_loop()`](./training_loop.md#_inner_training_loop) - 内部训练循环
- [`get_total_train_batch_size()`](./training_loop.md#get_total_train_batch_size) - 获取总训练批次大小
- [`get_tp_size()`](./training_loop.md#get_tp_size) - 获取张量并行大小
- [`set_initial_training_values()`](./training_loop.md#set_initial_training_values) - 设置初始训练值
- [`get_batch_samples()`](./training_loop.md#get_batch_samples) - 获取批次样本

### 📈 [评估和预测](./evaluation_prediction.md)
- [`evaluate()`](./evaluation_prediction.md#evaluate) - 评估模型
- [`predict()`](./evaluation_prediction.md#predict) - 进行预测
- [`evaluation_loop()`](./evaluation_prediction.md#evaluation_loop) - 评估循环
- [`prediction_loop()`](./evaluation_prediction.md#prediction_loop) - 预测循环
- [`prediction_step()`](./evaluation_prediction.md#prediction_step) - 单步预测
- [`_gather_and_numpify()`](./evaluation_prediction.md#_gather_and_numpify) - 收集并转换为 NumPy
- [`_nested_gather()`](./evaluation_prediction.md#_nested_gather) - 嵌套收集张量

### 💾 [模型保存和加载](./model_saving_loading.md)
- [`save_model()`](./model_saving_loading.md#save_model) - 保存模型
- [`_save_checkpoint()`](./model_saving_loading.md#_save_checkpoint) - 保存检查点
- [`_load_from_checkpoint()`](./model_saving_loading.md#_load_from_checkpoint) - 从检查点加载
- [`_save_optimizer_and_scheduler()`](./model_saving_loading.md#_save_optimizer_and_scheduler) - 保存优化器和调度器
- [`_load_optimizer_and_scheduler()`](./model_saving_loading.md#_load_optimizer_and_scheduler) - 加载优化器和调度器
- [`_save_rng_state()`](./model_saving_loading.md#_save_rng_state) - 保存随机数状态
- [`_load_rng_state()`](./model_saving_loading.md#_load_rng_state) - 加载随机数状态
- [`_load_best_model()`](./model_saving_loading.md#_load_best_model) - 加载最佳模型
- [`_sorted_checkpoints()`](./model_saving_loading.md#_sorted_checkpoints) - 获取排序的检查点列表
- [`_rotate_checkpoints()`](./model_saving_loading.md#_rotate_checkpoints) - 轮换检查点
- [`_save_scaler()`](./model_saving_loading.md#_save_scaler) - 保存混合精度缩放器
- [`_load_scaler()`](./model_saving_loading.md#_load_scaler) - 加载混合精度缩放器

### 🔄 [回调管理](./callback_management.md)
- [`add_callback()`](./callback_management.md#add_callback) - 添加回调
- [`remove_callback()`](./callback_management.md#remove_callback) - 移除回调
- [`pop_callback()`](./callback_management.md#pop_callback) - 弹出回调

### 🔍 [超参数搜索](./special_features.md#超参数搜索功能)
- [`hyperparameter_search()`](./special_features.md#hyperparameter_search) - 超参数搜索
- [`_hp_search_setup()`](./special_features.md#_hp_search_setup) - 超参数搜索设置
- [`_report_to_hp_search()`](./special_features.md#_report_to_hp_search) - 向超参数搜索报告

### 🖥️ [设备和分布式训练](./special_features.md#设备和分布式功能)
- [`_move_model_to_device()`](./special_features.md#_move_model_to_device) - 移动模型到设备
- [`_wrap_model()`](./special_features.md#_wrap_model) - 包装模型
- [`create_accelerator_and_postprocess()`](./special_features.md#create_accelerator_and_postprocess) - 创建加速器

### 🛠️ [实用工具](./utilities.md)
- [`is_local_process_zero()`](./utilities.md#is_local_process_zero) - 检查是否为本地主进程
- [`is_world_process_zero()`](./utilities.md#is_world_process_zero) - 检查是否为全局主进程
- [`log()`](./utilities.md#log) - 记录指标
- `_prepare_input()` - 准备单个输入
- `_prepare_inputs()` - 准备输入字典
- [`floating_point_ops()`](./utilities.md#floating_point_ops) - 计算浮点运算
- [`store_flos()`](./utilities.md#store_flos) - 存储 FLOPs

### 🤖 [特殊功能](./special_features.md)
- [`_activate_neftune()`](./special_features.md#_activate_neftune) - 激活 NEFTune
- [`_deactivate_neftune()`](./special_features.md#_deactivate_neftune) - 停用 NEFTune
- [`torch_jit_model_eval()`](./special_features.md#torch_jit_model_eval) - Torch JIT 模型评估
- [`autocast_smart_context_manager()`](./special_features.md#autocast_smart_context_manager) - 智能混合精度上下文
- [`compute_loss_context_manager()`](./special_features.md#compute_loss_context_manager) - 损失计算上下文
- [`_prepare_context_parallel_inputs()`](./special_features.md#_prepare_context_parallel_inputs) - 准备上下文并行输入

### 🌐 [模型卡和Hub集成](./hub_integration.md)
- [`create_model_card()`](./hub_integration.md#create_model_card) - 创建模型卡
- [`push_to_hub()`](./hub_integration.md#push_to_hub) - 推送到 Hub
- [`init_hf_repo()`](./hub_integration.md#init_hf_repo) - 初始化 HF 仓库
- [`create_accelerator_and_postprocess()`](./hub_integration.md#create_accelerator_and_postprocess) - 创建加速器
- [`_push_from_checkpoint()`](./hub_integration.md#_push_from_checkpoint) - 从检查点推送
- [`_finish_current_push()`](./hub_integration.md#_finish_current_push) - 完成推送

## 常用工作流程

### 1. 基础训练流程
```python
# 1. 初始化
trainer = Trainer(model=model, args=args, train_dataset=train_dataset)

# 2. 训练
trainer.train()

# 3. 评估
results = trainer.evaluate()

# 4. 保存
trainer.save_model()
```

### 2. 检查点恢复训练
```python
# 从检查点恢复
trainer.train(resume_from_checkpoint="./checkpoint-1000")

# 或从最新检查点恢复
trainer.train(resume_from_checkpoint=True)
```

### 3. 自定义训练循环
```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失计算"""
        outputs = model(**inputs)
        # 自定义损失逻辑
        loss = custom_loss_function(outputs.logits, inputs["labels"])
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """自定义训练步骤"""
        loss = super().training_step(model, inputs)
        # 添加自定义逻辑
        return loss

# 使用自定义训练器
trainer = CustomTrainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()
```

### 4. 超参数搜索
```python
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    n_trials=10,
    direction="minimize"
)
```

### 5. Hub 集成和分享
```python
# 配置 Hub 推送
args = TrainingArguments(
    output_dir="./results",
    push_to_hub=True,
    hub_model_id="username/my-model"
)

# 训练并自动推送
trainer.train()

# 手动推送最终模型
trainer.push_to_hub(commit_message="Training completed")
```

## 性能优化技巧

### 1. 内存优化
```python
# 使用梯度检查点
args.gradient_checkpointing = True

# 使用混合精度
args.fp16 = True  # 或 args.bf16 = True

# 启用优化器卸载（DeepSpeed）
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

### 2. 训练速度优化
```python
# 增加批次大小
args.per_device_train_batch_size = 32

# 使用数据加载器优化
args.dataloader_num_workers = 4
args.datloader_pin_memory = True

# 使用 fused AdamW
args.optim = "adamw_torch_fused"
```

### 3. 分布式训练
```python
# 多 GPU 训练
python -m torch.distributed.launch --nproc_per_node=4 train.py

# DeepSpeed 训练
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

## 故障排除

### 常见问题和解决方案

1. **内存不足**
   - 减小批次大小
   - 启用梯度累积
   - 使用混合精度训练
   - 启用梯度检查点

2. **训练速度慢**
   - 增加数据加载器工作进程
   - 使用更快的优化器
   - 启用自动混合精度
   - 检查 I/O 瓶颈

3. **收敛问题**
   - 调整学习率
   - 修改优化器参数
   - 检查数据质量
   - 调整批次大小

4. **分布式训练问题**
   - 检查网络连接
   - 确认所有节点环境一致
   - 检查 CUDA 版本兼容性

## 更多资源

- [Hugging Face 文档](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Accelerate 文档](https://huggingface.co/docs/accelerate/)
- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [Weights & Biases 集成](https://docs.wandb.ai/guides/integrations/huggingface)

---

*本文档基于 Transformers 4.x 版本。如有更新，请参考官方文档。*