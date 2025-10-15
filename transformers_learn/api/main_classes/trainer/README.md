# Trainer 类文档

Trainer 类是 🤗 Transformers 库中用于训练、评估和推理的核心类。它提供了一个简单但功能完整的训练和评估循环，专门为 PyTorch 优化，适用于 🤗 Transformers 模型。

## 主要功能模块

### 1. 初始化和配置 (Initialization & Configuration)
[初始化和配置](./initialization.md)
- `__init__()` - 初始化训练器
- `create_accelerator_and_postprocess()` - 创建加速器并进行后处理

### 2. 数据加载和处理 (Data Loading & Processing)
[数据加载和处理](./data_loading.md)
- `get_train_dataloader()` - 获取训练数据加载器
- `get_eval_dataloader()` - 获取评估数据加载器
- `get_test_dataloader()` - 获取测试数据加载器
- `_get_dataloader()` - 通用数据加载器创建方法
- `_remove_unused_columns()` - 移除未使用的列
- `_get_collator_with_removed_columns()` - 获取带有移除列功能的数据整理器

### 3. 优化器和调度器 (Optimizer & Scheduler)
[优化器和调度器](./optimizer_scheduler.md)
- `create_optimizer()` - 创建优化器
- `create_scheduler()` - 创建学习率调度器
- `create_optimizer_and_scheduler()` - 创建优化器和调度器
- `get_optimizer_cls_and_kwargs()` - 获取优化器类和参数
- `get_decay_parameter_names()` - 获取需要权重衰减的参数名

### 4. 训练循环 (Training Loop)
[训练循环](./training_loop.md)
- `train()` - 训练模型
- `training_step()` - 单步训练
- `_inner_training_loop()` - 内部训练循环
- `compute_loss()` - 计算损失

### 5. 评估和预测 (Evaluation & Prediction)
[评估和预测](./evaluation_prediction.md)
- `evaluate()` - 评估模型
- `predict()` - 进行预测
- `evaluation_loop()` - 评估循环
- `prediction_loop()` - 预测循环
- `prediction_step()` - 单步预测

### 6. 模型保存和加载 (Model Saving & Loading)
[模型保存和加载](./model_saving_loading.md)
- `save_model()` - 保存模型
- `_save_checkpoint()` - 保存检查点
- `_load_from_checkpoint()` - 从检查点加载
- `_save_optimizer_and_scheduler()` - 保存优化器和调度器
- `_load_optimizer_and_scheduler()` - 加载优化器和调度器

### 7. 回调管理 (Callback Management)
[回调管理](./callback_management.md)
- `add_callback()` - 添加回调
- `remove_callback()` - 移除回调
- `pop_callback()` - 弹出回调

### 8. 超参数搜索 (Hyperparameter Search)
[超参数搜索](./special_features.md)
- `hyperparameter_search()` - 超参数搜索
- `_hp_search_setup()` - 超参数搜索设置
- `_report_to_hp_search()` - 向超参数搜索报告

### 9. 设备和分布式训练 (Device & Distributed Training)
[设备和分布式训练](./special_features.md)
- `_move_model_to_device()` - 将模型移动到设备
- `_wrap_model()` - 包装模型
- `create_accelerator_and_postprocess()` - 创建加速器

### 10. 实用工具 (Utilities)
[实用工具](./utilities.md)
- `get_num_trainable_parameters()` - 获取可训练参数数量
- `get_learning_rates()` - 获取学习率
- `num_examples()` - 获取样本数量
- `num_tokens()` - 获取token数量
- `floating_point_ops()` - 浮点运算计算

### 11. 模型卡和Hub集成 (Model Card & Hub Integration)
[模型卡和Hub集成](./hub_integration.md)
- `create_model_card()` - 创建模型卡
- `push_to_hub()` - 推送到Hub
- `init_hf_repo()` - 初始化HF仓库

### 12. 特殊功能 (Special Features)
[特殊功能](./special_features.md)
- `_activate_neftune()` - 激活NEFTune
- `_deactivate_neftune()` - 停用NEFTune
- `torch_jit_model_eval()` - Torch JIT模型评估

## 方法详细说明

请查看各个子模块的文档以获取每个方法的详细说明、参数和使用示例。