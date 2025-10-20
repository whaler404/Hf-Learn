# 支持的优化器

`OptimizerNames` 枚举类存储了 HuggingFace Transformers 中所有可用的优化器标识符。这些优化器涵盖了从经典的 AdamW 到最新的内存高效优化器。

## 概述

优化器是深度学习训练过程中的核心组件，负责根据计算出的梯度更新模型参数。Transformers 库提供了丰富的优化器选择，适用于不同的训练场景和硬件条件。

## 优化器分类

### 🔥 AdamW 系列优化器

#### 标准实现
| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **PyTorch AdamW** | `"adamw_torch"` | PyTorch 原生 AdamW 实现 | 稳定可靠，兼容性最好 |
| **PyTorch Fused AdamW** | `"adamw_torch_fused"` | PyTorch 融合版本的 AdamW | 更快的计算速度，CUDA 优化 |
| **PyTorch XLA AdamW** | `"adamw_torch_xla"` | 支持 TPU 的 AdamW | 适用于 TPU 训练 |
| **NPU Fused AdamW** | `"adamw_torch_npu_fused"` | 支持 NPU 的融合 AdamW | 适用于华为 NPU |

#### 高精度和内存优化版本
| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **Apex Fused AdamW** | `"adamw_apex_fused"` | NVIDIA Apex 融合版本 | Apex 库提供，性能优异 |
| **AnyPrecision AdamW** | `"adamw_anyprecision"` | 支持任意精度的 AdamW | 可自定义不同部分的精度 |
| **AdamW 4-bit** | `"adamw_torch_4bit"` | 4-bit 量化版本 | 极大减少内存占用 |
| **AdamW 8-bit** | `"adamw_torch_8bit"` | 8-bit 量化版本 | 平衡内存和精度 |
| **BitsAndBytes 8-bit AdamW** | `"adamw_bnb_8bit"` | BnB 实现的 8-bit AdamW | 与 BnB 生态集成 |

### 🐯 Lion 优化器系列

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **Lion 32-bit** | `"lion_32bit"` | 标准精度 Lion 优化器 | 内存效率高，性能好 |
| **Lion 8-bit** | `"lion_8bit"` | 8-bit Lion 优化器 | 进一步节省内存 |
| **Paged Lion 32-bit** | `"paged_lion_32bit"` | 分页版本的 Lion | 大模型训练友好 |
| **Paged Lion 8-bit** | `"paged_lion_8bit"` | 分页 8-bit Lion | 结合内存和性能优化 |

### 🍯 AdEMAMix 优化器系列

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **AdEMAMix** | `"ademamix"` | 混合动量优化器 | 结合多种动量策略 |
| **AdEMAMix 8-bit** | `"ademamix_8bit"` | 8-bit 版本 | 内存优化版本 |
| **Paged AdEMAMix 32-bit** | `"paged_ademamix_32bit"` | 分页版本 | 大模型训练优化 |
| **Paged AdEMAMix 8-bit** | `"paged_ademamix_8bit"` | 分页 8-bit 版本 | 综合优化版本 |

### 📄 Paged 优化器系列

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **Paged AdamW 32-bit** | `"paged_adamw_32bit"` | 分页版本 AdamW | 适合大模型，避免 OOM |
| **Paged AdamW 8-bit** | `"paged_adamw_8bit"` | 分页 8-bit AdamW | 内存高效的大模型训练 |

### 🌟 GaLore (Gradient Low-Rank Projection) 优化器

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **GaLore AdamW** | `"galore_adamw"` | 梯度低秩投影 AdamW | 减少内存使用，保持性能 |
| **GaLore AdamW 8-bit** | `"galore_adamw_8bit"` | 8-bit GaLore AdamW | 进一步内存优化 |
| **GaLore Adafactor** | `"galore_adafactor"` | GaLore + Adafactor | 双重内存优化 |
| **GaLore AdamW Layerwise** | `"galore_adamw_layerwise"` | 逐层 GaLore AdamW | 更精细的内存控制 |
| **GaLore AdamW 8-bit Layerwise** | `"galore_adamw_8bit_layerwise"` | 逐层 8-bit GaLore | 最优内存效率 |
| **GaLore Adafactor Layerwise** | `"galore_adafactor_layerwise"` | 逐层 GaLore Adafactor | 内存优化逐层版本 |

### 🚀 APOLLO 优化器系列

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **APOLLO AdamW** | `"apollo_adamw"` | APOLLO 优化器 | 高效的二阶优化方法 |
| **APOLLO AdamW Layerwise** | `"apollo_adamw_layerwise"` | 逐层 APOLLO | 更精确的优化控制 |

### 🔬 传统优化器

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **SGD** | `"sgd"` | 随机梯度下降 | 简单经典，适合简单任务 |
| **Adagrad** | `"adagrad"` | 自适应梯度算法 | 适合稀疏数据 |
| **Adafactor** | `"adafactor"` | 内存高效的自适应优化器 | Google 开发，适合大模型 |
| **RMSprop** | `"rmsprop"` | 均方根传播 | 适合 RNN 和语音任务 |
| **RMSprop BnB** | `"rmsprop_bnb"` | BitsAndBytes RMSprop | 内存优化版本 |
| **RMSprop 8-bit** | `"rmsprop_bnb_8bit"` | 8-bit RMSprop | 极简内存占用 |
| **RMSprop 32-bit** | `"rmsprop_bnb_32bit"` | 32-bit BnB RMSprop | 标准精度 BnB 版本 |

### 🧪 实验性和专用优化器

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **LoMo** | `"lomo"` | 低内存优化器 | 专为内存受限环境设计 |
| **AdaLoMo** | `"adalomo"` | 自适应 LoMo | 智能内存管理 |
| **GrokAdamW** | `"grokadamw"` | Grok 优化器变种 | 基于最新研究成果 |
| **Stable AdamW** | `"stable_adamw"` | 稳定版 AdamW | 数值稳定性优化 |

### ⏰ Schedule-Free 优化器系列

| 优化器名称 | 标识符 | 描述 | 特点 |
|------------|--------|------|------|
| **Schedule-Free RAdam** | `"schedule_free_radam"` | 无调度器 RAdam | 内置学习率调度 |
| **Schedule-Free AdamW** | `"schedule_free_adamw"` | 无调度器 AdamW | 简化训练配置 |
| **Schedule-Free SGD** | `"schedule_free_sgd"` | 无调度器 SGD | 自动学习率调整 |

## 使用示例

### 基本使用

```python
from transformers import TrainingArguments, Trainer

# 使用默认的 AdamW 优化器
training_args = TrainingArguments(
    output_dir="./results",
    optim="adamw_torch",  # 默认值
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
```

### 内存高效优化器

```python
# 8-bit 量化优化器（适合大模型）
training_args = TrainingArguments(
    output_dir="./results",
    optim="adamw_8bit",  # 8-bit AdamW
    learning_rate=5e-5,
    weight_decay=0.01,
)

# GaLore 优化器（梯度低秩投影）
training_args = TrainingArguments(
    output_dir="./results",
    optim="galore_adamw",  # 需要 galore_torch 包
    learning_rate=5e-5,
    optim_target_modules=["q_proj", "v_proj", "k_proj"],  # 目标模块
)
```

### 高性能优化器

```python
# 融合版本优化器（CUDA 优化）
training_args = TrainingArguments(
    output_dir="./results",
    optim="adamw_torch_fused",  # 融合 AdamW
    learning_rate=5e-5,
    fp16=True,  # 配合混合精度
)

# Lion 优化器（新架构）
training_args = TrainingArguments(
    output_dir="./results",
    optim="lion_32bit",  # Lion 优化器
    learning_rate=1e-4,  # Lion 通常需要更高学习率
    weight_decay=0.1,
)
```

### 分页优化器（大模型训练）

```python
# 分页优化器（避免 OOM）
training_args = TrainingArguments(
    output_dir="./results",
    optim="paged_adamw_8bit",  # 分页 8-bit AdamW
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # 小批次
    gradient_accumulation_steps=32,  # 梯度累积
)
```

## 依赖要求

不同优化器可能需要安装额外的包：

```bash
# 8-bit 优化器
pip install bitsandbytes

# GaLore 优化器
pip install git+https://github.com/jiaweizzhao/GaLore

# APOLLO 优化器
pip install apollo-torch

# AnyPrecision 优化器
pip install git+https://github.com/pytorch/torchdistx

# GrokAdamW 优化器
pip install torch-optimi
```

## 选择指南

### 根据模型大小选择
- **小模型 (< 1B 参数)**: `adamw_torch`, `adamw_torch_fused`
- **中等模型 (1B-7B 参数)**: `adamw_8bit`, `lion_32bit`
- **大模型 (7B+ 参数)**: `galore_adamw`, `paged_adamw_8bit`

### 根据硬件条件选择
- **充足 GPU 内存**: `adamw_torch_fused`, `adamw_apex_fused`
- **有限 GPU 内存**: `adamw_8bit`, `lion_8bit`
- **极度内存受限**: `galore_adamw_8bit_layerwise`, `lomo`

### 根据任务类型选择
- **通用任务**: `adamw_torch`, `adamw_torch_fused`
- **大语言模型**: `galore_adamw`, `adamw_8bit`
- **计算机视觉**: `adamw_torch`, `lion_32bit`
- **推荐系统/稀疏数据**: `adagrad`

## 性能对比

### 内存使用（从低到高）
1. `galore_adamw_8bit_layerwise` - 最低
2. `adamw_8bit`, `lion_8bit`
3. `galore_adamw`, `paged_adamw_8bit`
4. `adamw_torch_8bit`
5. `lion_32bit`, `ademamix`
6. `adamw_torch`, `adamw_torch_fused`

### 训练速度（从快到慢）
1. `adamw_torch_fused`, `adamw_apex_fused` - 最快
2. `adamw_torch`
3. `lion_32bit`, `lion_8bit`
4. `adamw_8bit`
5. `galore_adamw`
6. `ademamix`, `apollo_adamw`

### 收敛性能（一般排序）
1. `adamw_torch`, `adamw_torch_fused` - 最稳定
2. `ademamix` - 收敛性好
3. `galore_adamw` - 内存效率高且性能好
4. `lion_32bit` - 新架构，表现优异
5. `adamw_8bit` - 略有精度损失

## 最佳实践

1. **默认选择**: 从 `adamw_torch` 开始，这是最稳定的选择
2. **内存不足**: 尝试 `adamw_8bit` 或 `galore_adamw`
3. **追求速度**: 使用 `adamw_torch_fused`（如果有 CUDA）
4. **大模型训练**: 优先考虑 `galore_adamw` 系列或 `paged_adamw_8bit`
5. **实验新方法**: 可以尝试 `lion_32bit` 或 `ademamix`

## 注意事项

- 8-bit 优化器需要 `bitsandbytes >= 0.41.1` 版本以避免已知 bug
- GaLore 和 APOLLO 优化器需要额外安装相应包
- 分页优化器适合大模型但可能有轻微性能开销
- 某些优化器对学习率敏感，需要相应调整
- 混合精度训练与量化优化器配合使用效果最佳