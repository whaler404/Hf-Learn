# 模型卡和 Hub 集成方法

## `init_hf_repo()`

初始化 Hugging Face Hub 仓库。

### 参数
- **token** (`str`, 可选): Hugging Face 访问令牌

### 说明
- 在 Hugging Face Hub 上创建或连接仓库
- 自动配置本地 Git 仓库
- 处理认证和权限设置
- 准备模型上传环境

### 示例
```python
# 使用本地认证初始化仓库
trainer.init_hf_repo()

# 使用显式令牌初始化仓库
trainer.init_hf_repo(token="hf_your_token_here")

# 仓库会在 args.output_dir 中创建
# 可以通过 trainer.hub_model_id 访问仓库ID
```

## `create_model_card()`

创建模型卡文档。

### 参数
- **language** (`str`, 可选): 模型语言
- **license** (`str`, 可选): 模型许可证
- **tags** (`list[str]`, 可选): 模型标签
- **model_name** (`str`, 可选): 模型名称
- **finetuned_from** (`str`, 可选): 微调的基础模型
- **tasks** (`list[str]`, 可选): 模型任务
- **dataset_tags** (`list[str]`, 可选): 数据集标签
- **dataset** (`str`, 可选): 数据集名称
- **dataset_args** (`list[str]`, 可选): 数据集参数
- **dataset_language** (`list[str]`, 可选): 数据集语言

### 返回值
- `ModelCard`: 创建的模型卡对象

### 说明
- 自动生成包含训练信息的模型卡
- 包含模型架构、训练参数、性能指标等
- 支持自定义内容和格式
- 遵循 Hugging Face Hub 模型卡标准

### 示例
```python
# 创建基础模型卡
model_card = trainer.create_model_card()

# 创建带有详细信息的模型卡
model_card = trainer.create_model_card(
    language="zh",
    license="mit",
    tags=["text-generation", "chinese", "gpt"],
    model_name="My-Chinese-GPT",
    finetuned_from="gpt2",
    tasks=["text-generation"],
    dataset_tags=["wikitext"],
    dataset="Wikitext-103",
    dataset_args=["zh"],
    dataset_language=["zh"]
)

# 模型卡会自动保存到输出目录
print(f"模型卡已创建: {model_card}")
```

## `push_to_hub()`

将模型和文件推送到 Hugging Face Hub。

### 参数
- **commit_message** (`str`, 可选): 提交消息
- **commit_description** (`str`, 可选): 提交描述
- **tags** (`list[str]`, 可选): 提交标签
- **private** (`bool`, 可选): 是否创建私有仓库
- **token** (`str`, 可选): 访问令牌
- **create_repo** (`bool`, 可选): 是否创建仓库
- **revision** (`str`, 可选): 分支或标签
- **delete_existing_repo** (`bool`, 可选): 是否删除现有仓库
- **run_as_future** (`bool`, 可选): 是否在后台运行

### 说明
- 推送训练好的模型到 Hugging Face Hub
- 自动包含模型权重、配置文件和模型卡
- 支持增量推送和版本控制
- 处理大文件的分块上传

### 示例
```python
# 基础推送
trainer.push_to_hub()

# 带详细信息的推送
trainer.push_to_hub(
    commit_message="训练完成，添加模型卡",
    commit_description="这是在中文数据集上微调的 GPT 模型",
    tags=["text-generation", "chinese", "fine-tuned"],
    private=True,
    create_repo=True
)

# 在后台推送（非阻塞）
future = trainer.push_to_hub(run_as_future=True)
# 可以继续其他工作
print("模型正在后台推送...")

# 推送到特定分支
trainer.push_to_hub(revision="main")
```

## `create_accelerator_and_postprocess()`

创建加速器并进行后处理配置。

### 说明
- 初始化 Accelerate 库的加速器
- 配置分布式训练环境
- 设置混合精度训练
- 处理设备放置和数据并行
- 配置内存优化和梯度累积

### 功能包括
- **分布式训练**: DDP、FSDP、DeepSpeed 配置
- **混合精度**: fp16、bf16、自动精度选择
- **设备管理**: GPU、TPU、CPU 自动选择
- **内存优化**: 梯度检查点、卸载等
- **数据并行**: 自动配置数据加载器并行

### 示例
```python
# 在 Trainer 初始化时自动调用
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset
)
# create_accelerator_and_postprocess() 已被自动调用

# 访问加速器属性
print(f"设备数量: {trainer.accelerator.num_processes}")
print(f"当前设备: {trainer.accelerator.device}")
print(f"分布式类型: {trainer.accelerator.state.distributed_type}")
```

## `_push_from_checkpoint()`

从检查点目录推送文件到 Hub。

### 参数
- **checkpoint_folder** (`str`): 检查点目录路径

### 说明
- 专门用于从检查点目录推送文件
- 自动识别和推送相关的模型文件
- 保持检查点的完整性

### 示例
```python
# 推送特定检查点
trainer._push_from_checkpoint("./checkpoint-1000")
```

## `_finish_current_push()`

完成当前的推送操作。

### 说明
- 等待后台推送操作完成
- 处理推送结果和错误
- 清理临时文件和资源

### 示例
```python
# 等待后台推送完成
if hasattr(trainer, '_push_in_progress'):
    trainer._finish_current_push()
```

## Hub 集成完整示例

```python
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
import torch

# 配置 Hub 集成参数
args = TrainingArguments(
    output_dir="./my-chinese-gpt",
    push_to_hub=True,  # 启用 Hub 推送
    hub_model_id="username/my-chinese-gpt",  # Hub 仓库名称
    hub_private_repo=True,  # 私有仓库
    hub_strategy="every_save",  # 每次保存时推送
    # ... 其他训练参数
)

class HubIntegratedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保仓库已初始化
        if self.args.push_to_hub:
            self.init_hf_repo()

    def train(self, *args, **kwargs):
        """带 Hub 集成的训练"""
        print("开始训练并集成到 Hugging Face Hub")

        # 创建初始模型卡
        model_card = self.create_model_card(
            language="zh",
            tags=["text-generation", "chinese"],
            model_name="Chinese GPT Model",
            tasks=["text-generation"]
        )
        print("初始模型卡已创建")

        # 执行训练
        result = super().train(*args, **kwargs)

        # 训练完成后更新模型卡
        final_model_card = self.create_model_card(
            language="zh",
            license="mit",
            tags=["text-generation", "chinese", "fine-tuned"],
            model_name="Fine-tuned Chinese GPT",
            finetuned_from="gpt2",
            tasks=["text-generation"]
        )
        print("最终模型卡已更新")

        return result

    def evaluate(self, *args, **kwargs):
        """评估并推送结果"""
        results = super().evaluate(*args, **kwargs)

        # 如果启用了推送，推送评估结果
        if self.args.push_to_hub:
            # 可以创建包含评估结果的模型卡更新
            self.push_to_hub(
                commit_message=f"评估完成: loss={results.get('eval_loss', 'N/A'):.4f}",
                tags=["evaluation", "metrics"]
            )
            print(f"评估结果已推送到 Hub: {results}")

        return results

# 使用 Hub 集成训练器
model = AutoModelForCausalLM.from_pretrained("gpt2")
trainer = HubIntegratedTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 执行训练（自动推送到 Hub）
trainer.train()

# 手动推送最终模型
trainer.push_to_hub(
    commit_message="训练完成",
    tags=["completed", "production-ready"]
)

print(f"模型已推送到: https://huggingface.co/{trainer.hub_model_id}")
```

## Hub 集成最佳实践

### 1. 认证设置
```python
# 使用环境变量设置认证
import os
os.environ["HF_TOKEN"] = "your_token_here"

# 或使用 huggingface-cli 命令行工具
# !huggingface-cli login
```

### 2. 版本控制
```python
# 使用语义版本标签
trainer.push_to_hub(
    commit_message="Release v1.0.0",
    tags=["v1.0.0", "release"]
)

# 推送到开发分支
trainer.push_to_hub(
    revision="dev",
    commit_message="开发版本更新"
)
```

### 3. 大文件处理
```python
# 自动处理大文件（如 safetensors）
# Trainer 会自动使用 Git LFS 或分块上传
trainer.push_to_hub()
```

### 4. 批量操作
```python
# 批量推送多个模型
for checkpoint_dir in ["./checkpoint-500", "./checkpoint-1000"]:
    trainer._push_from_checkpoint(checkpoint_dir)
```