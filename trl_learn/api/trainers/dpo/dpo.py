# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os
# 强制使用CPU，禁用CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 设置PyTorch使用CPU
# torch.cuda.is_available = lambda: False

# print(torch.cuda.memory_summary())

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny")
tokenizer = AutoTokenizer.from_pretrained("wheeler404/qwen2-tiny")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train").shuffle(seed=42).select(range(20))

# print(torch.cuda.memory_summary())

training_args = DPOConfig(
    output_dir="./trainer_output/qwen2-tiny-DPO",
    # no_cuda=True,  # 禁用CUDA
    # use_cpu=True,  # 强制使用CPU
    # dataloader_pin_memory=False,  # 禁用pin memory（GPU特性）
    # fp16=False,  # 禁用半精度（GPU优化）
    # bf16=False,  # 禁用bfloat16（GPU优化）
    per_device_train_batch_size=1,  # 减小批次大小以适应显存
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    # 启用梯度检查点以节省内存
    gradient_checkpointing_kwargs={
        "gradient_checkpointing": True,  # 启用梯度检查点
    },
    optim="paged_adamw_32bit",
    num_train_epochs=1,  # 减少训练轮数
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=10,
)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()

# {
#   "loss": 0.6931,  // 当前训练步骤的损失值
#   "grad_norm": 13.987984657287598,  // 梯度的范数，用于监控梯度爆炸或消失
#   "learning_rate": 5e-05,  // 当前使用的学习率
#   "rewards": {
#     "chosen": 0.0,  // 选中样本的奖励值
#     "rejected": 0.0,  // 被拒绝样本的奖励值
#     "accuracies": 0.0,  // 奖励的准确率
#     "margins": 0.0  // 奖励的边距
#   },
#   "logps": {
#     "chosen": -5826.65283203125,  // 选中样本的对数概率
#     "rejected": -5899.97216796875  // 被拒绝样本的对数概率
#   },
#   "logits": {
#     "chosen": 5.2085193601669744e-05,  // 选中样本的原始模型输出值
#     "rejected": 5.028907617088407e-05  // 被拒绝样本的原始模型输出值
#   },
#   "epoch": 0.2  // 当前训练的轮次进度
# }