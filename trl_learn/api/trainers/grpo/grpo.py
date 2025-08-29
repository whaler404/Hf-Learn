# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import os
# 强制使用CPU，禁用CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置PyTorch使用CPU
# torch.cuda.is_available = lambda: False

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny", torch_dtype=torch.float32)#, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
dataset = load_dataset("trl-lib/tldr", split="train").shuffle(seed=42).select(range(20))

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir="./trainer_output/qwen2-tiny-GRPO",
    # no_cuda=True,  # 禁用CUDA
    # use_cpu=True,  # 强制使用CPU
    # dataloader_pin_memory=False,  # 禁用pin memory（GPU特性）
    # fp16=False,  # 禁用半精度（GPU优化）
    # bf16=False,  # 禁用bfloat16（GPU优化）
    num_train_epochs=1,  # 减少训练轮数
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()