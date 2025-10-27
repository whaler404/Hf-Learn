import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torchvision.transforms as T

# 加载 beans 数据集（训练、验证、测试）
dataset = load_dataset("AI-Lab-Makerere/beans")

# 使用 ViT 的官方预处理器
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)

# 统一 transforms
transform = T.Compose([
    T.Resize(processor.size["height"]),
    T.CenterCrop(processor.size["height"]),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# 定义数据处理函数
def transform_examples(examples):
    examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["image"]]
    return examples

dataset = dataset.with_transform(transform_examples)

# DataLoader
def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["labels"] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset["validation"], batch_size=32, collate_fn=collate_fn)
