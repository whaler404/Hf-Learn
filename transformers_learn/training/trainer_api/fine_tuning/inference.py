from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 加载训练好的 checkpoint
checkpoint_path = "./trainer_output/SmolLM2_rotten_tomatoes_classification/checkpoint-4500"  # 修改为你实际的 checkpoint 路径

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to("cuda:0")
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# 推理样例
texts = [
    "if you sometimes like to go to the movies to have fun , wasabi is a good place to start .",
    "simplistic , silly and tedious .",
    "a good movie for the whole family .",
    "a terrible movie that I would not recommend to anyone .",
]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1)

for text, pred in zip(texts, preds):
    print(f"文本: {text}\t预测类别: {pred.item()}")
