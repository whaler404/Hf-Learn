from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model = AutoModelForSequenceClassification.from_pretrained("wheeler404/qwen2-tiny", num_labels=2).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("wheeler404/qwen2-tiny")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for causal language models

# set the pad token to eos token for causal language models
model.config.pad_token_id = model.config.eos_token_id

from datasets import load_dataset
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
train_dataset = dataset["train"].shuffle(seed=42).select(range(100))

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./trainer_output/qwen2-tiny-glue",
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
