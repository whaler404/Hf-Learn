import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./trainer_output/test-grokadamw",
    max_steps=1000,
    per_device_train_batch_size=4,
    optim="grokadamw",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="grokadamw",
)