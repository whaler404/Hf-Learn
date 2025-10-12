import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./trainer_output/test-apollo",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="apollo_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="apollo_adamw",
)


args = TrainingArguments(
    output_dir="./trainer_output/test-apollo_mini",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="apollo_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200",
)