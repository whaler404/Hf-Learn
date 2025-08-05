from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./trainer_output/test-lomo",
    max_steps=1000,
    per_device_train_batch_size=4,
    optim="adalomo",#
    gradient_checkpointing=True,
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="adalomo",
)