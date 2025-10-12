from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./SmolLM2_135M",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    # model=model,
    args=training_args,
    # train_dataset=dataset["train"],
    # eval_dataset=dataset["test"],
    # processing_class=tokenizer,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

trainer.train()