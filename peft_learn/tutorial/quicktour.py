import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("wheeler404/qwen2-tiny", num_labels=2)

from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("wheeler404/qwen2-tiny")
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.use_flash_attn = True

# import torchinfo

# torchinfo.summary(model)

# print(model)

model.push_to_hub(
    repo_id="qwen2-tiny-lora",
    commit_message="Add Qwen2.5-tiny model with lora",
    private=False
)

# from transformers import TrainingArguments
# training_args = TrainingArguments(
#     output_dir="trainer_output/wheeler404/qwen2-tiny-lora",
#     learning_rate=1e-3,
#     per_device_train_batch_size=4,
#     # gradient_accumulation_steps=4,
#     num_train_epochs=2,
#     save_strategy="no",
# )

# from datasets import load_dataset
# dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train").select(range(100))
# def tokenize(batch):
#     # fix bug: max length
#     return tokenizer(batch["text"], padding=True)

# tokenized_datasets = dataset.map(tokenize, batched=True)

# from transformers import Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,
# )

# trainer.train()