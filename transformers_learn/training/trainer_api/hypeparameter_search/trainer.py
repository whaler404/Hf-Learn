from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for causal language models

def tokenize(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

small_train = dataset["train"].shuffle(seed=42).select(range(100))
small_eval = dataset["test"].shuffle(seed=42).select(range(100))

from transformers import Trainer

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_args = {
    "model_name_or_path": "HuggingFaceTB/SmolLM2-135M",
    "cache_dir": None,
    "model_revision": "main",
    "use_auth_token": False,
}

def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        # config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./trainer_out/optuna",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    # compute_metrics=compute_metrics,
    model_init=model_init,
)