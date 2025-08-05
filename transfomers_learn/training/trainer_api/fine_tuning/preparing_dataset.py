from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for causal language models

def tokenize(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# small_train = dataset["train"].shuffle(seed=42).select(range(1000))
# small_eval = dataset["test"].shuffle(seed=42).select(range(1000))