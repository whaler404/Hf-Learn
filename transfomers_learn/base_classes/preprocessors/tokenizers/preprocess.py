from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

import os
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=os.environ.get("HF_TOKEN"))
# from tokenizers.pre_tokenizers import Whitespace
# tokenizer.pre_tokenizer = Whitespace()
tokenizer("We are very happy to show you the 🤗 Transformers library.", return_tensors="pt")

# 1. tokenize
tokens = tokenizer.tokenize("We are very happy to show you the 🤗 Transformers library")
print(tokens)

# 2. convert tokens to ids
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

# 3. decode ids back to text
decoded_string = tokenizer.decode(token_ids)
print(decoded_string)

# special tokens
model_inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(model_inputs)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
# decode with special tokens
decoded_string_with_special_tokens = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
print(decoded_string_with_special_tokens)

# batch tokenization
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True)
print(encoded_inputs)

# padding
encoded_inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")
print(encoded_inputs)

# truncation
encoded_inputs = tokenizer(batch_sentences, padding=True, max_length=8, truncation=True, return_tensors="pt")
print(encoded_inputs)