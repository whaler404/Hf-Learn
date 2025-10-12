# # 1. auto tokenizer
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# result = tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt")
# print(result)

# 2. model-specific tokenizer
# from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
# result = tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt")
# print(result)

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
result = tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt")
print(result)

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
# result = tokenizer.tokenize("We are very happy to show you the 🤗 Transformers library")
# print(result)