from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2TokenizerFast

tokenizer = Qwen2TokenizerFast.from_pretrained("wheeler404/qwen2-tiny")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

print(type(tokenizer))
# <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>

result = tokenizer.apply_chat_template(chat, tokenize=False)

print(result)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Hello, how are you?<|im_end|>
# <|im_start|>assistant
# I'm doing great. How can I help you today?<|im_end|>
# <|im_start|>user
# I'd like to show off how chat templating works!<|im_end|>