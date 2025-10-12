import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)

from transformers.cache_utils import DynamicCache
past_key_values = DynamicCache()
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, past_key_values=past_key_values)