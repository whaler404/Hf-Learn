import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda")
# explicitly set to default length because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=20)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# 'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'