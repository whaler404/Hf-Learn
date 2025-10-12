import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=100, penalty_alpha=0.6, top_k=4)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# 'Hugging Face is an open-source company that provides a platform for building and deploying AI models.\nHugging Face is an open-source company that provides a platform for building and deploying AI models. The platform allows developers to build and deploy AI models, as well as collaborate with other developers.\nHugging Face was founded in 2019 by Thibault Wittemberg and Clément Delangue. The company is based in Paris, France.\nHugging Face has'