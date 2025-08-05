import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=20, prompt_lookup_num_tokens=5)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# 'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'