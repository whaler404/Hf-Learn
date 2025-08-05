import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# "['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"