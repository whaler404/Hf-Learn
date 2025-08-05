import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, do_sample=False)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# 'Hugging Face is an open-source company 🤗\nWe are an open-source company. Our mission is to democratize AI and make it accessible to everyone. We believe that AI should be used for the benefit of humanity, not for the benefit of a'