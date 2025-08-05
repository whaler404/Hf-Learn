import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("What is the highest peak in the world??", return_tensors="pt").to("cuda")

# contrast with high layers
outputs = model.generate(**inputs, max_new_tokens=50, dola_layers="high", do_sample=False)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# " Mount EverestMount Everest, called Himalaya in Nepali, is the world's highest peak, lying almost 9.5 kilometers above the sea level and the tallest mountain from 19,036.91 ft. The mountain was"

# contrast with specific layers
outputs = model.generate(**inputs, max_new_tokens=50, dola_layers=[18,20], do_sample=False, repetition_penalty=1.2)
result = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(result)
# " Mount EverestMount Everest, called Himalaya in Nepali, is the world's highest peak above sea level and it rises to an incredible height of 29,028 feet above the ocean. Its summit is over a mile taller than Mt"
