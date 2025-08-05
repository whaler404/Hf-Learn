from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", device_map="auto")

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# `custom_generate` replaces the original `generate` by the custom decoding method defined in
# `transformers-community/custom_generate_example`

from custom_generate import generate
# gen_out = model.generate(**inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True)
custom_path = "./transfomers_learn/inference/llms/generation_strategies/custom_decoding_methods"
gen_out = model.generate(**inputs, custom_generate=custom_path, trust_remote_code=True)
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
# 'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'