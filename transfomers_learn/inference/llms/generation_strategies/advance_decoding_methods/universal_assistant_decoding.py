from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"

assistant_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
# ['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']