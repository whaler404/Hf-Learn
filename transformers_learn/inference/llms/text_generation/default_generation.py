from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", 
                                             device_map="cuda:0",
                                             quantization_config=quantization_config,
                                             )

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda:0")

generated_ids = model.generate(**model_inputs)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# "A list of colors: red, blue, green, yellow, orange, purple, pink,"
print(result)