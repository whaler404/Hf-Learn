from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", 
                                             device_map="cuda:0",
                                             )
print(model.generation_config)
# GenerationConfig {
#   "bos_token_id": 1,
#   "eos_token_id": 2
# }

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda:0")

from transformers import GenerationConfig
generation_config = GenerationConfig(
    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
)

generated_ids = model.generate(**model_inputs, generation_config=generation_config)
result = tokenizer.batch_decode(generated_ids)[0]
# "A list of colors: red, blue, green, yellow, orange, purple, pink,"
print(result)