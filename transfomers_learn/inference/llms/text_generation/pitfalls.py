from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", 
                                             device_map="cuda:0",
                                             )

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", padding_side="left")

# output length
print('Output length')
model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)
# 'A sequence of numbers: 1, 2, 3, 4, 5'

# decode strategy
model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")
# greedy decoding
print('Greedy decoding')
generated_ids = model.generate(**model_inputs)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)
# multinomial sampling
print('Multinomial sampling')
generated_ids = model.generate(**model_inputs, do_sample=True)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)

# padding
# right pad
print('Right pad')
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", padding_side="right")
tokenizer.pad_token = tokenizer.bos_token
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to("cuda")
generated_ids = model.generate(**model_inputs)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)
# '1, 2, 33333333333'

# left pad
print('Left pad')
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to("cuda")
generated_ids = model.generate(**model_inputs)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)
# '1, 2, 3, 4, 5, 6,'

# prompt format

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", device_map="cuda:0", #load_in_4bit=True
)

# no format
print('No format')
prompt = """How many cats does it take to change a light bulb? Reply as a pirate."""
model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=True)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
# "Aye, matey! 'Tis a simple task for a cat with a keen eye and nimble paws. First, the cat will climb up the ladder, carefully avoiding the rickety rungs. Then, with"

# chat template
print('Chat template')
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many cats does it take to change a light bulb?"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
input_length = model_inputs.shape[1]
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
# "Arr, matey! According to me beliefs, 'twas always one cat to hold the ladder and another to climb up it an’ change the light bulb, but if yer looking to save some catnip, maybe yer can