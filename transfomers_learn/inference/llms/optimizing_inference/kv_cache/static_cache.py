from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype="auto", device_map="cuda:0")

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)
prompt_length = input_ids.input_ids.shape[1]
model.generation_config.max_new_tokens = 16

past_key_values = StaticCache(
    config=model.config,
    max_batch_size=1,
    # If you plan to reuse the cache, make sure the cache length is large enough for all cases
    max_cache_len=prompt_length+(model.generation_config.max_new_tokens*2),
    device=model.device,
    dtype=model.dtype
)
outputs = model.generate(**input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference frames. 2']

# pass in the generated text and the same cache object to continue generation from where it left off. Optionally, in a
# multi-turn conversation, append the new user input to the generated text.
new_input_ids = outputs
outputs = model.generate(new_input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['The theory of special relativity states 1. The speed of light is constant in all inertial reference frames. 2. The speed of light is constant in all inertial reference frames. 3.']