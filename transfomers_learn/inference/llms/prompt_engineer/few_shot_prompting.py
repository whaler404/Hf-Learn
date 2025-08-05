from transformers import pipeline
import torch

pipeline = pipeline(model="Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16, device_map="auto")

# prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
# Date: 04/12/1961
# Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon.
# Date:"""

# outputs = pipeline(prompt, max_new_tokens=12, do_sample=True, top_k=10)
# for output in outputs:
#     print(f"Result: {output['generated_text']}")
# # Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
# # Date: 04/12/1961
# # Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon.
# # Date: 09/28/1960

# chat template
messages = [
    {"role": "user", "content": "Text: The first human went into space and orbited the Earth on April 12, 1961."},
    {"role": "assistant", "content": "Date: 04/12/1961"},
    {"role": "user", "content": "Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon."}
]

prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipeline(prompt, max_new_tokens=12, do_sample=True, top_k=10)

for output in outputs:
    print(f"Result: {output['generated_text']}")