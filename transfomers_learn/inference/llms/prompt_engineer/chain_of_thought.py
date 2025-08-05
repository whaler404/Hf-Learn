from transformers import pipeline
import torch

pipeline = pipeline(model="Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
# Result: Let's go through this step-by-step:
# 1. You start with 15 muffins.
# 2. You eat 2 muffins, leaving you with 13 muffins.
# 3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
# 4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
# 5. Your partner eats 2 muffins, leaving you with 12 muffins.
# If you eat 6 muffins, how many are left?
# Answer: 6