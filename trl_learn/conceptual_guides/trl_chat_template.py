from transformers import AutoTokenizer
from trl import apply_chat_template

# 加载预训练的LLaMA模型分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# ==================== Prompt-Only 类型示例 ====================
# 仅包含用户提示的示例
# 适用于需要模型生成回复的场景
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
result = apply_chat_template(prompt_only_example, tokenizer)
for key in result:
    print(f"{key}: {result[key]}")

# 预期输出格式:
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Cutting Knowledge Date: December 2023
# Today Date: 05 Aug 2025
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# What color is the sky?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# ==================== Language Modeling 类型示例 ====================
# 包含完整对话消息的示例
# 适用于语言模型训练场景
lm_example = {"messages": [{"role": "user", "content": "What color is the sky?"}]}
result = apply_chat_template(lm_example, tokenizer)
for key in result:
    print(f"{key}: {result[key]}")

# 预期输出格式:
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Cutting Knowledge Date: December 2023
# Today Date: 05 Aug 2025
# <|eot_id|><|start_header_id|>user<|end_header_id|>