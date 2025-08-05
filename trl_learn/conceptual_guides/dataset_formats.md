# 数据集格式概览

| 类型/格式 | 标准格式 | 对话格式 |
|----------|----------|----------|
| **Language modeling** | `{"text": "The sky is blue."}` | `{"messages": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}]}` |
| **Prompt-only** | `{"prompt": "The sky is"}` | `{"prompt": [{"role": "user", "content": "What color is the sky?"}]}` |
| **Prompt-completion** | `{"prompt": "The sky is", "completion": " blue."}` | `{"prompt": [{"role": "user", "content": "What color is the sky?"}], "completion": [{"role": "assistant", "content": "It is blue."}]}` |
| **Preference** | `{"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}` 或隐式提示：`{"chosen": "The sky is blue.", "rejected": "The sky is green."}` | `{"prompt": [{"role": "user", "content": "What color is the sky?"}], "chosen": [{"role": "assistant", "content": "It is blue."}], "rejected": [{"role": "assistant", "content": "It is green."}]}` 或隐式提示：`{"chosen": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}], "rejected": [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}]}` |
| **Unpaired preference** | `{"prompt": "The sky is", "completion": " blue.", "label": True}` | `{"prompt": [{"role": "user", "content": "What color is the sky?"}], "completion": [{"role": "assistant", "content": "It is green."}], "label": False}` |
| **Stepwise supervision** | `{"prompt": "Which number is larger, 9.8 or 9.11?", "completions": ["The fractional part of 9.8 is 0.8.", "The fractional part of 9.11 is 0.11.", "0.11 is greater than 0.8.", "Hence, 9.11 > 9.8."], "labels": [True, True, False, False]}` | |



标准数据集格式：纯文本
```python
# 自然语言建模
language_modeling_example = {"text": "The sky is blue."}
# 偏好
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# 无配对偏好
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
```

对话式：用户和助手的聊天交互
```python
# 提示词补全
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
# 偏好
preference_example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

对话数据集对训练聊天模型非常有用，但是需要转化为标准格式才能用于 trl 训练器

# 工具调用

为带有工具调用的 sft 准备数据集时，数据集必须包含 tools 列，用 json 的方式指定

```python
messages = [
    {"role": "user", "content": "Turn on the living room lights."},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {
            "name": "control_light",
            "arguments": {"room": "living room", "state": "on"}
        }}]
    },
    {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
    {"role": "assistant", "content": "Done!"}
]
```

# 数据集适用

| 训练器 | 所需数据集类型 |
|--------|----------------|
| **BCOTrainer** | Unpaired preference |
| **CPOTrainer** | Preference（推荐显式提示） |
| **DPOTrainer** | Preference（推荐显式提示） |
| **GKDTrainer** | Prompt-completion |
| **GRPOTrainer** | Prompt-only |
| **IterativeSFTTrainer** | Unpaired preference |
| **KTOTrainer** | Unpaired preference 或 Preference（推荐显式提示） |
| **NashMDTrainer** | Prompt-only |
| **OnlineDPOTrainer** | Prompt-only |
| **ORPOTrainer** | Preference（推荐显式提示） |
| **PPOTrainer** | Tokenized language modeling |
| **PRMTrainer** | Stepwise supervision |
| **RewardTrainer** | Preference（推荐隐式提示） |
| **SFTTrainer** | Language modeling 或 Prompt-completion |
| **XPOTrainer** | Prompt-only |

目前 trl 仅支持标准数据集格式，需要将对话数据集转化为标准数据集，请使用 `apply_chat_template()` 方法

聊天模板特定于模型，trl 提供了不同类型数据集之间转换的模板