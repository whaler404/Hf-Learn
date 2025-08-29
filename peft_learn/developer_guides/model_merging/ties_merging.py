from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny").eval()
tokenizer = AutoTokenizer.from_pretrained("wheeler404/qwen2-tiny")

model.config.vocab_size = 151940    # 151936
model.resize_token_embeddings(151940)

model = PeftModel.from_pretrained(model, "wheeler404/qwen2-tiny-lora", adapter_name="adapter_1")

from peft import LoraConfig, TaskType

peft_config_2 = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

peft_config_3 = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model.add_adapter(peft_config=peft_config_2, adapter_name="adapter_2")
model.add_adapter(peft_config=peft_config_3, adapter_name="adapter_3")

print(model.active_adapter)
# adapter_1
model.set_adapter("adapter_2")
print(model.active_adapter)
# adapter_2

adapters = ["adapter_1", "adapter_2", "adapter_3"]
weights = [2.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
# model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
model.set_adapter("merge")

print(model.active_adapter)
# merge

print(model.config.vocab_size)
# 151940