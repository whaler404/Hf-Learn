from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny")

from peft import LoraConfig

peft_config_1 = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model.add_adapter(peft_config_1, adapter_name="adapter_1")

# load peft model
# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny-lora")

# using more than one adapter
from peft import LoraConfig

peft_config_2 = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.2,
    r=32,
    bias="none",
    task_type="CAUSAL_LM"
)

print(model.active_adapters())
# ['adapter_1']
model.add_adapter(peft_config_2, adapter_name="adapter_2")
print(model.active_adapters())
# ['adapter_2']

model.set_adapter("adapter_1")
print(model.active_adapters())
# ['adapter_1']

model.disable_adapters()
print(model.active_adapters())
# ['adapter_1']

model.enable_adapters()
print(model.active_adapters())
# ['adapter_1']

from peft import LoHaConfig

adapter_3 = LoHaConfig()

model.add_adapter(adapter_3, adapter_name="adapter_3")
print(model.active_adapters())
# ['adapter_3']