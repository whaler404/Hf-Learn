from transformers.models.qwen3_next import Qwen3NextForCausalLM

from transformers import AutoConfig
from transformers import AutoModelForCausalLM

config = AutoConfig.from_pretrained("./transfomers_learn/api/models/text_models/qwen3next/config.json")
model = AutoModelForCausalLM.from_config(config)

print(model)

from torchinfo import summary
summary(model)