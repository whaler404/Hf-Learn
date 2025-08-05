from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

# config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/text_models/qwen3/config.json")
model = AutoModelForCausalLM.from_config(config)

print(model)

from torchinfo import summary
summary(model)