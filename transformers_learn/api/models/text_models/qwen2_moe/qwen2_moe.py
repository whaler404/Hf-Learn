from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

# config = AutoConfig.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/text_models/qwen2_moe/config.json")
model = AutoModelForCausalLM.from_config(config)

print(model)

from torchinfo import summary
summary(model)