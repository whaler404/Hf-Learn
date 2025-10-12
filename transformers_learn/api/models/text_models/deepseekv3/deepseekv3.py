from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

# config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/text_models/deepseekv3/config.json")
model = AutoModelForCausalLM.from_config(config)

print(model)

from torchinfo import summary
summary(model)