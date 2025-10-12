from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from transformers import AutoConfig, AutoModel

# config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/multimodal_models/qwen2.5_vl/config.json")
model = AutoModel.from_config(config)

print(model)

from torchinfo import summary
summary(model)