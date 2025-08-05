from transformers.models.blip_2.modeling_blip_2 import Blip2Model
from transformers import AutoConfig, AutoModel

# config = AutoConfig.from_pretrained("Salesforce/blip2-opt-2.7b")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/multimodal_models/blip_2/config.json")
model = AutoModel.from_config(config)

print(model)

from torchinfo import summary
summary(model)