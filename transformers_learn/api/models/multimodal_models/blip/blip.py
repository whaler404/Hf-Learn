from transformers.models.blip.modeling_blip import BlipModel
from transformers import AutoConfig, AutoModel

# config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/multimodal_models/blip/config.json")
model = AutoModel.from_config(config)

print(model)

from torchinfo import summary
summary(model)