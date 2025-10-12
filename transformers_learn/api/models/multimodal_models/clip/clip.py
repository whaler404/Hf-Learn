from transformers.models.clip import CLIPModel
from transformers import AutoConfig, AutoModel

# config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/multimodal_models/clip/config.json")
model = AutoModel.from_config(config)

print(model)

from torchinfo import summary
summary(model)