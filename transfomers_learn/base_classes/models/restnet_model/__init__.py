from configuration_resnet import ResnetConfig
from modeling_resnet import ResnetModel, ResnetModelForImageClassification

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("custom-resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)