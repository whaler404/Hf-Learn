# from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

# config = MaskFormerConfig(backbone="resnet50", use_timm_backbone=True, use_pretrained_backbone=True)
# model = MaskFormerForInstanceSegmentation(config)

# or

from transformers import TimmBackboneConfig

backbone_config = TimmBackboneConfig("resnet50", use_pretrained_backbone=True)
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)