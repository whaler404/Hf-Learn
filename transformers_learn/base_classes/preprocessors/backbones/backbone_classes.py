# # auto backbone
# from transformers import AutoImageProcessor, AutoBackbone

# model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))

# # specific-model backbone

# # load a ResNet backbone and neck for use in a MaskFormer instance segmentation head.

# from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

# config = MaskFormerConfig(backbone="microsoft/resnet-50", use_pretrained_backbone=True)
# model = MaskFormerForInstanceSegmentation(config)

from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig

# instantiate backbone configuration
backbone_config = ResNetConfig()
# load backbone in model
config = MaskFormerConfig(backbone_config=backbone_config)
# attach backbone to model head
model = MaskFormerForInstanceSegmentation(config)