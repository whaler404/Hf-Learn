from transformers.models.vit import ViTForImageClassification

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

print(model)

from torchinfo import summary
summary(model)