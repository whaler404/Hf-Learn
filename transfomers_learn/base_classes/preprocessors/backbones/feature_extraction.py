from transformers import AutoImageProcessor, AutoBackbone
import torch
from PIL import Image
import requests

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

with open("./datasets/pipeline-cat-chonk.jpeg", "rb") as image_file:
    image = Image.open(image_file).convert("RGB")

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)

feature_maps = outputs.feature_maps
print(list(feature_maps[0].shape))