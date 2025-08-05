from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("timm/resnet50.a1_in1k", use_fast=True)

from PIL import Image

with open("./datasets/pipeline-cat-chonk.jpeg", "rb") as image_file:
    image = Image.open(image_file).convert("RGB")
inputs = image_processor(image, return_tensors="pt")