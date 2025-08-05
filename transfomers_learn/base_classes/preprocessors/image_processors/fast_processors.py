# # model-specific image processors
# from transformers import ViTImageProcessor

# image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


# fast image processors
from transformers import ViTImageProcessorFast

image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")

from torchvision.io import read_image

images = read_image("./datasets/pipeline-cat-chonk.jpeg")
images_processed = image_processor(images, return_tensors="pt", device="cuda")

