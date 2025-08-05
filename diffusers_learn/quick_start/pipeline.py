from diffusers import DiffusionPipeline
# from diffusers import StableDiffusionInpaintPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
pipeline.to("cuda:5")

from diffusers import EulerDiscreteScheduler
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# print(pipeline)

image = pipeline("An image of a park in Monet style").images[0]

# 保存图像

image.save("./datasets/image_of_park_painting.png")